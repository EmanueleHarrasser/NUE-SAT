from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from pathlib import Path as _Path
import shutil

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import enue_sat
from tqdm import tqdm
from training import train


def collect_cnf_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.cnf"))


def iter_output_path(cnf_root: Path, out_dir: Path, cnf_path: Path, iteration: int) -> Path:
    if cnf_root.is_file():
        return (out_dir / f"iter_{iteration}" / cnf_path.name).with_suffix(".csv")
    rel = cnf_path.relative_to(cnf_root)
    return (out_dir / f"iter_{iteration}" / rel).with_suffix(".csv")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def enforce_sliding_window(buffer_dir: Path, current_iter: int, window_size: int) -> None:
    if current_iter >= window_size:
        obsolete_iter = current_iter - window_size
        obsolete_dir = buffer_dir / f"iter_{obsolete_iter}"
        assert obsolete_dir.is_dir(), f"Invariant violation: Expected obsolete directory {obsolete_dir} to exist."
        shutil.rmtree(obsolete_dir)
        print(f"🧹 Sliding Window: Deleted obsolete data in {obsolete_dir.name}")


import sys
import subprocess
import random

def generate_cnfgen_instance(vars_min: int, vars_max: int, rng: random.Random) -> str:
    """
    Uses CNFgen CLI to create a structured 3-Coloring problem on a gnm graph.
    Targets the phase transition (avg degree ~4.7) for a 50/50 SAT/UNSAT mix.
    """
    # 3-coloring creates 3 variables per vertex
    # Enforce minimum of 6 vertices so avg degree of 4.7 is mathematically possible
    v_min = max(6, vars_min // 3)
    v_max = max(7, vars_max // 3)

    vertices = rng.randint(v_min, v_max)

    # 3-Coloring phase transition is at average degree ≈ 4.7
    # Total Edges = (Vertices * Average Degree) / 2
    edges = int((vertices * 4.7) / 2)

    # Safety check: Cap edges at the absolute mathematical maximum for N vertices
    max_edges = (vertices * (vertices - 1)) // 2
    edges = min(edges, max_edges)

    cnfgen_exe = shutil.which("cnfgen")
    # Use sys.executable to prevent Windows subprocess -1 exit codes
    # Command structure for gnm: cnfgen kcolor 3 gnm <vertices> <edges>
    
    cmd = [
        cnfgen_exe,
        "kcolor", "3",
        "gnm", str(vertices), str(edges)
    ]

    try:
        # Call the CLI and capture the output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n--- CNFgen Failed ---")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error output:\n{e.stderr}")
        raise


def generate_and_solve_worker(vars_min: int, vars_max: int, seed: int) -> tuple[bool, int, str]:
    """Worker function to generate and solve a single CNF in a separate process."""
    rng = random.Random(seed)

    text = generate_cnfgen_instance(vars_min, vars_max, rng)

    is_sat = enue_sat.solve_cnf(text, None, epsilon=0.0, seed=0)

    num_vars = 0
    for line in text.splitlines():
        if line.startswith("p cnf "):
            num_vars = int(line.split()[2])
            break

    if num_vars == 0:
        raise ValueError("Could not find 'p cnf' header in CNFgen output")

    return is_sat, num_vars, text


def write_instance(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def generate_cnf_dataset(
    out_dir: Path,
    count: int,
    max_attempts: int,
    vars_min: int,
    vars_max: int,
    rng: random.Random,
) -> None:
    sat_target = count // 2
    unsat_target = count - sat_target
    sat_count = 0
    unsat_count = 0
    attempts = 0

    bar = tqdm(total=count, desc="Generate CNFs (Multi-Core)", unit="cnf")

    max_workers = os.cpu_count() or 4

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = set()

        for _ in range(max_workers * 2):
            if attempts < max_attempts:
                futures.add(
                    executor.submit(
                        generate_and_solve_worker,
                        vars_min,
                        vars_max,
                        rng.randint(0, 2**32 - 1),
                    )
                )
                attempts += 1

        while futures and (sat_count < sat_target or unsat_count < unsat_target):
            done, futures = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                try:
                    is_sat, num_vars, text = future.result()

                    if is_sat and sat_count < sat_target:
                        sat_count += 1
                        path = out_dir / f"kcolor_{sat_count}_v{num_vars}_sat.cnf"
                        write_instance(path, text)
                        bar.update(1)
                    elif (not is_sat) and unsat_count < unsat_target:
                        unsat_count += 1
                        path = out_dir / f"kcolor_{unsat_count}_v{num_vars}_unsat.cnf"
                        write_instance(path, text)
                        bar.update(1)

                except Exception as e:
                    print(f"\nWorker failed: {e}")

                if sat_count < sat_target or unsat_count < unsat_target:
                    if attempts < max_attempts:
                        futures.add(
                            executor.submit(
                                generate_and_solve_worker,
                                vars_min,
                                vars_max,
                                rng.randint(0, 2**32 - 1),
                            )
                        )
                        attempts += 1
                    elif not futures:
                        raise SystemExit(
                            f"\nReached max attempts ({max_attempts}) with SAT {sat_count} / UNSAT {unsat_count}"
                        )

    for future in futures:
        future.cancel()

    bar.close()


def process_single_cnf(
    cnf_path: Path, 
    cnf_root: Path, 
    buffer_dir: Path, 
    network_bin: Path, 
    seed: int, 
    bias_exp: float, 
    top_k: int,
    top_prob: float,
    iteration: int
) -> bool:
    """Worker function executed by ProcessPoolExecutor to process a single CNF."""
    out_path = iter_output_path(cnf_root, buffer_dir, cnf_path, iteration)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok, _, _ = enue_sat.perturb_dimacs_network(
            str(cnf_path),
            str(out_path),
            str(network_bin),
            seed=seed,
            bias_exp=bias_exp,
            top_k=top_k,
            top_prob=top_prob,
        )
        return ok
    except Exception as e:
        print(f"\nWorker failed for {cnf_path}: {e}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cnf-count", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    cnf_root = root / "data/rl_iter"
    buffer_dir = root / "output/rl_buffer"
    model_dir = root / "models"
    network_bin = model_dir / "network.bin"
    init_path = model_dir / "network.pth"
    best_overall = None

    max_attempts = 50000
    vars_min = 20
    vars_max = 100
    bias_exp = 2.0
    top_k = 5
    top_prob = 0
    epochs = 10
    batch_pairs = 256
    margin = 1.0
    lr = 5e-4
    window_size = 5

    rng = random.Random(args.seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    reset_dir(buffer_dir)

    for iteration in range(args.iterations):
        reset_dir(cnf_root)
        generate_cnf_dataset(
            out_dir=cnf_root,
            count=args.cnf_count,
            max_attempts=max_attempts,
            vars_min=vars_min,
            vars_max=vars_max,
            rng=rng,
        )

        cnf_files = collect_cnf_files(cnf_root)
        logged = 0
        
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_single_cnf,
                    cnf_path,
                    cnf_root,
                    buffer_dir,
                    network_bin,
                    args.seed,
                    bias_exp,
                    top_k,
                    top_prob,
                    iteration
                )
                for cnf_path in cnf_files
            ]
            
            for future in tqdm(as_completed(futures), total=len(cnf_files), desc=f"Iter {iteration} logs", unit="cnf"):
                if future.result():
                    logged += 1

        print(f"Iter {iteration}: logged {logged} / {len(cnf_files)}")

        enforce_sliding_window(buffer_dir, iteration, window_size)

        best_pth, best_bin, best_val, last_train_loss, last_val_loss = train.train_model(
            data_dir=buffer_dir,
            model_dir=model_dir,
            init_path=init_path,
            epochs=epochs,
            batch_pairs=batch_pairs,
            margin=margin,
            lr=lr,
            seed=args.seed,
            best_name="network_rl_iter_best",
            latest_name="network_rl_latest",
        )

        print(
            "Iter {}: train {:.4f} val {:.4f} best {:.4f}".format(
                iteration,
                last_train_loss,
                last_val_loss,
                best_val,
            )
        )

        
        network_bin = model_dir / "network_rl_best.bin"
        init_path = model_dir / "network_rl_best.pth"

        if best_overall is None or best_val < best_overall:
            shutil.copy2(best_pth, model_dir / "network_rl_best.pth")
            shutil.copy2(best_bin, model_dir / "network_rl_best.bin")
            best_overall = best_val

    return 0


if __name__ == "__main__":
    raise SystemExit(main())