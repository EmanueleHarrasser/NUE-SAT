from __future__ import annotations

import argparse
import random
import shutil
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from pathlib import Path as _Path

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


def sample_clause_length(k_min: int, k_max: int, alpha: float, rng: random.Random) -> int:
    weights = [k ** (-alpha) for k in range(k_min, k_max + 1)]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for k, w in zip(range(k_min, k_max + 1), weights):
        acc += w
        if r <= acc:
            return k
    return k_max


def make_clause(num_vars: int, k_min: int, k_max: int, alpha: float, rng: random.Random) -> list[int]:
    k = sample_clause_length(k_min, k_max, alpha, rng)
    vars_sample = rng.sample(range(1, num_vars + 1), k)
    clause = []
    for var in vars_sample:
        sign = rng.choice([1, -1])
        clause.append(sign * var)
    return clause


def cnf_text(num_vars: int, clauses: list[list[int]]) -> str:
    lines = [f"p cnf {num_vars} {len(clauses)}"]
    for clause in clauses:
        lines.append(" ".join(str(lit) for lit in clause) + " 0")
    return "\n".join(lines) + "\n"


def clause_count(num_vars: int, ratio_min: float, ratio_max: float, rng: random.Random) -> int:
    ratio = rng.uniform(ratio_min, ratio_max)
    return max(1, int(round(ratio * num_vars)))


def generate_random_instance(
    num_vars: int,
    num_clauses: int,
    k_min: int,
    k_max: int,
    alpha: float,
    rng: random.Random,
) -> str:
    clauses = [make_clause(num_vars, k_min, k_max, alpha, rng) for _ in range(num_clauses)]
    return cnf_text(num_vars, clauses)


def write_instance(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def generate_cnf_dataset(
    out_dir: Path,
    count: int,
    ratio_min: float,
    ratio_max: float,
    k_min: int,
    k_max: int,
    k_alpha: float,
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

    bar = tqdm(total=count, desc="Generate CNFs", unit="cnf")
    try:
        while sat_count < sat_target or unsat_count < unsat_target:
            if attempts >= max_attempts:
                raise SystemExit(
                    f"Reached max attempts ({max_attempts}) with SAT {sat_count} / UNSAT {unsat_count}"
                )

            num_vars = rng.randint(vars_min, vars_max)
            num_clauses = clause_count(num_vars, ratio_min, ratio_max, rng)
            text = generate_random_instance(num_vars, num_clauses, k_min, k_max, k_alpha, rng)
            is_sat = enue_sat.solve_cnf(text, None, epsilon=0.0, seed=0)
            attempts += 1

            if is_sat and sat_count < sat_target:
                sat_count += 1
                path = out_dir / f"sr_{sat_count}_u{num_vars}_c{num_clauses}_sat.cnf"
                write_instance(path, text)
                bar.update(1)
            elif (not is_sat) and unsat_count < unsat_target:
                unsat_count += 1
                path = out_dir / f"sr_{unsat_count}_u{num_vars}_c{num_clauses}_unsat.cnf"
                write_instance(path, text)
                bar.update(1)
    finally:
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
    
    ok, base_decisions, new_decisions = enue_sat.perturb_dimacs_network(
        str(cnf_path),
        str(out_path),
        str(network_bin),
        seed=seed,
        bias_exp=bias_exp,
        top_k=top_k,
        top_prob=top_prob,
    )
    return ok


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

    ratio_min = 1.5
    ratio_max = 3.0
    k_min = 2
    k_max = 4
    k_alpha = 1.5
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
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            k_min=k_min,
            k_max=k_max,
            k_alpha=k_alpha,
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