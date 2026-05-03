from pathlib import Path
import argparse

import enue_sat


def iter_cnf_files(data_dir: Path):
    return sorted(data_dir.rglob("*.cnf"))


def output_path_for(data_dir: Path, output_dir: Path, cnf_path: Path) -> Path:
    rel = cnf_path.relative_to(data_dir)
    return (output_dir / rel).with_suffix(".csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bias-exp", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--perturb", action="store_true")
    group.add_argument("--no-perturb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    use_perturb = not args.no_perturb

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cnf_files = iter_cnf_files(data_dir)
    if not cnf_files:
        print(f"No .cnf files found under {data_dir}")
        return 1

    total = 0
    sat_count = 0
    improved_count = 0
    worse_count = 0
    equal_count = 0
    delta_sum = 0
    for cnf_path in cnf_files:
        out_path = output_path_for(data_dir, output_dir, cnf_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if use_perturb:
                logged, base_decisions, new_decisions = enue_sat.perturb_dimacs(
                    str(cnf_path),
                    str(out_path),
                    seed=args.seed,
                    bias_exp=args.bias_exp,
                )
                if not logged:
                    continue
                total += 1
                delta = int(new_decisions) - int(base_decisions)
                delta_sum += delta
                if new_decisions < base_decisions:
                    improved_count += 1
                elif new_decisions > base_decisions:
                    worse_count += 1
                else:
                    equal_count += 1
                if new_decisions < base_decisions:
                    sat_count += 1
            else:
                is_sat = enue_sat.solve_dimacs(
                    str(cnf_path),
                    str(out_path),
                    epsilon=args.epsilon,
                    seed=args.seed,
                )
                total += 1
                if is_sat:
                    sat_count += 1
        except Exception as exc:
            print(f"Failed on {cnf_path}: {exc}")
            continue

        if total % 100 == 0:
            label = "improved" if use_perturb else "SAT"
            print(f"Processed {total} files ({sat_count} {label})")

    label = "improved" if use_perturb else "SAT"
    print(f"Done. Files: {total}, {label}: {sat_count}")
    if use_perturb and total > 0:
        avg_delta = delta_sum / total
        print(
            "Perturbation vs JW: "
            f"better={improved_count} worse={worse_count} equal={equal_count} "
            f"avg_delta={avg_delta:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
