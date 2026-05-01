from pathlib import Path
import sys

import enue_sat


def iter_cnf_files(data_dir: Path):
    return sorted(data_dir.rglob("*.cnf"))


def output_path_for(data_dir: Path, output_dir: Path, cnf_path: Path) -> Path:
    rel = cnf_path.relative_to(data_dir)
    return (output_dir / rel).with_suffix(".csv")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon = 0.1
    seed = 0

    cnf_files = iter_cnf_files(data_dir)
    if not cnf_files:
        print(f"No .cnf files found under {data_dir}")
        return 1

    total = 0
    sat_count = 0
    for cnf_path in cnf_files:
        out_path = output_path_for(data_dir, output_dir, cnf_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            is_sat = enue_sat.solve_dimacs(
                str(cnf_path),
                str(out_path),
                epsilon=epsilon,
                seed=seed,
            )
        except Exception as exc:
            print(f"Failed on {cnf_path}: {exc}")
            continue

        total += 1
        if is_sat:
            sat_count += 1

        if total % 100 == 0:
            print(f"Processed {total} files ({sat_count} SAT)")

    print(f"Done. Files: {total}, SAT: {sat_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
