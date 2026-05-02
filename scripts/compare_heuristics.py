from __future__ import annotations

import argparse
import time
from pathlib import Path

import enue_sat


def collect_cnf_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.cnf"))


def run_solver(
    cnf_files: list[Path],
    heuristic: str,
    nnue_path: str | None,
) -> tuple[int, int, float, float, float, float]:
    sat_count = 0
    unsat_count = 0
    total_decisions = 0
    total_backtracks = 0
    total_conflicts = 0
    start = time.perf_counter()

    for cnf_path in cnf_files:
        is_sat, decisions, backtracks, conflicts = enue_sat.solve_dimacs_stats(
            str(cnf_path),
            None,
            epsilon=0.0,
            seed=0,
            heuristic=heuristic,
            nnue_path=nnue_path,
        )
        if is_sat:
            sat_count += 1
        else:
            unsat_count += 1

        total_decisions += decisions
        total_backtracks += backtracks
        total_conflicts += conflicts

    elapsed = time.perf_counter() - start
    total = len(cnf_files)
    avg_decisions = total_decisions / total
    avg_backtracks = total_backtracks / total
    avg_conflicts = total_conflicts / total
    return sat_count, unsat_count, elapsed, avg_decisions, avg_backtracks, avg_conflicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cnf_path", help="File or directory containing .cnf files")
    parser.add_argument("--nnue", required=True, help="Path to NNUE .bin weights")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cnf_root = Path(args.cnf_path)
    cnf_files = collect_cnf_files(cnf_root)
    if not cnf_files:
        raise SystemExit(f"No .cnf files found under {cnf_root}")

    jw_sat, jw_unsat, jw_time, jw_dec, jw_back, jw_conf = run_solver(cnf_files, "jw", None)
    nn_sat, nn_unsat, nn_time, nn_dec, nn_back, nn_conf = run_solver(cnf_files, "nnue", args.nnue)

    total = len(cnf_files)
    jw_avg = jw_time / total
    nn_avg = nn_time / total

    print("JW (epsilon=0)")
    print(f"  SAT: {jw_sat}  UNSAT: {jw_unsat}  total: {jw_time:.3f}s  avg: {jw_avg:.6f}s")
    print(f"  avg decisions: {jw_dec:.2f}  avg backtracks: {jw_back:.2f}  avg conflicts: {jw_conf:.2f}")
    print("NNUE")
    print(f"  SAT: {nn_sat}  UNSAT: {nn_unsat}  total: {nn_time:.3f}s  avg: {nn_avg:.6f}s")
    print(f"  avg decisions: {nn_dec:.2f}  avg backtracks: {nn_back:.2f}  avg conflicts: {nn_conf:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
