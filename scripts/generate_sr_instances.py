from __future__ import annotations

import argparse
import random
from pathlib import Path

import enue_sat


def next_sr_dir(data_dir: Path) -> Path:
    n = 1
    while True:
        candidate = data_dir / f"sr_{n}"
        if not candidate.exists():
            return candidate
        n += 1


def sample_clause_length(k_min: int, k_max: int, alpha: float) -> int:
    weights = [k ** (-alpha) for k in range(k_min, k_max + 1)]
    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for k, w in zip(range(k_min, k_max + 1), weights):
        acc += w
        if r <= acc:
            return k
    return k_max


def make_clause(num_vars: int, k_min: int, k_max: int, alpha: float) -> list[int]:
    k = sample_clause_length(k_min, k_max, alpha)
    vars_sample = random.sample(range(1, num_vars + 1), k)
    clause = []
    for var in vars_sample:
        sign = random.choice([1, -1])
        clause.append(sign * var)
    return clause


def cnf_text(num_vars: int, clauses: list[list[int]]) -> str:
    lines = [f"p cnf {num_vars} {len(clauses)}"]
    for clause in clauses:
        lines.append(" ".join(str(lit) for lit in clause) + " 0")
    return "\n".join(lines) + "\n"


def random_ratio(ratio_min: float, ratio_max: float) -> float:
    return random.uniform(ratio_min, ratio_max)


def clause_count(num_vars: int, mode: str, fixed: int, ratio_min: float, ratio_max: float) -> int:
    if mode == "fixed":
        return fixed
    ratio = random_ratio(ratio_min, ratio_max)
    return max(1, int(round(ratio * num_vars)))


def generate_random_instance(
    num_vars: int,
    num_clauses: int,
    k_min: int,
    k_max: int,
    alpha: float,
) -> str:
    clauses = [make_clause(num_vars, k_min, k_max, alpha) for _ in range(num_clauses)]
    return cnf_text(num_vars, clauses)


def write_instance(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--clauses", type=int, default=1000)
    parser.add_argument("--mode", choices=["ratio", "fixed"], default="ratio")
    parser.add_argument("--ratio-min", type=float, default=1.5)
    parser.add_argument("--ratio-max", type=float, default=3.5)
    parser.add_argument("--vars-min", type=int, default=20)
    parser.add_argument("--vars-max", type=int, default=100)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--k-alpha", type=float, default=1.5)
    parser.add_argument("--max-attempts", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    output_dir = next_sr_dir(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sat_target = args.count // 2
    unsat_target = args.count - sat_target
    sat_count = 0
    unsat_count = 0
    attempts = 0

    while sat_count < sat_target or unsat_count < unsat_target:
        if attempts >= args.max_attempts:
            raise SystemExit(
                f"Reached max attempts ({args.max_attempts}) with SAT {sat_count} / UNSAT {unsat_count}"
            )

        num_vars = random.randint(args.vars_min, args.vars_max)
        num_clauses = clause_count(num_vars, args.mode, args.clauses, args.ratio_min, args.ratio_max)
        text = generate_random_instance(num_vars, num_clauses, args.k_min, args.k_max, args.k_alpha)
        is_sat = enue_sat.solve_cnf(text, None, epsilon=0.0, seed=0)
        attempts += 1

        if is_sat and sat_count < sat_target:
            sat_count += 1
            path = output_dir / f"sr_{sat_count}_u{num_vars}_c{num_clauses}_sat.cnf"
            write_instance(path, text)
        elif (not is_sat) and unsat_count < unsat_target:
            unsat_count += 1
            path = output_dir / f"sr_{unsat_count}_u{num_vars}_c{num_clauses}_unsat.cnf"
            write_instance(path, text)

    print(f"Wrote {args.count} instances to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
