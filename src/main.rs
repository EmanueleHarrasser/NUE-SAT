use std::env;
use std::fs;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: enue-sat <path.cnf>");
        std::process::exit(2);
    });

    let input = fs::read_to_string(&path).unwrap_or_else(|err| {
        eprintln!("Failed to read {}: {}", path, err);
        std::process::exit(2);
    });

    let cnf = match enue_sat::parser::dimacs::parse_dimacs(&input) {
        Ok(cnf) => cnf,
        Err(err) => {
            eprintln!("Parse error: {}", err);
            std::process::exit(2);
        }
    };

    let result = enue_sat::solver::solve(&cnf);
    if result.is_some() {
        println!("SAT");
    } else {
        println!("UNSAT");
    }
}
