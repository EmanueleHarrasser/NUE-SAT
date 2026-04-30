use enue_sat::cnf::{Cnf, Lit};
use enue_sat::solver;

fn is_model(cnf: &Cnf, model: &solver::Model) -> bool {
    for clause in &cnf.clauses {
        if !clause
            .iter()
            .any(|&lit| model.eval_lit(lit) == Some(true))
        {
            return false;
        }
    }
    true
}

#[test]
fn solves_sat_formula() {
    let cnf = Cnf::new(
        2,
        vec![
            vec![Lit::from_dimacs(1), Lit::from_dimacs(2)],
            vec![Lit::from_dimacs(-1), Lit::from_dimacs(2)],
        ],
    );

    let model = solver::solve(&cnf).expect("sat");
    assert!(is_model(&cnf, &model));
}

#[test]
fn solves_unsat_formula() {
    let cnf = Cnf::new(
        1,
        vec![vec![Lit::from_dimacs(1)], vec![Lit::from_dimacs(-1)]],
    );

    assert!(solver::solve(&cnf).is_none());
}
