use crate::cnf::Cnf;
use crate::solver::assignment::Assignment;
use crate::solver::{branching, propagation};

pub fn solve(cnf: &Cnf, assignment: Assignment) -> Option<Assignment> {
    let mut assignment = assignment;

    if !propagation::unit_propagate(cnf, &mut assignment) {
        return None;
    }
    if propagation::all_clauses_satisfied(cnf, &assignment) {
        return Some(assignment);
    }

    let var = match branching::choose_unassigned_var(cnf.num_vars, &assignment) {
        Some(var) => var,
        None => return None,
    };

    let mut try_true = assignment.clone();
    if try_true.assign(var, true) {
        if let Some(model) = solve(cnf, try_true) {
            return Some(model);
        }
    }

    let mut try_false = assignment;
    if try_false.assign(var, false) {
        return solve(cnf, try_false);
    }

    None
}
