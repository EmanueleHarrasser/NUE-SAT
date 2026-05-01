use crate::cnf::{Cnf, Lit};
use crate::solver::assignment::Assignment;
use crate::solver::stats::Stats;

enum ClauseStatus {
    Satisfied,
    Unit(Lit),
    Unresolved,
    Conflict,
}

pub fn unit_propagate(cnf: &Cnf, assignment: &mut Assignment, stats: &mut Stats) -> bool {
    loop {
        let mut changed = false;
        for clause in &cnf.clauses {
            match clause_status(clause, assignment) {
                ClauseStatus::Satisfied | ClauseStatus::Unresolved => {}
                ClauseStatus::Conflict => {
                    stats.record_conflict_clause(clause);
                    return false;
                }
                ClauseStatus::Unit(lit) => {
                    let value = !lit.neg;
                    if !assignment.assign(lit.var, value) {
                        stats.record_conflict_clause(clause);
                        return false;
                    }
                    changed = true;
                }
            }
        }
        if !changed {
            return true;
        }
    }
}

pub fn all_clauses_satisfied(cnf: &Cnf, assignment: &Assignment) -> bool {
    cnf.clauses
        .iter()
        .all(|clause| clause_satisfied(clause, assignment))
}

fn clause_satisfied(clause: &[Lit], assignment: &Assignment) -> bool {
    clause
        .iter()
        .any(|&lit| assignment.eval_lit(lit) == Some(true))
}

fn clause_status(clause: &[Lit], assignment: &Assignment) -> ClauseStatus {
    let mut unit: Option<Lit> = None;
    let mut multiple = false;

    for &lit in clause {
        match assignment.eval_lit(lit) {
            Some(true) => return ClauseStatus::Satisfied,
            Some(false) => {}
            None => {
                if unit.is_some() {
                    multiple = true;
                } else {
                    unit = Some(lit);
                }
            }
        }
    }

    if multiple {
        ClauseStatus::Unresolved
    } else if let Some(lit) = unit {
        ClauseStatus::Unit(lit)
    } else {
        ClauseStatus::Conflict
    }
}
