pub mod assignment;
pub mod branching;
pub mod dpll;
pub mod propagation;

use crate::cnf::Cnf;

pub use assignment::{Assignment, Model};

pub fn solve(cnf: &Cnf) -> Option<Model> {
    let assignment = Assignment::new(cnf.num_vars);
    dpll::solve(cnf, assignment)
}
