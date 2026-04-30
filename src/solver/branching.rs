use crate::cnf::Var;
use crate::solver::assignment::Assignment;

pub fn choose_unassigned_var(num_vars: u32, assignment: &Assignment) -> Option<Var> {
    for i in 1..=num_vars {
        let var = Var::new(i);
        if !assignment.is_assigned(var) {
            return Some(var);
        }
    }
    None
}
