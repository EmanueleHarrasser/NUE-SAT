use crate::cnf::{Lit, Var};

#[derive(Clone, Debug)]
pub struct Assignment {
    values: Vec<Option<bool>>,
}

pub type Model = Assignment;

impl Assignment {
    pub fn new(num_vars: u32) -> Self {
        Assignment {
            values: vec![None; num_vars as usize],
        }
    }

    pub fn get(&self, var: Var) -> Option<bool> {
        self.values[var.index()]
    }

    pub fn assign(&mut self, var: Var, value: bool) -> bool {
        let idx = var.index();
        match self.values[idx] {
            None => {
                self.values[idx] = Some(value);
                true
            }
            Some(existing) => existing == value,
        }
    }

    pub fn is_assigned(&self, var: Var) -> bool {
        self.get(var).is_some()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn eval_lit(&self, lit: Lit) -> Option<bool> {
        match self.get(lit.var) {
            None => None,
            Some(val) => Some(if lit.neg { !val } else { val }),
        }
    }
}
