use crate::cnf::{Lit, Var};

#[derive(Clone, Debug)]
pub struct Stats {
    conflict_heat: Vec<u32>,
    recent_flips: Vec<u32>,
    conflict_count: u32,
}

impl Stats {
    pub fn new(num_vars: u32) -> Self {
        Stats {
            conflict_heat: vec![0; num_vars as usize],
            recent_flips: vec![0; num_vars as usize],
            conflict_count: 0,
        }
    }

    pub fn conflict_heat(&self, var: Var) -> u32 {
        self.conflict_heat[var.index()]
    }

    pub fn recent_flips(&self, var: Var) -> u32 {
        self.recent_flips[var.index()]
    }

    pub fn inc_flip(&mut self, var: Var) {
        let idx = var.index();
        self.recent_flips[idx] += 1;
    }

    pub fn record_conflict_clause(&mut self, clause: &[Lit]) {
        for &lit in clause {
            let idx = lit.var.index();
            self.conflict_heat[idx] += 1;
        }

        self.conflict_count += 1;
        if self.conflict_count % 256 == 0 {
            for heat in &mut self.conflict_heat {
                *heat /= 2;
            }
        }
    }
}
