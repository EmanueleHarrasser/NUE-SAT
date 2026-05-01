use rand::rngs::StdRng;
use rand::Rng;

use crate::cnf::{Cnf, Var};
use crate::solver::assignment::Assignment;

pub struct Decision {
    pub var: Var,
    pub value: bool,
}

pub fn choose_decision(
    cnf: &Cnf,
    assignment: &Assignment,
    rng: &mut StdRng,
    epsilon: f64,
) -> Option<Decision> {
    let mut unassigned: Vec<Var> = Vec::new();
    for i in 1..=cnf.num_vars {
        let var = Var::new(i);
        if !assignment.is_assigned(var) {
            unassigned.push(var);
        }
    }
    if unassigned.is_empty() {
        return None;
    }

    let mut pos_scores = vec![0.0f64; cnf.num_vars as usize];
    let mut neg_scores = vec![0.0f64; cnf.num_vars as usize];

    for clause in &cnf.clauses {
        let mut satisfied = false;
        let mut unassigned_count = 0u32;

        for &lit in clause {
            match assignment.eval_lit(lit) {
                Some(true) => {
                    satisfied = true;
                    break;
                }
                Some(false) => {}
                None => {
                    unassigned_count += 1;
                }
            }
        }

        if satisfied || unassigned_count == 0 {
            continue;
        }

        let weight = 2f64.powi(-(unassigned_count as i32));
        for &lit in clause {
            if assignment.eval_lit(lit).is_none() {
                let idx = lit.var.index();
                if lit.neg {
                    neg_scores[idx] += weight;
                } else {
                    pos_scores[idx] += weight;
                }
            }
        }
    }

    let pick_random = rng.gen::<f64>() < epsilon;
    let var = if pick_random {
        let idx = rng.gen_range(0..unassigned.len());
        unassigned[idx]
    } else {
        let mut best_var = unassigned[0];
        let mut best_score = pos_scores[best_var.index()] + neg_scores[best_var.index()];
        for &candidate in &unassigned[1..] {
            let score = pos_scores[candidate.index()] + neg_scores[candidate.index()];
            if score > best_score {
                best_score = score;
                best_var = candidate;
            }
        }
        best_var
    };

    let value = pos_scores[var.index()] >= neg_scores[var.index()];
    Some(Decision { var, value })
}
