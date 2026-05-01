use crate::cnf::{Cnf, Var};
use crate::solver::assignment::Assignment;
use crate::solver::stats::Stats;

#[derive(Clone, Copy, Debug)]
pub struct FeatureVector {
    pub pos_len_2: u32,
    pub neg_len_2: u32,
    pub pos_len_3: u32,
    pub neg_len_3: u32,
    pub pos_len_4p: u32,
    pub neg_len_4p: u32,
    pub conflict_heat: u32,
    pub recent_flips: u32,
    pub trail_depth: u32,
    pub active_clause_ratio: u32,
}

pub fn compute_features(
    cnf: &Cnf,
    assignment: &Assignment,
    stats: &Stats,
    var: Var,
    trail_depth: u32,
) -> FeatureVector {
    let mut pos_len_2 = 0u32;
    let mut neg_len_2 = 0u32;
    let mut pos_len_3 = 0u32;
    let mut neg_len_3 = 0u32;
    let mut pos_len_4p = 0u32;
    let mut neg_len_4p = 0u32;
    let mut active_clauses = 0u32;

    for clause in &cnf.clauses {
        let mut satisfied = false;
        let mut unassigned = 0u32;
        let mut pos_present = false;
        let mut neg_present = false;

        for &lit in clause {
            match assignment.eval_lit(lit) {
                Some(true) => {
                    satisfied = true;
                    break;
                }
                Some(false) => {}
                None => {
                    unassigned += 1;
                }
            }
            if lit.var == var {
                if lit.neg {
                    neg_present = true;
                } else {
                    pos_present = true;
                }
            }
        }

        if satisfied {
            continue;
        }

        active_clauses += 1;
        if unassigned == 0 {
            continue;
        }

        match unassigned {
            2 => {
                if pos_present {
                    pos_len_2 += 1;
                }
                if neg_present {
                    neg_len_2 += 1;
                }
            }
            3 => {
                if pos_present {
                    pos_len_3 += 1;
                }
                if neg_present {
                    neg_len_3 += 1;
                }
            }
            _ => {
                if unassigned >= 4 {
                    if pos_present {
                        pos_len_4p += 1;
                    }
                    if neg_present {
                        neg_len_4p += 1;
                    }
                }
            }
        }
    }

    let total_clauses = cnf.clauses.len() as u32;
    let active_clause_ratio = if total_clauses == 0 {
        0
    } else {
        (active_clauses * 100) / total_clauses
    };

    FeatureVector {
        pos_len_2,
        neg_len_2,
        pos_len_3,
        neg_len_3,
        pos_len_4p,
        neg_len_4p,
        conflict_heat: stats.conflict_heat(var),
        recent_flips: stats.recent_flips(var),
        trail_depth,
        active_clause_ratio,
    }
}
