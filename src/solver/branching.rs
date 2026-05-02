use rand::rngs::StdRng;
use rand::Rng;

use crate::cnf::{Cnf, Var};
use crate::solver::assignment::Assignment;
use crate::solver::features;
use crate::solver::nnue::NnueModel;
use crate::solver::stats::Stats;
use crate::solver::HeuristicKind;

const RUNNER_UPS: usize = 3;

pub struct RunnerUp {
    pub var: Var,
    pub value: bool,
}

pub struct Decision {
    pub var: Var,
    pub value: bool,
    pub runner_ups: Vec<RunnerUp>,
}

pub fn choose_decision(
    cnf: &Cnf,
    assignment: &Assignment,
    stats: &Stats,
    trail_depth: u32,
    rng: &mut StdRng,
    epsilon: f64,
    heuristic: HeuristicKind,
    nnue: Option<&NnueModel>,
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

    let (pos_scores, neg_scores) = jw_scores(cnf, assignment);

    match heuristic {
        HeuristicKind::JwEpsilon => {
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

            let mut scored: Vec<(f64, Var)> = unassigned
                .iter()
                .map(|&candidate| {
                    let score = pos_scores[candidate.index()] + neg_scores[candidate.index()];
                    (score, candidate)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let mut runner_ups: Vec<RunnerUp> = Vec::new();
            for (_, candidate) in scored {
                if candidate == var {
                    continue;
                }
                let candidate_value = pos_scores[candidate.index()] >= neg_scores[candidate.index()];
                runner_ups.push(RunnerUp {
                    var: candidate,
                    value: candidate_value,
                });
                if runner_ups.len() == RUNNER_UPS {
                    break;
                }
            }

            Some(Decision {
                var,
                value,
                runner_ups,
            })
        }
        HeuristicKind::Nnue => {
            let model = nnue.expect("nnue model required");
            let mut scored: Vec<(f32, Var)> = Vec::with_capacity(unassigned.len());
            for &candidate in &unassigned {
                let feats = features::compute_features(cnf, assignment, stats, candidate, trail_depth);
                let score = model.score(&feats);
                scored.push((score, candidate));
            }

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let var = scored[0].1;
            let value = pos_scores[var.index()] >= neg_scores[var.index()];

            let mut runner_ups: Vec<RunnerUp> = Vec::new();
            for (_, candidate) in scored.into_iter() {
                if candidate == var {
                    continue;
                }
                let candidate_value = pos_scores[candidate.index()] >= neg_scores[candidate.index()];
                runner_ups.push(RunnerUp {
                    var: candidate,
                    value: candidate_value,
                });
                if runner_ups.len() == RUNNER_UPS {
                    break;
                }
            }

            Some(Decision {
                var,
                value,
                runner_ups,
            })
        }
    }
}

fn jw_scores(cnf: &Cnf, assignment: &Assignment) -> (Vec<f64>, Vec<f64>) {
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

    (pos_scores, neg_scores)
}
