use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::cnf::Cnf;
use crate::solver::assignment::Assignment;
use crate::solver::features;
use crate::solver::logging::DecisionRecord;
use crate::solver::propagation;
use crate::solver::stats::Stats;
use crate::solver::{branching, SolveConfig};

pub(crate) struct SolveState {
    pub stats: Stats,
    pub log_stack: Vec<DecisionRecord>,
    pub best_unsat: Option<Vec<DecisionRecord>>,
    pub rng: StdRng,
    pub epsilon: f64,
}

impl SolveState {
    pub(crate) fn new(num_vars: u32, config: SolveConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        SolveState {
            stats: Stats::new(num_vars),
            log_stack: Vec::new(),
            best_unsat: None,
            rng,
            epsilon: config.epsilon,
        }
    }

    fn record_unsat_path(&mut self) {
        let current_len = self.log_stack.len();
        match &self.best_unsat {
            Some(best) if best.len() <= current_len => {}
            _ => {
                self.best_unsat = Some(self.log_stack.clone());
            }
        }
    }
}

pub fn solve(cnf: &Cnf, assignment: Assignment, state: &mut SolveState) -> Option<Assignment> {
    let mut assignment = assignment;

    if !propagation::unit_propagate(cnf, &mut assignment, &mut state.stats) {
        state.record_unsat_path();
        return None;
    }
    if propagation::all_clauses_satisfied(cnf, &assignment) {
        return Some(assignment);
    }

    let decision = match branching::choose_decision(cnf, &assignment, &mut state.rng, state.epsilon) {
        Some(decision) => decision,
        None => return None,
    };

    let trail_depth = state.log_stack.len() as u32;
    let feats = features::compute_features(cnf, &assignment, &state.stats, decision.var, trail_depth);
    state.log_stack.push(DecisionRecord {
        var: decision.var,
        value: decision.value,
        features: feats,
    });

    let mut try_first = assignment.clone();
    assert!(try_first.assign(decision.var, decision.value));
    if let Some(model) = solve(cnf, try_first, state) {
        return Some(model);
    }

    state.stats.inc_flip(decision.var);
    if let Some(last) = state.log_stack.last_mut() {
        last.value = !decision.value;
    }

    let mut try_second = assignment;
    assert!(try_second.assign(decision.var, !decision.value));
    if let Some(model) = solve(cnf, try_second, state) {
        return Some(model);
    }

    state.log_stack.pop();
    None
}
