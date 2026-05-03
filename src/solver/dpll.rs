use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::cnf::Cnf;
use crate::solver::assignment::Assignment;
use crate::solver::features;
use crate::solver::logging::{DecisionGroup, DecisionSample};
use crate::solver::nnue::NnueModel;
use crate::solver::propagation;
use crate::solver::stats::Stats;
use crate::solver::{branching, HeuristicKind, SolveConfig, SolveStats};

pub(crate) struct SolveState {
    pub stats: Stats,
    pub log_stack: Vec<DecisionGroup>,
    pub best_unsat: Option<Vec<DecisionGroup>>,
    pub rng: StdRng,
    pub epsilon: f64,
    pub decision_id: u32,
    pub heuristic: HeuristicKind,
    pub nnue: Option<NnueModel>,
    pub decisions: u64,
    pub backtracks: u64,
}

impl SolveState {
    pub(crate) fn new(num_vars: u32, config: SolveConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let nnue = match config.heuristic {
            HeuristicKind::Nnue => {
                let path = config.nnue_path.expect("nnue_path required for NNUE");
                Some(NnueModel::from_bin(&path).expect("failed to load nnue weights"))
            }
            HeuristicKind::JwEpsilon => None,
        };

        SolveState {
            stats: Stats::new(num_vars),
            log_stack: Vec::new(),
            best_unsat: None,
            rng,
            epsilon: config.epsilon,
            decision_id: 0,
            heuristic: config.heuristic,
            nnue,
            decisions: 0,
            backtracks: 0,
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

    pub(crate) fn metrics(&self) -> SolveStats {
        SolveStats {
            decisions: self.decisions,
            backtracks: self.backtracks,
            conflicts: self.stats.conflict_count(),
        }
    }
}

pub(crate) fn solve(cnf: &Cnf, assignment: Assignment, state: &mut SolveState) -> Option<Assignment> {
    let mut assignment = assignment;

    if !propagation::unit_propagate(cnf, &mut assignment, &mut state.stats) {
        state.record_unsat_path();
        return None;
    }
    if propagation::all_clauses_satisfied(cnf, &assignment) {
        return Some(assignment);
    }

    let trail_depth = state.log_stack.len() as u32;
    let decision = match branching::choose_decision(
        cnf,
        &assignment,
        &state.stats,
        trail_depth,
        &mut state.rng,
        state.epsilon,
        state.heuristic,
        state.nnue.as_ref(),
    ) {
        Some(decision) => decision,
        None => return None,
    };

    state.decisions += 1;

    let decision_id = state.decision_id;
    state.decision_id += 1;

    let mut samples: Vec<DecisionSample> = Vec::with_capacity(1 + decision.runner_ups.len());
    let feats = features::compute_features(cnf, &assignment, &state.stats, decision.var, trail_depth);
    samples.push(DecisionSample {
        label: 1,
        var: decision.var,
        value: decision.value,
        features: feats,
    });

    for runner in &decision.runner_ups {
        let runner_feats =
            features::compute_features(cnf, &assignment, &state.stats, runner.var, trail_depth);
        samples.push(DecisionSample {
            label: 0,
            var: runner.var,
            value: runner.value,
            features: runner_feats,
        });
    }

    state.log_stack.push(DecisionGroup {
        decision_id,
        samples,
    });

    let mut try_first = assignment.clone();
    assert!(try_first.assign(decision.var, decision.value));
    if let Some(model) = solve(cnf, try_first, state) {
        return Some(model);
    }

    state.backtracks += 1;
    state.stats.inc_flip(decision.var);
    if let Some(last) = state.log_stack.last_mut() {
        if let Some(first) = last.samples.first_mut() {
            first.value = !decision.value;
        }
    }

    let mut try_second = assignment;
    assert!(try_second.assign(decision.var, !decision.value));
    if let Some(model) = solve(cnf, try_second, state) {
        return Some(model);
    }

    state.log_stack.pop();
    None
}
