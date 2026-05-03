pub mod assignment;
pub mod branching;
pub mod dpll;
pub mod features;
pub mod logging;
pub mod nnue;
pub mod perturb;
pub mod propagation;
pub mod stats;

use crate::cnf::Cnf;
use std::path::{Path, PathBuf};

pub use assignment::{Assignment, Model};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeuristicKind {
    JwEpsilon,
    Nnue,
}

impl HeuristicKind {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "jw" | "jw-epsilon" | "jw_epsilon" => Some(HeuristicKind::JwEpsilon),
            "nnue" => Some(HeuristicKind::Nnue),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolveConfig {
    pub epsilon: f64,
    pub seed: Option<u64>,
    pub heuristic: HeuristicKind,
    pub nnue_path: Option<PathBuf>,
}

impl SolveConfig {
    pub fn new(epsilon: f64, seed: Option<u64>) -> Self {
        assert!(epsilon >= 0.0 && epsilon <= 1.0);
        SolveConfig {
            epsilon,
            seed,
            heuristic: HeuristicKind::JwEpsilon,
            nnue_path: None,
        }
    }
}

impl Default for SolveConfig {
    fn default() -> Self {
        SolveConfig {
            epsilon: 0.1,
            seed: None,
            heuristic: HeuristicKind::JwEpsilon,
            nnue_path: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SolveStats {
    pub decisions: u64,
    pub backtracks: u64,
    pub conflicts: u64,
}

#[derive(Clone, Debug)]
pub struct SolveOutcome {
    pub model: Option<Model>,
    pub stats: SolveStats,
}

pub fn solve(cnf: &Cnf) -> Option<Model> {
    solve_with_config(cnf, SolveConfig::default())
}

pub fn solve_with_config(cnf: &Cnf, config: SolveConfig) -> Option<Model> {
    solve_with_stats(cnf, config).model
}

pub fn solve_with_stats(cnf: &Cnf, config: SolveConfig) -> SolveOutcome {
    let assignment = Assignment::new(cnf.num_vars);
    let mut state = dpll::SolveState::new(cnf.num_vars, config);
    let model = dpll::solve(cnf, assignment, &mut state);
    let stats = state.metrics();
    SolveOutcome { model, stats }
}

pub fn solve_with_log(
    cnf: &Cnf,
    config: SolveConfig,
    log_path: &Path,
) -> std::io::Result<Option<Model>> {
    Ok(solve_with_log_and_stats(cnf, config, log_path)?.model)
}

pub fn solve_with_log_and_stats(
    cnf: &Cnf,
    config: SolveConfig,
    log_path: &Path,
) -> std::io::Result<SolveOutcome> {
    let assignment = Assignment::new(cnf.num_vars);
    let mut state = dpll::SolveState::new(cnf.num_vars, config);
    let model = dpll::solve(cnf, assignment, &mut state);
    let groups = if model.is_some() {
        state.log_stack.as_slice()
    } else {
        state.best_unsat.as_deref().unwrap_or(&[])
    };
    logging::write_csv(log_path, groups)?;
    let stats = state.metrics();
    Ok(SolveOutcome { model, stats })
}

pub fn generate_perturbation_log(
    cnf: &Cnf,
    log_path: &Path,
    seed: Option<u64>,
    bias_exp: f64,
) -> std::io::Result<perturb::PerturbationOutcome> {
    perturb::generate_perturbation_log(cnf, log_path, seed, bias_exp)
}
