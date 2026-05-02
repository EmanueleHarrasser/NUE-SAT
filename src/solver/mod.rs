pub mod assignment;
pub mod branching;
pub mod dpll;
pub mod features;
pub mod logging;
pub mod nnue;
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

pub fn solve(cnf: &Cnf) -> Option<Model> {
    solve_with_config(cnf, SolveConfig::default())
}

pub fn solve_with_config(cnf: &Cnf, config: SolveConfig) -> Option<Model> {
    let assignment = Assignment::new(cnf.num_vars);
    let mut state = dpll::SolveState::new(cnf.num_vars, config);
    dpll::solve(cnf, assignment, &mut state)
}

pub fn solve_with_log(
    cnf: &Cnf,
    config: SolveConfig,
    log_path: &Path,
) -> std::io::Result<Option<Model>> {
    let assignment = Assignment::new(cnf.num_vars);
    let mut state = dpll::SolveState::new(cnf.num_vars, config);
    let model = dpll::solve(cnf, assignment, &mut state);
    let groups = if model.is_some() {
        state.log_stack.as_slice()
    } else {
        state.best_unsat.as_deref().unwrap_or(&[])
    };
    logging::write_csv(log_path, groups)?;
    Ok(model)
}
