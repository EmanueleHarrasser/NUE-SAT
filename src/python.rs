use std::fs;
use std::path::{Path, PathBuf};

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use crate::parser;
use crate::solver::{self, HeuristicKind, SolveConfig};

fn parse_heuristic(value: Option<&str>) -> Result<HeuristicKind, String> {
    let raw = value.unwrap_or("jw");
    HeuristicKind::from_str(raw).ok_or_else(|| format!("unknown heuristic: {}", raw))
}

#[pyfunction]
fn solve_dimacs(
    path: &str,
    log_path: Option<&str>,
    epsilon: Option<f64>,
    seed: Option<u64>,
    heuristic: Option<&str>,
    network_path: Option<&str>,
) -> PyResult<bool> {
    let input = fs::read_to_string(path).map_err(PyIOError::new_err)?;
    let cnf = parser::dimacs::parse_dimacs(&input)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let heuristic = parse_heuristic(heuristic).map_err(PyValueError::new_err)?;
    if heuristic == HeuristicKind::Network && network_path.is_none() {
        return Err(PyValueError::new_err("network_path is required for heuristic=network"));
    }
    let mut config = SolveConfig::new(epsilon.unwrap_or(0.1), seed);
    config.heuristic = heuristic;
    config.network_path = network_path.map(PathBuf::from);
    let result = match log_path {
        Some(log_path) => solver::solve_with_log(&cnf, config, Path::new(log_path))
            .map_err(PyIOError::new_err)?,
        None => solver::solve_with_config(&cnf, config),
    };

    Ok(result.is_some())
}

#[pyfunction]
fn solve_cnf(
    cnf_text: &str,
    log_path: Option<&str>,
    epsilon: Option<f64>,
    seed: Option<u64>,
    heuristic: Option<&str>,
    network_path: Option<&str>,
) -> PyResult<bool> {
    let cnf = parser::dimacs::parse_dimacs(cnf_text)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let heuristic = parse_heuristic(heuristic).map_err(PyValueError::new_err)?;
    if heuristic == HeuristicKind::Network && network_path.is_none() {
        return Err(PyValueError::new_err("network_path is required for heuristic=network"));
    }
    let mut config = SolveConfig::new(epsilon.unwrap_or(0.1), seed);
    config.heuristic = heuristic;
    config.network_path = network_path.map(PathBuf::from);
    let result = match log_path {
        Some(log_path) => solver::solve_with_log(&cnf, config, Path::new(log_path))
            .map_err(PyIOError::new_err)?,
        None => solver::solve_with_config(&cnf, config),
    };

    Ok(result.is_some())
}

#[pyfunction]
fn solve_dimacs_stats(
    path: &str,
    log_path: Option<&str>,
    epsilon: Option<f64>,
    seed: Option<u64>,
    heuristic: Option<&str>,
    network_path: Option<&str>,
) -> PyResult<(bool, u64, u64, u64)> {
    let input = fs::read_to_string(path).map_err(PyIOError::new_err)?;
    let cnf = parser::dimacs::parse_dimacs(&input)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let heuristic = parse_heuristic(heuristic).map_err(PyValueError::new_err)?;
    if heuristic == HeuristicKind::Network && network_path.is_none() {
        return Err(PyValueError::new_err("network_path is required for heuristic=network"));
    }
    let mut config = SolveConfig::new(epsilon.unwrap_or(0.1), seed);
    config.heuristic = heuristic;
    config.network_path = network_path.map(PathBuf::from);

    let outcome = match log_path {
        Some(log_path) => solver::solve_with_log_and_stats(&cnf, config, Path::new(log_path))
            .map_err(PyIOError::new_err)?,
        None => solver::solve_with_stats(&cnf, config),
    };

    Ok((
        outcome.model.is_some(),
        outcome.stats.decisions,
        outcome.stats.backtracks,
        outcome.stats.conflicts,
    ))
}

#[pyfunction]
fn solve_cnf_stats(
    cnf_text: &str,
    log_path: Option<&str>,
    epsilon: Option<f64>,
    seed: Option<u64>,
    heuristic: Option<&str>,
    network_path: Option<&str>,
) -> PyResult<(bool, u64, u64, u64)> {
    let cnf = parser::dimacs::parse_dimacs(cnf_text)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let heuristic = parse_heuristic(heuristic).map_err(PyValueError::new_err)?;
    if heuristic == HeuristicKind::Network && network_path.is_none() {
        return Err(PyValueError::new_err("network_path is required for heuristic=network"));
    }
    let mut config = SolveConfig::new(epsilon.unwrap_or(0.1), seed);
    config.heuristic = heuristic;
    config.network_path = network_path.map(PathBuf::from);

    let outcome = match log_path {
        Some(log_path) => solver::solve_with_log_and_stats(&cnf, config, Path::new(log_path))
            .map_err(PyIOError::new_err)?,
        None => solver::solve_with_stats(&cnf, config),
    };

    Ok((
        outcome.model.is_some(),
        outcome.stats.decisions,
        outcome.stats.backtracks,
        outcome.stats.conflicts,
    ))
}

#[pyfunction]
fn perturb_dimacs(
    path: &str,
    log_path: &str,
    seed: Option<u64>,
    bias_exp: Option<f64>,
) -> PyResult<(bool, u64, u64)> {
    let input = fs::read_to_string(path).map_err(PyIOError::new_err)?;
    let cnf = parser::dimacs::parse_dimacs(&input)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let outcome = solver::generate_perturbation_log(
        &cnf,
        Path::new(log_path),
        seed,
        bias_exp.unwrap_or(2.0),
    )
    .map_err(PyIOError::new_err)?;

    Ok((
        outcome.logged,
        outcome.base_decisions,
        outcome.new_decisions,
    ))
}

#[pyfunction]
fn perturb_dimacs_network(
    path: &str,
    log_path: &str,
    network_path: &str,
    seed: Option<u64>,
    bias_exp: Option<f64>,
    top_k: Option<usize>,
    top_prob: Option<f64>,
) -> PyResult<(bool, u64, u64)> {
    let input = fs::read_to_string(path).map_err(PyIOError::new_err)?;
    let cnf = parser::dimacs::parse_dimacs(&input)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let outcome = solver::generate_network_perturbation_log(
        &cnf,
        Path::new(log_path),
        seed,
        bias_exp.unwrap_or(2.0),
        Path::new(network_path),
        top_k.unwrap_or(5),
        top_prob.unwrap_or(0.5),
    )
    .map_err(PyIOError::new_err)?;

    Ok((
        outcome.logged,
        outcome.base_decisions,
        outcome.new_decisions,
    ))
}

#[pymodule]
fn enue_sat(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_dimacs, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cnf, m)?)?;
    m.add_function(wrap_pyfunction!(solve_dimacs_stats, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cnf_stats, m)?)?;
    m.add_function(wrap_pyfunction!(perturb_dimacs, m)?)?;
    m.add_function(wrap_pyfunction!(perturb_dimacs_network, m)?)?;
    Ok(())
}
