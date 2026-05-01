use std::fs;
use std::path::Path;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use crate::parser;
use crate::solver::{self, SolveConfig};

#[pyfunction]
fn solve_dimacs(
    path: &str,
    log_path: Option<&str>,
    epsilon: Option<f64>,
    seed: Option<u64>,
) -> PyResult<bool> {
    let input = fs::read_to_string(path).map_err(PyIOError::new_err)?;
    let cnf = parser::dimacs::parse_dimacs(&input).map_err(|err| PyValueError::new_err(err.to_string()))?;

    let config = SolveConfig::new(epsilon.unwrap_or(0.1), seed);
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
) -> PyResult<bool> {
    let cnf = parser::dimacs::parse_dimacs(cnf_text)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let config = SolveConfig::new(epsilon.unwrap_or(0.1), seed);
    let result = match log_path {
        Some(log_path) => solver::solve_with_log(&cnf, config, Path::new(log_path))
            .map_err(PyIOError::new_err)?,
        None => solver::solve_with_config(&cnf, config),
    };

    Ok(result.is_some())
}

#[pymodule]
fn enue_sat(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_dimacs, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cnf, m)?)?;
    Ok(())
}
