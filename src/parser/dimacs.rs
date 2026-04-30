use crate::cnf::{Cnf, Clause, Lit};
use crate::parser::ParseError;

pub fn parse_dimacs(input: &str) -> Result<Cnf, ParseError> {
    let mut num_vars: Option<u32> = None;
    let mut num_clauses: Option<usize> = None;
    let mut clauses: Vec<Clause> = Vec::new();
    let mut current: Clause = Vec::new();
    let mut saw_problem = false;

    for (line_idx, line) in input.lines().enumerate() {
        let line_no = line_idx + 1;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('c') {
            continue;
        }
        if trimmed.starts_with('%') {
            break;
        }
        if trimmed.starts_with('p') {
            if saw_problem {
                return Err(ParseError::new(line_no, "duplicate problem line"));
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() != 4 || parts[1] != "cnf" {
                return Err(ParseError::new(
                    line_no,
                    "expected: p cnf <vars> <clauses>",
                ));
            }
            let vars: u32 = parts[2]
                .parse()
                .map_err(|_| ParseError::new(line_no, "invalid variable count"))?;
            let clauses_count: usize = parts[3]
                .parse()
                .map_err(|_| ParseError::new(line_no, "invalid clause count"))?;

            num_vars = Some(vars);
            num_clauses = Some(clauses_count);
            saw_problem = true;
            continue;
        }

        if !saw_problem {
            return Err(ParseError::new(line_no, "missing problem line"));
        }

        for token in trimmed.split_whitespace() {
            let value: i32 = token
                .parse()
                .map_err(|_| ParseError::new(line_no, "invalid literal"))?;
            if value == 0 {
                clauses.push(current);
                current = Vec::new();
                continue;
            }

            let lit = Lit::from_dimacs(value);
            if let Some(vars) = num_vars {
                if lit.var.0 > vars {
                    return Err(ParseError::new(
                        line_no,
                        "literal exceeds declared variable count",
                    ));
                }
            }
            current.push(lit);
        }
    }

    if !saw_problem {
        return Err(ParseError::new(1, "missing problem line"));
    }
    if !current.is_empty() {
        let line_no = input.lines().count().max(1);
        return Err(ParseError::new(line_no, "clause missing trailing 0"));
    }
    if let Some(expected) = num_clauses {
        if clauses.len() != expected {
            let line_no = input.lines().count().max(1);
            return Err(ParseError::new(
                line_no,
                "clause count does not match header",
            ));
        }
    }

    Ok(Cnf::new(num_vars.unwrap(), clauses))
}
