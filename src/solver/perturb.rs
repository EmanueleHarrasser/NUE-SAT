use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::cnf::{Cnf, Var};
use crate::solver::assignment::Assignment;
use crate::solver::features;
use crate::solver::logging::{DecisionGroup, DecisionSample};
use crate::solver::nnue::NnueModel;
use crate::solver::propagation;
use crate::solver::stats::Stats;
use crate::solver::{dpll, HeuristicKind, SolveConfig};

struct PerturbRecord {
    jw_var: Var,
    jw_value: bool,
    jw_features: features::FeatureVector,
    rand_var: Var,
    rand_value: bool,
    rand_features: features::FeatureVector,
}

struct NnuePerturbRecord {
    base_var: Var,
    base_value: bool,
    base_features: features::FeatureVector,
    perturb_var: Var,
    perturb_value: bool,
    perturb_features: features::FeatureVector,
}

struct NnuePerturbState {
    stats: Stats,
    decisions: u64,
    backtracks: u64,
}

pub struct PerturbationOutcome {
    pub model: Option<crate::solver::Model>,
    pub base_decisions: u64,
    pub new_decisions: u64,
    pub logged: bool,
}

pub fn generate_perturbation_log(
    cnf: &Cnf,
    log_path: &Path,
    seed: Option<u64>,
    bias_exp: f64,
) -> std::io::Result<PerturbationOutcome> {
    assert!(bias_exp > 0.0);

    let mut base_config = SolveConfig::new(0.0, seed);
    base_config.heuristic = HeuristicKind::JwEpsilon;

    let assignment = Assignment::new(cnf.num_vars);
    let mut base_state = dpll::SolveState::new(cnf.num_vars, base_config.clone());
    let base_model = dpll::solve(cnf, assignment, &mut base_state);
    let base_decisions = base_state.decisions;

    let base_path = if base_model.is_some() {
        base_state.log_stack.as_slice()
    } else {
        base_state.best_unsat.as_deref().unwrap_or(&[])
    };

    if base_path.is_empty() {
        return Ok(PerturbationOutcome {
            model: base_model,
            base_decisions,
            new_decisions: base_decisions,
            logged: false,
        });
    }

    let mut base_choices: Vec<(Var, bool)> = Vec::with_capacity(base_path.len());
    for group in base_path {
        let sample = group
            .samples
            .iter()
            .find(|sample| sample.label == 1)
            .expect("missing chosen sample");
        base_choices.push((sample.var, sample.value));
    }

    let mut rng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    let perturb_idx = biased_index(base_choices.len(), bias_exp, &mut rng);

    let assignment = Assignment::new(cnf.num_vars);
    let mut perturb_state = dpll::SolveState::new(cnf.num_vars, base_config);
    let mut record: Option<PerturbRecord> = None;

    let model = solve_with_perturbation(
        cnf,
        assignment,
        &mut perturb_state,
        &base_choices,
        perturb_idx,
        &mut rng,
        &mut record,
        0,
    );

    let new_decisions = perturb_state.decisions;
    let record = record.expect("missing perturbation record");

    let mut jw_sample = DecisionSample {
        label: 0,
        var: record.jw_var,
        value: record.jw_value,
        features: record.jw_features,
    };
    let mut rand_sample = DecisionSample {
        label: 0,
        var: record.rand_var,
        value: record.rand_value,
        features: record.rand_features,
    };

    if new_decisions < base_decisions {
        rand_sample.label = 1;
    } else {
        jw_sample.label = 1;
    }

    let group = DecisionGroup {
        decision_id: 0,
        samples: vec![jw_sample, rand_sample],
    };

    crate::solver::logging::write_csv(log_path, &[group])?;

    Ok(PerturbationOutcome {
        model,
        base_decisions,
        new_decisions,
        logged: true,
    })
}

pub fn generate_nnue_perturbation_log(
    cnf: &Cnf,
    log_path: &Path,
    seed: Option<u64>,
    bias_exp: f64,
    nnue_path: &Path,
    top_k: usize,
) -> std::io::Result<PerturbationOutcome> {
    assert!(bias_exp > 0.0);
    assert!(top_k > 0);

    let mut base_config = SolveConfig::new(0.0, seed);
    base_config.heuristic = HeuristicKind::Nnue;
    base_config.nnue_path = Some(nnue_path.to_path_buf());

    let assignment = Assignment::new(cnf.num_vars);
    let mut base_state = dpll::SolveState::new(cnf.num_vars, base_config);
    let base_model = dpll::solve(cnf, assignment, &mut base_state);
    let base_decisions = base_state.decisions;

    let base_path = if base_model.is_some() {
        base_state.log_stack.as_slice()
    } else {
        base_state.best_unsat.as_deref().unwrap_or(&[])
    };

    if base_path.is_empty() {
        return Ok(PerturbationOutcome {
            model: base_model,
            base_decisions,
            new_decisions: base_decisions,
            logged: false,
        });
    }

    let mut base_choices: Vec<(Var, bool)> = Vec::with_capacity(base_path.len());
    for group in base_path {
        let sample = group
            .samples
            .iter()
            .find(|sample| sample.label == 1)
            .expect("missing chosen sample");
        base_choices.push((sample.var, sample.value));
    }

    let mut rng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    let perturb_idx = biased_index(base_choices.len(), bias_exp, &mut rng);

    let mut nnue = NnueModel::from_bin(nnue_path).expect("failed to load nnue weights");
    let mut perturb_state = NnuePerturbState {
        stats: Stats::new(cnf.num_vars),
        decisions: 0,
        backtracks: 0,
    };
    let mut record: Option<NnuePerturbRecord> = None;

    let assignment = Assignment::new(cnf.num_vars);
    let model = solve_with_nnue_perturbation(
        cnf,
        assignment,
        &mut perturb_state,
        &base_choices,
        perturb_idx,
        &mut rng,
        &mut record,
        0,
        &mut nnue,
        top_k,
    );

    let new_decisions = perturb_state.decisions;
    let record = record.expect("missing perturbation record");

    let mut base_sample = DecisionSample {
        label: 0,
        var: record.base_var,
        value: record.base_value,
        features: record.base_features,
    };
    let mut perturb_sample = DecisionSample {
        label: 0,
        var: record.perturb_var,
        value: record.perturb_value,
        features: record.perturb_features,
    };

    if new_decisions < base_decisions {
        perturb_sample.label = 1;
    } else {
        base_sample.label = 1;
    }

    let group = DecisionGroup {
        decision_id: 0,
        samples: vec![base_sample, perturb_sample],
    };

    crate::solver::logging::write_csv(log_path, &[group])?;

    Ok(PerturbationOutcome {
        model,
        base_decisions,
        new_decisions,
        logged: true,
    })
}

fn solve_with_perturbation(
    cnf: &Cnf,
    assignment: Assignment,
    state: &mut dpll::SolveState,
    base_choices: &[(Var, bool)],
    perturb_idx: usize,
    rng: &mut StdRng,
    record: &mut Option<PerturbRecord>,
    depth: usize,
) -> Option<Assignment> {
    let mut assignment = assignment;

    if !propagation::unit_propagate(cnf, &mut assignment, &mut state.stats) {
        return None;
    }
    if propagation::all_clauses_satisfied(cnf, &assignment) {
        return Some(assignment);
    }

    if depth < perturb_idx {
        let (var, value) = base_choices[depth];
        state.decisions += 1;

        let mut try_first = assignment.clone();
        assert!(try_first.assign(var, value));
        if let Some(model) = solve_with_perturbation(
            cnf,
            try_first,
            state,
            base_choices,
            perturb_idx,
            rng,
            record,
            depth + 1,
        ) {
            return Some(model);
        }

        state.backtracks += 1;
        state.stats.inc_flip(var);

        let mut try_second = assignment;
        assert!(try_second.assign(var, !value));
        if let Some(model) = solve_with_perturbation(
            cnf,
            try_second,
            state,
            base_choices,
            perturb_idx,
            rng,
            record,
            depth + 1,
        ) {
            return Some(model);
        }

        return None;
    }

    let unassigned = unassigned_vars(cnf, &assignment);
    let (pos_scores, neg_scores) = jw_scores(cnf, &assignment);
    let jw_var = best_var(&unassigned, &pos_scores, &neg_scores);
    let jw_value = pos_scores[jw_var.index()] >= neg_scores[jw_var.index()];

    let (var, value) = if depth == perturb_idx {
        let rand_var = random_other(&unassigned, jw_var, rng);
        let rand_value = pos_scores[rand_var.index()] >= neg_scores[rand_var.index()];
        if record.is_none() {
            let trail_depth = depth as u32;
            let jw_features =
                features::compute_features(cnf, &assignment, &state.stats, jw_var, trail_depth);
            let rand_features =
                features::compute_features(cnf, &assignment, &state.stats, rand_var, trail_depth);
            *record = Some(PerturbRecord {
                jw_var,
                jw_value,
                jw_features,
                rand_var,
                rand_value,
                rand_features,
            });
        }
        (rand_var, rand_value)
    } else {
        (jw_var, jw_value)
    };

    state.decisions += 1;

    let mut try_first = assignment.clone();
    assert!(try_first.assign(var, value));
    if let Some(model) = solve_with_perturbation(
        cnf,
        try_first,
        state,
        base_choices,
        perturb_idx,
        rng,
        record,
        depth + 1,
    ) {
        return Some(model);
    }

    state.backtracks += 1;
    state.stats.inc_flip(var);

    let mut try_second = assignment;
    assert!(try_second.assign(var, !value));
    if let Some(model) = solve_with_perturbation(
        cnf,
        try_second,
        state,
        base_choices,
        perturb_idx,
        rng,
        record,
        depth + 1,
    ) {
        return Some(model);
    }

    None
}

fn solve_with_nnue_perturbation(
    cnf: &Cnf,
    assignment: Assignment,
    state: &mut NnuePerturbState,
    base_choices: &[(Var, bool)],
    perturb_idx: usize,
    rng: &mut StdRng,
    record: &mut Option<NnuePerturbRecord>,
    depth: usize,
    nnue: &mut NnueModel,
    top_k: usize,
) -> Option<Assignment> {
    let mut assignment = assignment;

    if !propagation::unit_propagate(cnf, &mut assignment, &mut state.stats) {
        return None;
    }
    if propagation::all_clauses_satisfied(cnf, &assignment) {
        return Some(assignment);
    }

    if depth < perturb_idx {
        let (var, value) = base_choices[depth];
        state.decisions += 1;

        let mut try_first = assignment.clone();
        assert!(try_first.assign(var, value));
        if let Some(model) = solve_with_nnue_perturbation(
            cnf,
            try_first,
            state,
            base_choices,
            perturb_idx,
            rng,
            record,
            depth + 1,
            nnue,
            top_k,
        ) {
            return Some(model);
        }

        state.backtracks += 1;
        state.stats.inc_flip(var);

        let mut try_second = assignment;
        assert!(try_second.assign(var, !value));
        if let Some(model) = solve_with_nnue_perturbation(
            cnf,
            try_second,
            state,
            base_choices,
            perturb_idx,
            rng,
            record,
            depth + 1,
            nnue,
            top_k,
        ) {
            return Some(model);
        }

        return None;
    }

    let unassigned = unassigned_vars(cnf, &assignment);
    let (pos_scores, neg_scores) = jw_scores(cnf, &assignment);
    let trail_depth = depth as u32;
    let ranked = rank_nnue(cnf, &assignment, &state.stats, trail_depth, nnue, &unassigned);

    let (var, value) = if depth == perturb_idx {
        let base_var = base_choices[depth].0;
        let base_value = base_choices[depth].1;

        let mut candidates: Vec<Var> = ranked.iter().take(top_k.min(ranked.len())).copied().collect();
        let rand_var = unassigned[rng.gen_range(0..unassigned.len())];
        candidates.push(rand_var);

        let perturb_var = if candidates.len() == 1 {
            candidates[0]
        } else {
            loop {
                let candidate = candidates[rng.gen_range(0..candidates.len())];
                if candidate != base_var {
                    break candidate;
                }
            }
        };
        let perturb_value = pos_scores[perturb_var.index()] >= neg_scores[perturb_var.index()];

        if record.is_none() {
            let base_features =
                features::compute_features(cnf, &assignment, &state.stats, base_var, trail_depth);
            let perturb_features = features::compute_features(
                cnf,
                &assignment,
                &state.stats,
                perturb_var,
                trail_depth,
            );
            *record = Some(NnuePerturbRecord {
                base_var,
                base_value,
                base_features,
                perturb_var,
                perturb_value,
                perturb_features,
            });
        }

        (perturb_var, perturb_value)
    } else {
        let var = ranked[0];
        let value = pos_scores[var.index()] >= neg_scores[var.index()];
        (var, value)
    };

    state.decisions += 1;

    let mut try_first = assignment.clone();
    assert!(try_first.assign(var, value));
    if let Some(model) = solve_with_nnue_perturbation(
        cnf,
        try_first,
        state,
        base_choices,
        perturb_idx,
        rng,
        record,
        depth + 1,
        nnue,
        top_k,
    ) {
        return Some(model);
    }

    state.backtracks += 1;
    state.stats.inc_flip(var);

    let mut try_second = assignment;
    assert!(try_second.assign(var, !value));
    if let Some(model) = solve_with_nnue_perturbation(
        cnf,
        try_second,
        state,
        base_choices,
        perturb_idx,
        rng,
        record,
        depth + 1,
        nnue,
        top_k,
    ) {
        return Some(model);
    }

    None
}

fn biased_index(len: usize, bias_exp: f64, rng: &mut StdRng) -> usize {
    let u: f64 = rng.gen();
    let idx = (u.powf(bias_exp) * len as f64).floor() as usize;
    if idx >= len {
        len - 1
    } else {
        idx
    }
}

fn unassigned_vars(cnf: &Cnf, assignment: &Assignment) -> Vec<Var> {
    let mut vars = Vec::new();
    for i in 1..=cnf.num_vars {
        let var = Var::new(i);
        if !assignment.is_assigned(var) {
            vars.push(var);
        }
    }
    vars
}

fn random_other(unassigned: &[Var], jw_var: Var, rng: &mut StdRng) -> Var {
    if unassigned.len() == 1 {
        return jw_var;
    }

    loop {
        let idx = rng.gen_range(0..unassigned.len());
        let candidate = unassigned[idx];
        if candidate != jw_var {
            return candidate;
        }
    }
}

fn best_var(unassigned: &[Var], pos_scores: &[f64], neg_scores: &[f64]) -> Var {
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

fn rank_nnue(
    cnf: &Cnf,
    assignment: &Assignment,
    stats: &Stats,
    trail_depth: u32,
    nnue: &mut NnueModel,
    unassigned: &[Var],
) -> Vec<Var> {
    let mut scored: Vec<(f32, Var)> = Vec::with_capacity(unassigned.len());
    for &candidate in unassigned {
        let feats = features::compute_features(cnf, assignment, stats, candidate, trail_depth);
        let score = nnue.score(&feats);
        scored.push((score, candidate));
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.into_iter().map(|(_, var)| var).collect()
}
