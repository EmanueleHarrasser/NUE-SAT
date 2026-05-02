use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use crate::cnf::Var;
use crate::solver::features::FeatureVector;

#[derive(Clone, Debug)]
pub struct DecisionSample {
    pub label: u8,
    pub var: Var,
    pub value: bool,
    pub features: FeatureVector,
}

#[derive(Clone, Debug)]
pub struct DecisionGroup {
    pub decision_id: u32,
    pub samples: Vec<DecisionSample>,
}

impl DecisionSample {
    pub fn header() -> &'static str {
        "decision_id,label,var,value,pos_len_2,neg_len_2,pos_len_3,neg_len_3,pos_len_4p,neg_len_4p,conflict_heat,recent_flips,trail_depth,active_clause_ratio"
    }

    pub fn write_csv_row(&self, decision_id: u32, out: &mut dyn Write) -> io::Result<()> {
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            decision_id,
            self.label,
            self.var.0,
            if self.value { 1 } else { 0 },
            self.features.pos_len_2,
            self.features.neg_len_2,
            self.features.pos_len_3,
            self.features.neg_len_3,
            self.features.pos_len_4p,
            self.features.neg_len_4p,
            self.features.conflict_heat,
            self.features.recent_flips,
            self.features.trail_depth,
            self.features.active_clause_ratio,
        )
    }
}

pub fn write_csv(path: &Path, groups: &[DecisionGroup]) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "{}", DecisionSample::header())?;
    for group in groups {
        for sample in &group.samples {
            sample.write_csv_row(group.decision_id, &mut file)?;
        }
    }
    Ok(())
}
