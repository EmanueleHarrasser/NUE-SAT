use std::fs;
use std::path::Path;

use crate::solver::features::FeatureVector;

const INPUT_DIM: usize = 10;
const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 256;
const OUTPUT_DIM: usize = 1;
const TOTAL_WEIGHTS: usize =
    HIDDEN_1 * INPUT_DIM + HIDDEN_1 + HIDDEN_2 * HIDDEN_1 + HIDDEN_2 + OUTPUT_DIM * HIDDEN_2 + OUTPUT_DIM;

pub struct NnueModel {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
}

impl NnueModel {
    pub fn from_bin(path: &Path) -> std::io::Result<Self> {
        let bytes = fs::read(path)?;
        assert!(bytes.len() % 4 == 0, "nnue bin must be f32-aligned");

        let mut values = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            values.push(f32::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ]));
        }

        assert!(
            values.len() == TOTAL_WEIGHTS,
            "nnue weight count mismatch: expected {}, got {}",
            TOTAL_WEIGHTS,
            values.len()
        );

        let mut idx = 0;
        let w1 = values[idx..idx + HIDDEN_1 * INPUT_DIM].to_vec();
        idx += HIDDEN_1 * INPUT_DIM;
        let b1 = values[idx..idx + HIDDEN_1].to_vec();
        idx += HIDDEN_1;
        let w2 = values[idx..idx + HIDDEN_2 * HIDDEN_1].to_vec();
        idx += HIDDEN_2 * HIDDEN_1;
        let b2 = values[idx..idx + HIDDEN_2].to_vec();
        idx += HIDDEN_2;
        let w3 = values[idx..idx + HIDDEN_2].to_vec();
        idx += HIDDEN_2;
        let b3 = values[idx];

        Ok(NnueModel { w1, b1, w2, b2, w3, b3 })
    }

    pub fn score(&self, features: &FeatureVector) -> f32 {
        let input = [
            features.pos_len_2 as f32,
            features.neg_len_2 as f32,
            features.pos_len_3 as f32,
            features.neg_len_3 as f32,
            features.pos_len_4p as f32,
            features.neg_len_4p as f32,
            features.conflict_heat as f32,
            features.recent_flips as f32,
            features.trail_depth as f32,
            features.active_clause_ratio as f32,
        ];

        let mut h1 = vec![0.0f32; HIDDEN_1];
        for i in 0..HIDDEN_1 {
            let mut sum = self.b1[i];
            let base = i * INPUT_DIM;
            for j in 0..INPUT_DIM {
                sum += self.w1[base + j] * input[j];
            }
            if sum < 0.0 {
                sum = 0.0;
            }
            h1[i] = sum;
        }

        let mut h2 = vec![0.0f32; HIDDEN_2];
        for i in 0..HIDDEN_2 {
            let mut sum = self.b2[i];
            let base = i * HIDDEN_1;
            for j in 0..HIDDEN_1 {
                sum += self.w2[base + j] * h1[j];
            }
            if sum < 0.0 {
                sum = 0.0;
            }
            h2[i] = sum;
        }

        let mut out = self.b3;
        for j in 0..HIDDEN_2 {
            out += self.w3[j] * h2[j];
        }
        out
    }
}
