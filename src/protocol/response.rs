//! Jev Wire Protocol Response definitions.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Top-level response payload for `POST /v1/systemone`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    /// Unique request identification string with prefix `sys1_`.
    pub id: String,

    /// Serving model identifier.
    pub model: String,

    /// Evaluated answers for each submitted question.
    pub answers: IndexMap<String, Answer>,

    /// Token consumption metrics.
    pub usage: Usage,
}

/// A single evaluated answer conforming to Jev protocol specifications.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    /// Result for a `noul` boolean question.
    Noul {
        /// Probability of "Yes", calibrated in [0.0, 1.0].
        noul: f32,
    },

    /// Result for a `choice` multi-class discrete question.
    Choice {
        /// The chosen criterion key with the highest probability.
        choice: String,

        /// Probability distribution across all candidate keys.
        probabilities: IndexMap<String, f32>,

        /// Maximum probability (confidence score).
        confidence: f32,
    },

    /// Result for a `score` continuous expectation question.
    Score {
        /// Continuous expected score across discrete criteria.
        score: f32,

        /// Maximum probability across criteria levels.
        confidence: f32,

        /// Legend mapping numeric indices to criteria descriptions.
        legend: IndexMap<usize, String>,

        /// Probability distribution across criteria index levels.
        probabilities: IndexMap<usize, f32>,
    },
}

/// Token usage metadata. System One models do not generate text, so output_tokens is 0.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

impl Usage {
    pub fn new(input_tokens: usize) -> Self {
        Self {
            input_tokens,
            output_tokens: 0,
        }
    }
}

/// Rounds a float to the given number of decimal places.
#[inline]
pub fn round_to(val: f32, decimals: u32) -> f32 {
    let factor = 10f32.powi(decimals as i32);
    (val * factor).round() / factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_official_spec_response() {
        let mut answers = IndexMap::new();
        answers.insert("is_urgent".to_string(), Answer::Noul { noul: 0.884 });

        let mut cat_probs = IndexMap::new();
        cat_probs.insert("billing".to_string(), 0.962);
        cat_probs.insert("tech_support".to_string(), 0.031);
        cat_probs.insert("sales".to_string(), 0.007);
        answers.insert(
            "category".to_string(),
            Answer::Choice {
                choice: "billing".to_string(),
                probabilities: cat_probs,
                confidence: 0.962,
            },
        );

        let mut legend = IndexMap::new();
        legend.insert(0, "Negligible impact".to_string());
        legend.insert(1, "Moderate annoyance".to_string());
        legend.insert(2, "Critical blocker".to_string());

        let mut sev_probs = IndexMap::new();
        sev_probs.insert(0, 0.02);
        sev_probs.insert(1, 0.24);
        sev_probs.insert(2, 0.74);
        answers.insert(
            "severity".to_string(),
            Answer::Score {
                score: 1.72,
                confidence: 0.74,
                legend,
                probabilities: sev_probs,
            },
        );

        let resp = SystemOneResponse {
            id: "sys1_01j8x7y9k0z2".to_string(),
            model: "log-pi".to_string(),
            answers,
            usage: Usage::new(142),
        };

        let json = serde_json::to_string_pretty(&resp).expect("valid response serialization");
        assert!(json.contains(r#""type": "noul""#));
        assert!(json.contains(r#""choice": "billing""#));
        assert!(json.contains(r#""score": 1.72"#));
        assert!(json.contains(r#""output_tokens": 0"#));
    }
}
