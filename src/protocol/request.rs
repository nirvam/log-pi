//! Jev Wire Protocol Request definitions.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::errors::AppError;

/// Maximum options per `choice` / `score` question: one capital letter (`A..Z`) per option.
pub const MAX_OPTIONS: usize = 26;

/// Content that can be either plain text, a JSON object, or a JSON array.
///
/// In the official Jev Wire Protocol, `state`, `instructions`, and criteria descriptions
/// can be strings or structured JSON values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptContent {
    String(String),
    Structured(serde_json::Value),
}

impl PromptContent {
    pub fn to_prompt_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Structured(serde_json::Value::Null) => String::new(),
            Self::Structured(serde_json::Value::String(s)) => s.clone(),
            Self::Structured(val) => serde_json::to_string(val).unwrap_or_default(),
        }
    }
}

impl Default for PromptContent {
    fn default() -> Self {
        Self::String(String::new())
    }
}

impl From<String> for PromptContent {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for PromptContent {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<serde_json::Value> for PromptContent {
    fn from(v: serde_json::Value) -> Self {
        Self::Structured(v)
    }
}

/// Optional descriptions of the yes and no outcomes for `noul` questions.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(default, rename = "true")]
    pub true_case: Option<PromptContent>,
    #[serde(default, rename = "false")]
    pub false_case: Option<PromptContent>,
}

/// Top-level request payload for `POST /v1/systemone`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneRequest {
    /// Model name requested by caller. Defaults to "log-pi" if omitted.
    #[serde(default = "default_model_name")]
    pub model: String,

    /// The environment state / document text / input content (string, object, or array).
    pub state: PromptContent,

    /// Typed questions to evaluate against the state.
    pub questions: IndexMap<String, Question>,
}

impl SystemOneRequest {
    /// Rejects requests the engine cannot (or should not) evaluate, before any queueing.
    pub fn validate(&self, max_questions: usize) -> Result<(), AppError> {
        if self.questions.is_empty() {
            return Err(AppError::InvalidRequest(
                "Request must contain at least one question".to_string(),
            ));
        }
        if self.questions.len() > max_questions {
            return Err(AppError::InvalidRequest(format!(
                "Request has {} questions; the maximum is {max_questions}",
                self.questions.len()
            )));
        }

        for (id, question) in &self.questions {
            let n_options = question.options().len();
            if n_options == 0 || n_options > MAX_OPTIONS {
                return Err(AppError::InvalidRequest(format!(
                    "Question '{id}' must have between 1 and {MAX_OPTIONS} criteria, got {n_options}"
                )));
            }
        }

        Ok(())
    }
}

fn default_model_name() -> String {
    "log-pi".to_string()
}

/// A single typed question conforming to the Jev protocol primitives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    /// Binary boolean classification (Yes / No).
    Noul {
        #[serde(default)]
        instructions: Option<PromptContent>,
        #[serde(default)]
        criteria: Option<NoulCriteria>,
    },

    /// Multi-class discrete classification over named keys and descriptions.
    Choice {
        #[serde(default)]
        instructions: Option<PromptContent>,
        criteria: IndexMap<String, Option<PromptContent>>,
    },

    /// Continuous calibrated score across an ordered scale of descriptions.
    Score {
        #[serde(default)]
        instructions: Option<PromptContent>,
        criteria: Vec<PromptContent>,
    },
}

impl Question {
    pub fn noul(instructions: impl Into<PromptContent>) -> Self {
        Self::Noul {
            instructions: Some(instructions.into()),
            criteria: None,
        }
    }

    pub fn choice(
        instructions: impl Into<PromptContent>,
        criteria: impl IntoIterator<Item = (impl Into<String>, impl Into<PromptContent>)>,
    ) -> Self {
        Self::Choice {
            instructions: Some(instructions.into()),
            criteria: criteria
                .into_iter()
                .map(|(k, v)| (k.into(), Some(v.into())))
                .collect(),
        }
    }

    pub fn score(
        instructions: impl Into<PromptContent>,
        criteria: impl IntoIterator<Item = impl Into<PromptContent>>,
    ) -> Self {
        Self::Score {
            instructions: Some(instructions.into()),
            criteria: criteria.into_iter().map(Into::into).collect(),
        }
    }

    /// The options offered to the model as `(key, description)`, in label order.
    pub fn options(&self) -> Vec<(String, String)> {
        match self {
            Question::Noul { criteria, .. } => {
                let yes_desc = criteria
                    .as_ref()
                    .and_then(|c| c.true_case.as_ref())
                    .map(|c| c.to_prompt_string())
                    .filter(|s| !s.is_empty())
                    .map(|s| format!("Yes ({s})"))
                    .unwrap_or_else(|| "Yes".to_string());
                let no_desc = criteria
                    .as_ref()
                    .and_then(|c| c.false_case.as_ref())
                    .map(|c| c.to_prompt_string())
                    .filter(|s| !s.is_empty())
                    .map(|s| format!("No ({s})"))
                    .unwrap_or_else(|| "No".to_string());
                vec![("Yes".to_string(), yes_desc), ("No".to_string(), no_desc)]
            }
            Question::Choice { criteria, .. } => criteria
                .iter()
                .map(|(key, desc_opt)| {
                    let desc = desc_opt
                        .as_ref()
                        .map(|c| c.to_prompt_string())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| key.clone());
                    (key.clone(), desc)
                })
                .collect(),
            Question::Score { criteria, .. } => criteria
                .iter()
                .map(|c| {
                    let text = c.to_prompt_string();
                    (text.clone(), text)
                })
                .collect(),
        }
    }

    pub fn instructions(&self) -> String {
        let opt = match self {
            Question::Noul { instructions, .. } => instructions.as_ref(),
            Question::Choice { instructions, .. } => instructions.as_ref(),
            Question::Score { instructions, .. } => instructions.as_ref(),
        };
        opt.map(|c| c.to_prompt_string()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with(questions: Vec<(&str, Question)>) -> SystemOneRequest {
        SystemOneRequest {
            model: default_model_name(),
            state: "state".into(),
            questions: questions
                .into_iter()
                .map(|(id, q)| (id.to_string(), q))
                .collect(),
        }
    }

    fn score_with(n: usize) -> Question {
        Question::Score {
            instructions: Some("rate".into()),
            criteria: (0..n).map(|i| format!("level {i}").into()).collect(),
        }
    }

    #[test]
    fn validate_accepts_boundary_option_counts() {
        let req = request_with(vec![
            ("one", score_with(1)),
            ("max", score_with(MAX_OPTIONS)),
            ("yn", Question::noul("ok?")),
        ]);
        assert!(req.validate(8).is_ok());
    }

    #[test]
    fn validate_rejects_empty_and_oversized_requests() {
        assert!(request_with(vec![]).validate(8).is_err());

        let req = request_with(vec![("a", score_with(2)), ("b", score_with(2))]);
        assert!(req.validate(1).is_err());
    }

    #[test]
    fn validate_rejects_bad_option_counts() {
        assert!(request_with(vec![("empty", score_with(0))])
            .validate(8)
            .is_err());
        assert!(request_with(vec![("many", score_with(MAX_OPTIONS + 1))])
            .validate(8)
            .is_err());

        let empty_choice = Question::Choice {
            instructions: Some("pick".into()),
            criteria: IndexMap::new(),
        };
        assert!(request_with(vec![("c", empty_choice)]).validate(8).is_err());
    }

    #[test]
    fn test_deserialize_official_spec_request() {
        let json = r#"{
            "model": "log-pi",
            "state": "The user complains that their billing card was rejected twice with code 4001.",
            "questions": {
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Is this an urgent matter requiring immediate escalation?"
                },
                "category": {
                    "type": "choice",
                    "instructions": "Which department should handle this ticket?",
                    "criteria": {
                        "billing": "Subscription, credit card, or payment errors",
                        "tech_support": "API bugs, 5xx server errors, system crashes",
                        "sales": "Plan upgrades and enterprise licensing"
                    }
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is this issue for customer operations?",
                    "criteria": [
                        "Negligible impact",
                        "Moderate annoyance",
                        "Critical blocker"
                    ]
                }
            }
        }"#;

        let req: SystemOneRequest = serde_json::from_str(json).expect("valid request json");
        assert_eq!(req.model, "log-pi");
        assert_eq!(req.questions.len(), 3);

        match &req.questions["category"] {
            Question::Choice { criteria, .. } => {
                assert_eq!(criteria.len(), 3);
                assert!(criteria.contains_key("billing"));
            }
            _ => panic!("expected Choice question"),
        }
    }

    #[test]
    fn test_deserialize_structured_state_and_criteria() {
        let json = r#"{
            "model": "log-pi",
            "state": {
                "user_id": 1234,
                "error_code": "ERR_CARD_DECLINED",
                "details": ["attempt 1 failed", "attempt 2 failed"]
            },
            "questions": {
                "is_fraud": {
                    "type": "noul",
                    "instructions": {"task": "Determine whether this transaction appears fraudulent"},
                    "criteria": {
                        "true": "Patterns indicating card testing or stolen credentials",
                        "false": "Legitimate customer billing issue"
                    }
                },
                "routing": {
                    "type": "choice",
                    "criteria": {
                        "support": {"team": "Tier 1 Support"},
                        "finance": null,
                        "security": "Dedicated fraud inspection"
                    }
                },
                "severity": {
                    "type": "score",
                    "criteria": [
                        {"level": 0, "name": "low"},
                        {"level": 1, "name": "medium"},
                        {"level": 2, "name": "high"}
                    ]
                }
            }
        }"#;

        let req: SystemOneRequest = serde_json::from_str(json).expect("valid structured json");
        assert_eq!(req.model, "log-pi");
        let state_str = req.state.to_prompt_string();
        assert!(state_str.contains("ERR_CARD_DECLINED"));
        assert!(state_str.contains("user_id"));

        let q_fraud = &req.questions["is_fraud"];
        assert_eq!(q_fraud.options().len(), 2);
        assert!(q_fraud.instructions().contains("Determine whether"));

        let q_routing = &req.questions["routing"];
        assert_eq!(q_routing.options().len(), 3);

        let q_severity = &req.questions["severity"];
        assert_eq!(q_severity.options().len(), 3);
    }
}
