//! Turns protocol questions into chat prompts and next-token logits into protocol answers.
//!
//! Everything here is pure (no FFI, no model), so the decoding math is testable with synthetic
//! logits. One code path serves every question type and every batch size.

use indexmap::IndexMap;

use crate::engine::math;
use crate::engine::template::ChatMessage;
use crate::protocol::{round_to, Answer, Question};

/// Probability mass on the option labels below which the model is most likely not answering in
/// the requested format (the reported probabilities are then conditioned on a rare event).
pub const LOW_COVERAGE_THRESHOLD: f32 = 0.5;

const SYSTEM_PROMPT: &str =
    "You are an accurate decision classifier. Choose exactly one option letter.";

/// Maps an option index (`< MAX_OPTIONS`) to its label letter `A..Z`.
pub fn option_letter(idx: usize) -> String {
    debug_assert!(idx < crate::protocol::MAX_OPTIONS);
    ((b'A' + idx as u8) as char).to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Noul,
    Choice,
    Score,
}

/// The result of decoding one question's logits.
pub struct Decoded {
    pub answer: Answer,
    /// Probability mass the model puts on the option labels (over the full vocabulary).
    pub coverage: f32,
}

/// A question bound to its prompt messages and the label tokens whose logits answer it.
pub struct PreparedQuestion {
    id: String,
    kind: Kind,
    /// Option keys in label order: criteria keys (choice), descriptions (score), Yes/No (noul).
    keys: Vec<String>,
    label_tokens: Vec<i32>,
    messages: Vec<ChatMessage>,
}

impl PreparedQuestion {
    /// `label_tokens` must hold one token per option, in option order.
    pub fn new(id: &str, state: &str, question: &Question, label_tokens: Vec<i32>) -> Self {
        let options = question.options();
        debug_assert_eq!(options.len(), label_tokens.len());

        let options_text = options
            .iter()
            .enumerate()
            .map(|(idx, (_, text))| format!("{}. {text}", option_letter(idx)))
            .collect::<Vec<_>>()
            .join("\n");
        let user_prompt = format!(
            "State:\n{}\n\nQuestion:\n{}\n\nOptions:\n{}\n\nAnswer with only the letter:",
            state.trim(),
            question.instructions().trim(),
            options_text.trim()
        );

        Self {
            id: id.to_string(),
            kind: match question {
                Question::Noul { .. } => Kind::Noul,
                Question::Choice { .. } => Kind::Choice,
                Question::Score { .. } => Kind::Score,
            },
            keys: options.into_iter().map(|(key, _)| key).collect(),
            label_tokens,
            messages: vec![
                ChatMessage::new("system", SYSTEM_PROMPT),
                ChatMessage::new("user", user_prompt),
            ],
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Converts the next-token logits (one value per vocabulary entry) into an answer.
    pub fn decode(&self, logits: &[f32]) -> Decoded {
        let candidates: Vec<f32> = self
            .label_tokens
            .iter()
            .map(|&token| logits[token as usize])
            .collect();
        let probs = math::softmax(&candidates);
        let coverage = math::candidate_coverage(logits, &candidates);

        // argmax / confidence use the unrounded probabilities; rounding is only
        // applied to the serialized display values so the winner is never chosen
        // off a rounded magnitude (matters when two options are within half an ulp).
        let (best_idx, confidence) =
            probs
                .iter()
                .copied()
                .enumerate()
                .fold(
                    (0, -1.0_f32),
                    |best, (idx, p)| {
                        if p > best.1 {
                            (idx, p)
                        } else {
                            best
                        }
                    },
                );

        // Display probabilities are rounded independently; their sum is only
        // approximately 1.0 (e.g. three-way split rounds to 0.333 + 0.333 + 0.333).
        let rounded: Vec<f32> = probs.iter().map(|&p| round_to(p, 3)).collect();

        let answer = match self.kind {
            Kind::Noul => Answer::Noul { noul: rounded[0] },
            Kind::Choice => Answer::Choice {
                choice: self.keys[best_idx].clone(),
                probabilities: self.keys.iter().cloned().zip(rounded).collect(),
                confidence,
            },
            Kind::Score => Answer::Score {
                score: round_to(math::expected_score(&probs), 2),
                confidence,
                legend: self.keys.iter().cloned().enumerate().collect(),
                probabilities: rounded.into_iter().enumerate().collect::<IndexMap<_, _>>(),
            },
        };

        Decoded { answer, coverage }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 12-token "vocabulary" whose option labels live at ids 3, 4, 5.
    const LABELS: [i32; 3] = [3, 4, 5];

    fn logits_with(label_logits: [f32; 3], filler: f32) -> Vec<f32> {
        let mut logits = vec![filler; 12];
        for (token, value) in LABELS.iter().zip(label_logits) {
            logits[*token as usize] = value;
        }
        logits
    }

    fn choice() -> Question {
        Question::choice(
            " Which team? ",
            [
                ("billing", "Payments"),
                ("infra", "Servers"),
                ("sales", "Upgrades"),
            ],
        )
    }

    fn score() -> Question {
        Question::score("How bad?", ["Low", "Mid", "High"])
    }

    fn noul() -> Question {
        Question::noul("Is it up?")
    }

    #[test]
    fn prompt_lists_lettered_options_and_trims_inputs() {
        let q = PreparedQuestion::new("q", "  state  ", &choice(), LABELS.to_vec());
        let [system, user] = q.messages() else {
            panic!("expected system + user messages");
        };
        assert_eq!(system.role, "system");
        assert_eq!(user.role, "user");
        assert_eq!(
            user.content,
            "State:\nstate\n\nQuestion:\nWhich team?\n\nOptions:\nA. Payments\nB. Servers\nC. Upgrades\n\nAnswer with only the letter:"
        );
    }

    #[test]
    fn noul_is_offered_as_yes_no() {
        let q = PreparedQuestion::new("q", "s", &noul(), vec![3, 4]);
        assert!(q.messages()[1].content.contains("A. Yes\nB. No"));
    }

    #[test]
    fn choice_picks_the_most_likely_option() {
        let q = PreparedQuestion::new("q", "s", &choice(), LABELS.to_vec());
        let Decoded { answer, coverage } = q.decode(&logits_with([0.0, 6.0, 1.0], -30.0));

        let Answer::Choice {
            choice,
            probabilities,
            confidence,
        } = answer
        else {
            panic!("expected a choice answer");
        };
        assert_eq!(choice, "infra");
        // confidence is the unrounded winner probability; the serialized
        // probabilities are rounded to 3 decimals for display, so the two may
        // differ by up to half an ulp and their sum is only approximating 1.0.
        assert!((confidence - probabilities["infra"]).abs() < 5e-4);
        assert_eq!(
            probabilities.keys().collect::<Vec<_>>(),
            ["billing", "infra", "sales"]
        );
        assert!((probabilities.values().sum::<f32>() - 1.0).abs() < 5e-3);
        assert!(coverage > 0.999);
    }

    #[test]
    fn ties_resolve_to_the_first_option() {
        let q = PreparedQuestion::new("q", "s", &choice(), LABELS.to_vec());
        let Answer::Choice { choice, .. } = q.decode(&logits_with([2.0, 2.0, 2.0], -30.0)).answer
        else {
            panic!("expected a choice answer");
        };
        assert_eq!(choice, "billing");
    }

    #[test]
    fn noul_reports_probability_of_yes() {
        let q = PreparedQuestion::new("q", "s", &noul(), vec![3, 4]);
        let mut logits = vec![-30.0; 12];
        logits[3] = 1.0;
        logits[4] = 1.0;
        let Answer::Noul { noul } = q.decode(&logits).answer else {
            panic!("expected a noul answer");
        };
        assert_eq!(noul, 0.5);

        logits[3] = 11.0;
        let Answer::Noul { noul } = q.decode(&logits).answer else {
            panic!("expected a noul answer");
        };
        assert!(noul > 0.999);
    }

    #[test]
    fn score_is_the_expected_option_index() {
        let q = PreparedQuestion::new("q", "s", &score(), LABELS.to_vec());
        // P ≈ [0, 0.5, 0.5] -> expected index 1.5.
        let Answer::Score {
            score,
            confidence,
            legend,
            probabilities,
        } = q.decode(&logits_with([-30.0, 1.0, 1.0], -30.0)).answer
        else {
            panic!("expected a score answer");
        };
        assert_eq!(score, 1.5);
        assert_eq!(confidence, 0.5);
        assert_eq!(legend[&2], "High");
        assert_eq!(probabilities[&0], 0.0);
    }

    #[test]
    fn coverage_drops_when_the_model_prefers_other_tokens() {
        let q = PreparedQuestion::new("q", "s", &choice(), LABELS.to_vec());
        // Labels at 0.0, but a non-label token sits at 10.0: the labels hold almost no mass.
        let mut logits = logits_with([0.0, 0.0, 0.0], -30.0);
        logits[9] = 10.0;
        let Decoded { coverage, .. } = q.decode(&logits);
        assert!(coverage < LOW_COVERAGE_THRESHOLD);
    }
}
