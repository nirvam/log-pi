//! Mathematical primitives for log-probability and numerically stable softmax calculations.
//!
//! Outer brand identity: log(π)
//! Inner kernel identity: log(P_i) = z_i - ln \sum_j e^{z_j}

/// Computes the numerically stable Log-Sum-Exp reduction of a slice of logits:
///
/// LSE(z) = M + ln \sum_i e^{z_i - M}, where M = \max_i(z_i)
#[inline]
pub fn log_sum_exp(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return f32::NEG_INFINITY;
    }

    let mut max_val = f32::NEG_INFINITY;
    for &val in logits {
        if val > max_val {
            max_val = val;
        }
    }

    if max_val.is_infinite() {
        return max_val;
    }

    let sum_exp: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Computes numerically stable log-softmax:
///
/// \log(P_i) = z_i - LSE(z)
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let lse = log_sum_exp(logits);
    logits.iter().map(|&z| z - lse).collect()
}

/// Computes normalized probabilities from logits using Softmax:
///
/// P_i = \exp(\log(P_i))
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    log_softmax(logits).iter().map(|&lp| lp.exp()).collect()
}

/// Probability mass the softmax over the *full* vocabulary assigns to the candidate tokens:
///
/// \text{coverage} = \frac{\sum_{c \in C} e^{z_c}}{\sum_j e^{z_j}} = e^{LSE(z_C) - LSE(z)}
///
/// Values near 0 mean the model wants to say something other than the offered options.
pub fn candidate_coverage(logits: &[f32], candidates: &[f32]) -> f32 {
    (log_sum_exp(candidates) - log_sum_exp(logits)).exp()
}

/// Computes the expected score for an ordered discrete distribution:
///
/// \text{Score} = \sum_{i=0}^{N-1} i \times P_i
pub fn expected_score(probabilities: &[f32]) -> f32 {
    probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as f32) * p)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp_basic() {
        let logits = [1000.0, 1000.0];
        // ln(e^1000 + e^1000) = 1000 + ln(2) = 1000.693147
        let lse = log_sum_exp(&logits);
        assert!((lse - 1000.6931_f32).abs() < 1e-3);
    }

    #[test]
    fn test_prototype_values() {
        // From myfev.py output:
        // Logits: Legitimate: 15.997, Spam: 22.06, Phishing: 25.321
        let logits = [15.997_f32, 22.06_f32, 25.321_f32];
        let logprobs = log_softmax(&logits);
        let probs = softmax(&logits);

        // Expected logprobs: [-9.362, -3.299, -0.038]
        assert!((logprobs[0] - (-9.362)).abs() < 1e-2);
        assert!((logprobs[1] - (-3.299)).abs() < 1e-2);
        assert!((logprobs[2] - (-0.038)).abs() < 1e-2);

        // Expected probs: [0.000, 0.037, 0.963]
        assert!((probs[0] - 0.000).abs() < 1e-3);
        assert!((probs[1] - 0.037).abs() < 1e-3);
        assert!((probs[2] - 0.963).abs() < 1e-3);

        // Probability sum should be 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_candidate_coverage() {
        // Uniform logits over 10 tokens: 2 candidates carry 2/10 of the mass.
        let logits = [3.0_f32; 10];
        let coverage = candidate_coverage(&logits, &logits[..2]);
        assert!((coverage - 0.2).abs() < 1e-5);

        // All candidates dominate: coverage approaches 1.
        let mut peaked = [-20.0_f32; 10];
        peaked[4] = 5.0;
        peaked[5] = 5.0;
        assert!(candidate_coverage(&peaked, &[5.0, 5.0]) > 0.999);
    }

    #[test]
    fn test_expected_score() {
        // Negligible (0): 0.02, Moderate (1): 0.24, Critical (2): 0.74
        // Expected: 0*0.02 + 1*0.24 + 2*0.74 = 0.24 + 1.48 = 1.72
        let probs = [0.02_f32, 0.24_f32, 0.74_f32];
        let score = expected_score(&probs);
        assert!((score - 1.72).abs() < 1e-4);
    }
}
