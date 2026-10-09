use std::path::Path;
use std::sync::Arc;

use indexmap::IndexMap;
use log_pi::engine::template::ChatTemplate;
use log_pi::engine::{ContextParams, LlamaEngine, LlamaModel};
use log_pi::protocol::{Answer, Question};

const MODEL_PATH: &str = "/home/nirvam/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/bfc15c382204943c3a8fff0c750b94ae2364d7a3/gemma-4-E4B-it-Q4_K_M.gguf";

/// Builds an engine whose chat template comes from the model's own GGUF metadata.
fn engine_for(model: &Arc<LlamaModel>, max_sequences: u32) -> LlamaEngine {
    let template = ChatTemplate::for_model(model, None).expect("chat template");
    let params = ContextParams {
        n_ctx: 512,
        n_threads: 4,
        n_seq_max: max_sequences,
        ..Default::default()
    };
    LlamaEngine::new(model.clone(), template, params).expect("failed to create engine")
}

#[test]
fn test_live_gemma_engine_inference() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("Model not found at {}, skipping live test.", MODEL_PATH);
        return;
    }

    let model =
        Arc::new(LlamaModel::load(MODEL_PATH, 0, None).expect("failed to load Gemma model"));
    assert!(model.vocab_size() > 0);
    assert!(model.chat_template().is_some());
    assert_eq!(model.bos_text(), Some("<bos>"));

    let mut engine = engine_for(&model, 16);

    let questions = IndexMap::from([(
        "risk".to_string(),
        Question::choice(
            "Classify the security risk of this incoming email message.",
            [
                ("legitimate", "Normal business or personal message"),
                (
                    "spam",
                    "Commercial marketing or promotional unsolicited message",
                ),
                (
                    "phishing",
                    "Malicious deception or fraud attempt targeting sensitive information or funds",
                ),
            ],
        ),
    )]);
    let state =
        "Win one million dollars in cash today from the international lottery! Click here now.";
    let (answers, tokens) = engine.solve_batch(state, &questions).expect("solve choice");
    assert!(tokens > 0);

    match &answers["risk"] {
        Answer::Choice {
            choice,
            probabilities,
            confidence,
        } => {
            println!(
                "Choice: {}, confidence: {}, probs: {:?}",
                choice, confidence, probabilities
            );
            assert!(choice == "phishing" || choice == "spam");
            assert!(*confidence > 0.5);
        }
        _ => panic!("Expected Choice answer"),
    }

    let questions = IndexMap::from([(
        "healthy".to_string(),
        Question::noul("Is the application running in a healthy operational state?"),
    )]);
    let server_state =
        "All database connection pools exhausted, HTTP 500 returned on all endpoints.";
    let (answers, _) = engine
        .solve_batch(server_state, &questions)
        .expect("solve noul");
    match &answers["healthy"] {
        Answer::Noul { noul } => {
            println!("Noul (P_yes): {}", noul);
            assert!(
                *noul < 0.3,
                "Healthy state probability should be low for server crash"
            );
        }
        _ => panic!("Expected Noul answer"),
    }
}

#[test]
fn test_live_option_labels_are_distinct_single_tokens() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("Model not found at {}, skipping live test.", MODEL_PATH);
        return;
    }

    let model = LlamaModel::load(MODEL_PATH, 0, None).expect("failed to load Gemma model");
    let mut seen = std::collections::HashSet::new();
    for label in ('A'..='Z').map(String::from) {
        let tokens = model
            .tokenize(&label, false, false)
            .expect("tokenize label");
        assert_eq!(tokens.len(), 1, "label {label} -> {tokens:?}");
        let id = tokens[0];
        assert!(
            (0..model.vocab_size()).contains(&id),
            "label {label} id {id}"
        );
        assert!(seen.insert(id), "label {label} collides with another label");
    }
}

#[test]
fn test_live_multiseq_matches_sequential() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("Model not found at {}, skipping live test.", MODEL_PATH);
        return;
    }

    let model = Arc::new(LlamaModel::load(MODEL_PATH, 0, None).expect("failed to load model"));
    let state = "All database connection pools exhausted, HTTP 500 returned on all endpoints.";

    let mut questions = IndexMap::new();
    questions.insert(
        "healthy".to_string(),
        Question::noul("Is the application running in a healthy operational state?"),
    );
    questions.insert(
        "team".to_string(),
        Question::choice(
            "Which team should be paged?",
            [
                ("billing", "Payment problems"),
                ("infra", "Servers, databases, outages"),
                ("sales", "Licensing and upgrades"),
            ],
        ),
    );
    questions.insert(
        "severity".to_string(),
        Question::score(
            "How severe is this incident?",
            ["Negligible", "Moderate", "Critical"],
        ),
    );

    // max_seq = 2 forces the multi-sequence path to chunk (3 questions > 2 sequences).
    let mut multi = engine_for(&model, 2);
    let mut sequential = engine_for(&model, 1);

    let (multi_answers, multi_tokens) = multi.solve_batch(state, &questions).expect("multi");
    let (seq_answers, seq_tokens) = sequential.solve_batch(state, &questions).expect("seq");

    assert_eq!(multi_tokens, seq_tokens);
    assert_eq!(
        multi_answers.keys().collect::<Vec<_>>(),
        seq_answers.keys().collect::<Vec<_>>()
    );

    let multi_json = serde_json::to_value(&multi_answers).expect("serialize multi");
    let seq_json = serde_json::to_value(&seq_answers).expect("serialize seq");
    println!("multi: {multi_json}\nseq:   {seq_json}");
    assert_json_close(&multi_json, &seq_json, 0.02);
}

/// Recursively compares two JSON values, allowing `tol` absolute drift on numbers.
fn assert_json_close(a: &serde_json::Value, b: &serde_json::Value, tol: f64) {
    use serde_json::Value;
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!((x - y).abs() <= tol, "{x} vs {y}");
        }
        (Value::Object(x), Value::Object(y)) => {
            assert_eq!(x.len(), y.len());
            for (k, v) in x {
                assert_json_close(v, &y[k], tol);
            }
        }
        _ => assert_eq!(a, b),
    }
}

#[test]
fn test_live_chunked_prefill_long_context() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("Model not found at {}, skipping live test.", MODEL_PATH);
        return;
    }

    let model = Arc::new(LlamaModel::load(MODEL_PATH, 0, None).expect("failed to load model"));
    let template = ChatTemplate::for_model(&model, None).expect("chat template");
    // Allocate 4096 context size to accommodate the long input (> 2048 default n_batch)
    let params = ContextParams {
        n_ctx: 4096,
        n_threads: 4,
        n_seq_max: 1,
        ..Default::default()
    };
    let mut engine = LlamaEngine::new(model, template, params).expect("create engine");

    // Construct a ~2500 token state by repeating policy contract text
    let paragraph = "Section 4.2: The insurance policy covers accidental property damage up to $500,000, excluding intentional misconduct and normal wear and tear. Claims must be submitted within 30 days of the incident with complete documentation. ";
    let state = paragraph.repeat(50);

    let questions = IndexMap::from([(
        "covered".to_string(),
        Question::noul("Does the policy cover accidental property damage?"),
    )]);

    let (answers, total_tokens) = engine
        .solve_batch(&state, &questions)
        .expect("solve long batch");
    println!("Chunked prefill evaluated {total_tokens} tokens successfully!");
    assert!(
        total_tokens > 2048,
        "Expected > 2048 tokens to cross the default n_batch boundary, got {total_tokens}"
    );

    match &answers["covered"] {
        Answer::Noul { noul } => {
            println!("Coverage noul (P_yes): {noul}");
            assert!(*noul > 0.7, "Expected high probability of covered");
        }
        _ => panic!("Expected Noul answer"),
    }
}
