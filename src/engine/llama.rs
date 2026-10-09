//! High-level RAII safe wrappers around llama.cpp C-FFI and decision solver logic.

use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::sync::{Arc, Once};

use indexmap::IndexMap;
use tracing::{info, warn};

use crate::cli::FlashAttnSetting;
use crate::engine::ffi;
use crate::engine::prefix::shared_prefix_len;
use crate::engine::question::{option_letter, PreparedQuestion, LOW_COVERAGE_THRESHOLD};
use crate::engine::template::ChatTemplate;
use crate::errors::AppError;
use crate::protocol::{Answer, Question, MAX_OPTIONS};

/// Parameters for configuring inference contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_threads: i32,
    pub n_seq_max: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub flash_attn: FlashAttnSetting,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_threads: 4,
            n_seq_max: 16,
            n_batch: 2048,
            n_ubatch: 512,
            flash_attn: FlashAttnSetting::Auto,
        }
    }
}

static BACKEND_INIT: Once = Once::new();

/// Initializes the underlying llama.cpp backend once per process lifecycle.
///
/// `Once` blocks concurrent callers until initialization has finished, so no thread can load a
/// model against a half-initialized backend.
pub fn ensure_backend_initialized() {
    BACKEND_INIT.call_once(|| {
        unsafe {
            ffi::bridge_backend_init();
        }
        info!("llama.cpp backend initialized successfully");
    });
}

/// Copies a nullable C string; invalid UTF-8 is treated as absent.
///
/// # Safety
/// `ptr` must be null or point to a NUL-terminated string that outlives this call.
unsafe fn optional_string(ptr: *const std::os::raw::c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok().map(str::to_string)
}

/// RAII wrapper around a loaded GGUF model.
pub struct LlamaModel {
    raw: *mut ffi::BridgeModel,
    vocab_size: i32,
    chat_template: Option<String>,
    bos_text: Option<String>,
    eos_text: Option<String>,
    add_bos: bool,
    model_size: u64,
    n_params: u64,
}

// Safety:
// In llama.cpp, `struct llama_model` is read-only after loading. Reading vocabulary,
// metadata, tokenization rules, and model hyperparameters is thread-safe.
// Multiple threads can safely access `LlamaModel` concurrently or create contexts from it.
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

impl LlamaModel {
    pub fn print_devices() {
        ensure_backend_initialized();
        unsafe {
            ffi::bridge_print_devices();
        }
    }

    pub fn load(path: &str, n_gpu_layers: i32, device: Option<&str>) -> Result<Self, AppError> {
        ensure_backend_initialized();

        let c_path = CString::new(path).map_err(|e| AppError::ModelLoadError {
            path: path.to_string(),
            message: e.to_string(),
        })?;

        let c_device =
            device
                .map(CString::new)
                .transpose()
                .map_err(|e| AppError::ModelLoadError {
                    path: path.to_string(),
                    message: format!("Invalid device string: {e}"),
                })?;
        let dev_ptr = c_device.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        let raw = unsafe { ffi::bridge_load_model(c_path.as_ptr(), n_gpu_layers, dev_ptr) };
        if raw.is_null() {
            return Err(AppError::ModelLoadError {
                path: path.to_string(),
                message: "llama_model_load_from_file returned null".to_string(),
            });
        }

        let vocab_size = unsafe { ffi::bridge_get_vocab_size(raw) };
        let model_size = unsafe { ffi::bridge_get_model_size_bytes(raw) };
        let n_params = unsafe { ffi::bridge_get_model_n_params(raw) };

        let (chat_template, bos_text, eos_text, add_bos) = unsafe {
            (
                optional_string(ffi::bridge_get_chat_template(raw)),
                optional_string(ffi::bridge_get_bos_text(raw)),
                optional_string(ffi::bridge_get_eos_text(raw)),
                ffi::bridge_get_add_bos(raw),
            )
        };

        info!(
            vocab_size = vocab_size,
            size_mb = model_size / (1024 * 1024),
            params = n_params,
            has_template = chat_template.is_some(),
            "GGUF model loaded into memory"
        );

        Ok(Self {
            raw,
            vocab_size,
            chat_template,
            bos_text,
            eos_text,
            add_bos,
            model_size,
            n_params,
        })
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn bos_text(&self) -> Option<&str> {
        self.bos_text.as_deref()
    }

    pub fn eos_text(&self) -> Option<&str> {
        self.eos_text.as_deref()
    }

    pub fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    pub fn model_size_bytes(&self) -> u64 {
        self.model_size
    }

    pub fn model_params_count(&self) -> u64 {
        self.n_params
    }

    pub fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>, AppError> {
        let c_text = CString::new(text).map_err(|_| AppError::tokenize_error(text))?;
        let text_len = text.len() as i32;

        // Initial buffer sizing: 2 tokens per byte + 16 (clamped between 32 and 1M tokens)
        let mut max_tokens = (text.len() * 2 + 16).clamp(32, 1024 * 1024) as i32;
        let mut tokens = vec![0i32; max_tokens as usize];

        let mut n = unsafe {
            ffi::bridge_tokenize(
                self.raw,
                c_text.as_ptr(),
                text_len,
                tokens.as_mut_ptr(),
                max_tokens,
                add_special,
                parse_special,
            )
        };

        // If buffer was too small, llama_tokenize returns negative of the required token count.
        if n < 0 {
            let required = (-n) as usize;
            if required > 0 && required <= (16 * 1024 * 1024) {
                tokens.resize(required, 0);
                max_tokens = required as i32;
                n = unsafe {
                    ffi::bridge_tokenize(
                        self.raw,
                        c_text.as_ptr(),
                        text_len,
                        tokens.as_mut_ptr(),
                        max_tokens,
                        add_special,
                        parse_special,
                    )
                };
            }
        }

        if n < 0 {
            return Err(AppError::tokenize_error(text));
        }

        tokens.truncate(n as usize);
        Ok(tokens)
    }

    /// Tokenizes a fully rendered chat prompt.
    ///
    /// Special tokens in the text (`<bos>`, `<|turn>`, ...) are parsed. A BOS is added only if the
    /// vocabulary asks for one *and* the template did not already emit it, mirroring llama-server.
    pub fn tokenize_prompt(&self, rendered: &str) -> Result<Vec<i32>, AppError> {
        let template_emits_bos = self
            .bos_text
            .as_deref()
            .is_some_and(|bos| !bos.is_empty() && rendered.starts_with(bos));
        self.tokenize(rendered, self.add_bos && !template_emits_bos, true)
    }

    /// Resolves an option label to its single, in-vocabulary token id.
    ///
    /// Answer extraction reads exactly one logit per option, so a label that splits into several
    /// tokens (or maps outside the vocabulary) would silently read the wrong distribution.
    pub fn label_token(&self, label: &str) -> Result<i32, AppError> {
        match self.tokenize(label, false, false)?.as_slice() {
            &[id] if (0..self.vocab_size).contains(&id) => Ok(id),
            tokens => Err(AppError::LabelTokenError(format!(
                "label '{label}' tokenized to {tokens:?}"
            ))),
        }
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::bridge_free_model(self.raw);
            }
        }
    }
}

/// RAII wrapper around an active inference context.
pub struct LlamaContext {
    raw: *mut ffi::BridgeContext,
    n_ctx: u32,
    n_seq_max: u32,
    n_batch: u32,
    n_ubatch: u32,
    flash_attn: FlashAttnSetting,
}

unsafe impl Send for LlamaContext {}

impl LlamaContext {
    pub fn new(model: &LlamaModel, params: &ContextParams) -> Result<Self, AppError> {
        let raw = unsafe {
            ffi::bridge_create_context(
                model.raw,
                params.n_ctx,
                params.n_threads,
                params.n_seq_max,
                params.n_batch,
                params.n_ubatch,
                params.flash_attn.to_c_int(),
            )
        };
        if raw.is_null() {
            return Err(AppError::ContextCreationError(
                "llama_init_from_model returned null".to_string(),
            ));
        }

        Ok(Self {
            raw,
            n_ctx: params.n_ctx,
            n_seq_max: params.n_seq_max.max(1),
            n_batch: params.n_batch,
            n_ubatch: params.n_ubatch,
            flash_attn: params.flash_attn,
        })
    }

    pub fn clear_memory(&mut self) {
        unsafe {
            ffi::bridge_clear_memory(self.raw);
        }
    }

    /// Acquires an RAII memory guard that clears the context KV cache both on entry and upon drop.
    pub fn memory_guard(&mut self) -> ContextMemoryGuard<'_> {
        self.clear_memory();
        ContextMemoryGuard(self)
    }

    pub fn rollback_to(&mut self, n_past: usize) -> Result<(), AppError> {
        let success = unsafe { ffi::bridge_kv_cache_seq_rm(self.raw, -1, n_past as i32, -1) };
        if !success {
            return Err(AppError::ContextCreationError(
                "llama_memory_seq_rm failed to rollback KV cache".to_string(),
            ));
        }
        Ok(())
    }

    pub fn decode(&mut self, tokens: &[i32]) -> Result<(), AppError> {
        if tokens.is_empty() {
            return Ok(());
        }

        let res =
            unsafe { ffi::bridge_decode_tokens(self.raw, tokens.as_ptr(), tokens.len() as i32) };

        if res != 0 {
            return Err(AppError::DecodeError(res));
        }

        Ok(())
    }

    pub fn eval_prefix_multiseq(&mut self, tokens: &[i32], n_seqs: i32) -> Result<(), AppError> {
        if tokens.is_empty() || n_seqs <= 0 {
            return Ok(());
        }

        let res = unsafe {
            ffi::bridge_eval_prefix_multiseq(self.raw, tokens.as_ptr(), tokens.len() as i32, n_seqs)
        };

        if res != 0 {
            return Err(AppError::DecodeError(res));
        }

        Ok(())
    }

    /// Decodes one suffix per sequence and returns the last-token logits of each suffix as a
    /// flat row-major `[suffix_lens.len() x vocab_size]` buffer (copied out of llama.cpp).
    pub fn eval_suffixes_multiseq(
        &mut self,
        tokens: &[i32],
        suffix_lens: &[i32],
        pos_start: i32,
        vocab_size: usize,
    ) -> Result<Vec<f32>, AppError> {
        if tokens.is_empty() || suffix_lens.is_empty() || suffix_lens.iter().any(|&len| len <= 0) {
            return Err(AppError::LogitsError(
                "every suffix must contain at least one token".to_string(),
            ));
        }

        if tokens.len() > self.n_batch as usize {
            return Err(AppError::InvalidRequest(format!(
                "Suffix tokens count ({}) exceeds batch size ({})",
                tokens.len(),
                self.n_batch
            )));
        }

        let mut logits = vec![0.0_f32; suffix_lens.len() * vocab_size];
        let res = unsafe {
            ffi::bridge_eval_suffixes_multiseq(
                self.raw,
                tokens.as_ptr(),
                suffix_lens.as_ptr(),
                suffix_lens.len() as i32,
                pos_start,
                logits.as_mut_ptr(),
                logits.len(),
            )
        };

        if res != 0 {
            return Err(AppError::DecodeError(res));
        }

        Ok(logits)
    }

    pub fn synchronize(&self) {
        unsafe {
            ffi::bridge_synchronize(self.raw);
        }
    }

    pub fn copy_last_logits(&self, out: &mut [f32]) -> Result<(), AppError> {
        let res = unsafe { ffi::bridge_copy_last_logits(self.raw, out.as_mut_ptr(), out.len()) };
        if res != 0 {
            return Err(AppError::LogitsError(format!(
                "bridge_copy_last_logits failed with code {res}"
            )));
        }
        Ok(())
    }

    pub fn get_last_logits(&self, vocab_size: i32) -> Result<&[f32], AppError> {
        let ptr = unsafe { ffi::bridge_get_last_logits(self.raw) };
        if ptr.is_null() {
            return Err(AppError::LogitsError("logits pointer is null".to_string()));
        }

        let slice = unsafe { std::slice::from_raw_parts(ptr, vocab_size as usize) };
        Ok(slice)
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    pub fn n_seq_max(&self) -> u32 {
        self.n_seq_max
    }

    pub fn n_batch(&self) -> u32 {
        self.n_batch
    }

    pub fn n_ubatch(&self) -> u32 {
        self.n_ubatch
    }

    pub fn flash_attn(&self) -> FlashAttnSetting {
        self.flash_attn
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::bridge_free_context(self.raw);
            }
        }
    }
}

/// RAII guard ensuring the context KV cache is cleanly reset upon exit (even on early return or error).
pub struct ContextMemoryGuard<'a>(&'a mut LlamaContext);

impl<'a> std::ops::Deref for ContextMemoryGuard<'a> {
    type Target = LlamaContext;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> std::ops::DerefMut for ContextMemoryGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

impl<'a> Drop for ContextMemoryGuard<'a> {
    fn drop(&mut self) {
        self.0.clear_memory();
    }
}


/// Immutable model facts, captured once so request handlers never need the engine.
#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    pub n_ctx: u32,
    pub vocab_size: i32,
    pub n_params: u64,
    pub size_bytes: u64,
    pub created: u64,
}

/// A question together with its fully rendered and tokenized prompt.
struct PendingQuestion {
    question: PreparedQuestion,
    tokens: Vec<i32>,
}

impl PendingQuestion {
    fn decode_into(&self, answers: &mut IndexMap<String, Answer>, logits: &[f32]) {
        let decoded = self.question.decode(logits);
        if decoded.coverage < LOW_COVERAGE_THRESHOLD {
            warn!(
                question = self.question.id(),
                coverage = decoded.coverage,
                "Model put little probability on the option labels; answer may be unreliable"
            );
        }
        answers.insert(self.question.id().to_string(), decoded.answer);
    }
}

/// High-level single-pass Decision Engine.
pub struct LlamaEngine {
    model: Arc<LlamaModel>,
    context: LlamaContext,
    template: ChatTemplate,
    /// Token ids of the option labels `A..Z`, validated once at startup.
    label_tokens: Vec<i32>,
}

impl LlamaEngine {
    pub fn new(
        model: Arc<LlamaModel>,
        template: ChatTemplate,
        params: ContextParams,
    ) -> Result<Self, AppError> {
        let context = LlamaContext::new(&model, &params)?;

        let label_tokens = (0..MAX_OPTIONS)
            .map(|idx| model.label_token(&option_letter(idx)))
            .collect::<Result<Vec<_>, _>>()?;
        if label_tokens.iter().collect::<HashSet<_>>().len() != label_tokens.len() {
            return Err(AppError::LabelTokenError(format!(
                "option labels collide: {label_tokens:?}"
            )));
        }

        Ok(Self {
            model,
            context,
            template,
            label_tokens,
        })
    }

    pub fn model_info(&self) -> ModelInfo {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        ModelInfo {
            n_ctx: self.context.n_ctx(),
            vocab_size: self.model.vocab_size(),
            n_params: self.model.model_params_count(),
            size_bytes: self.model.model_size_bytes(),
            created,
        }
    }

    /// Returns the label token ids for a question with `n_options` options.
    fn label_tokens_for(&self, n_options: usize) -> Result<Vec<i32>, AppError> {
        if n_options == 0 || n_options > MAX_OPTIONS {
            return Err(AppError::InvalidRequest(format!(
                "A question must have between 1 and {MAX_OPTIONS} options, got {n_options}"
            )));
        }
        Ok(self.label_tokens[..n_options].to_vec())
    }

    /// Renders and tokenizes the full prompt of every question.
    fn prepare(
        &self,
        state: &str,
        questions: &IndexMap<String, Question>,
    ) -> Result<Vec<PendingQuestion>, AppError> {
        let n_ctx = self.context.n_ctx as usize;

        questions
            .iter()
            .map(|(id, question)| {
                let label_tokens = self.label_tokens_for(question.options().len())?;
                let prepared = PreparedQuestion::new(id, state, question, label_tokens);
                let rendered = self.template.render(prepared.messages())?;
                let tokens = self.model.tokenize_prompt(&rendered)?;

                if tokens.len() > n_ctx {
                    return Err(AppError::InvalidRequest(format!(
                        "Prompt for question '{id}' is {} tokens, exceeding the context window ({n_ctx})",
                        tokens.len()
                    )));
                }
                Ok(PendingQuestion {
                    question: prepared,
                    tokens,
                })
            })
            .collect()
    }

    /// Answers every question about the same state in one pass.
    ///
    /// Each question is rendered through the model's chat template as a complete prompt. The
    /// token-level common prefix (system prompt + state) is decoded once and the per-question
    /// remainders branch off it, so the shared text is never evaluated twice.
    pub fn solve_batch(
        &mut self,
        state: &str,
        questions: &IndexMap<String, Question>,
    ) -> Result<(IndexMap<String, Answer>, usize), AppError> {
        if questions.is_empty() {
            return Ok((IndexMap::new(), 0));
        }

        let pending = self.prepare(state, questions)?;
        let vocab_size = self.model.vocab_size as usize;
        let mut answers = IndexMap::with_capacity(pending.len());
        // RAII guard automatically cleans up KV cache memory (and synchronizes GPU) on normal exit or early return
        let mut ctx = self.context.memory_guard();

        // Fast path for single question: evaluate the full prompt in a single pass without
        // prefix extraction or multi-sequence broadcast overhead.
        if pending.len() == 1 {
            let item = &pending[0];
            let total_tokens = item.tokens.len();
            let n_ctx = ctx.n_ctx as usize;
            if total_tokens > n_ctx {
                return Err(AppError::InvalidRequest(format!(
                    "Prompt length ({total_tokens} tokens) exceeds context window ({n_ctx})."
                )));
            }
            ctx.decode(&item.tokens)?;
            let mut logits = vec![0.0_f32; vocab_size];
            ctx.copy_last_logits(&mut logits)?;
            item.decode_into(&mut answers, &logits);
            return Ok((answers, total_tokens));
        }

        let prompts: Vec<&[i32]> = pending.iter().map(|p| p.tokens.as_slice()).collect();
        let n_prefix = shared_prefix_len(&prompts);
        let prefix = &pending[0].tokens[..n_prefix];

        let total_tokens = n_prefix
            + pending
                .iter()
                .map(|p| p.tokens.len() - n_prefix)
                .sum::<usize>();
        let max_seq = ctx.n_seq_max as usize;
        let n_ctx = ctx.n_ctx as usize;

        // Verify that unified KV cache can accommodate concurrent sequences
        let required_kv = if max_seq <= 1 {
            let max_suffix = pending
                .iter()
                .map(|p| p.tokens.len() - n_prefix)
                .max()
                .unwrap_or(0);
            n_prefix + max_suffix
        } else {
            let n_broadcast = pending.len().min(max_seq);
            pending
                .chunks(max_seq)
                .map(|chunk| {
                    let chunk_suffix_sum: usize =
                        chunk.iter().map(|p| p.tokens.len() - n_prefix).sum();
                    n_prefix * n_broadcast + chunk_suffix_sum
                })
                .max()
                .unwrap_or(0)
        };

        if required_kv > n_ctx {
            return Err(AppError::InvalidRequest(format!(
                "Required KV cache size ({required_kv} tokens) exceeds context window ({n_ctx}). Shorten inputs or reduce --max-sequences."
            )));
        }

        if max_seq <= 1 {
            // Sequential: keep the prefix in the KV cache and roll back after each suffix.
            ctx.decode(prefix)?;

            for item in &pending {
                ctx.decode(&item.tokens[n_prefix..])?;
                let mut logits = vec![0.0_f32; vocab_size];
                ctx.copy_last_logits(&mut logits)?;
                item.decode_into(&mut answers, &logits);
                ctx.rollback_to(n_prefix)?;
            }
        } else {
            // Multi-sequence: broadcast the prefix to every sequence, then decode suffixes
            // concurrently in chunks of at most `max_seq` sequences.
            let n_broadcast = pending.len().min(max_seq);
            ctx.eval_prefix_multiseq(prefix, n_broadcast as i32)?;

            for (chunk_idx, chunk) in pending.chunks(max_seq).enumerate() {
                let mut flat_tokens = Vec::new();
                let mut suffix_lens = Vec::with_capacity(chunk.len());
                for item in chunk {
                    let suffix = &item.tokens[n_prefix..];
                    flat_tokens.extend_from_slice(suffix);
                    suffix_lens.push(suffix.len() as i32);
                }

                let logits_block = ctx.eval_suffixes_multiseq(
                    &flat_tokens,
                    &suffix_lens,
                    n_prefix as i32,
                    vocab_size,
                )?;
                for (item, logits) in chunk.iter().zip(logits_block.chunks_exact(vocab_size)) {
                    item.decode_into(&mut answers, logits);
                }

                if (chunk_idx + 1) * max_seq < pending.len() {
                    ctx.rollback_to(n_prefix)?;
                }
            }
        }

        Ok((answers, total_tokens))
    }
}
