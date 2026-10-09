//! C FFI bridge bindings for llama.cpp.

use std::os::raw::c_char;

pub enum BridgeModelOpaque {}
pub enum BridgeContextOpaque {}

pub type BridgeModel = BridgeModelOpaque;
pub type BridgeContext = BridgeContextOpaque;

#[allow(dead_code)]
extern "C" {
    pub fn bridge_backend_init();
    pub fn bridge_backend_free();
    pub fn bridge_print_devices();

    pub fn bridge_load_model(
        path: *const c_char,
        n_gpu_layers: i32,
        devices_str: *const c_char,
    ) -> *mut BridgeModel;
    pub fn bridge_free_model(model: *mut BridgeModel);

    pub fn bridge_get_chat_template(model: *const BridgeModel) -> *const c_char;
    pub fn bridge_get_vocab_size(model: *const BridgeModel) -> i32;
    pub fn bridge_get_bos_text(model: *const BridgeModel) -> *const c_char;
    pub fn bridge_get_eos_text(model: *const BridgeModel) -> *const c_char;
    pub fn bridge_get_add_bos(model: *const BridgeModel) -> bool;
    pub fn bridge_get_model_size_bytes(model: *const BridgeModel) -> u64;
    pub fn bridge_get_model_n_params(model: *const BridgeModel) -> u64;

    pub fn bridge_create_context(
        model: *mut BridgeModel,
        n_ctx: u32,
        n_threads: i32,
        n_seq_max: u32,
        n_batch: u32,
        n_ubatch: u32,
        flash_attn_type: i32,
    ) -> *mut BridgeContext;
    pub fn bridge_free_context(ctx: *mut BridgeContext);

    pub fn bridge_synchronize(ctx: *mut BridgeContext);
    pub fn bridge_clear_memory(ctx: *mut BridgeContext);
    pub fn bridge_kv_cache_seq_rm(ctx: *mut BridgeContext, seq_id: i32, p0: i32, p1: i32) -> bool;
    pub fn bridge_tokenize(
        model: *const BridgeModel,
        text: *const c_char,
        text_len: i32,
        tokens_out: *mut i32,
        max_tokens: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
    pub fn bridge_decode_tokens(ctx: *mut BridgeContext, tokens: *const i32, n_tokens: i32) -> i32;
    pub fn bridge_get_last_logits(ctx: *mut BridgeContext) -> *const f32;
    pub fn bridge_copy_last_logits(
        ctx: *mut BridgeContext,
        out_logits: *mut f32,
        out_len: usize,
    ) -> i32;
    pub fn bridge_eval_prefix_multiseq(
        ctx: *mut BridgeContext,
        tokens: *const i32,
        n_tokens: i32,
        n_seqs: i32,
    ) -> i32;
    pub fn bridge_eval_suffixes_multiseq(
        ctx: *mut BridgeContext,
        tokens: *const i32,
        suffix_lens: *const i32,
        n_suffixes: i32,
        pos_start: i32,
        out_logits: *mut f32,
        out_len: usize,
    ) -> i32;
}
