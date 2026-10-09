#ifndef LLAMA_BRIDGE_H
#define LLAMA_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_model BridgeModel;
typedef struct llama_context BridgeContext;

void bridge_backend_init(void);
void bridge_backend_free(void);
void bridge_print_devices(void);

BridgeModel *bridge_load_model(const char *path, int32_t n_gpu_layers, const char *devices_str);
void bridge_free_model(BridgeModel *model);

const char *bridge_get_chat_template(const BridgeModel *model);
int32_t bridge_get_vocab_size(const BridgeModel *model);
const char *bridge_get_bos_text(const BridgeModel *model);
const char *bridge_get_eos_text(const BridgeModel *model);
bool bridge_get_add_bos(const BridgeModel *model);
uint64_t bridge_get_model_size_bytes(const BridgeModel *model);
uint64_t bridge_get_model_n_params(const BridgeModel *model);

BridgeContext *bridge_create_context(
    BridgeModel *model,
    uint32_t n_ctx,
    int32_t n_threads,
    uint32_t n_seq_max,
    uint32_t n_batch,
    uint32_t n_ubatch,
    int32_t flash_attn_type
);
void bridge_free_context(BridgeContext *ctx);

void bridge_synchronize(BridgeContext *ctx);
void bridge_clear_memory(BridgeContext *ctx);
bool bridge_kv_cache_seq_rm(BridgeContext *ctx, int32_t seq_id, int32_t p0, int32_t p1);
int32_t bridge_tokenize(const BridgeModel *model, const char *text, int32_t text_len, int32_t *tokens_out, int32_t max_tokens, bool add_special, bool parse_special);
int32_t bridge_decode_tokens(BridgeContext *ctx, const int32_t *tokens, int32_t n_tokens);
const float *bridge_get_last_logits(BridgeContext *ctx);
int32_t bridge_copy_last_logits(BridgeContext *ctx, float *out_logits, size_t out_len);
int32_t bridge_eval_prefix_multiseq(BridgeContext *ctx, const int32_t *tokens, int32_t n_tokens, int32_t n_seqs);
int32_t bridge_eval_suffixes_multiseq(
    BridgeContext *ctx,
    const int32_t *tokens,
    const int32_t *suffix_lens,
    int32_t n_suffixes,
    int32_t pos_start,
    float *out_logits,
    size_t out_len
);
#ifdef __cplusplus
}
#endif

#endif // LLAMA_BRIDGE_H
