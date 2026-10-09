#include "llama_bridge.h"
#include <llama.h>
#include <string.h>
#include <stdlib.h>

#include <ggml-backend.h>
#include <stdio.h>

void bridge_backend_init(void) {
    llama_backend_init();
}

void bridge_backend_free(void) {
    llama_backend_free();
}

void bridge_print_devices(void) {
    llama_backend_init();
    size_t count = ggml_backend_dev_count();
    printf("Available devices (%zu):\n", count);
    if (count == 0) {
        printf("  (none)\n");
        return;
    }
    for (size_t i = 0; i < count; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        printf("  - %s: %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    }
}

BridgeModel *bridge_load_model(const char *path, int32_t n_gpu_layers, const char *devices_str) {
    if (!path) return NULL;
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    ggml_backend_dev_t *dev_array = NULL;
    if (devices_str && strcmp(devices_str, "none") == 0) {
        mparams.n_gpu_layers = 0;
        mparams.devices = NULL;
    } else if (devices_str && strlen(devices_str) > 0) {
        char *buf = strdup(devices_str);
        if (buf) {
            size_t max_devs = 16;
            dev_array = (ggml_backend_dev_t *)calloc(max_devs + 1, sizeof(ggml_backend_dev_t));
            size_t dev_idx = 0;

            char *token = strtok(buf, ",");
            while (token && dev_idx < max_devs) {
                while (*token == ' ') token++;
                char *end = token + strlen(token) - 1;
                while (end > token && *end == ' ') *end-- = '\0';

                ggml_backend_dev_t dev = ggml_backend_dev_by_name(token);
                if (dev) {
                    dev_array[dev_idx++] = dev;
                } else {
                    fprintf(stderr, "Warning: GGML device '%s' not found\n", token);
                }
                token = strtok(NULL, ",");
            }
            free(buf);
            dev_array[dev_idx] = NULL;
            mparams.devices = dev_array;
        }
    }

    struct llama_model *model = llama_model_load_from_file(path, mparams);
    if (dev_array) {
        free(dev_array);
    }
    return (BridgeModel *)model;
}

void bridge_free_model(BridgeModel *model) {
    if (model) {
        llama_model_free((struct llama_model *)model);
    }
}

const char *bridge_get_chat_template(const BridgeModel *model) {
    if (!model) return NULL;
    return llama_model_chat_template((const struct llama_model *)model, NULL);
}

int32_t bridge_get_vocab_size(const BridgeModel *model) {
    if (!model) return 0;
    const struct llama_vocab *vocab = llama_model_get_vocab((const struct llama_model *)model);
    if (!vocab) return 0;
    return llama_vocab_n_tokens(vocab);
}

static const char *special_token_text(const BridgeModel *model, bool want_bos) {
    if (!model) return NULL;
    const struct llama_vocab *vocab = llama_model_get_vocab((const struct llama_model *)model);
    if (!vocab) return NULL;
    llama_token tok = want_bos ? llama_vocab_bos(vocab) : llama_vocab_eos(vocab);
    return tok == LLAMA_TOKEN_NULL ? NULL : llama_vocab_get_text(vocab, tok);
}

const char *bridge_get_bos_text(const BridgeModel *model) {
    return special_token_text(model, true);
}

const char *bridge_get_eos_text(const BridgeModel *model) {
    return special_token_text(model, false);
}

bool bridge_get_add_bos(const BridgeModel *model) {
    if (!model) return false;
    const struct llama_vocab *vocab = llama_model_get_vocab((const struct llama_model *)model);
    return vocab && llama_vocab_get_add_bos(vocab);
}

uint64_t bridge_get_model_size_bytes(const BridgeModel *model) {
    if (!model) return 0;
    return llama_model_size((const struct llama_model *)model);
}

uint64_t bridge_get_model_n_params(const BridgeModel *model) {
    if (!model) return 0;
    return llama_model_n_params((const struct llama_model *)model);
}

BridgeContext *bridge_create_context(
    BridgeModel *model,
    uint32_t n_ctx,
    int32_t n_threads,
    uint32_t n_seq_max,
    uint32_t n_batch,
    uint32_t n_ubatch,
    int32_t flash_attn_type
) {
    if (!model) return NULL;
    struct llama_context_params cparams = llama_context_default_params();
    if (n_ctx > 0) {
        cparams.n_ctx = n_ctx;
    }
    if (n_threads > 0) {
        cparams.n_threads = n_threads;
        cparams.n_threads_batch = n_threads;
    }
    cparams.n_seq_max = (n_seq_max > 1) ? n_seq_max : 1;
    cparams.kv_unified = true;
    cparams.n_batch = (n_batch > 0) ? n_batch : 2048;
    cparams.n_ubatch = (n_ubatch > 0) ? n_ubatch : 512;
    if (cparams.n_ubatch > cparams.n_batch) {
        cparams.n_ubatch = cparams.n_batch;
    }
    cparams.flash_attn_type = (enum llama_flash_attn_type)flash_attn_type;

    return (BridgeContext *)llama_init_from_model((struct llama_model *)model, cparams);
}

void bridge_free_context(BridgeContext *ctx) {
    if (ctx) {
        llama_synchronize((struct llama_context *)ctx);
        llama_free((struct llama_context *)ctx);
    }
}

void bridge_synchronize(BridgeContext *ctx) {
    if (ctx) {
        llama_synchronize((struct llama_context *)ctx);
    }
}

void bridge_clear_memory(BridgeContext *ctx) {
    if (!ctx) return;
    llama_synchronize((struct llama_context *)ctx);
    llama_memory_t mem = llama_get_memory((const struct llama_context *)ctx);
    if (mem) {
        llama_memory_clear(mem, false);
    }
}

bool bridge_kv_cache_seq_rm(BridgeContext *ctx, int32_t seq_id, int32_t p0, int32_t p1) {
    if (!ctx) return false;
    llama_memory_t mem = llama_get_memory((const struct llama_context *)ctx);
    if (!mem) return false;
    return llama_memory_seq_rm(mem, seq_id, p0, p1);
}

int32_t bridge_tokenize(const BridgeModel *model, const char *text, int32_t text_len, int32_t *tokens_out, int32_t max_tokens, bool add_special, bool parse_special) {
    if (!model || !text || !tokens_out || max_tokens <= 0) return -1;
    const struct llama_vocab *vocab = llama_model_get_vocab((const struct llama_model *)model);
    if (!vocab) return -1;
    if (text_len < 0) {
        text_len = (int32_t)strlen(text);
    }
    return llama_tokenize(vocab, text, text_len, tokens_out, max_tokens, add_special, parse_special);
}

int32_t bridge_decode_tokens(BridgeContext *ctx, const int32_t *tokens, int32_t n_tokens) {
    if (!ctx || !tokens || n_tokens <= 0) return -1;
    uint32_t ubatch = llama_n_ubatch((const struct llama_context *)ctx);
    const int32_t chunk_size = (ubatch > 0) ? (int32_t)ubatch : 512;
    int8_t *logits_flags = (int8_t *)calloc((size_t)chunk_size, sizeof(int8_t));
    if (!logits_flags) return -1;

    for (int32_t i = 0; i < n_tokens; i += chunk_size) {
        int32_t cur = (n_tokens - i < chunk_size) ? (n_tokens - i) : chunk_size;
        struct llama_batch batch = llama_batch_get_one((int32_t *)(tokens + i), cur);
        if (i + cur == n_tokens) {
            logits_flags[cur - 1] = 1;
        }
        batch.logits = logits_flags;
        int32_t res = llama_decode((struct llama_context *)ctx, batch);
        if (i + cur == n_tokens) {
            logits_flags[cur - 1] = 0;
        }
        if (res != 0) {
            free(logits_flags);
            return res;
        }
        llama_synchronize((struct llama_context *)ctx);
    }
    free(logits_flags);
    return 0;
}

const float *bridge_get_last_logits(BridgeContext *ctx) {
    if (!ctx) return NULL;
    return llama_get_logits_ith((struct llama_context *)ctx, -1);
}

int32_t bridge_copy_last_logits(BridgeContext *ctx, float *out_logits, size_t out_len) {
    if (!ctx || !out_logits) return -1;
    const struct llama_model *model = llama_get_model((const struct llama_context *)ctx);
    if (!model) return -1;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) return -1;
    const size_t n_vocab = (size_t)llama_vocab_n_tokens(vocab);
    if (out_len < n_vocab) return -2;

    const float *logits = llama_get_logits_ith((struct llama_context *)ctx, -1);
    if (!logits) return -3;
    memcpy(out_logits, logits, n_vocab * sizeof(float));
    return 0;
}

int32_t bridge_eval_prefix_multiseq(BridgeContext *ctx, const int32_t *tokens, int32_t n_tokens, int32_t n_seqs) {
    if (!ctx || !tokens || n_tokens <= 0 || n_seqs <= 0) return -1;
    uint32_t ubatch = llama_n_ubatch((const struct llama_context *)ctx);
    const int32_t chunk_size = (ubatch > 0) ? (int32_t)ubatch : 512;
    for (int32_t i = 0; i < n_tokens; i += chunk_size) {
        int32_t cur = (n_tokens - i < chunk_size) ? (n_tokens - i) : chunk_size;
        struct llama_batch batch = llama_batch_init(cur, 0, n_seqs);
        for (int32_t j = 0; j < cur; j++) {
            batch.token[j] = tokens[i + j];
            batch.pos[j] = i + j;
            batch.n_seq_id[j] = n_seqs;
            for (int32_t s = 0; s < n_seqs; s++) {
                batch.seq_id[j][s] = s;
            }
            batch.logits[j] = 0;
        }
        batch.n_tokens = cur;
        int32_t res = llama_decode((struct llama_context *)ctx, batch);
        llama_batch_free(batch);
        if (res != 0) {
            return res;
        }
        llama_synchronize((struct llama_context *)ctx);
    }
    return 0;
}

int32_t bridge_eval_suffixes_multiseq(
    BridgeContext *ctx,
    const int32_t *tokens,
    const int32_t *suffix_lens,
    int32_t n_suffixes,
    int32_t pos_start,
    float *out_logits,
    size_t out_len
) {
    if (!ctx || !tokens || !suffix_lens || n_suffixes <= 0 || !out_logits) return -1;
    const struct llama_model *model = llama_get_model((const struct llama_context *)ctx);
    const size_t n_vocab = (size_t)llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (out_len < (size_t)n_suffixes * n_vocab) return -2;
    int32_t total_tokens = 0;
    for (int32_t s = 0; s < n_suffixes; s++) {
        total_tokens += suffix_lens[s];
    }
    if (total_tokens <= 0) return -1;
    if (total_tokens > (int32_t)llama_n_batch((const struct llama_context *)ctx)) return -4;

    struct llama_batch batch = llama_batch_init(total_tokens, 0, 1);
    int32_t idx = 0;
    int32_t token_offset = 0;
    for (int32_t s = 0; s < n_suffixes; s++) {
        int32_t len = suffix_lens[s];
        for (int32_t j = 0; j < len; j++) {
            batch.token[idx] = tokens[token_offset + j];
            batch.pos[idx] = pos_start + j;
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = s;
            batch.logits[idx] = (j == len - 1) ? 1 : 0;
            idx++;
        }
        token_offset += len;
    }
    batch.n_tokens = total_tokens;

    int32_t res = llama_decode((struct llama_context *)ctx, batch);
    if (res != 0) {
        llama_batch_free(batch);
        return res;
    }

    idx = 0;
    for (int32_t s = 0; s < n_suffixes; s++) {
        int32_t last_token_idx = idx + suffix_lens[s] - 1;
        const float *row = llama_get_logits_ith((struct llama_context *)ctx, last_token_idx);
        if (!row) {
            llama_batch_free(batch);
            return -3;
        }
        memcpy(out_logits + (size_t)s * n_vocab, row, n_vocab * sizeof(float));
        idx += suffix_lens[s];
    }

    llama_batch_free(batch);
    return 0;
}
