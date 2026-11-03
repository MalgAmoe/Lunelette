#include "block.h"
#include "attention.h"
#include "lunelette.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 3072
#define QUERY 4096
#define KV 1024
#define INNER 9216
#define PREFIX "language_model.model.layers.0."

static const st_tensor *require(st_file *file, const char *name,
                                size_t rows, size_t cols, char *error, size_t capacity)
{
    const st_tensor *tensor = st_find(file, name);
    if (!tensor || tensor->dtype != ST_BF16 || tensor->rank != (cols ? 2u : 1u) ||
        tensor->shape[0] != rows || (cols && tensor->shape[1] != cols)) {
        if (error && capacity) {
            snprintf(error, capacity, "%s: expected BF16 %zu rows, %zu columns (0=vector)",
                     name, rows, cols);
        }
        return NULL;
    }
    return tensor;
}

static int project(st_file *file, const char *name, size_t rows, size_t cols,
                   const float *input, float *output, size_t count,
                   float *matrix, char *error, size_t capacity)
{
    const st_tensor *tensor = require(file, name, rows, cols, error, capacity);
    if (!tensor) {
        return -1;
    }
    if (st_load_bf16(file, tensor, matrix, rows * cols)) {
        if (error && capacity) {
            snprintf(error, capacity, "%s: BF16 read failed", name);
        }
        return -1;
    }
    for (size_t t = 0; t < count; t++) {
        if (lunelette_matvec(matrix, input + t * cols, output + t * rows, rows, cols)) {
            if (error && capacity) {
                snprintf(error, capacity, "%s: matvec failed at token %zu", name, t);
            }
            return -1;
        }
    }
    return 0;
}

int lunelette_first_block(st_file *shard, const int64_t *ids, size_t count,
                          float *output, char *error, size_t capacity)
{
    int result = -1;
    float *workspace = NULL, *matrix = NULL;
    if (error && capacity) {
        snprintf(error, capacity, "first block: invalid arguments");
    }
    if (!shard || !shard->file || !ids || !output || !count || count > 256) {
        return -1;
    }
    for (size_t t = 0; t < count; t++) {
        if (ids[t] < 0 || ids[t] >= 131072) {
            if (error && capacity) {
                snprintf(error, capacity, "embedding: invalid token %lld at row %zu",
                         (long long)ids[t], t);
            }
            return -1;
        }
    }
    const st_tensor *embedding = require(shard, "language_model.model.embed_tokens.weight",
                                         131072, HIDDEN, error, capacity);
    if (!embedding) {
        return -1;
    }
    const st_tensor *norm = require(shard, PREFIX "input_layernorm.weight",
                                    HIDDEN, 0, error, capacity);
    if (!norm) {
        return -1;
    }
    /* All dimensions and token count are fixed/bounded before multiplication.
     * Token-major activations, row-major matrices. Never expand embeddings. */
    size_t stride = 3 * HIDDEN + 2 * QUERY + 2 * KV + 2 * INNER;
    workspace = malloc(count * stride * sizeof(float));
    matrix = malloc((size_t)INNER * HIDDEN * sizeof(float));
    if (!workspace || !matrix) {
        if (error && capacity) {
            snprintf(error, capacity, "first block: workspace/matrix allocation failed");
        }
        goto done;
    }
    float *hidden = workspace;
    float *normalized = hidden + count * HIDDEN;
    float *projected = normalized + count * HIDDEN;
    float *q = projected + count * HIDDEN;
    float *k = q + count * QUERY;
    float *v = k + count * KV;
    float *attention = v + count * KV;
    float *gate = attention + count * QUERY;
    float *up = gate + count * INNER;
    float norm_weight[HIDDEN], scratch[256];
    uint32_t positions[256];
    if (error && capacity) {
        snprintf(error, capacity, PREFIX "input_layernorm.weight: read failed");
    }
    if (st_load_bf16(shard, norm, norm_weight, HIDDEN)) {
        goto done;
    }
    for (size_t t = 0; t < count; t++) {
        positions[t] = (uint32_t)t;
        if (st_embedding(shard, embedding, ids[t], hidden + t * HIDDEN, HIDDEN)) {
            if (error && capacity) {
                snprintf(error, capacity, "embedding: read failed at token row %zu", t);
            }
            goto done;
        }
        lunelette_rmsnorm(hidden + t * HIDDEN, norm_weight,
                          normalized + t * HIDDEN, HIDDEN, 1e-5f);
    }
    if (project(shard, PREFIX "self_attn.q_proj.weight", QUERY, HIDDEN,
                normalized, q, count, matrix, error, capacity) ||
        project(shard, PREFIX "self_attn.k_proj.weight", KV, HIDDEN,
                normalized, k, count, matrix, error, capacity) ||
        project(shard, PREFIX "self_attn.v_proj.weight", KV, HIDDEN,
                normalized, v, count, matrix, error, capacity)) {
        goto done;
    }
    if (lunelette_ministral3_yarn(q, k, positions, count) ||
        lunelette_causal_gqa(q, k, v, attention, scratch, count, 32, 8, 128)) {
        if (error && capacity) {
            snprintf(error, capacity, "first block: YaRN/causal GQA failed");
        }
        goto done;
    }
    if (project(shard, PREFIX "self_attn.o_proj.weight", HIDDEN, QUERY,
                attention, projected, count, matrix, error, capacity)) {
        goto done;
    }
    for (size_t i = 0; i < count * HIDDEN; i++) {
        hidden[i] += projected[i];
    }
    norm = require(shard, PREFIX "post_attention_layernorm.weight",
                   HIDDEN, 0, error, capacity);
    if (!norm) {
        goto done;
    }
    if (st_load_bf16(shard, norm, norm_weight, HIDDEN)) {
        if (error && capacity) {
            snprintf(error, capacity, PREFIX "post_attention_layernorm.weight: read failed");
        }
        goto done;
    }
    for (size_t t = 0; t < count; t++) {
        lunelette_rmsnorm(hidden + t * HIDDEN, norm_weight,
                          normalized + t * HIDDEN, HIDDEN, 1e-5f);
    }
    if (project(shard, PREFIX "mlp.gate_proj.weight", INNER, HIDDEN,
                normalized, gate, count, matrix, error, capacity) ||
        project(shard, PREFIX "mlp.up_proj.weight", INNER, HIDDEN,
                normalized, up, count, matrix, error, capacity)) {
        goto done;
    }
    lunelette_silu_gate(gate, up, gate, count * INNER);
    if (project(shard, PREFIX "mlp.down_proj.weight", HIDDEN, INNER,
                gate, projected, count, matrix, error, capacity)) {
        goto done;
    }
    for (size_t i = 0; i < count * HIDDEN; i++) {
        hidden[i] += projected[i];
    }
    memcpy(output, hidden, count * HIDDEN * sizeof(float));
    if (error && capacity) {
        error[0] = '\0';
    }
    result = 0;
done:
    free(matrix);
    free(workspace);
    return result;
}
