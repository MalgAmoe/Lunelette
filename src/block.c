#include "block.h"
#include "forward_internal.h"
#include "attention.h"
#include "lunelette.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 3072
#define QUERY 4096
#define KV 1024
#define INNER 9216

/* FP32 products/output with a wider sum prevent thousands of positive squares
 * losing low bits at every addition. Across 26 layers that norm error compounds.
 * Keep the public sequential-FP32 kernel unchanged as the comparison baseline.
 * Fixed width and epsilon belong to this model, not a generic norm interface. */
void lunelette_hidden_norm(const float *input, const float *weight, float *output)
{
    double sum = 0.0;
    for (size_t i = 0; i < HIDDEN; i++) {
        float square = input[i] * input[i];
        sum += square;
    }
    float mean = (float)(sum / HIDDEN);
    float scale = 1.0f / sqrtf(mean + 1e-5f);
    for (size_t i = 0; i < HIDDEN; i++) {
        output[i] = (input[i] * scale) * weight[i];
    }
}

const st_tensor *lunelette_require(st_file **files, size_t nfiles, const char *name,
                                  size_t rows, size_t cols, st_file **owner,
                                  char *error, size_t capacity)
{
    const st_tensor *found = NULL;
    for (size_t f = 0; f < nfiles; f++) {
        for (size_t i = 0; i < files[f]->count; i++) {
            const st_tensor *tensor = &files[f]->tensors[i];
            if (strcmp(tensor->name, name)) {
                continue;
            }
            if (found) {
                if (error && capacity) {
                    snprintf(error, capacity, "%s: duplicate tensor across supplied shards", name);
                }
                return NULL;
            }
            found = tensor;
            *owner = files[f];
        }
    }
    if (!found || found->dtype != ST_BF16 || found->rank != (cols ? 2u : 1u) ||
        found->shape[0] != rows || (cols && found->shape[1] != cols)) {
        if (error && capacity) {
            snprintf(error, capacity, "%s: expected BF16 %zu rows, %zu columns (0=vector)",
                     name, rows, cols);
        }
        return NULL;
    }
    return found;
}

static int project(st_file **files, size_t nfiles, const char *name, size_t rows, size_t cols,
                   const float *input, float *output, size_t count,
                   float *matrix, char *error, size_t capacity)
{
    st_file *file = NULL;
    const st_tensor *tensor = lunelette_require(files, nfiles, name, rows, cols, &file, error, capacity);
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

int lunelette_layer(st_file **files, size_t nfiles, size_t layer,
                     const float *input, size_t count, float *output,
                     char *error, size_t capacity)
{
    int result = -1;
    float *workspace = NULL, *matrix = NULL;
    char names[9][128];
    const char *suffixes[] = {
        "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
    };
    if (!input || !output || !count || count > 256 || layer >= 26) {
        return -1;
    }
    for (size_t i = 0; i < 9; i++) {
        snprintf(names[i], sizeof names[i], "language_model.model.layers.%zu.%s", layer, suffixes[i]);
    }
    st_file *owner = NULL;
    const st_tensor *norm = lunelette_require(files, nfiles, names[0], HIDDEN, 0,
                                             &owner, error, capacity);
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
            snprintf(error, capacity, "layer %zu: workspace/matrix allocation failed", layer);
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
        snprintf(error, capacity, "%s: read failed", names[0]);
    }
    if (st_load_bf16(owner, norm, norm_weight, HIDDEN)) {
        goto done;
    }
    memcpy(hidden, input, count * HIDDEN * sizeof(float));
    for (size_t t = 0; t < count; t++) {
        positions[t] = (uint32_t)t;
        lunelette_hidden_norm(hidden + t * HIDDEN, norm_weight, normalized + t * HIDDEN);
    }
    if (project(files, nfiles, names[1], QUERY, HIDDEN,
                normalized, q, count, matrix, error, capacity) ||
        project(files, nfiles, names[2], KV, HIDDEN,
                normalized, k, count, matrix, error, capacity) ||
        project(files, nfiles, names[3], KV, HIDDEN,
                normalized, v, count, matrix, error, capacity)) {
        goto done;
    }
    if (lunelette_ministral3_yarn(q, k, positions, count) ||
        lunelette_causal_gqa(q, k, v, attention, scratch, count, 32, 8, 128)) {
        if (error && capacity) {
            snprintf(error, capacity, "layer %zu: YaRN/causal GQA failed", layer);
        }
        goto done;
    }
    if (project(files, nfiles, names[4], HIDDEN, QUERY,
                attention, projected, count, matrix, error, capacity)) {
        goto done;
    }
    for (size_t i = 0; i < count * HIDDEN; i++) {
        hidden[i] += projected[i];
    }
    norm = lunelette_require(files, nfiles, names[5],
                   HIDDEN, 0, &owner, error, capacity);
    if (!norm) {
        goto done;
    }
    if (st_load_bf16(owner, norm, norm_weight, HIDDEN)) {
        if (error && capacity) {
            snprintf(error, capacity, "%s: read failed", names[5]);
        }
        goto done;
    }
    for (size_t t = 0; t < count; t++) {
        lunelette_hidden_norm(hidden + t * HIDDEN, norm_weight, normalized + t * HIDDEN);
    }
    if (project(files, nfiles, names[6], INNER, HIDDEN,
                normalized, gate, count, matrix, error, capacity) ||
        project(files, nfiles, names[7], INNER, HIDDEN,
                normalized, up, count, matrix, error, capacity)) {
        goto done;
    }
    lunelette_silu_gate(gate, up, gate, count * INNER);
    if (project(files, nfiles, names[8], HIDDEN, INNER,
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

int lunelette_first_block(st_file *shard, const int64_t *ids, size_t count,
                          float *output, char *error, size_t capacity)
{
    if (error && capacity) {
        snprintf(error, capacity, "first block: invalid arguments");
    }
    if (!shard || !shard->file || !ids || !output || !count || count > 256) {
        return -1;
    }
    st_file *files[] = {shard};
    float *hidden = malloc(count * HIDDEN * sizeof(float));
    if (!hidden) {
        if (error && capacity) {
            snprintf(error, capacity, "first block: embedding allocation failed");
        }
        return -1;
    }
    int result = lunelette_embed(files, 1, ids, count, hidden, error, capacity);
    if (!result) {
        result = lunelette_layer(files, 1, 0, hidden, count, output, error, capacity);
    }
    free(hidden);
    return result;
}

int lunelette_embed(st_file **files, size_t nfiles, const int64_t *ids, size_t count,
                     float *hidden, char *error, size_t capacity)
{
    for (size_t t = 0; t < count; t++) {
        if (ids[t] < 0 || ids[t] >= 131072) {
            if (error && capacity) {
                snprintf(error, capacity, "embedding: invalid token %lld at row %zu",
                         (long long)ids[t], t);
            }
            return -1;
        }
    }
    st_file *owner = NULL;
    const st_tensor *embedding = lunelette_require(files, nfiles,
        "language_model.model.embed_tokens.weight", 131072, HIDDEN, &owner, error, capacity);
    if (!embedding) {
        return -1;
    }
    for (size_t t = 0; t < count; t++) {
        if (st_embedding(owner, embedding, ids[t], hidden + t * HIDDEN, HIDDEN)) {
            if (error && capacity) {
                snprintf(error, capacity, "embedding: read failed at row %zu", t);
            }
            return -1;
        }
    }
    return 0;
}
