#include "block.h"
#include "forward_internal.h"
#include "attention.h"
#include "lunelette.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN LUNELETTE_MODEL_HIDDEN
#define QUERY LUNELETTE_MODEL_QUERY
#define KV LUNELETTE_MODEL_KV
#define INNER LUNELETTE_MODEL_INNER

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
    float scale = 1.0f / sqrtf(mean + LUNELETTE_MODEL_NORM_EPS);
    for (size_t i = 0; i < HIDDEN; i++) {
        output[i] = (input[i] * scale) * weight[i];
    }
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

int lunelette_layer_cached(st_file **files, size_t nfiles, size_t layer,
                     const float *input, size_t count, float *output,
                     char *error, size_t capacity, float *cached_key, float *cached_value, size_t offset)
{
    int result = -1;
    float *workspace = NULL, *matrix = NULL;
    char names[LUNELETTE_MODEL_BLOCK_WEIGHTS][128];
    if (!input || !output || !count || count > LUNELETTE_CONTEXT_LIMIT ||
        layer >= LUNELETTE_MODEL_LAYERS ||
        offset > LUNELETTE_CONTEXT_LIMIT - count || (!!cached_key != !!cached_value) ||
        (offset && !cached_key)) {
        return -1;
    }
    for (size_t i = 0; i < LUNELETTE_MODEL_BLOCK_WEIGHTS; i++) {
        snprintf(names[i], sizeof names[i], "language_model.model.layers.%zu.%s", layer,
                 lunelette_model_block_weights[i].suffix);
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
    float norm_weight[HIDDEN], scratch[LUNELETTE_CONTEXT_LIMIT];
    uint32_t positions[LUNELETTE_CONTEXT_LIMIT];
    if (error && capacity) {
        snprintf(error, capacity, "%s: read failed", names[0]);
    }
    if (st_load_bf16(owner, norm, norm_weight, HIDDEN)) {
        goto done;
    }
    memcpy(hidden, input, count * HIDDEN * sizeof(float));
    for (size_t t = 0; t < count; t++) {
        positions[t] = (uint32_t)(offset + t);
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
    if (lunelette_ministral3_yarn(q, k, positions, count)) {
        goto done;
    }
    if (cached_key) {
        memcpy(cached_key + offset * KV, k, count * KV * sizeof(float));
        memcpy(cached_value + offset * KV, v, count * KV * sizeof(float));
    }
    if (lunelette_offset_gqa(q, cached_key ? cached_key : k,
                            cached_value ? cached_value : v, attention, scratch,
                            count, LUNELETTE_MODEL_QUERY_HEADS, LUNELETTE_MODEL_KV_HEADS,
                            LUNELETTE_MODEL_HEAD_DIM, offset)) {
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
    if (!shard || !shard->file || !ids || !output || !count ||
        count > LUNELETTE_CONTEXT_LIMIT) {
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
        if (ids[t] < 0 || ids[t] >= LUNELETTE_MODEL_VOCAB) {
            if (error && capacity) {
                snprintf(error, capacity, "embedding: invalid token %lld at row %zu",
                         (long long)ids[t], t);
            }
            return -1;
        }
    }
    st_file *owner = NULL;
    const st_tensor *embedding = lunelette_require(files, nfiles,
        "language_model.model.embed_tokens.weight", LUNELETTE_MODEL_VOCAB, HIDDEN, &owner, error, capacity);
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

int lunelette_layer(st_file **files, size_t nfiles, size_t layer,
                     const float *input, size_t count, float *output,
                     char *error, size_t capacity)
{
    return lunelette_layer_cached(files, nfiles, layer, input, count, output,
                                   error, capacity, NULL, NULL, 0);
}

int lunelette_cpu_tail(st_file **files, size_t count, float *state, float *scores,
    float *norm_weight, float *row, char *error, size_t capacity)
{
    st_file *owner = NULL;
    const st_tensor *norm = lunelette_require(files, 2, "language_model.model.norm.weight",
                                             LUNELETTE_MODEL_HIDDEN, 0, &owner, error, capacity);
    if (!norm || st_load_bf16(owner, norm, norm_weight, LUNELETTE_MODEL_HIDDEN)) {
        if (error && capacity) {
            snprintf(error, capacity, "language_model.model.norm.weight: read failed");
        }
        return -1;
    }
    for (size_t t = 0; t < count; t++) {
        lunelette_hidden_norm(state + t * LUNELETTE_MODEL_HIDDEN, norm_weight,
                              state + t * LUNELETTE_MODEL_HIDDEN);
    }
    const st_tensor *embedding = lunelette_require(files, 2,
        "language_model.model.embed_tokens.weight", LUNELETTE_MODEL_VOCAB,
        LUNELETTE_MODEL_HIDDEN, &owner, error, capacity);
    if (!embedding) {
        return -1;
    }
    /* Tied output head: expand one BF16 vocabulary row, never the 1.5 GiB
     * embedding matrix. The same scalar reduction as all layer projections. */
    for (size_t token = 0; token < LUNELETTE_MODEL_VOCAB; token++) {
        if (st_embedding(owner, embedding, (int64_t)token, row, LUNELETTE_MODEL_HIDDEN)) {
            if (error && capacity) {
                snprintf(error, capacity, "tied logits: embedding row %zu read failed", token);
            }
            return -1;
        }
        lunelette_matvec(row, state + (count - 1) * LUNELETTE_MODEL_HIDDEN,
                         scores + token, 1, LUNELETTE_MODEL_HIDDEN);
    }
    return 0;
}
