#include "forward.h"
#include "forward_internal.h"
#include "lunelette.h"

#include <stdlib.h>
#include <string.h>

/* Validate the whole supported text layout before expensive execution, including
 * duplicate names. Unrelated vision tensors are deliberately ignored. */
static int validate(st_file **files, char *error, size_t capacity)
{
    const char *suffix[] = {
        "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
    };
    const size_t rows[] = {3072, 4096, 1024, 1024, 3072, 3072, 9216, 9216, 3072};
    const size_t cols[] = {0, 3072, 3072, 3072, 4096, 0, 3072, 3072, 9216};
    st_file *owner = NULL;
    if (!lunelette_require(files, 2, "language_model.model.embed_tokens.weight",
                           131072, 3072, &owner, error, capacity) ||
        !lunelette_require(files, 2, "language_model.model.norm.weight",
                           3072, 0, &owner, error, capacity)) {
        return -1;
    }
    for (size_t layer = 0; layer < 26; layer++) {
        for (size_t i = 0; i < 9; i++) {
            char name[128];
            snprintf(name, sizeof name, "language_model.model.layers.%zu.%s", layer, suffix[i]);
            if (!lunelette_require(files, 2, name, rows[i], cols[i], &owner, error, capacity)) {
                return -1;
            }
        }
    }
    return 0;
}

int lunelette_forward(st_file *shard1, st_file *shard2, const int64_t *ids, size_t count,
                       float *hidden, float *logits, float *hidden_13,
                       char *error, size_t capacity)
{
    if (error && capacity) {
        snprintf(error, capacity, "forward: require two distinct open shards, IDs, outputs, 1..256 tokens");
    }
    if (!shard1 || !shard2 || shard1 == shard2 || !shard1->file || !shard2->file ||
        shard1->file == shard2->file || !ids || !hidden || !logits || !count || count > 256) {
        return -1;
    }
    for (size_t t = 0; t < count; t++) {
        if (ids[t] < 0 || ids[t] >= 131072) {
            if (error && capacity) {
                snprintf(error, capacity, "forward: invalid token %lld at row %zu",
                         (long long)ids[t], t);
            }
            return -1;
        }
    }
    st_file *files[] = {shard1, shard2};
    if (validate(files, error, capacity)) {
        return -1;
    }
    int result = -1;
    size_t elements = count * 3072;
    float *state = malloc(elements * sizeof(float));
    float *checkpoint = hidden_13 ? malloc(elements * sizeof(float)) : NULL;
    float *scores = malloc(131072 * sizeof(float));
    float norm_weight[3072], row[3072];
    if (!state || !scores || (hidden_13 && !checkpoint)) {
        if (error && capacity) {
            snprintf(error, capacity, "forward: allocation failed");
        }
        goto done;
    }
    if (lunelette_embed(files, 2, ids, count, state, error, capacity)) {
        goto done;
    }
    for (size_t layer = 0; layer < 26; layer++) {
        if (lunelette_layer(files, 2, layer, state, count, state, error, capacity)) {
            goto done;
        }
        if (layer == 12 && checkpoint) {
            memcpy(checkpoint, state, elements * sizeof(float));
        }
    }
    st_file *owner = NULL;
    const st_tensor *norm = lunelette_require(files, 2, "language_model.model.norm.weight",
                                             3072, 0, &owner, error, capacity);
    if (!norm || st_load_bf16(owner, norm, norm_weight, 3072)) {
        if (error && capacity) {
            snprintf(error, capacity, "language_model.model.norm.weight: read failed");
        }
        goto done;
    }
    for (size_t t = 0; t < count; t++) {
        lunelette_hidden_norm(state + t * 3072, norm_weight, state + t * 3072);
    }
    const st_tensor *embedding = lunelette_require(files, 2,
        "language_model.model.embed_tokens.weight", 131072, 3072, &owner, error, capacity);
    if (!embedding) {
        goto done;
    }
    /* Tied output head: expand one BF16 vocabulary row, never the 1.5 GiB
     * embedding matrix. The same scalar reduction as all layer projections. */
    for (size_t token = 0; token < 131072; token++) {
        if (st_embedding(owner, embedding, (int64_t)token, row, 3072)) {
            if (error && capacity) {
                snprintf(error, capacity, "tied logits: embedding row %zu read failed", token);
            }
            goto done;
        }
        lunelette_matvec(row, state + (count - 1) * 3072, scores + token, 1, 3072);
    }
    memcpy(hidden, state, elements * sizeof(float));
    memcpy(logits, scores, 131072 * sizeof(float));
    if (checkpoint) {
        memcpy(hidden_13, checkpoint, elements * sizeof(float));
    }
    if (error && capacity) {
        error[0] = '\0';
    }
    result = 0;
done:
    free(scores);
    free(checkpoint);
    free(state);
    return result;
}
