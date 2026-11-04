#include "forward.h"
#include "forward_internal.h"
#include "lunelette.h"

#include <stdlib.h>
#include <string.h>

int lunelette_forward(st_file *shard1, st_file *shard2, const int64_t *ids, size_t count,
                       float *hidden, float *logits, float *hidden_13,
                       char *error, size_t capacity)
{
    if (error && capacity) {
        snprintf(error, capacity, "forward: require two distinct open shards, IDs, outputs, 1..256 tokens");
    }
    if (!shard1 || !shard2 || shard1 == shard2 || !shard1->file || !shard2->file ||
        shard1->file == shard2->file || !ids || !hidden || !logits || !count ||
        count > LUNELETTE_CONTEXT_LIMIT) {
        return -1;
    }
    for (size_t t = 0; t < count; t++) {
        if (ids[t] < 0 || ids[t] >= LUNELETTE_MODEL_VOCAB) {
            if (error && capacity) {
                snprintf(error, capacity, "forward: invalid token %lld at row %zu",
                         (long long)ids[t], t);
            }
            return -1;
        }
    }
    st_file *files[] = {shard1, shard2};
    if (lunelette_cpu_validate_layout(files, error, capacity)) {
        return -1;
    }
    int result = -1;
    size_t elements = count * LUNELETTE_MODEL_HIDDEN;
    float *state = malloc(elements * sizeof(float));
    float *checkpoint = hidden_13 ? malloc(elements * sizeof(float)) : NULL;
    float *scores = malloc(LUNELETTE_MODEL_VOCAB * sizeof(float));
    float norm_weight[LUNELETTE_MODEL_HIDDEN], row[LUNELETTE_MODEL_HIDDEN];
    if (!state || !scores || (hidden_13 && !checkpoint)) {
        if (error && capacity) {
            snprintf(error, capacity, "forward: allocation failed");
        }
        goto done;
    }
    if (lunelette_embed(files, 2, ids, count, state, error, capacity)) {
        goto done;
    }
    for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
        if (lunelette_layer(files, 2, layer, state, count, state, error, capacity)) {
            goto done;
        }
        if (layer == 12 && checkpoint) {
            memcpy(checkpoint, state, elements * sizeof(float));
        }
    }
    st_file *owner = NULL;
    const st_tensor *norm = lunelette_require(files, 2, "language_model.model.norm.weight",
                                             LUNELETTE_MODEL_HIDDEN, 0, &owner, error, capacity);
    if (!norm || st_load_bf16(owner, norm, norm_weight, LUNELETTE_MODEL_HIDDEN)) {
        if (error && capacity) {
            snprintf(error, capacity, "language_model.model.norm.weight: read failed");
        }
        goto done;
    }
    for (size_t t = 0; t < count; t++) {
        lunelette_hidden_norm(state + t * LUNELETTE_MODEL_HIDDEN, norm_weight,
                              state + t * LUNELETTE_MODEL_HIDDEN);
    }
    const st_tensor *embedding = lunelette_require(files, 2,
        "language_model.model.embed_tokens.weight", LUNELETTE_MODEL_VOCAB,
        LUNELETTE_MODEL_HIDDEN, &owner, error, capacity);
    if (!embedding) {
        goto done;
    }
    /* Tied output head: expand one BF16 vocabulary row, never the 1.5 GiB
     * embedding matrix. The same scalar reduction as all layer projections. */
    for (size_t token = 0; token < LUNELETTE_MODEL_VOCAB; token++) {
        if (st_embedding(owner, embedding, (int64_t)token, row, LUNELETTE_MODEL_HIDDEN)) {
            if (error && capacity) {
                snprintf(error, capacity, "tied logits: embedding row %zu read failed", token);
            }
            goto done;
        }
        lunelette_matvec(row, state + (count - 1) * LUNELETTE_MODEL_HIDDEN,
                         scores + token, 1, LUNELETTE_MODEL_HIDDEN);
    }
    memcpy(hidden, state, elements * sizeof(float));
    memcpy(logits, scores, LUNELETTE_MODEL_VOCAB * sizeof(float));
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
