#include "cache.h"
#include "forward_internal.h"
#include "lunelette.h"
#include <stdlib.h>
#include <string.h>

struct lunelette_cache {
    st_file *first, *second;
    size_t capacity, length;
    float *kv;
};

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

int lunelette_cache_forward(lunelette_cache *cache, const int64_t *ids, size_t count,
                       float *logits,
                       char *error, size_t capacity)
{
    st_file *shard1 = cache ? cache->first : NULL;
    st_file *shard2 = cache ? cache->second : NULL;
    if (error && capacity) {
        snprintf(error, capacity, "cache: require two distinct open shards, IDs, outputs, 1..256 tokens");
    }
    if (!shard1 || !shard2 || shard1 == shard2 || !shard1->file || !shard2->file ||
        shard1->file == shard2->file || !ids || !logits || !count || !cache ||
        count > cache->capacity - cache->length ||
        (cache->length && count != 1)) {
        return -1;
    }
    for (size_t t = 0; t < count; t++) {
        if (ids[t] < 0 || ids[t] >= 131072) {
            if (error && capacity) {
                snprintf(error, capacity, "cache: invalid token %lld at row %zu",
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
    float *scores = malloc(131072 * sizeof(float));
    float norm_weight[3072], row[3072];
    if (!state || !scores) {
        if (error && capacity) {
            snprintf(error, capacity, "cache: allocation failed");
        }
        goto done;
    }
    if (lunelette_embed(files, 2, ids, count, state, error, capacity)) {
        goto done;
    }
    for (size_t layer = 0; layer < 26; layer++) {
        if (lunelette_layer_cached(files, 2, layer, state, count, state, error, capacity,
            cache->kv + layer * 2 * cache->capacity * 1024,
            cache->kv + (layer * 2 + 1) * cache->capacity * 1024, cache->length)) {
            goto done;
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
    memcpy(logits, scores, 131072 * sizeof(float));
    cache->length += count;
    if (error && capacity) {
        error[0] = '\0';
    }
    result = 0;
done:
    free(scores);
    free(state);
    return result;
}

lunelette_cache *lunelette_cache_init(st_file *first, st_file *second,
    size_t token_capacity, char *error, size_t error_capacity)
{
    if (error && error_capacity) {
        snprintf(error, error_capacity, "cache: require distinct open shards and capacity 1..256");
    }
    if (!first || !second || first == second || !first->file || !second->file ||
        first->file == second->file || !token_capacity || token_capacity > 256) {
        return NULL;
    }
    st_file *files[] = {first, second};
    if (validate(files, error, error_capacity)) {
        return NULL;
    }
    lunelette_cache *cache = calloc(1, sizeof *cache);
    if (cache) {
        cache->kv = malloc(26 * 2 * token_capacity * 1024 * sizeof(float));
        if (!cache->kv) {
            free(cache);
            cache = NULL;
        }
    }
    if (!cache) {
        if (error && error_capacity) {
            snprintf(error, error_capacity, "cache: allocation failed");
        }
        return NULL;
    }
    cache->first = first;
    cache->second = second;
    cache->capacity = token_capacity;
    if (error && error_capacity) {
        error[0] = '\0';
    }
    return cache;
}

void lunelette_cache_reset(lunelette_cache *cache)
{
    if (cache) {
        cache->length = 0;
    }
}

void lunelette_cache_free(lunelette_cache *cache)
{
    if (cache) {
        free(cache->kv);
        free(cache);
    }
}

size_t lunelette_cache_length(const lunelette_cache *cache)
{
    return cache ? cache->length : 0;
}

static int next_cached(void *opaque, const int64_t *ids, size_t count,
                       float *logits, char *error, size_t capacity)
{
    lunelette_cache *cache = opaque;
    if (cache->length && count != cache->length + 1) {
        return -1;
    }
    return lunelette_cache_forward(cache, ids + cache->length,
        count - cache->length, logits, error, capacity);
}

int lunelette_generate_cached(lunelette_cache *cache, const int64_t *prompt,
    size_t count, const lunelette_generation_config *config,
    lunelette_generation_result *result, char *error, size_t capacity)
{
    if (!cache || cache->capacity != 256) {
        if (error && capacity) {
            snprintf(error, capacity, "cached generation: require capacity 256");
        }
        return -1;
    }
    lunelette_cache_reset(cache);
    return lunelette_generate_with_logits(next_cached, cache, prompt, count,
                                           config, result, error, capacity);
}
