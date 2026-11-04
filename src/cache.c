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
        if (ids[t] < 0 || ids[t] >= LUNELETTE_MODEL_VOCAB) {
            if (error && capacity) {
                snprintf(error, capacity, "cache: invalid token %lld at row %zu",
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
    float *scores = malloc(LUNELETTE_MODEL_VOCAB * sizeof(float));
    float norm_weight[LUNELETTE_MODEL_HIDDEN], row[LUNELETTE_MODEL_HIDDEN];
    if (!state || !scores) {
        if (error && capacity) {
            snprintf(error, capacity, "cache: allocation failed");
        }
        goto done;
    }
    if (lunelette_embed(files, 2, ids, count, state, error, capacity)) {
        goto done;
    }
    for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
        if (lunelette_layer_cached(files, 2, layer, state, count, state, error, capacity,
            cache->kv + layer * 2 * cache->capacity * LUNELETTE_MODEL_KV,
            cache->kv + (layer * 2 + 1) * cache->capacity * LUNELETTE_MODEL_KV, cache->length)) {
            goto done;
        }
    }
    if (lunelette_cpu_tail(files, count, state, scores, norm_weight, row, error, capacity)) {
        goto done;
    }
    memcpy(logits, scores, LUNELETTE_MODEL_VOCAB * sizeof(float));
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
        first->file == second->file || !token_capacity || token_capacity > LUNELETTE_CONTEXT_LIMIT) {
        return NULL;
    }
    st_file *files[] = {first, second};
    if (lunelette_cpu_validate_layout(files, error, error_capacity)) {
        return NULL;
    }
    lunelette_cache *cache = calloc(1, sizeof *cache);
    if (cache) {
        cache->kv = malloc(LUNELETTE_MODEL_LAYERS * 2 * token_capacity * LUNELETTE_MODEL_KV *
                           sizeof(float));
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
    if (!cache || cache->capacity != LUNELETTE_CONTEXT_LIMIT) {
        if (error && capacity) {
            snprintf(error, capacity, "cached generation: require capacity 256");
        }
        return -1;
    }
    lunelette_cache_reset(cache);
    return lunelette_generate_with_logits(next_cached, cache, prompt, count,
                                           config, result, error, capacity);
}
