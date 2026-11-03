#ifndef LUNELETTE_CACHE_H
#define LUNELETTE_CACHE_H
#include "generation.h"

typedef struct lunelette_cache lunelette_cache;
/* Single3B, batch one, explicit capacity 1..256. Borrows distinct open shards
 * until free; never close/reopen/mutate them during that lifetime. Not concurrent.
 * Returns NULL on validation/allocation failure. Stores only FP32 K/V, not weights. */
lunelette_cache *lunelette_cache_init(st_file *first, st_file *second,
    size_t token_capacity, char *error, size_t error_capacity);
void lunelette_cache_reset(lunelette_cache *cache);
void lunelette_cache_free(lunelette_cache *cache);
size_t lunelette_cache_length(const lunelette_cache *cache);
/* Empty cache accepts prefill; otherwise exactly one token. Output 131072 logits
 * is committed only on success, as is length. Failed writes affect only unused
 * tail slots: prior prefix remains usable, retry or reset is safe. Buffers disjoint. */
int lunelette_cache_forward(lunelette_cache *cache, const int64_t *ids, size_t count,
    float *logits, char *error, size_t error_capacity);
/* Starts a fresh prompt (reset), uses the existing greedy/EOS/config contract.
 * Requires capacity 256 so generation's context bound is unambiguous. The last
 * sampled token is not evaluated/stored. On failure result is unchanged; cache
 * may contain completed steps and may be reset or reused via forward. */
int lunelette_generate_cached(lunelette_cache *cache, const int64_t *prompt,
    size_t count, const lunelette_generation_config *config,
    lunelette_generation_result *result, char *error, size_t error_capacity);
#endif
