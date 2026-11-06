#ifndef LUNELETTE_METAL_H
#define LUNELETTE_METAL_H

#include <stddef.h>
#include <stdint.h>

/* Production Darwin runtime. No Apple types cross this API.
 * Each context owns text weights, scratch and a 256-position KV cache.
 * Single caller only; no concurrent calls, including destroy/reset.
 */
typedef struct lunelette_metal lunelette_metal;

/* One official HF mixed FP8/BF16 model.safetensors; no index file needed.
 * Validates all required text tensor shapes/dtypes before weight allocation.
 * NULL on failure; diagnostic written when error/capacity permit. */
lunelette_metal *lunelette_metal_create(const char *path,
    char *error, size_t capacity);
typedef enum {
    LUNELETTE_METAL_PREFILL_MPS = 0,
    LUNELETTE_METAL_PREFILL_CUSTOM = 1
} lunelette_metal_prefill_mode;

typedef enum {
    LUNELETTE_METAL_RESIDENT_FP32 = 0,
    LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC = 1
} lunelette_metal_strategy;

typedef struct {
    lunelette_metal_prefill_mode prefill;
    lunelette_metal_strategy strategy;
} lunelette_metal_options;

#define LUNELETTE_METAL_OPTIONS_DEFAULT { LUNELETTE_METAL_PREFILL_MPS, LUNELETTE_METAL_RESIDENT_FP32 }

/* NULL options selects defaults. Invalid enum values fail before opening the checkpoint.
 * create() is equivalent to create_with_options(..., NULL, ...).
 * MPS is the default for multi-token block projections; decode/head stay precise.
 * MPS uses FP32 storage, but internal accumulation precision is unspecified.
 * CUSTOM retains precise FP32 custom arithmetic. Compact storage uses arithmetic
 * FP8 decode and FP32 scaling in single-token block projections; multi-token
 * blocks first convert into one stable FP32 stage, then use the selected policy.
 * Embedding, tied head and norms remain FP32. Compact payload admission requires
 * positive FP32-normal widened BF16 weight scales and finite codes with normal
 * nonzero reconstructed products. Unsupported inputs fail without fallback. */
lunelette_metal *lunelette_metal_create_with_options(const char *path,
    const lunelette_metal_options *options, char *error, size_t capacity);
/* Named owned GPU allocations only, not driver/process memory. NULL arguments fail.
 * Query does not mutate the context or invalidate borrowed logits. */
typedef struct {
    lunelette_metal_strategy strategy;
    uint64_t weight_bytes, conversion_stage_bytes, scratch_bytes, kv_bytes, total_bytes;
} lunelette_metal_memory;
int lunelette_metal_get_memory(const lunelette_metal *context, lunelette_metal_memory *memory);
void lunelette_metal_destroy(lunelette_metal *context);

/* Reset makes old cache tails inaccessible; no GPU work or weight reload. */
int lunelette_metal_reset(lunelette_metal *context, char *error, size_t capacity);

/* Prefill requires an empty, valid sequence and 1..256 valid token IDs.
 * Decode appends one token to a nonempty sequence of length <256.
 * On success *logits borrows 131072 FP32 values until the next mutating context
 * call or destroy (the read-only memory query is exempt).
 * On failure *logits is NULL. Invalid arguments are rejected before GPU/cache
 * mutation; execution/allocation/nonfinite failures invalidate the sequence,
 * which must be reset before reuse. Never publish partial KV as a valid prefix.
 * Context must be non-NULL; destroy(NULL) is allowed. Output pointer is required.
 */
int lunelette_metal_prefill(lunelette_metal *context, const int64_t *ids, size_t count,
    const float **logits, char *error, size_t capacity);
int lunelette_metal_decode(lunelette_metal *context, int64_t token,
    const float **logits, char *error, size_t capacity);

#endif
