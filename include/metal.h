#ifndef LUNELETTE_METAL_H
#define LUNELETTE_METAL_H

#include <stddef.h>
#include <stdint.h>

/* Optional Darwin runtime, linked only with METAL=1. No Apple types cross this API.
 * Each context owns FP32 text weights, scratch and a 256-position KV cache.
 * Single caller only; no concurrent calls, including destroy/reset.
 */
typedef struct lunelette_metal lunelette_metal;

/* Two explicit BF16 safetensors shards, in either order; no index file needed.
 * Validates all required text tensor shapes/dtypes before weight allocation.
 * NULL on failure; diagnostic written when error/capacity permit. */
lunelette_metal *lunelette_metal_create(const char *first, const char *second,
    char *error, size_t capacity);
typedef enum {
    LUNELETTE_METAL_PREFILL_MPS = 0,
    LUNELETTE_METAL_PREFILL_CUSTOM = 1
} lunelette_metal_prefill_mode;

typedef struct {
    lunelette_metal_prefill_mode prefill;
} lunelette_metal_options;

#define LUNELETTE_METAL_OPTIONS_DEFAULT { LUNELETTE_METAL_PREFILL_MPS }

/* NULL options selects defaults. Invalid enum values fail before opening shards.
 * create() is equivalent to create_with_options(..., NULL, ...).
 * MPS is the default for multi-token block projections; decode/head stay precise.
 * MPS uses FP32 storage, but internal accumulation precision is unspecified.
 * CUSTOM retains the precise FP32 custom arithmetic for all projections. */
lunelette_metal *lunelette_metal_create_with_options(const char *first, const char *second,
    const lunelette_metal_options *options, char *error, size_t capacity);
void lunelette_metal_destroy(lunelette_metal *context);

/* Reset makes old cache tails inaccessible; no GPU work or weight reload. */
int lunelette_metal_reset(lunelette_metal *context, char *error, size_t capacity);

/* Prefill requires an empty, valid sequence and 1..256 valid token IDs.
 * Decode appends one token to a nonempty sequence of length <256.
 * On success *logits borrows 131072 FP32 values until ANY next context call.
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
