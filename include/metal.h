#ifndef LUNELETTE_METAL_H
#define LUNELETTE_METAL_H

#include <stddef.h>
#include <stdint.h>

/* Production Darwin runtime. No Apple types cross this API.
 * Each context owns text weights, scratch and an immutable configured KV capacity (default 256).
 * Single caller only; no concurrent calls, including destroy/reset.
 */
typedef struct lunelette_metal lunelette_metal;
typedef struct lunelette_metal_checkpoint lunelette_metal_checkpoint;

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
    LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC = 1,
    LUNELETTE_METAL_COMPACT_FP8_LOOKUP = 2
} lunelette_metal_strategy;

typedef struct {
    lunelette_metal_prefill_mode prefill;
    lunelette_metal_strategy strategy;
} lunelette_metal_options;

#define LUNELETTE_METAL_OPTIONS_DEFAULT \
    { LUNELETTE_METAL_PREFILL_MPS, LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC }

/* NULL options selects compact arithmetic/MPS defaults.
 * Zero initialization (including an omitted strategy field) still selects resident FP32;
 * an omitted prefill field still selects MPS. Use the macro for product defaults.
 * Explicit resident and lookup remain supported, without automatic fallback.
 * Invalid enum values fail before opening the checkpoint.
 * create() is equivalent to create_with_options(..., NULL, ...).
 * MPS is the default for multi-token block projections; decode/head stay precise.
 * MPS uses FP32 storage, but internal accumulation precision is unspecified.
 * CUSTOM retains precise FP32 custom arithmetic. Compact storage uses arithmetic
 * FP8 decode (or an unscaled constant lookup when selected) and FP32 scaling
 * in single-token block projections; multi-token
 * blocks first convert into one stable FP32 stage, then use the selected policy.
 * Embedding, tied head and norms remain FP32. Compact payload admission requires
 * positive FP32-normal widened BF16 weight scales and finite codes with normal
 * nonzero reconstructed products. Unsupported inputs fail without fallback. */
lunelette_metal *lunelette_metal_create_with_options(const char *path,
    const lunelette_metal_options *options, char *error, size_t capacity);
/* Additive profile API; options ABI and old constructors remain unchanged.
 * NULL limits selects C256/B256. Supported explicit pairs: 256/256,
 * 4352/256 and 4352/128. No live resizing or implicit chunk orchestration. */
typedef struct {
    size_t context, block;
} lunelette_metal_limits;
lunelette_metal *lunelette_metal_create_configured(const char *path,
    const lunelette_metal_options *options, const lunelette_metal_limits *limits,
    char *error, size_t capacity);
/* Query is nonmutating; NULL arguments fail and leave output unchanged. */
int lunelette_metal_get_limits(const lunelette_metal *context, lunelette_metal_limits *limits);

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

/* Prefill requires an empty, valid sequence and 1..B valid token IDs and total length <=C.
 * Decode appends one token to a nonempty sequence of length <C.
 * On success *logits borrows 131072 FP32 values until the next mutating context
 * call or context destroy (memory/limits queries and checkpoint create/destroy are exempt).
 * On failure *logits is NULL. Invalid arguments are rejected before GPU/cache
 * mutation; execution/allocation/nonfinite failures invalidate the sequence,
 * which must be reset before reuse. Never publish partial KV as a valid prefix.
 * Context must be non-NULL; destroy(NULL) is allowed. Output pointer is required.
 */
int lunelette_metal_prefill(lunelette_metal *context, const int64_t *ids, size_t count,
    const float **logits, char *error, size_t capacity);
int lunelette_metal_decode(lunelette_metal *context, int64_t token,
    const float **logits, char *error, size_t capacity);

/* Batched nonempty suffix, requiring a valid nonempty prefix. Same validation,
 * failure and borrowed-output contracts as decode; count<=B and length+count<=C.
 * One block per call, not a loop of scalar decodes or logical-ingestion orchestration. */
int lunelette_metal_append(lunelette_metal *context, const int64_t *ids, size_t count,
    const float **logits, char *error, size_t capacity);

/* Single caller, including all handle operations. Handles own a full finite logit
 * copy, not KV; destroy every handle before its context. Only one handle is active
 * per context. Successful create replaces it; older live handles reject restore.
 * Create requires a materialized nonempty endpoint. Allocation/argument failure
 * preserves the previous checkpoint and context; *checkpoint is NULL on failure.
 * Use a separate output slot when replacing: retain the old handle for destruction.
 * Detection of nonfinite endpoint logits invalidates sequence and checkpoint.
 * Restore repeatedly rewinds to the saved prefix and copies exact saved logits,
 * with no GPU work or KV copy. Output borrows runtime storage as above. Append/
 * decode preserve the checkpoint; reset or execution failure invalidates it.
 * Invalid restore preserves state and sets *logits NULL. No arbitrary retain API.
 * Snapshot payload is 524288 host bytes per handle, excluded from the GPU-only
 * memory query. Replacement temporarily needs two payloads; promptly destroy old
 * handles (the context never frees caller-owned handles). Create/restore do not
 * execute the model. Epoch exhaustion rejects create without replacing the handle.
 * Handles are opaque: forged pointers and use after destroy are invalid C usage. */
int lunelette_metal_checkpoint_create(lunelette_metal *context,
    lunelette_metal_checkpoint **checkpoint, char *error, size_t capacity);
int lunelette_metal_checkpoint_restore(lunelette_metal *context,
    const lunelette_metal_checkpoint *checkpoint, const float **logits,
    char *error, size_t capacity);
void lunelette_metal_checkpoint_destroy(lunelette_metal_checkpoint *checkpoint);

#endif
