#ifndef LUNELETTE_FORWARD_INTERNAL_H
#define LUNELETTE_FORWARD_INTERNAL_H
#include "cpu_weights.h"
#include "model_facts.h"
#include "runtime_limits.h"
/* Private fixed-layout helpers; callers validate open handles and bounded sizes.
 * Layer supports input==output and commits output only on success. */
/* Fixed 3072-wide norm, epsilon 1e-5; FP32 products, double sum, FP32 mean/output.
 * Exact in-place input/output supported; no partial overlap. */
void lunelette_hidden_norm(const float *input, const float *weight, float *output);
int lunelette_embed(st_file **files, size_t nfiles, const int64_t *ids, size_t count,
                     float *hidden, char *error, size_t capacity);
int lunelette_layer(st_file **files, size_t nfiles, size_t layer,
                     const float *input, size_t count, float *output,
                     char *error, size_t capacity);
int lunelette_layer_cached(st_file **files, size_t nfiles, size_t layer,
    const float *input, size_t count, float *output, char *error, size_t capacity,
    float *key, float *value, size_t offset);
/* Internal offset attention: scratch holds offset+tokens floats. Same math as
 * baseline; queries start at absolute offset, keys/values start at zero. */
int lunelette_offset_gqa(const float *query, const float *key, const float *value,
    float *output, float *scratch, size_t tokens, size_t query_heads,
    size_t kv_heads, size_t head_dim, size_t offset);
#endif
