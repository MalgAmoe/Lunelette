#ifndef LUNELETTE_FORWARD_INTERNAL_H
#define LUNELETTE_FORWARD_INTERNAL_H
#include "safetensors.h"
/* Private fixed-layout helpers; callers validate open handles and bounded sizes.
 * Layer supports input==output and commits output only on success. */
/* Fixed 3072-wide norm, epsilon 1e-5; FP32 products, double sum, FP32 mean/output.
 * Exact in-place input/output supported; no partial overlap. */
void lunelette_hidden_norm(const float *input, const float *weight, float *output);
const st_tensor *lunelette_require(st_file **files, size_t nfiles, const char *name,
                                  size_t rows, size_t cols, st_file **owner,
                                  char *error, size_t capacity);
int lunelette_embed(st_file **files, size_t nfiles, const int64_t *ids, size_t count,
                     float *hidden, char *error, size_t capacity);
int lunelette_layer(st_file **files, size_t nfiles, size_t layer,
                     const float *input, size_t count, float *output,
                     char *error, size_t capacity);
#endif
