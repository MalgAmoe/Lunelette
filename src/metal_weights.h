#ifndef LUNELETTE_METAL_WEIGHTS_H
#define LUNELETTE_METAL_WEIGHTS_H

#include "metal.h"
#include "model_facts.h"
#include "safetensors.h"

#define LUNELETTE_METAL_WEIGHT_COUNT (2 + LUNELETTE_MODEL_LAYERS * LUNELETTE_MODEL_BLOCK_WEIGHTS)
/* Descriptor-only staging. Files stay open until all bounded uploads finish. */
typedef struct {
    st_file file;
    struct {
        float scale;
        const st_tensor *tensor;
        size_t elements;
    } weights[LUNELETTE_METAL_WEIGHT_COUNT];
    uint64_t bytes;
} lunelette_metal_weights;
int lunelette_metal_weights_open(lunelette_metal_weights *weights,
    const char *path, char *error, size_t capacity);
int lunelette_metal_weights_validate(lunelette_metal_weights *weights,
    char *error, size_t capacity);
/* Reads and validates actual scalar payloads; called by open, including zero-work CLI.
 * Descriptor-only validate deliberately performs no payload reads. */
int lunelette_metal_weights_read_scales(lunelette_metal_weights *weights,
    char *error, size_t capacity);
/* Fill exactly the staged element count. Failure may modify out partially;
 * runtime construction must not publish that destination. */
int lunelette_metal_weights_load(lunelette_metal_weights *weights, size_t index,
    float *out, size_t count);
/* Compact-only admission domain and bounded raw loading. Exact count, F8 only.
 * Failure may partially modify out; never publish a failed destination. */
int lunelette_metal_weights_load_raw(lunelette_metal_weights *weights, size_t index,
    unsigned char *out, size_t count, char *error, size_t capacity);
/* Validates every allocation size and sums before allocation/admission. */
int lunelette_metal_weights_account_configured(const lunelette_metal_weights *weights,
    lunelette_metal_strategy strategy, lunelette_metal_limits limits,
    lunelette_metal_memory *memory);
void lunelette_metal_weights_close(lunelette_metal_weights *weights);
#endif
