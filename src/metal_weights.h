#ifndef LUNELETTE_METAL_WEIGHTS_H
#define LUNELETTE_METAL_WEIGHTS_H

#include "model_facts.h"
#include "safetensors.h"

#define LUNELETTE_METAL_WEIGHT_COUNT (2 + LUNELETTE_MODEL_LAYERS * LUNELETTE_MODEL_BLOCK_WEIGHTS)
/* Descriptor-only staging. Files stay open until all bounded uploads finish. */
typedef struct {
    st_file files[2];
    struct {
        st_file *file;
        const st_tensor *tensor;
        size_t elements;
    } weights[LUNELETTE_METAL_WEIGHT_COUNT];
    uint64_t bytes;
} lunelette_metal_weights;
int lunelette_metal_weights_open(lunelette_metal_weights *weights,
    const char *first, const char *second, char *error, size_t capacity);
int lunelette_metal_weights_validate(lunelette_metal_weights *weights,
    char *error, size_t capacity);
void lunelette_metal_weights_close(lunelette_metal_weights *weights);
#endif
