#include "metal_weights.h"
#include <stdio.h>
#include <string.h>

void lunelette_metal_weights_close(lunelette_metal_weights *weights)
{
    st_close(&weights->files[0]);
    st_close(&weights->files[1]);
}

static int validate_weight(lunelette_metal_weights *weights, size_t index,
    const char *name, size_t rows, size_t cols, char *error, size_t capacity)
{
    const st_tensor *tensor = NULL;
    st_file *file = NULL;
    for (size_t i = 0; i < 2; i++) {
        const st_tensor *candidate = st_find(&weights->files[i], name);
        if (candidate) {
            if (tensor) {
                snprintf(error, capacity, "Duplicate required Metal tensor: %s", name);
                return -1;
            }
            tensor = candidate;
            file = &weights->files[i];
        }
    }
    if (!tensor || tensor->dtype != ST_BF16 || tensor->rank != (cols ? 2u : 1u) ||
        tensor->shape[0] != rows || (cols && tensor->shape[1] != cols)) {
        if (cols) {
            snprintf(error, capacity, "Metal tensor %s: expected BF16 [%zu,%zu]",
                name, rows, cols);
        } else {
            snprintf(error, capacity, "Metal tensor %s: expected BF16 [%zu]", name, rows);
        }
        return -1;
    }
    size_t elements = rows * (cols ? cols : 1);
    weights->weights[index].file = file;
    weights->weights[index].tensor = tensor;
    weights->weights[index].elements = elements;
    weights->bytes += elements * sizeof(float);
    return 0;
}

int lunelette_metal_weights_open(lunelette_metal_weights *weights,
    const char *first, const char *second, char *error, size_t capacity)
{
    char fallback[512];
    if (!error || !capacity) {
        error = fallback;
        capacity = sizeof fallback;
    }
    memset(weights, 0, sizeof *weights);
    if (!first || !second) {
        snprintf(error, capacity, "Metal requires two explicit safetensors paths");
        return -1;
    }
    if (st_open(&weights->files[0], first, error, capacity) != 0 ||
        st_open(&weights->files[1], second, error, capacity) != 0) {
        goto failure;
    }
    if (lunelette_metal_weights_validate(weights, error, capacity) != 0) {
        goto failure;
    }
    return 0;
failure:
    lunelette_metal_weights_close(weights);
    return -1;
}

/* Separate metadata preflight permits synthetic all-layer validation without
 * model-sized allocations or fabricating sparse checkpoint files. */
int lunelette_metal_weights_validate(lunelette_metal_weights *weights,
    char *error, size_t capacity)
{
    weights->bytes = 0;
    if (validate_weight(weights, 0, "language_model.model.embed_tokens.weight",
            LUNELETTE_MODEL_VOCAB, LUNELETTE_MODEL_HIDDEN, error, capacity) != 0 ||
        validate_weight(weights, 1, "language_model.model.norm.weight",
            LUNELETTE_MODEL_HIDDEN, 0, error, capacity) != 0) {
        return -1;
    }
    for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
        for (size_t i = 0; i < LUNELETTE_MODEL_BLOCK_WEIGHTS; i++) {
            char name[ST_MAX_NAME];
            const lunelette_model_weight *fact = &lunelette_model_block_weights[i];
            snprintf(name, sizeof name, "language_model.model.layers.%zu.%s", layer,
                fact->suffix);
            if (validate_weight(weights, 2 + layer * LUNELETTE_MODEL_BLOCK_WEIGHTS + i,
                    name, fact->rows, fact->cols, error, capacity) != 0) {
                return -1;
            }
        }
    }
    return 0;
}
