#include "cpu_weights.h"
#include "model_facts.h"

#include <string.h>

const st_tensor *lunelette_require(st_file **files, size_t nfiles, const char *name,
                                  size_t rows, size_t cols, st_file **owner,
                                  char *error, size_t capacity)
{
    const st_tensor *found = NULL;
    for (size_t f = 0; f < nfiles; f++) {
        for (size_t i = 0; i < files[f]->count; i++) {
            const st_tensor *tensor = &files[f]->tensors[i];
            if (strcmp(tensor->name, name)) {
                continue;
            }
            if (found) {
                if (error && capacity) {
                    snprintf(error, capacity, "%s: duplicate tensor across supplied shards", name);
                }
                return NULL;
            }
            found = tensor;
            *owner = files[f];
        }
    }
    if (!found || found->dtype != ST_BF16 || found->rank != (cols ? 2u : 1u) ||
        found->shape[0] != rows || (cols && found->shape[1] != cols)) {
        if (error && capacity) {
            snprintf(error, capacity, "%s: expected BF16 %zu rows, %zu columns (0=vector)",
                     name, rows, cols);
        }
        return NULL;
    }
    return found;
}

/* Validate the whole supported text layout before expensive execution, including
 * duplicate names. Unrelated vision tensors are deliberately ignored. */
int lunelette_cpu_validate_layout(st_file **files, char *error, size_t capacity)
{
    st_file *owner = NULL;
    if (!lunelette_require(files, 2, "language_model.model.embed_tokens.weight",
                           LUNELETTE_MODEL_VOCAB, LUNELETTE_MODEL_HIDDEN, &owner, error, capacity) ||
        !lunelette_require(files, 2, "language_model.model.norm.weight",
                           LUNELETTE_MODEL_HIDDEN, 0, &owner, error, capacity)) {
        return -1;
    }
    for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
        for (size_t i = 0; i < LUNELETTE_MODEL_BLOCK_WEIGHTS; i++) {
            char name[128];
            snprintf(name, sizeof name, "language_model.model.layers.%zu.%s", layer,
                     lunelette_model_block_weights[i].suffix);
            if (!lunelette_require(files, 2, name, lunelette_model_block_weights[i].rows,
                                    lunelette_model_block_weights[i].cols, &owner, error, capacity)) {
                return -1;
            }
        }
    }
    return 0;
}
