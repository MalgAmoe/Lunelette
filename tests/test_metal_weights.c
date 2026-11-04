#include "metal_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void tensor(st_tensor *value, const char *name, size_t rows, size_t cols)
{
    snprintf(value->name, sizeof value->name, "%s", name);
    value->dtype = ST_BF16;
    value->rank = cols ? 2 : 1;
    value->shape[0] = rows;
    value->shape[1] = cols;
}

int main(void)
{
    lunelette_metal_weights weights = {0};
    st_tensor tensors[LUNELETTE_METAL_WEIGHT_COUNT] = {0};
    tensor(&tensors[0], "language_model.model.embed_tokens.weight",
        LUNELETTE_MODEL_VOCAB, LUNELETTE_MODEL_HIDDEN);
    tensor(&tensors[1], "language_model.model.norm.weight", LUNELETTE_MODEL_HIDDEN, 0);
    size_t index = 2;
    for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
        for (size_t i = 0; i < LUNELETTE_MODEL_BLOCK_WEIGHTS; i++) {
            char name[ST_MAX_NAME];
            const lunelette_model_weight *fact = &lunelette_model_block_weights[i];
            snprintf(name, sizeof name, "language_model.model.layers.%zu.%s", layer, fact->suffix);
            tensor(&tensors[index++], name, fact->rows, fact->cols);
        }
    }
    /* Parser byte-range validation is covered by test_safetensors. These borrowed
     * descriptors test only complete text metadata, not fabricated weight data. */
    weights.files[0].tensors = tensors;
    weights.files[0].count = 100;
    weights.files[1].tensors = tensors + 100;
    weights.files[1].count = LUNELETTE_METAL_WEIGHT_COUNT - 100;
    char error[512];
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) == 0);
    CHECK(weights.bytes == 3429006336ULL * sizeof(float));
    CHECK(weights.weights[index - 1].file == &weights.files[1]);
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) == 0);
    CHECK(weights.bytes == 3429006336ULL * sizeof(float));
    for (size_t i = 0; i < LUNELETTE_METAL_WEIGHT_COUNT; i++) {
        tensors[i].dtype = ST_F32;
        CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
        CHECK(strstr(error, tensors[i].name) != NULL);
        tensors[i].dtype = ST_BF16;
        tensors[i].shape[0]++;
        CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
        tensors[i].shape[0]--;
        tensors[i].rank++;
        CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
        tensors[i].rank--;
        if (tensors[i].rank == 2) {
            tensors[i].shape[1]++;
            CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
            tensors[i].shape[1]--;
        }
    }
    weights.files[1].count--;
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
    weights.files[1].count++;
    weights.files[1].tensors = tensors;
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
    CHECK(strstr(error, "Duplicate") != NULL);
    /* Do not close the synthetic borrowed descriptors. Real failed open closes
     * its own descriptors; repeat to catch accidental live/reused handles. */
    for (size_t i = 0; i < 10; i++) {
        lunelette_metal_weights failed;
        CHECK(lunelette_metal_weights_open(&failed, "missing-metal-shard", NULL,
            error, sizeof error) != 0);
        lunelette_metal_weights_close(&failed);
        CHECK(lunelette_metal_weights_open(&failed, "missing-metal-shard",
            "missing-second-shard", error, sizeof error) != 0);
        lunelette_metal_weights_close(&failed);
    }
    puts("Metal metadata preflight tests passed (all 26 layers, no weights allocated)");
    return 0;
}
