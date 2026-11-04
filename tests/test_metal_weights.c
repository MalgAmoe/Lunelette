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

/* Expected acceptance/rejection is asserted directly at each call below.
 * Parser rejects within-shard duplicates; layout rejects cross-shard duplicates. */
static void duplicate_shard_test(const char *path, const char *name)
{
    char header[1024];
    int length = snprintf(header, sizeof header,
        "{\"%s\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},"
        "\"%s\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]}}",
        name, name);
    CHECK(length > 0 && (size_t)length < sizeof header);
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    for (size_t i = 0; i < 8; i++) {
        CHECK(fputc((int)(((uint64_t)length >> (8 * i)) & 255), file) != EOF);
    }
    CHECK(fwrite(header, 1, (size_t)length, file) == (size_t)length);
    CHECK(fwrite("\0\0\0\0", 1, 4, file) == 4);
    CHECK(fclose(file) == 0);
    char error[512];
    st_file parsed;
    CHECK(st_open(&parsed, path, error, sizeof error) != 0);
    CHECK(strstr(error, "unique") != NULL);
    CHECK(parsed.file == NULL && parsed.tensors == NULL);
    st_close(&parsed);
    lunelette_metal_weights metal;
    CHECK(lunelette_metal_weights_open(&metal, path, path, error, sizeof error) != 0);
    CHECK(strstr(error, "unique") != NULL);
    CHECK(metal.files[0].file == NULL && metal.files[0].tensors == NULL);
    CHECK(metal.files[1].file == NULL && metal.files[1].tensors == NULL);
    lunelette_metal_weights_close(&metal);
    CHECK(remove(path) == 0);
}

int main(int argc, char **argv)
{
    CHECK(argc <= 2);
    const char *path = argc == 2 ? argv[1] : "build/test-metal-duplicates.tmp";
    duplicate_shard_test(path, "language_model.model.embed_tokens.weight");
    duplicate_shard_test(path, "vision.unrelated.weight");
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
        char saved_name[ST_MAX_NAME];
        memcpy(saved_name, tensors[i].name, sizeof saved_name);
        snprintf(tensors[i].name, sizeof tensors[i].name, "unrelated.tensor");
        CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
        memcpy(tensors[i].name, saved_name, sizeof saved_name);
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
    /* Shard placement is not policy: all required weights may be in one file,
     * and unrelated (e.g. vision) tensors may have arbitrary metadata. */
    st_tensor unrelated = {0};
    snprintf(unrelated.name, sizeof unrelated.name, "vision.unrelated.weight");
    weights.files[0].count = LUNELETTE_METAL_WEIGHT_COUNT;
    weights.files[1].tensors = &unrelated;
    weights.files[1].count = 1;
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) == 0);
    for (size_t i = 0; i < LUNELETTE_METAL_WEIGHT_COUNT; i++) {
        weights.files[1].tensors = &tensors[i];
        CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
        CHECK(strstr(error, tensors[i].name) != NULL);
    }
    weights.files[0].count = 100;
    weights.files[1].tensors = tensors + 100;
    weights.files[1].count = LUNELETTE_METAL_WEIGHT_COUNT - 100;
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
