#include "metal_weights.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void descriptor(st_tensor *tensor, const char *name, size_t rows, size_t cols,
    st_dtype dtype)
{
    snprintf(tensor->name, sizeof tensor->name, "%s", name);
    tensor->dtype = dtype;
    tensor->rank = rows ? (cols ? 2 : 1) : 0;
    tensor->shape[0] = rows;
    tensor->shape[1] = cols;
    tensor->end = (rows ? rows * (cols ? cols : 1) : 1) * dtype;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    lunelette_metal_weights weights = {0};
    st_tensor tensors[LUNELETTE_METAL_WEIGHT_COUNT + 364 + 1] = {0};
    descriptor(&tensors[0], "language_model.model.embed_tokens.weight",
        LUNELETTE_MODEL_VOCAB, LUNELETTE_MODEL_HIDDEN, ST_BF16);
    descriptor(&tensors[1], "language_model.model.norm.weight", LUNELETTE_MODEL_HIDDEN,
        0, ST_BF16);
    size_t index = 2;
    size_t scales = LUNELETTE_METAL_WEIGHT_COUNT;
    for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
        for (size_t i = 0; i < LUNELETTE_MODEL_BLOCK_WEIGHTS; i++) {
            char name[ST_MAX_NAME];
            const lunelette_model_weight *fact = &lunelette_model_block_weights[i];
            snprintf(name, sizeof name, "language_model.model.layers.%zu.%s", layer,
                fact->suffix);
            descriptor(&tensors[index++], name, fact->rows, fact->cols,
                fact->cols ? ST_F8_E4M3 : ST_BF16);
            if (fact->cols) {
                const char *suffixes[] = {"weight_scale_inv", "activation_scale"};
                for (size_t j = 0; j < 2; j++) {
                    char companion[ST_MAX_NAME];
                    snprintf(companion, sizeof companion, "%.*s%s",
                        (int)(strlen(name) - 6), name, suffixes[j]);
                    descriptor(&tensors[scales++], companion, 0, 0, ST_BF16);
                }
            }
        }
    }
    CHECK(scales == LUNELETTE_METAL_WEIGHT_COUNT + 364);
    weights.file.tensors = tensors;
    weights.file.count = scales;
    char error[512];
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) == 0);
    CHECK(weights.bytes == 13716025344ULL);
    for (size_t i = 0; i < scales; i++) {
        char saved_name[ST_MAX_NAME];
        memcpy(saved_name, tensors[i].name, sizeof saved_name);
        strcpy(tensors[i].name, "unrelated");
        CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
        memcpy(tensors[i].name, saved_name, sizeof saved_name);
        st_dtype saved = tensors[i].dtype;
        tensors[i].dtype = saved == ST_F8_E4M3 ? ST_BF16 : ST_F32;
        CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
        CHECK(strstr(error, tensors[i].name) != NULL);
        tensors[i].dtype = saved;
        tensors[i].rank++;
        CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
        tensors[i].rank--;
        if (tensors[i].rank) {
            tensors[i].shape[0]++;
            CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
            tensors[i].shape[0]--;
            if (tensors[i].rank == 2) {
                tensors[i].shape[1]++;
                CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
                tensors[i].shape[1]--;
            }
        }
    }
    weights.file.count--;
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
    weights.file.count = scales + 1;
    descriptor(&tensors[scales], "vision.unrelated.weight", 1, 0, ST_F32);
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) == 0);
    strcpy(tensors[scales].name, "language_model.unexpected.weight");
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) != 0);
    weights.file.count = scales;

    /* Actual small scalar payloads, with synthetic projection descriptors: no
     * model-size sparse file and no GPU. Exercise both companion values. */
    FILE *payload = tmpfile();
    CHECK(payload != NULL);
    CHECK(fwrite("\201\076", 1, 2, payload) == 2); /* BF16 0.251953125 */
    CHECK(fflush(payload) == 0);
    weights.file.file = payload;
    CHECK(lunelette_metal_weights_validate(&weights, error, sizeof error) == 0);
    CHECK(lunelette_metal_weights_read_scales(&weights, error, sizeof error) == 0);
    CHECK(weights.weights[3].scale == 0.251953125f);
    const unsigned bad[] = {0, 0x8000, 0xbf80, 0x7f80, 0xff80, 0x7fc0, 0xffc0};
    for (size_t j = 0; j < 2; j++) {
        /* Isolate weight or activation scalar while others use valid bytes. */
        tensors[LUNELETTE_METAL_WEIGHT_COUNT + j].begin = 2;
        tensors[LUNELETTE_METAL_WEIGHT_COUNT + j].end = 4;
        for (size_t k = 0; k < sizeof bad / sizeof bad[0]; k++) {
            CHECK(fseek(payload, 2, SEEK_SET) == 0);
            CHECK(fputc(bad[k] & 255, payload) != EOF);
            CHECK(fputc(bad[k] >> 8, payload) != EOF);
            CHECK(fflush(payload) == 0);
            CHECK(lunelette_metal_weights_read_scales(&weights, error, sizeof error) != 0);
        }
        tensors[LUNELETTE_METAL_WEIGHT_COUNT + j].begin = 0;
        tensors[LUNELETTE_METAL_WEIGHT_COUNT + j].end = 2;
    }
    tensors[LUNELETTE_METAL_WEIGHT_COUNT].begin = 4;
    tensors[LUNELETTE_METAL_WEIGHT_COUNT].end = 6;
    CHECK(lunelette_metal_weights_read_scales(&weights, error, sizeof error) != 0);
    CHECK(fclose(payload) == 0);
    /* Borrowed descriptors aren't passed to close. Failed open owns its cleanup. */
    lunelette_metal_weights failed;
    CHECK(lunelette_metal_weights_open(&failed, "missing-model", error, sizeof error) != 0);
    CHECK(failed.file.file == NULL && failed.file.tensors == NULL && failed.bytes == 0);
    lunelette_metal_weights_close(&failed);
    CHECK(lunelette_metal_weights_open(&failed, NULL, error, sizeof error) != 0);
    CHECK(failed.file.file == NULL && failed.file.tensors == NULL && failed.bytes == 0);
    lunelette_metal_weights_close(&failed);
    puts("Mixed FP8/BF16 metadata and actual scalar reads passed (182 FP8, 53 norms, embedding)");
    return 0;
}
