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

static void raw_tests(void)
{
    lunelette_metal_weights weights = {0};
    st_tensor tensor = {0};
    strcpy(tensor.name, "synthetic.compact");
    tensor.dtype = ST_F8_E4M3;
    tensor.end = 254;
    weights.file.tensors = &tensor;
    weights.file.count = 1;
    weights.file.file = tmpfile();
    CHECK(weights.file.file);
    weights.weights[3].tensor = &tensor;
    weights.weights[3].elements = 254;
    weights.weights[3].scale = st_bf16_to_f32(0x39fb);
    unsigned char bytes[65538], out[65538];
    size_t count = 0;
    for (unsigned code = 0; code < 256; code++) {
        if (code != 127 && code != 255) {
            bytes[count++] = (unsigned char)code;
        }
    }
    CHECK(count == 254 && bytes[127] == 128);
    CHECK(fwrite(bytes, 1, count, weights.file.file) == count);
    CHECK(fflush(weights.file.file) == 0);
    char error[512];
    CHECK(lunelette_metal_weights_load_raw(&weights, 3, out, count, error, sizeof error) == 0);
    CHECK(!memcmp(bytes, out, count));
    float decoded[254];
    CHECK(st_load_fp8(&weights.file, &tensor, weights.weights[3].scale, decoded, count) == 0);
    CHECK(decoded[0] == 0 && !signbit(decoded[0]));
    CHECK(decoded[127] == 0 && signbit(decoded[127]));
    for (size_t i = 0; i < count; i++) {
        CHECK(isfinite(decoded[i]) && (decoded[i] == 0 || isnormal(decoded[i])));
    }
    const unsigned bad_scales[] = {0, 0x8000, 0xbf80, 0x7fc0, 0x7f80, 0x0001,
        0x007f, 0x0080, 0x0100, 0x04ff, 0x7f00};
    for (size_t i = 0; i < sizeof bad_scales / sizeof bad_scales[0]; i++) {
        weights.weights[3].scale = st_bf16_to_f32((uint16_t)bad_scales[i]);
        CHECK(lunelette_metal_weights_load_raw(&weights, 3, out, count, error, sizeof error));
    }
    /* A subnormal scale times a large code yields a normal product: still reject scale. */
    weights.weights[3].scale = st_bf16_to_f32(0x007f);
    bytes[0] = 126;
    CHECK(isnormal(st_f8_e4m3_to_f32(bytes[0]) * weights.weights[3].scale));
    rewind(weights.file.file);
    CHECK(fwrite(bytes, 1, count, weights.file.file) == count);
    CHECK(fflush(weights.file.file) == 0);
    CHECK(lunelette_metal_weights_load_raw(&weights, 3, out, count, error, sizeof error));
    CHECK(strstr(error, "scale"));
    weights.weights[3].scale = st_bf16_to_f32(0x39fb);
    CHECK(lunelette_metal_weights_load_raw(&weights, LUNELETTE_METAL_WEIGHT_COUNT,
        out, count, error, sizeof error));
    CHECK(lunelette_metal_weights_load_raw(&weights, 3, out, count - 1, error, sizeof error));
    CHECK(lunelette_metal_weights_load_raw(&weights, 3, NULL, count, error, sizeof error));
    tensor.dtype = ST_BF16;
    CHECK(lunelette_metal_weights_load_raw(&weights, 3, out, count, error, sizeof error));
    tensor.dtype = ST_F8_E4M3;
    tensor.end--;
    CHECK(lunelette_metal_weights_load_raw(&weights, 3, out, count, error, sizeof error));
    tensor.end = count = sizeof bytes;
    weights.weights[3].elements = count;
    memset(bytes, 0, count);
    for (unsigned bad = 127; bad <= 255; bad += 128) {
        bytes[count - 1] = (unsigned char)bad;
        rewind(weights.file.file);
        CHECK(fwrite(bytes, 1, count, weights.file.file) == count);
        CHECK(fflush(weights.file.file) == 0);
        CHECK(lunelette_metal_weights_load_raw(&weights, 3, out, count, error, sizeof error));
        CHECK(strstr(error, "byte 65537"));
        CHECK(out[0] == 0); /* Failure after the first bounded chunk. */
    }
    bytes[count - 1] = 128;
    rewind(weights.file.file);
    CHECK(fwrite(bytes, 1, count, weights.file.file) == count);
    CHECK(fflush(weights.file.file) == 0);
    CHECK(!lunelette_metal_weights_load_raw(&weights, 3, out, count, error, sizeof error));
    CHECK(!memcmp(bytes, out, count)); /* All signed zeros admitted at normal scale. */
    weights.weights[3].scale = st_bf16_to_f32(0x7f7f);
    CHECK(!lunelette_metal_weights_load_raw(&weights, 3, out, count, error, sizeof error));
    weights.weights[3].scale = 1.00001f; /* Normal but not widened BF16. */
    CHECK(lunelette_metal_weights_load_raw(&weights, 3, out, count, error, sizeof error));
    weights.weights[3].scale = st_bf16_to_f32(0x39fb);
    tensor.end++;
    weights.weights[3].elements++;
    CHECK(lunelette_metal_weights_load_raw(&weights, 3, out, count + 1, error, sizeof error));
    CHECK(fclose(weights.file.file) == 0);
}

int main(int argc, char **argv)
{
    raw_tests();
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
    lunelette_metal_memory memory;
    CHECK(!lunelette_metal_weights_account(&weights, LUNELETTE_METAL_RESIDENT_FP32, &memory));
    CHECK(memory.weight_bytes == 13716025344ULL && memory.conversion_stage_bytes == 0);
    CHECK(memory.total_bytes == 13858370560ULL);
    CHECK(!lunelette_metal_weights_account(&weights, LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC, &memory));
    CHECK(memory.weight_bytes == 4637454336ULL && memory.conversion_stage_bytes == 113246208ULL);
    CHECK(memory.weight_bytes + memory.conversion_stage_bytes == 4750700544ULL);
    CHECK(memory.scratch_bytes == 87819264 && memory.kv_bytes == 54525952);
    CHECK(memory.total_bytes == 4893045760ULL);
    lunelette_metal_memory arithmetic = memory;
    CHECK(!lunelette_metal_weights_account(&weights, LUNELETTE_METAL_COMPACT_FP8_LOOKUP, &memory));
    CHECK(memory.strategy == LUNELETTE_METAL_COMPACT_FP8_LOOKUP);
    CHECK(memory.weight_bytes == arithmetic.weight_bytes);
    CHECK(memory.conversion_stage_bytes == arithmetic.conversion_stage_bytes);
    CHECK(memory.scratch_bytes == arithmetic.scratch_bytes);
    CHECK(memory.kv_bytes == arithmetic.kv_bytes);
    CHECK(memory.total_bytes == arithmetic.total_bytes);
    for (size_t block = 128; block <= 256; block *= 2) {
        for (int strategy = 0; strategy <= 2; strategy++) {
            CHECK(!lunelette_metal_weights_account_configured(&weights, strategy,
                (lunelette_metal_limits){4352, block}, &memory));
            CHECK(memory.scratch_bytes == (block == 128 ? 111280640 : 222036992));
            CHECK(memory.kv_bytes == 926941184);
            CHECK(memory.total_bytes == memory.weight_bytes + memory.conversion_stage_bytes +
                memory.scratch_bytes + memory.kv_bytes);
        }
    }
    CHECK(lunelette_metal_weights_account_configured(&weights, 0,
        (lunelette_metal_limits){4352, 64}, &memory));
    CHECK(lunelette_metal_weights_account(&weights, (lunelette_metal_strategy)3, &memory));
    size_t saved_elements = weights.weights[3].elements;
    weights.weights[3].elements = SIZE_MAX;
    CHECK(lunelette_metal_weights_account(&weights, LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC, &memory));
    weights.weights[3].elements = saved_elements;
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
