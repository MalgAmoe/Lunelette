#include "block.h"
#include "safetensors.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Fixed before measurement: scalar FP32 versus PyTorch parallel FP32 reductions.
 * Absolute allowance handles cancellation; relative allowance scales large values.
 */
static int compare(const float *actual, const float *expected, size_t count)
{
    float max_abs = 0, max_rel = 0;
    int failed = 0;
    for (size_t i = 0; i < count; i++) {
        float absolute = fabsf(actual[i] - expected[i]);
        float relative = absolute / fmaxf(fabsf(expected[i]), 1e-12f);
        max_abs = fmaxf(max_abs, absolute);
        max_rel = fmaxf(max_rel, relative);
        if (!isfinite(actual[i]) || !isfinite(expected[i]) ||
            absolute > 2e-4f + 1e-4f * fabsf(expected[i])) {
            failed = 1;
        }
    }
    printf("max_abs=%.9g max_rel=%.9g atol=2e-4 rtol=1e-4 %s\n",
           max_abs, max_rel, failed ? "FAIL" : "PASS");
    return failed;
}

static uint64_t decode(const unsigned char *bytes, size_t count)
{
    uint64_t result = 0;
    for (size_t i = 0; i < count; i++) {
        result |= (uint64_t)bytes[i] << (8 * i);
    }
    return result;
}

static int load(st_file *file, const char *name, size_t tokens, size_t width, float *out)
{
    const st_tensor *tensor = st_find(file, name);
    if (!tensor || tensor->dtype != ST_F32 || tensor->rank != 3 ||
        tensor->shape[0] != 1 || tensor->shape[1] != tokens || tensor->shape[2] != width) {
        fprintf(stderr, "%s: expected F32 [1,%zu,%zu]\n", name, tokens, width);
        return -1;
    }
    unsigned char raw[4];
    for (size_t i = 0; i < tokens * width; i++) {
        if (st_read(file, tensor, i * 4, raw, 4)) {
            return -1;
        }
        uint32_t bits = (uint32_t)decode(raw, 4);
        memcpy(out + i, &bits, 4);
    }
    return 0;
}

static int self_test(void)
{
    float a[] = {0, 1, -2}, b[] = {0, 1, -2};
    if (compare(a, b, 3)) {
        return 1;
    }
    b[1] += 1;
    if (!compare(a, b, 3)) {
        return 1;
    }
    b[1] = NAN;
    if (!compare(a, b, 3)) {
        return 1;
    }
    st_file empty = {0};
    int64_t id = 0;
    float sentinel = 123;
    char error[256];
    if (!lunelette_first_block(NULL, &id, 1, &sentinel, error, sizeof error) ||
        !lunelette_first_block(&empty, &id, 1, &sentinel, error, sizeof error) ||
        sentinel != 123) {
        return 1;
    }
    /* A valid synthetic open handle exercises argument rejection before reads. */
    empty.file = tmpfile();
    if (!empty.file) {
        return 1;
    }
    id = -1;
    int failed = !lunelette_first_block(&empty, &id, 1, &sentinel, error, sizeof error);
    id = 131072;
    failed |= !lunelette_first_block(&empty, &id, 1, &sentinel, error, sizeof error);
    id = 0;
    failed |= !lunelette_first_block(&empty, &id, 0, &sentinel, error, sizeof error);
    failed |= !lunelette_first_block(&empty, &id, 257, &sentinel, error, sizeof error);
    failed |= !lunelette_first_block(&empty, NULL, 1, &sentinel, error, sizeof error);
    failed |= !lunelette_first_block(&empty, &id, 1, NULL, error, sizeof error);
    failed |= !lunelette_first_block(&empty, &id, 1, &sentinel, error, sizeof error);
    st_tensor tensor = {0};
    strcpy(tensor.name, "language_model.model.embed_tokens.weight");
    tensor.dtype = ST_F32;
    tensor.rank = 2;
    tensor.shape[0] = 131072;
    tensor.shape[1] = 3072;
    empty.count = 1;
    empty.tensors = &tensor;
    failed |= !lunelette_first_block(&empty, &id, 1, &sentinel, error, sizeof error);
    tensor.dtype = ST_BF16;
    tensor.shape[1] = 3071;
    failed |= !lunelette_first_block(&empty, &id, 1, &sentinel, error, sizeof error);
    tensor.shape[1] = 3072;
    /* Correct embedding descriptor but missing required norm. */
    failed |= !lunelette_first_block(&empty, &id, 1, &sentinel, error, sizeof error);
    failed |= sentinel != 123;
    fclose(empty.file);
    return failed;
}

static int fixture(st_file *weights, const char *path)
{
    st_file file = {0};
    char error[512] = {0};
    int rc = 1;
    float *out = NULL, *expected = NULL;
    int64_t ids[256];
    if (st_open(&file, path, error, sizeof error)) {
        goto done;
    }
    const st_tensor *tokens = st_find(&file, "input_ids");
    if (!tokens || tokens->dtype != ST_I64 || tokens->rank != 2 ||
        tokens->shape[0] != 1 || !tokens->shape[1] || tokens->shape[1] > 256) {
        snprintf(error, sizeof error, "input_ids: expected I64 [1,N], 1<=N<=256");
        goto done;
    }
    size_t count = (size_t)tokens->shape[1];
    for (size_t i = 0; i < count; i++) {
        unsigned char raw[8];
        if (st_read(&file, tokens, i * 8, raw, 8) || decode(raw, 8) >= 131072) {
            snprintf(error, sizeof error, "input_ids: invalid/read failure at %zu", i);
            goto done;
        }
        ids[i] = (int64_t)decode(raw, 8);
    }
    out = malloc(count * 3072 * sizeof(float));
    expected = malloc(count * 3072 * sizeof(float));
    if (!out || !expected) {
        snprintf(error, sizeof error, "fixture allocation failed");
        goto done;
    }
    if (load(&file, "hidden_1", count, 3072, expected) ||
        lunelette_first_block(weights, ids, count, out, error, sizeof error)) {
        goto done;
    }
    printf("%s hidden_1 (%zu tokens): ", path, count);
    rc = compare(out, expected, count * 3072);
done:
    if (rc) {
        fprintf(stderr, "%s: block comparison failed: %s\n", path, error);
    }
    free(out);
    free(expected);
    st_close(&file);
    return rc;
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "--self-test")) {
        return self_test();
    }
    if (argc < 3) {
        fprintf(stderr, "Usage: %s WEIGHT_SHARD FIXTURE [MORE_FIXTURES...] | --self-test\n", argv[0]);
        return 1;
    }
    st_file weights = {0};
    char error[256];
    if (st_open(&weights, argv[1], error, sizeof error)) {
        fprintf(stderr, "%s: %s\n", argv[1], error);
        st_close(&weights);
        return 1;
    }
    int rc = 0;
    for (int i = 2; i < argc; i++) {
        rc |= fixture(&weights, argv[i]);
    }
    st_close(&weights);
    return rc;
}
