#include "attention.h"
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

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "--self-test")) {
        float a[] = {0, 1, -2}, b[] = {0, 1, -2};
        if (compare(a, b, 3)) {
            return 1;
        }
        b[1] += 1;
        if (!compare(a, b, 3)) {
            return 1;
        }
        b[1] = NAN;
        return !compare(a, b, 3);
    }
    if (argc != 2) {
        fprintf(stderr, "Usage: %s FIXTURE | --self-test\n", argv[0]);
        return 1;
    }
    st_file fixture = {0};
    char error[256];
    int rc = 1;
    float *q = NULL, *k = NULL, *v = NULL, *out = NULL, *expected = NULL;
    uint32_t positions[256];
    float scratch[256];
    if (st_open(&fixture, argv[1], error, sizeof error)) {
        fprintf(stderr, "%s: %s\n", argv[1], error);
        goto done;
    }
    const st_tensor *pos = st_find(&fixture, "positions");
    if (!pos || pos->dtype != ST_I64 || pos->rank != 1 ||
        !pos->shape[0] || pos->shape[0] > 256) {
        fprintf(stderr, "positions: expected I64 [N], 1<=N<=256\n");
        goto done;
    }
    size_t tokens = (size_t)pos->shape[0];
    for (size_t i = 0; i < tokens; i++) {
        unsigned char raw[8];
        if (st_read(&fixture, pos, i * 8, raw, 8) || decode(raw, 8) >= 262144) {
            goto done;
        }
        positions[i] = (uint32_t)decode(raw, 8);
    }
    q = malloc(tokens * 4096 * sizeof(float));
    k = malloc(tokens * 1024 * sizeof(float));
    v = malloc(tokens * 1024 * sizeof(float));
    out = malloc(tokens * 4096 * sizeof(float));
    expected = malloc(tokens * 4096 * sizeof(float));
    if (!q || !k || !v || !out || !expected ||
        load(&fixture, "q", tokens, 4096, q) ||
        load(&fixture, "k", tokens, 1024, k) ||
        load(&fixture, "v", tokens, 1024, v) ||
        lunelette_ministral3_yarn(q, k, positions, tokens)) {
        goto done;
    }
    printf("%s rotated query: ", argv[1]);
    if (load(&fixture, "rotated_q", tokens, 4096, expected) ||
        compare(q, expected, tokens * 4096)) {
        goto done;
    }
    printf("%s rotated key: ", argv[1]);
    if (load(&fixture, "rotated_k", tokens, 1024, expected) ||
        compare(k, expected, tokens * 1024)) {
        goto done;
    }
    if (lunelette_causal_gqa(q, k, v, out, scratch, tokens, 32, 8, 128)) {
        goto done;
    }
    printf("%s attention before o_proj: ", argv[1]);
    if (load(&fixture, "attention", tokens, 4096, expected) ||
        compare(out, expected, tokens * 4096)) {
        goto done;
    }
    rc = 0;
done:
    if (rc) {
        fprintf(stderr, "%s: attention comparison failed\n", argv[1]);
    }
    free(q);
    free(k);
    free(v);
    free(out);
    free(expected);
    st_close(&fixture);
    return rc;
}
