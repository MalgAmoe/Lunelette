#include "forward.h"
#include "safetensors.h"
#include "full_depth.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Fixed before measurement: scalar FP32 versus PyTorch parallel FP32 reductions.
 * Absolute allowance handles cancellation; relative allowance scales large values.
 */
static int compare(const float *actual, const float *expected, size_t count)
{
    float max_abs = 0, max_rel = 0;
    int failed = 0;
    size_t failures = 0, worst = 0;
    float max_ratio = 0;
    double squared = 0;
    for (size_t i = 0; i < count; i++) {
        float absolute = fabsf(actual[i] - expected[i]);
        float relative = absolute / fmaxf(fabsf(expected[i]), 1e-12f);
        squared += (double)absolute * absolute;
        float ratio = absolute / (2e-4f + 1e-4f * fabsf(expected[i]));
        if (ratio > max_ratio) {
            max_ratio = ratio;
            worst = i;
        }
        max_abs = fmaxf(max_abs, absolute);
        max_rel = fmaxf(max_rel, relative);
        if (!isfinite(actual[i]) || !isfinite(expected[i]) ||
            absolute > 2e-4f + 1e-4f * fabsf(expected[i])) {
            failed = 1;
            failures++;
        }
    }
    printf("max_abs=%.9g max_rel=%.9g atol=2e-4 rtol=1e-4 %s\n",
           max_abs, max_rel, failed ? "FAIL" : "PASS");
    printf("  rmse=%.9g failing=%zu/%zu worst_index=%zu actual=%.9g expected=%.9g allowance_ratio=%.6g\n",
           sqrt(squared / count), failures, count, worst, actual[worst], expected[worst], max_ratio);
    fflush(stdout);
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
    if (!tensor || tensor->dtype != ST_F32 || tensor->rank != (tokens ? 3u : 2u) ||
        tensor->shape[0] != 1 ||
        (tokens && (tensor->shape[1] != tokens || tensor->shape[2] != width)) ||
        (!tokens && tensor->shape[1] != width)) {
        fprintf(stderr, "%s: expected F32 %s with width %zu (tokens=%zu)\n",
                name, tokens ? "[1,N,D]" : "[1,D]", width, tokens);
        return -1;
    }
    unsigned char raw[4];
    for (size_t i = 0; i < (tokens ? tokens : 1) * width; i++) {
        if (st_read(file, tensor, i * 4, raw, 4)) {
            return -1;
        }
        uint32_t bits = (uint32_t)decode(raw, 4);
        memcpy(out + i, &bits, 4);
    }
    return 0;
}

static int depth_self_test(void)
{
    /* A single damaged row cannot be diluted by the matching rows. */
    float expected[4 * 3072], actual[4 * 3072];
    for (size_t i = 0; i < 4 * 3072; i++) {
        expected[i] = (i % 2) ? -1.0f : 1.0f;
    }
    memcpy(actual, expected, sizeof actual);
    if (compare_full_depth(actual, expected, 4, 3072)) {
        return 1;
    }
    for (int mutation = 0; mutation < 9; mutation++) {
        memcpy(actual, expected, sizeof actual);
        size_t start = 2 * 3072;
        switch (mutation) {
        case 0:
            actual[start] = NAN;
            break;
        case 1:
            actual[start] = INFINITY;
            break;
        case 2:
            memset(actual + start, 0, 3072 * sizeof(float));
            break;
        case 3:
            for (size_t i = 0; i < 3072; i++) {
                actual[start + i] = -expected[start + i];
            }
            break;
        case 4:
            for (size_t i = 0; i < 3072; i += 2) {
                actual[start + i] = expected[start + i + 1];
                actual[start + i + 1] = expected[start + i];
            }
            break;
        case 5:
            /* RMS/cosine pass: the independent absolute ceiling must reject. */
            actual[start] += 0.02f;
            break;
        case 6:
            for (size_t i = 0; i < 3072; i++) {
                actual[start + i] *= 1.01f;
            }
            break;
        case 7:
            actual[start] = -INFINITY;
            break;
        case 8:
            /* Orthogonal perturbation: RMS/absolute pass, cosine rejects. */
            for (size_t i = 0; i < 3072; i++) {
                actual[start + i] += 0.001f;
            }
            break;
        }
        if (!compare_full_depth(actual, expected, 4, 3072) ||
            !compare_full_depth(expected, actual, 4, 3072)) {
            fprintf(stderr, "full-depth mutation %d unexpectedly accepted\n", mutation);
            return 1;
        }
    }
    memset(actual, 0, sizeof actual);
    memset(expected, 0, sizeof expected);
    if (compare_full_depth(actual, expected, 4, 3072)) {
        return 1;
    }
    actual[3072] = 1e-9f;
    if (!compare_full_depth(actual, expected, 4, 3072) ||
        !compare_full_depth(expected, actual, 4, 3072)) {
        return 1;
    }
    /* Nearzero nonzero rows exercise the explicit RMS floor. */
    for (size_t i = 0; i < 4 * 3072; i++) {
        expected[i] = 1e-9f;
        actual[i] = 2e-9f;
    }
    if (compare_full_depth(actual, expected, 4, 3072)) {
        return 1;
    }
    for (size_t i = 0; i < 3072; i++) {
        actual[i] = 3e-9f;
    }
    if (!compare_full_depth(actual, expected, 4, 3072) ||
        !compare_full_depth(actual, expected, 0, 3072) ||
        !compare_full_depth(actual, expected, 257, 3072) ||
        !compare_full_depth(actual, expected, 4, 0) ||
        !compare_full_depth(actual, expected, 4, 3073)) {
        return 1;
    }
    return 0;
}

static int self_test(void)
{
    if (depth_self_test()) {
        return 1;
    }
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
    return 0;
}

static int fixture(st_file *weights, st_file *second, const char *path)
{
    st_file file = {0};
    char error[512] = {0};
    int rc = 1;
    float *out = NULL, *expected = NULL, *checkpoint = NULL, *logits = NULL;
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
    expected = malloc((count * 3072 + 131072) * sizeof(float));
    checkpoint = malloc(count * 3072 * sizeof(float));
    logits = malloc(131072 * sizeof(float));
    if (!out || !expected || !checkpoint || !logits) {
        snprintf(error, sizeof error, "fixture allocation failed");
        goto done;
    }
    clock_t start = clock();
    if (lunelette_forward(weights, second, ids, count, out, logits, checkpoint, error, sizeof error)) {
        goto done;
    }
    printf("%s (%zu tokens), CPU seconds %.3f\n", path, count,
           (double)(clock() - start) / CLOCKS_PER_SEC);
    rc = 0;
    if (load(&file, "hidden_13", count, 3072, expected)) {
        rc = 1;
        goto done;
    }
    printf("hidden_13: ");
    (void)compare(checkpoint, expected, count * 3072); /* Strict diagnostic only. */
    rc |= compare_full_depth(checkpoint, expected, count, 3072);
    if (load(&file, "hidden_26", count, 3072, expected)) {
        rc = 1;
        goto done;
    }
    printf("hidden_26: ");
    (void)compare(out, expected, count * 3072); /* Strict diagnostic only. */
    rc |= compare_full_depth(out, expected, count, 3072);
    if (load(&file, "last_logits", 0, 131072, expected)) {
        rc = 1;
        goto done;
    }
    printf("last_logits: ");
    rc |= compare(logits, expected, 131072);
    size_t actual_top = 0, expected_top = 0;
    for (size_t i = 1; i < 131072; i++) {
        if (logits[i] > logits[actual_top]) {
            actual_top = i;
        }
        if (expected[i] > expected[expected_top]) {
            expected_top = i;
        }
    }
    size_t actual_ties = 0, expected_ties = 0;
    float actual_second = -INFINITY, expected_second = -INFINITY;
    for (size_t i = 0; i < 131072; i++) {
        actual_ties += logits[i] == logits[actual_top];
        expected_ties += expected[i] == expected[expected_top];
        if (i != actual_top) {
            actual_second = fmaxf(actual_second, logits[i]);
        }
        if (i != expected_top) {
            expected_second = fmaxf(expected_second, expected[i]);
        }
    }
    int agreement = logits[expected_top] == logits[actual_top] &&
                    expected[actual_top] == expected[expected_top];
    printf("top1 actual=%zu expected=%zu %s ties=%zu/%zu margins=%.9g/%.9g "
           "(diagnostic; smallest index on exact ties)\n",
           actual_top, expected_top, agreement ? "MATCH" : "MISMATCH",
           actual_ties, expected_ties, logits[actual_top] - actual_second,
           expected[expected_top] - expected_second);

done:
    if (rc) {
        fprintf(stderr, "%s: forward comparison failed: %s\n", path,
                error[0] ? error : "tensor/shape or numerical mismatch (see metrics)");
    }
    free(checkpoint);
    free(logits);
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
    if (argc < 4) {
        fprintf(stderr, "Usage: %s SHARD1 SHARD2 FIXTURE [MORE_FIXTURES...] | --self-test\n", argv[0]);
        return 1;
    }
    st_file weights = {0}, second = {0};
    char error[256];
    if (st_open(&weights, argv[1], error, sizeof error)) {
        fprintf(stderr, "%s: %s\n", argv[1], error);
        st_close(&weights);
        return 1;
    }
    if (st_open(&second, argv[2], error, sizeof error)) {
        fprintf(stderr, "%s: %s\n", argv[2], error);
        st_close(&weights);
        return 1;
    }
    int rc = 0;
    for (int i = 3; i < argc; i++) {
        rc |= fixture(&weights, &second, argv[i]);
    }
    st_close(&second);
    st_close(&weights);
    return rc;
}
