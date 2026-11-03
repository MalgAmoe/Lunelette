#include "cache.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t decode(const unsigned char *raw, size_t count)
{
    uint64_t value = 0;
    for (size_t i = 0; i < count; i++) {
        value |= (uint64_t)raw[i] << (8 * i);
    }
    return value;
}

static int compare(const float *actual, const float *expected, size_t count)
{
    float max_abs = 0, max_rel = 0;
    size_t failures = 0;
    for (size_t i = 0; i < count; i++) {
        float absolute = fabsf(actual[i] - expected[i]);
        max_abs = fmaxf(max_abs, absolute);
        max_rel = fmaxf(max_rel, absolute / fmaxf(fabsf(expected[i]), 1e-12f));
        failures += !isfinite(actual[i]) || !isfinite(expected[i]) ||
                    absolute > 2e-4f + 1e-4f * fabsf(expected[i]);
    }
    printf("logits max_abs=%.9g max_rel=%.9g failures=%zu/%zu atol=2e-4 rtol=1e-4\n",
           max_abs, max_rel, failures, count);
    fflush(stdout);
    return failures ? -1 : 0;
}

struct context {
    st_file first, second;
    lunelette_cache *cache;
    float *baseline;
    char **paths;
    size_t steps, step;
    float *hidden, *expected;
    int64_t expected_tokens[256];
};

static int read_ids(st_file *file, int64_t *ids, size_t *count)
{
    const st_tensor *tensor = st_find(file, "input_ids");
    if (!tensor || tensor->dtype != ST_I64 || tensor->rank != 2 ||
        tensor->shape[0] != 1 || !tensor->shape[1] || tensor->shape[1] > 256) {
        return -1;
    }
    *count = tensor->shape[1];
    for (size_t i = 0; i < *count; i++) {
        unsigned char raw[8];
        if (st_read(file, tensor, i * 8, raw, 8) || decode(raw, 8) >= 131072) {
            return -1;
        }
        ids[i] = (int64_t)decode(raw, 8);
    }
    return 0;
}

static int next(void *opaque, const int64_t *ids, size_t count, float *logits,
                char *error, size_t capacity)
{
    struct context *context = opaque;
    st_file fixture = {0};
    int rc = -1;
    if (context->step >= context->steps ||
        st_open(&fixture, context->paths[context->step], error, capacity)) {
        return -1;
    }
    int64_t expected_ids[256];
    size_t expected_count;
    if (read_ids(&fixture, expected_ids, &expected_count) || expected_count != count ||
        memcmp(ids, expected_ids, count * sizeof *ids)) {
        goto done;
    }
    const st_tensor *tensor = st_find(&fixture, "last_logits");
    if (!tensor || tensor->dtype != ST_F32 || tensor->rank != 2 ||
        tensor->shape[0] != 1 || tensor->shape[1] != 131072) {
        goto done;
    }
    for (size_t i = 0; i < 131072; i++) {
        unsigned char raw[4];
        if (st_read(&fixture, tensor, i * 4, raw, 4)) {
            goto done;
        }
        uint32_t bits = (uint32_t)decode(raw, 4);
        memcpy(context->expected + i, &bits, 4);
    }
    if (lunelette_forward(&context->first, &context->second, ids, count,
                          context->hidden, logits, NULL, error, capacity)) {
        goto done;
    }
    memcpy(context->baseline, logits, 131072 * sizeof(float));
    size_t length = lunelette_cache_length(context->cache);
    if (lunelette_cache_forward(context->cache, ids + length, count - length,
                                 logits, error, capacity)) {
        goto done;
    }
    puts("cached vs uncached C:");
    if (compare(logits, context->baseline, 131072)) {
        goto done;
    }
    printf("step=%zu prefix=%zu %s\n", context->step, count, context->paths[context->step]);
    if (compare(logits, context->expected, 131072)) {
        goto done;
    }
    size_t best = 0, actual = 0;
    for (size_t i = 1; i < 131072; i++) {
        if (context->expected[i] > context->expected[best]) {
            best = i;
        }
        if (logits[i] > logits[actual]) {
            actual = i;
        }
    }
    printf("token actual=%zu expected=%zu\n", actual, best);
    fflush(stdout);
    if (actual != best) {
        goto done;
    }
    context->expected_tokens[context->step++] = (int64_t)best;
    rc = 0;
done:
    st_close(&fixture);
    return rc;
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "--self-test")) {
        float a[] = {0, 1, -2}, b[] = {0, 1, -2};
        if (compare(a, b, 3)) {
            return 1;
        }
        b[1] = 2;
        if (!compare(a, b, 3)) {
            return 1;
        }
        b[1] = NAN;
        return compare(a, b, 3) ? 0 : 1;
    }
    int boundary = argc > 1 && !strcmp(argv[1], "--forward-only");
    if (boundary) {
        argc--;
        argv++;
    }
    if (argc < 4 || argc > 259 || (boundary && argc != 4)) {
        fprintf(stderr, "Usage: compare_generation [--forward-only] SHARD1 SHARD2 STEP_FIXTURE...\n");
        return 1;
    }
    struct context context = {0};
    char error[512] = {0};
    int rc = 1;
    st_file fixture = {0};
    int64_t ids[256];
    size_t count;
    context.paths = argv + 3;
    context.steps = (size_t)argc - 3;
    context.hidden = malloc(256 * 3072 * sizeof(float));
    context.expected = malloc(131072 * sizeof(float));
    float *logits = malloc(131072 * sizeof(float));
    if (!context.hidden || !context.expected || !logits ||
        st_open(&context.first, argv[1], error, sizeof error) ||
        st_open(&context.second, argv[2], error, sizeof error) ||
        st_open(&fixture, argv[3], error, sizeof error) || read_ids(&fixture, ids, &count)) {
        goto done;
    }
    context.cache = lunelette_cache_init(&context.first, &context.second, 256, error, sizeof error);
    context.baseline = malloc(131072 * sizeof(float));
    if (!context.cache || !context.baseline) {
        goto done;
    }
    if (boundary) {
        rc = next(&context, ids, count, logits, error, sizeof error) ? 1 : 0;
    } else {
        lunelette_generation_config config = LUNELETTE_GENERATION_DEFAULT;
        config.max_new_tokens = context.steps;
        lunelette_generation_result result;
        if (lunelette_generate_with_logits(next, &context, ids, count, &config,
                                            &result, error, sizeof error)) {
            goto done;
        }
        rc = result.count != context.steps || context.step != context.steps ||
             memcmp(result.tokens, context.expected_tokens, result.count * sizeof(int64_t));
        printf("generation count=%zu stop=%d %s\n", result.count, result.stop, rc ? "FAIL" : "PASS");
        if (!rc) {
            lunelette_generation_result adapted;
            if (lunelette_generate_cached(context.cache, ids, count, &config,
                    &adapted, error, sizeof error) || adapted.count != result.count ||
                adapted.stop != result.stop ||
                memcmp(adapted.tokens, result.tokens, result.count * sizeof(int64_t))) {
                rc = 1;
            }
            size_t before = lunelette_cache_length(context.cache);
            int64_t invalid = 131072;
            if (!lunelette_cache_forward(context.cache, &invalid, 1, logits, error, sizeof error) ||
                lunelette_cache_length(context.cache) != before) {
                rc = 1;
            }
            lunelette_cache_reset(context.cache);
            if (lunelette_cache_length(context.cache)) {
                rc = 1;
            }
            printf("cached generation adapter/reset/invalid token %s\n", rc ? "FAIL" : "PASS");
        }
    }
done:
    if (rc) {
        fprintf(stderr, "generation comparison failed: %s\n", error);
    }
    lunelette_cache_free(context.cache);
    free(context.baseline);
    free(logits);
    free(context.hidden);
    free(context.expected);
    st_close(&fixture);
    st_close(&context.first);
    st_close(&context.second);
    return rc ? 1 : 0;
}
