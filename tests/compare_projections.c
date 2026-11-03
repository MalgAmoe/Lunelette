#include "safetensors.h"
#include "lunelette.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 3072
#define MAX_WIDTH 4096
#define MAX_SHARDS 16

static uint64_t decode_le(const unsigned char *bytes, size_t count)
{
    uint64_t value = 0;
    for (size_t i = 0; i < count; i++) {
        value |= (uint64_t)bytes[i] << (8 * i);
    }
    return value;
}

/* Sequential float dot products versus parallel PyTorch reductions. Absolute
 * allowance protects near-zero cancellation; relative allowance scales large
 * outputs. These are fixed before comparison, not fitted per fixture. */
static int compare(const float *actual, const float *expected, size_t count,
                   float *max_absolute, float *max_relative)
{
    int failed = 0;
    for (size_t i = 0; i < count; i++) {
        float absolute = fabsf(actual[i] - expected[i]);
        float relative = absolute / fmaxf(fabsf(expected[i]), 1e-12f);
        if (!isfinite(actual[i]) || !isfinite(expected[i])) {
            failed = 1;
        }
        if (absolute > *max_absolute) {
            *max_absolute = absolute;
        }
        if (relative > *max_relative) {
            *max_relative = relative;
        }
        if (absolute > 2e-4f + 1e-4f * fabsf(expected[i])) {
            failed = 1;
        }
    }
    return failed;
}

static const st_tensor *require(st_file *file, const char *name, st_dtype dtype,
                                size_t rank, uint64_t first, uint64_t second)
{
    const st_tensor *tensor = st_find(file, name);
    if (!tensor || tensor->dtype != dtype || tensor->rank != rank ||
        tensor->shape[0] != first || (rank == 2 && tensor->shape[1] != second)) {
        fprintf(stderr, "%s: expected dtype %d shape [%llu%s%llu] (rank %zu)\n",
                name, (int)dtype, (unsigned long long)first, rank == 2 ? "," : ";unused=",
                (unsigned long long)second, rank);
        return NULL;
    }
    return tensor;
}

static int find_weight(st_file *shards, size_t count, const char *name,
                       st_file **owner, const st_tensor **tensor)
{
    *tensor = NULL;
    for (size_t i = 0; i < count; i++) {
        const st_tensor *found = st_find(&shards[i], name);
        if (found) {
            if (*tensor) {
                fprintf(stderr, "%s: duplicate across supplied shards\n", name);
                return -1;
            }
            *tensor = found;
            *owner = &shards[i];
        }
    }
    if (!*tensor) {
        fprintf(stderr, "%s: missing from supplied shards\n", name);
        return -1;
    }
    return 0;
}

static int self_test(void)
{
    float input[] = {1, -2}, weight[] = {2, 3}, normalized[2], projected[1];
    const float matrix[] = {1, -1};
    float expected[1], absolute = 0, relative = 0;
    if (lunelette_rmsnorm(input, weight, normalized, 2, 1e-5f) ||
        lunelette_matvec(matrix, normalized, projected, 1, 2)) {
        return 1;
    }
    expected[0] = 8.0f / sqrtf(2.5f + 1e-5f);
    if (compare(projected, expected, 1, &absolute, &relative)) {
        return 1;
    }
    expected[0] += 1;
    if (!compare(projected, expected, 1, &absolute, &relative)) {
        return 1;
    }
    expected[0] = NAN;
    if (!compare(projected, expected, 1, &absolute, &relative)) {
        return 1;
    }
    puts("projection comparator synthetic match/mismatch/nonfinite tests passed");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "--self-test")) {
        return self_test();
    }
    if (argc < 3 || argc > MAX_SHARDS + 2) {
        fprintf(stderr, "Usage: %s FIXTURE WEIGHT_SHARD [MORE_SHARDS...]\n", argv[0]);
        return 1;
    }
    st_file shards[MAX_SHARDS] = {{0}}, fixture = {0};
    size_t shard_count = (size_t)argc - 2;
    int rc = 1;
    char error[256];
    float *matrix = NULL;
    float norm_weight[HIDDEN], embedding[HIDDEN], normalized[HIDDEN];
    float actual[MAX_WIDTH], expected[MAX_WIDTH];
    unsigned char raw[MAX_WIDTH * 4];
    const char *names[] = {"input_norm", "q", "k", "v"};
    const char *weight_names[] = {
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight"
    };
    const size_t widths[] = {HIDDEN, 4096, 1024, 1024};
    for (size_t i = 0; i < shard_count; i++) {
        if (st_open(&shards[i], argv[i + 2], error, sizeof error)) {
            fprintf(stderr, "%s: %s\n", argv[i + 2], error);
            goto done;
        }
    }
    if (st_open(&fixture, argv[1], error, sizeof error)) {
        fprintf(stderr, "%s: %s\n", argv[1], error);
        goto done;
    }
    const st_tensor *ids = st_find(&fixture, "input_ids");
    if (!ids || ids->dtype != ST_I64 || ids->rank != 2 || ids->shape[0] != 1 ||
        !ids->shape[1] || ids->shape[1] > 256) {
        fprintf(stderr, "input_ids: expected I64 [1,N], 1 <= N <= 256\n");
        goto done;
    }
    st_file *embedding_file = NULL, *owner = NULL;
    const st_tensor *embedding_tensor = NULL, *tensor = NULL;
    if (find_weight(shards, shard_count, "language_model.model.embed_tokens.weight",
                    &embedding_file, &embedding_tensor) ||
        !require(embedding_file, embedding_tensor->name, ST_BF16, 2, 131072, HIDDEN) ||
        find_weight(shards, shard_count, weight_names[0], &owner, &tensor) ||
        !require(owner, weight_names[0], ST_BF16, 1, HIDDEN, 0) ||
        st_load_bf16(owner, tensor, norm_weight, HIDDEN)) {
        goto done;
    }
    /* One projection at a time: at most 48 MiB F32 matrix, not the model. */
    matrix = malloc((size_t)MAX_WIDTH * HIDDEN * sizeof *matrix);
    if (!matrix) {
        fprintf(stderr, "cannot allocate projection matrix\n");
        goto done;
    }
    for (size_t stage = 0; stage < 4; stage++) {
        size_t width = widths[stage];
        const st_tensor *reference = st_find(&fixture, names[stage]);
        if (!reference || reference->dtype != ST_F32 || reference->rank != 3 ||
            reference->shape[0] != 1 || reference->shape[1] != ids->shape[1] ||
            reference->shape[2] != width) {
            fprintf(stderr, "%s: expected F32 [1,%llu,%zu]\n", names[stage],
                    (unsigned long long)ids->shape[1], width);
            goto done;
        }
        if (stage && (find_weight(shards, shard_count, weight_names[stage], &owner, &tensor) ||
            !require(owner, weight_names[stage], ST_BF16, 2, width, HIDDEN) ||
            st_load_bf16(owner, tensor, matrix, width * HIDDEN))) {
            goto done;
        }
        float max_absolute = 0, max_relative = 0;
        int failed = 0;
        for (uint64_t row = 0; row < ids->shape[1]; row++) {
            if (st_read(&fixture, ids, row * 8, raw, 8)) {
                goto done;
            }
            uint64_t token = decode_le(raw, 8);
            if (token > INT64_MAX || st_embedding(embedding_file, embedding_tensor,
                    (int64_t)token, embedding, HIDDEN) ||
                lunelette_rmsnorm(embedding, norm_weight, normalized, HIDDEN, 1e-5f)) {
                fprintf(stderr, "invalid token or failed embedding/norm at row %llu\n",
                        (unsigned long long)row);
                goto done;
            }
            const float *output = normalized;
            if (stage) {
                if (lunelette_matvec(matrix, normalized, actual, width, HIDDEN)) {
                    goto done;
                }
                output = actual;
            }
            if (st_read(&fixture, reference, row * width * 4, raw, width * 4)) {
                goto done;
            }
            for (size_t i = 0; i < width; i++) {
                uint32_t bits = (uint32_t)decode_le(raw + i * 4, 4);
                memcpy(&expected[i], &bits, 4);
            }
            if (compare(output, expected, width, &max_absolute, &max_relative)) {
                fprintf(stderr, "%s: mismatch at row %llu\n", names[stage],
                        (unsigned long long)row);
                failed = 1;
            }
        }
        printf("%s %s: max_abs=%.9g max_rel=%.9g atol=2e-4 rtol=1e-4 (%llu x %zu)\n",
               argv[1], names[stage], max_absolute, max_relative,
               (unsigned long long)ids->shape[1], width);
        if (failed) {
            goto done;
        }
    }
    rc = 0;
done:
    if (rc) {
        fprintf(stderr, "%s: projection comparison failed (schema, read, or numerical error)\n",
                argv[1]);
    }
    free(matrix);
    st_close(&fixture);
    for (size_t i = 0; i < shard_count; i++) {
        st_close(&shards[i]);
    }
    return rc;
}
