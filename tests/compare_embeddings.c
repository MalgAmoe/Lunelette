#include "safetensors.h"

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

static uint64_t decode_le(const unsigned char *p, size_t n)
{
    uint64_t value = 0;
    for (size_t i = 0; i < n; i++)
        value |= (uint64_t)p[i] << (8 * i);
    return value;
}

/* Keep schema errors next to the file and tensor that need correction. */
static const st_tensor *require_tensor(const st_file *file, const char *path,
                                     const char *name, st_dtype dtype, size_t rank)
{
    const st_tensor *tensor = st_find(file, name);
    if (!tensor) {
        fprintf(stderr, "%s: missing tensor '%s'\n", path, name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->rank != rank) {
        fprintf(stderr, "%s: '%s': expected dtype %d, rank %zu; got dtype %d, rank %zu\n",
                path, name, (int)dtype, rank, (int)tensor->dtype, tensor->rank);
        return NULL;
    }
    return tensor;
}

int main(int argc, char **argv)
{
    st_file weights = {0}, fixture = {0};
    char error[256];
    int rc = 1;
    float *row = NULL;
    unsigned char *expected = NULL;
    if (argc < 3) {
        fprintf(stderr, "Usage: %s WEIGHT_SHARD FIXTURE...\n", argv[0]);
        return 1;
    }
    if (st_open(&weights, argv[1], error, sizeof error)) {
        fprintf(stderr, "%s: %s\n", argv[1], error);
        goto done;
    }
    const st_tensor *w = require_tensor(&weights, argv[1],
        "language_model.model.embed_tokens.weight", ST_BF16, 2);
    if (!w)
        goto done;
    if (!w->shape[1] || w->shape[1] > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "%s: embedding width %" PRIu64 " is zero or exceeds allocation limit\n",
                argv[1], w->shape[1]);
        goto done;
    }
    size_t width = (size_t)w->shape[1];
    row = malloc(width * sizeof *row);
    expected = malloc(width * 4);
    if (!row || !expected) {
        fprintf(stderr, "%s: cannot allocate two embedding rows of %zu bytes\n", argv[1], width * 4);
        goto done;
    }
    for (int f = 2; f < argc; f++) {
        const char *path = argv[f];
        if (st_open(&fixture, path, error, sizeof error)) {
            fprintf(stderr, "%s: %s\n", path, error);
            goto done;
        }
        const st_tensor *ids = require_tensor(&fixture, path, "input_ids", ST_I64, 2);
        const st_tensor *h = require_tensor(&fixture, path, "hidden_0", ST_F32, 3);
        if (!ids || !h)
            goto done;
        if (ids->shape[0] != 1 || h->shape[0] != 1 ||
            h->shape[1] != ids->shape[1] || h->shape[2] != width) {
            fprintf(stderr, "%s: expected input_ids [1,N], hidden_0 [1,N,%zu]; got "
                    "[%" PRIu64 ",%" PRIu64 "], [%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]\n",
                    path, width, ids->shape[0], ids->shape[1],
                    h->shape[0], h->shape[1], h->shape[2]);
            goto done;
        }
        for (uint64_t i = 0; i < ids->shape[1]; i++) {
            unsigned char raw[8];
            if (st_read(&fixture, ids, i * 8, raw, 8)) {
                fprintf(stderr, "%s: cannot read input_ids at position %" PRIu64 "\n", path, i);
                goto done;
            }
            uint64_t token = decode_le(raw, 8);
            if (token > INT64_MAX || token >= w->shape[0]) {
                fprintf(stderr, "%s: input_ids position %" PRIu64 ": token %" PRIu64
                        " outside [0,%" PRIu64 ") in %s\n", path, i, token, w->shape[0], argv[1]);
                goto done;
            }
            if (st_embedding(&weights, w, (int64_t)token, row, width)) {
                fprintf(stderr, "%s: embedding row %" PRIu64
                        " allocation/read failed (fixture %s, position %" PRIu64 ")\n",
                        argv[1], token, path, i);
                goto done;
            }
            if (st_read(&fixture, h, i * width * 4, expected, width * 4)) {
                fprintf(stderr, "%s: cannot read hidden_0 row %" PRIu64 "\n", path, i);
                goto done;
            }
            for (size_t k = 0; k < width; k++) {
                uint32_t actual;
                memcpy(&actual, &row[k], 4);
                uint32_t reference = (uint32_t)decode_le(expected + k * 4, 4);
                if (actual != reference) {
                    fprintf(stderr, "%s: mismatch token position %" PRIu64
                            " column %zu: expected %08" PRIx32 ", got %08" PRIx32 " (weights %s)\n",
                            path, i, k, reference, actual, argv[1]);
                    goto done;
                }
            }
        }
        printf("%s: exact hidden_0 match (%" PRIu64 " rows x %zu)\n", path, ids->shape[1], width);
        st_close(&fixture);
    }
    rc = 0;
done:
    free(row);
    free(expected);
    st_close(&fixture);
    st_close(&weights);
    return rc;
}
