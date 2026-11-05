/* Optional macOS host-only exact decoded-weight check. No GPU/model execution.
 * CommonCrypto is the platform SHA256 implementation, not a runtime dependency. */
#include "metal_weights.h"
#include <CommonCrypto/CommonDigest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: compare_fp8 MODEL.safetensors DECODED-SHA256.txt\n");
        return 1;
    }
    lunelette_metal_weights weights;
    char error[512];
    if (lunelette_metal_weights_open(&weights, argv[1], error, sizeof error) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    FILE *hashes = fopen(argv[2], "r");
    int status = 1;
    size_t checked = 0;
    unsigned char seen[LUNELETTE_METAL_WEIGHT_COUNT] = {0};
    int scanned;
    float *expanded = NULL;
    if (!hashes) {
        goto cleanup;
    }
    char name[ST_MAX_NAME], expected[65];
    while ((scanned = fscanf(hashes, "%255s %64s", name, expected)) == 2) {
        size_t index;
        for (index = 0; index < LUNELETTE_METAL_WEIGHT_COUNT; index++) {
            if (strcmp(weights.weights[index].tensor->name, name) == 0) {
                break;
            }
        }
        if (index == LUNELETTE_METAL_WEIGHT_COUNT || seen[index]) {
            goto cleanup;
        }
        seen[index] = 1;
        const st_tensor *tensor = weights.weights[index].tensor;
        CC_SHA256_CTX digest;
        CC_SHA256_Init(&digest);
        if (tensor->dtype == ST_F8_E4M3) {
            size_t count = weights.weights[index].elements;
            expanded = malloc(count * sizeof(float));
            if (!expanded || lunelette_metal_weights_load(&weights, index, expanded, count)) {
                goto cleanup;
            }
            CC_SHA256_Update(&digest, expanded, (CC_LONG)(count * sizeof(float)));
            free(expanded);
            expanded = NULL;
        } else {
            /* Includes tied embedding/head: stream its 1.6 GB FP32 expansion. */
            unsigned char bytes[65536];
            float values[32768];
            for (uint64_t offset = 0; offset < tensor->end - tensor->begin;) {
                size_t count = (size_t)((tensor->end - tensor->begin - offset) / 2);
                if (count > 32768) {
                    count = 32768;
                }
                if (st_read(&weights.file, tensor, offset, bytes, count * 2)) {
                    goto cleanup;
                }
                for (size_t i = 0; i < count; i++) {
                    values[i] = st_bf16_to_f32((uint16_t)(bytes[2 * i] |
                        (unsigned)bytes[2 * i + 1] << 8));
                }
                CC_SHA256_Update(&digest, values, (CC_LONG)(count * sizeof(float)));
                offset += count * 2;
            }
        }
        unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
        char actual[65];
        CC_SHA256_Final(bytes, &digest);
        for (size_t i = 0; i < sizeof bytes; i++) {
            snprintf(actual + 2 * i, 3, "%02x", bytes[i]);
        }
        if (strcmp(actual, expected) != 0) {
            fprintf(stderr, "FAIL %s expected%s actual%s\n", name, expected, actual);
            goto cleanup;
        }
        checked++;
        printf("PASS %s %s\n", name, actual);
    }
    status = scanned != EOF || ferror(hashes) || checked != LUNELETTE_METAL_WEIGHT_COUNT;
    printf("%s exact decoded FP32 SHA256 tensors%zu bytes%llu\n", status ? "FAIL" : "PASS",
        checked, (unsigned long long)weights.bytes);
cleanup:
    free(expanded);
    if (hashes) {
        fclose(hashes);
    }
    lunelette_metal_weights_close(&weights);
    return status;
}
