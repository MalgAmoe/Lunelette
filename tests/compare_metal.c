/* Optional real-weight validation. No tokenizer/Python/model paths embedded.
 * IDs: whitespace-separated decimal prompt IDs. Fixture: generated I64 [N],
 * logits_0..N-1 F32 [1,131072], captured by an independent reference.
 */
#include "metal.h"
#include "model_facts.h"
#include "runtime_limits.h"
#include "safetensors.h"
#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static uint64_t little_u64(const unsigned char *bytes)
{
    uint64_t value = 0;
    for (size_t i = 0; i < 8; i++) {
        value |= (uint64_t)bytes[i] << (8 * i);
    }
    return value;
}

static float little_float(const unsigned char *bytes)
{
    uint32_t bits = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
        (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static double seconds(void)
{
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec + time.tv_nsec * 1e-9;
}

static int compare(const float *actual, const unsigned char *reference,
    size_t count, double *maximum)
{
    *maximum = 0;
    for (size_t i = 0; i < count; i++) {
        float expected = little_float(reference + i * 4);
        double difference = fabs((double)actual[i] - expected);
        if (!isfinite(actual[i]) || !isfinite(expected) ||
            difference > 2e-4 + 1e-4 * fabs(expected)) {
            return -1;
        }
        if (difference > *maximum) {
            *maximum = difference;
        }
    }
    return 0;
}

/* A numerical fixture must contain at least one oracle comparison. */
static int valid_generated(const st_tensor *generated, size_t prompt_count)
{
    return generated && generated->dtype == ST_I64 && generated->rank == 1 &&
        generated->shape[0] > 0 && prompt_count <= LUNELETTE_CONTEXT_LIMIT &&
        generated->shape[0] <= LUNELETTE_CONTEXT_LIMIT - prompt_count;
}

static int self_test(void)
{
    const unsigned char one[4] = {0, 0, 128, 63};
    float actual = 1;
    double maximum;
    if (compare(&actual, one, 1, &maximum) != 0) {
        return 1;
    }
    actual = 2;
    if (compare(&actual, one, 1, &maximum) == 0) {
        return 1;
    }
    actual = NAN;
    if (compare(&actual, one, 1, &maximum) == 0) {
        return 1;
    }
    st_tensor generated = {0};
    generated.dtype = ST_I64;
    generated.rank = 1;
    if (valid_generated(&generated, 1) || valid_generated(NULL, 1)) {
        return 1;
    }
    generated.shape[0] = 1;
    if (!valid_generated(&generated, 255) || valid_generated(&generated, 256)) {
        return 1;
    }
    generated.rank = 2;
    if (valid_generated(&generated, 1)) {
        return 1;
    }
    generated.rank = 1;
    generated.dtype = ST_F32;
    if (valid_generated(&generated, 1)) {
        return 1;
    }
    puts("Metal comparator match/mismatch/nonfinite/empty/malformed tests passed");
    return 0;
}

static int run_case(lunelette_metal *runtime, const char *ids_path, const char *fixture_path,
    int cycle, size_t *steps)
{
    char error[512] = "Invalid Metal fixture";
    st_file fixture = {0};
    FILE *input = fopen(ids_path, "r");
    unsigned char *reference = NULL;
    int status = -1;
    int64_t ids[LUNELETTE_CONTEXT_LIMIT];
    size_t count = 0;
    if (!input) {
        snprintf(error, sizeof error, "Cannot open IDs: %s", ids_path);
        goto cleanup;
    }
    char text[64];
    int scanned;
    while ((scanned = fscanf(input, "%63s", text)) == 1) {
        char *end;
        errno = 0;
        long long token = strtoll(text, &end, 10);
        if (errno || *end || count == LUNELETTE_CONTEXT_LIMIT ||
            token < 0 || token >= LUNELETTE_MODEL_VOCAB) {
            goto cleanup;
        }
        ids[count++] = token;
    }
    if (scanned != EOF || ferror(input) || !count ||
        st_open(&fixture, fixture_path, error, sizeof error) != 0) {
        goto cleanup;
    }
    const st_tensor *generated = st_find(&fixture, "generated");
    if (!valid_generated(generated, count)) {
        snprintf(error, sizeof error,
            "Invalid generated fixture: require nonempty I64 [N] within context: %s",
            fixture_path);
        goto cleanup;
    }
    reference = malloc(LUNELETTE_MODEL_VOCAB * sizeof(float));
    if (!reference) {
        goto cleanup;
    }
    if (lunelette_metal_reset(runtime, error, sizeof error) != 0) {
        goto cleanup;
    }
    const float *logits = NULL;
    /* Rejections must not mutate the empty sequence; real execution follows. */
    int64_t invalid = -1;
    if (lunelette_metal_prefill(runtime, &invalid, 1, &logits, error, sizeof error) == 0 ||
        lunelette_metal_prefill(runtime, ids, 0, &logits, error, sizeof error) == 0 ||
        lunelette_metal_decode(runtime, 0, &logits, error, sizeof error) == 0) {
        goto cleanup;
    }
    int64_t previous = 0;
    for (size_t step = 0; step < generated->shape[0]; step++) {
        double start = seconds();
        int result = step == 0 ?
            lunelette_metal_prefill(runtime, ids, count, &logits, error, sizeof error) :
            lunelette_metal_decode(runtime, previous, &logits, error, sizeof error);
        double elapsed = seconds() - start;
        if (result != 0) {
            goto cleanup;
        }
        char name[64];
        snprintf(name, sizeof name, "logits_%zu", step);
        const st_tensor *tensor = st_find(&fixture, name);
        if (!tensor || tensor->dtype != ST_F32 || tensor->rank != 2 ||
            tensor->shape[0] != 1 || tensor->shape[1] != LUNELETTE_MODEL_VOCAB ||
            st_read(&fixture, tensor, 0, reference, LUNELETTE_MODEL_VOCAB * 4) != 0) {
            goto cleanup;
        }
        double maximum;
        if (compare(logits, reference, LUNELETTE_MODEL_VOCAB, &maximum) != 0) {
            snprintf(error, sizeof error, "Strict logits mismatch: %s cycle%d step%zu",
                fixture_path, cycle, step);
            goto cleanup;
        }
        size_t chosen = 0;
        for (size_t i = 1; i < LUNELETTE_MODEL_VOCAB; i++) {
            if (logits[i] > logits[chosen]) {
                chosen = i;
            }
        }
        unsigned char expected[8];
        if (st_read(&fixture, generated, step * 8, expected, 8) != 0 ||
            little_u64(expected) != chosen) {
            snprintf(error, sizeof error, "Greedy ID mismatch: %s step%zu", fixture_path, step);
            goto cleanup;
        }
        previous = (int64_t)chosen;
        printf("PASS %s cycle%d step%zu id%zu maxabs%.9g seconds%.6f\n",
            fixture_path, cycle, step, chosen, maximum, elapsed);
        fflush(stdout);
        (*steps)++;
        if (step == 0) {
            const float *rejected = NULL;
            if (lunelette_metal_decode(runtime, -1, &rejected, error, sizeof error) == 0 ||
                rejected != NULL || lunelette_metal_prefill(runtime, ids, count,
                    &rejected, error, sizeof error) == 0 || rejected != NULL) {
                goto cleanup;
            }
        }
    }
    if (count == LUNELETTE_CONTEXT_LIMIT - 1 && generated->shape[0] == 1) {
        /* Secondary state check, not an additional independent oracle logit step:
         * fill the final KV slot, reject overflow, then next case/cycle resets. */
        if (lunelette_metal_decode(runtime, previous, &logits, error, sizeof error) != 0 ||
            lunelette_metal_decode(runtime, previous, &logits, error, sizeof error) == 0 ||
            logits != NULL) {
            goto cleanup;
        }
        printf("PASS capacity256 append then reject %s cycle%d\n", fixture_path, cycle);
        fflush(stdout);
    }
    status = 0;
cleanup:
    if (status != 0) {
        fprintf(stderr, "%s\n", error);
    }
    free(reference);
    st_close(&fixture);
    if (input) {
        fclose(input);
    }
    return status;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    }
    if (argc < 5 || (argc - 3) % 2 != 0) {
        fputs("usage: compare_metal SHARD1 SHARD2 IDS_FILE FIXTURE [IDS_FILE FIXTURE ...]\n", stderr);
        return 1;
    }
    char error[512];
    double start = seconds();
    lunelette_metal *runtime = lunelette_metal_create(argv[1], argv[2], error, sizeof error);
    if (!runtime) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    printf("load_seconds %.6f\n", seconds() - start);
    fflush(stdout);
    int status = 0;
    size_t steps = 0;
    for (int cycle = 0; cycle < 2 && status == 0; cycle++) {
        for (int i = 3; i < argc; i += 2) {
            if (run_case(runtime, argv[i], argv[i + 1], cycle, &steps) != 0) {
                status = 1;
                break;
            }
        }
    }
    lunelette_metal_destroy(runtime);
    if (steps == 0) {
        fputs("No numerical steps compared\n", stderr);
        status = 1;
    }
    if (status == 0) {
        /* Recreate in the same process: exercises ARC owner/reader teardown,
         * not just the OS reclaiming allocations at process exit. */
        /* Driver/VM page reclamation can lag ARC/Metal buffer release. Preserve
         * the conservative guard; allow at most three one-second settling waits. */
        runtime = NULL;
        for (size_t attempt = 0; attempt < 3; attempt++) {
            sleep(1);
            runtime = lunelette_metal_create(argv[2], argv[1], error, sizeof error);
            if (runtime || !strstr(error, "Metal needs conservative")) {
                break;
            }
        }
        int64_t ids[] = {1, 42};
        const float *logits = NULL;
        if (!runtime || lunelette_metal_prefill(runtime, ids, 2, &logits,
                error, sizeof error) != 0) {
            fprintf(stderr, "Metal recreate failed: %s\n", error);
            status = 1;
        } else {
            puts("PASS destroy/recreate reversed shards and prefill (state check, not oracle)");
        }
        lunelette_metal_destroy(runtime);
    }
    printf("%s steps%zu cycles2\n", status ? "FAIL" : "PASS", steps);
    return status;
}
