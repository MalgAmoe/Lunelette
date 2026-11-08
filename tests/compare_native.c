/* Native regression reference, NOT an independent mathematical oracle.
 * Versioned LE32 format: magic, strategy (LUN2 only), mode, context, prompt count,
 * steps, vocabulary,
 * prompt IDs, eight teacher-forced IDs, then eight full FP32 logit vectors.
 * No production instrumentation. Offline build needs only the C library. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "comparator_strategy.h"
static lunelette_metal_strategy selected_strategy = LUNELETTE_METAL_RESIDENT_FP32;
static int legacy_capture; /* Omitted selector preserves archived LUN1 capture callers. */
#ifndef NATIVE_OFFLINE
#include "metal.h"
#include "safetensors.h"
#endif
#define VOCAB 131072
#define STEPS 8
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static uint32_t read_u32(FILE *file, int *bad)
{
    unsigned char b[4];
    if (fread(b, 1, 4, file) != 4) {
        *bad = 1;
        return 0;
    }
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

static int write_u32(FILE *file, uint32_t value)
{
    unsigned char b[4];
    for (int i = 0; i < 4; i++) {
        b[i] = (unsigned char)(value >> (8 * i));
    }
    return fwrite(b, 1, 4, file) != 4;
}

static void top64(const float *values, size_t count, size_t *ids)
{
    for (size_t k = 0; k < 64; k++) {
        ids[k] = count;
    }
    for (size_t i = 0; i < count; i++) {
        for (size_t k = 0; k < 64; k++) {
            if (ids[k] == count || values[i] > values[ids[k]] ||
                (values[i] == values[ids[k]] && i < ids[k])) {
                memmove(ids + k + 1, ids + k, (63 - k) * sizeof *ids);
                ids[k] = i;
                break;
            }
        }
    }
}

static size_t intersection(const size_t *a, const size_t *b, size_t k)
{
    size_t result = 0;
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < k; j++) {
            result += a[i] == b[j];
        }
    }
    return result;
}

static int compare(const float *a, const float *b, size_t count, int report)
{
    size_t finite_a = 0, finite_b = 0;
    double maximum = 0, square = 0, aa = 0, bb = 0, ab = 0;
    for (size_t i = 0; i < count; i++) {
        finite_a += !!isfinite(a[i]);
        finite_b += !!isfinite(b[i]);
        double d = (double)a[i] - b[i];
        maximum = fmax(maximum, fabs(d));
        square += d * d;
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
        ab += (double)a[i] * b[i];
    }
    size_t at[64], bt[64];
    top64(a, count, at);
    top64(b, count, bt);
    int exact = memcmp(a, b, count * sizeof(float)) == 0;
    if (report) {
        printf("finite=%zu/%zu reference_finite=%zu/%zu exact=%d top1=%zu/%zu top5=%zu top20=%zu top64=%zu maxabs=%.17g RMS=%.17g cosine=%.17g\n",
            finite_a, count, finite_b, count, exact, at[0], bt[0],
            intersection(at, bt, 5), intersection(at, bt, 20), intersection(at, bt, 64),
            maximum, sqrt(square / count), aa && bb ? ab / sqrt(aa * bb) : (exact ? 1.0 : 0.0));
    }
    return !(finite_a == count && finite_b == count && exact &&
        at[0] == bt[0] && intersection(at, bt, 5) == 5 &&
        intersection(at, bt, 20) == 20 && intersection(at, bt, 64) == 64);
}

static int header(FILE *file, int mode, size_t count, const int64_t *ids,
    const int64_t *forced, int writing)
{
    int bad = 0;
    uint32_t magic = writing ? (legacy_capture ? 0x314e554c : 0x324e554c) :
        read_u32(file, &bad);
    if (magic != 0x314e554c && magic != 0x324e554c) { return 1; }
    if (writing && write_u32(file, magic)) { return 1; }
    uint32_t strategy = magic == 0x324e554c ?
        (writing ? (uint32_t)selected_strategy : read_u32(file, &bad)) : 0;
    if (strategy != (uint32_t)selected_strategy || strategy > 2) {
        fputs("Native artifact strategy mismatch or invalid strategy\n", stderr);
        return 1;
    }
    if (writing && magic == 0x324e554c && write_u32(file, strategy)) { return 1; }
    uint32_t fixed[] = { (uint32_t)mode, 256, (uint32_t)count, STEPS, VOCAB};
    for (size_t i = 0; i < 5; i++) {
        if (writing) {
            bad |= write_u32(file, fixed[i]);
        } else if (read_u32(file, &bad) != fixed[i]) {
            bad = 1;
        }
    }
    for (size_t i = 0; i < count + STEPS; i++) {
        uint32_t value = (uint32_t)(i < count ? ids[i] : forced[i - count]);
        if (value >= VOCAB) {
            bad = 1;
        }
        if (writing) {
            bad |= write_u32(file, value);
        } else if (read_u32(file, &bad) != value) {
            bad = 1;
        }
    }
    return bad;
}

static int vector(FILE *file, float *values, int writing)
{
    int bad = 0;
    for (size_t i = 0; i < VOCAB; i++) {
        uint32_t bits;
        if (writing) {
            memcpy(&bits, values + i, 4);
            bad |= write_u32(file, bits);
        } else {
            bits = read_u32(file, &bad);
            memcpy(values + i, &bits, 4);
        }
        bad |= !isfinite(values[i]);
    }
    return bad;
}

/* Offline actual-file validator: validates structure and every vector, then exact parity.
 * Used for deliberate artifact faults without constructing a model. */
static int files(const char *left, const char *right, int mode, size_t count)
{
    FILE *a = fopen(left, "rb"), *b = fopen(right, "rb");
    int status = 1, bad = 0;
    float *av = malloc(VOCAB * 4), *bv = malloc(VOCAB * 4);
    int64_t ids[256], forced[STEPS];
    if (!a || !b || !av || !bv || count > 248 || !count) {
        goto cleanup;
    }
    /* Read trajectory from left, then validate both complete headers against it. */
    uint32_t magic = read_u32(a, &bad);
    if (bad || (magic != 0x314e554c && magic != 0x324e554c) ||
        fseek(a, magic == 0x324e554c ? 28 : 24, SEEK_SET)) {
        goto cleanup;
    }
    for (size_t i = 0; i < count + STEPS; i++) {
        uint32_t token = read_u32(a, &bad);
        if (i < count) {
            ids[i] = token;
        } else {
            forced[i - count] = token;
        }
    }
    rewind(a);
    if (bad || header(a, mode, count, ids, forced, 0) ||
        header(b, mode, count, ids, forced, 0)) {
        goto cleanup;
    }
    for (int step = 0; step < STEPS; step++) {
        if (vector(a, av, 0) || vector(b, bv, 0) || compare(av, bv, VOCAB, 1)) {
            goto cleanup;
        }
    }
    status = fgetc(a) != EOF || fgetc(b) != EOF || ferror(a) || ferror(b);
cleanup:
    if (a) { fclose(a); }
    if (b) { fclose(b); }
    free(av);
    free(bv);
    return status;
}

static int self_test(void)
{
    legacy_capture = 0;
    selected_strategy = LUNELETTE_METAL_RESIDENT_FP32;
    char *legacy[] = {"tool", "check"};
    int arguments = 2;
    CHECK(!comparator_strategy_args(&arguments, legacy, &selected_strategy));
    CHECK(arguments == 2 && selected_strategy == LUNELETTE_METAL_RESIDENT_FP32);
    for (int strategy = 0; strategy < 3; strategy++) {
        char selector[128];
        snprintf(selector, sizeof selector, "--strategy=%s",
            comparator_strategy_name((lunelette_metal_strategy)strategy));
        char *args[] = {"tool", selector};
        arguments = 2;
        CHECK(!comparator_strategy_args(&arguments, args, &selected_strategy));
        CHECK(arguments == 1 && (int)selected_strategy == strategy);
        int64_t token = 1, forced_ids[STEPS] = {0};
        FILE *artifact = tmpfile();
        CHECK(artifact);
        CHECK(!header(artifact, 0, 1, &token, forced_ids, 1));
        rewind(artifact);
        CHECK(!header(artifact, 0, 1, &token, forced_ids, 0));
        selected_strategy = (lunelette_metal_strategy)((strategy + 1) % 3);
        rewind(artifact);
        CHECK(header(artifact, 0, 1, &token, forced_ids, 0));
        CHECK(!fseek(artifact, 4, SEEK_SET));
        CHECK(!write_u32(artifact, 3));
        rewind(artifact);
        CHECK(header(artifact, 0, 1, &token, forced_ids, 0));
        CHECK(!fclose(artifact));
    }
    char *invalid[] = {"tool", "--strategy=invalid"};
    arguments = 2;
    CHECK(comparator_strategy_args(&arguments, invalid, &selected_strategy));
    /* Original LUN1 layout, no strategy field: resident only. */
    FILE *old = tmpfile();
    CHECK(old);
    const uint32_t old_header[] = {0x314e554c, 0, 256, 1, STEPS, VOCAB,
        1, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < sizeof old_header / sizeof *old_header; i++) {
        CHECK(!write_u32(old, old_header[i]));
    }
    int64_t old_token = 1, old_forced[STEPS] = {0};
    for (int strategy = 0; strategy < 3; strategy++) {
        selected_strategy = (lunelette_metal_strategy)strategy;
        rewind(old);
        CHECK(!!header(old, 0, 1, &old_token, old_forced, 0) == (strategy != 0));
    }
    CHECK(!fclose(old));
    selected_strategy = LUNELETTE_METAL_RESIDENT_FP32;
    legacy_capture = 1;
    old = tmpfile();
    CHECK(old);
    CHECK(!header(old, 0, 1, &old_token, old_forced, 1));
    CHECK(ftell(old) == (long)sizeof old_header);
    rewind(old);
    for (size_t i = 0; i < sizeof old_header / sizeof *old_header; i++) {
        int bad = 0;
        CHECK(read_u32(old, &bad) == old_header[i] && !bad);
    }
    CHECK(!fclose(old));
    legacy_capture = 0;
    selected_strategy = LUNELETTE_METAL_RESIDENT_FP32;
    float a[128], b[128];
    for (int i = 0; i < 128; i++) {
        a[i] = b[i] = (float)(127 - i);
    }
    CHECK(!compare(a, b, 128, 0));
    b[127] += 0.25f; /* Same argmax and same top64: still must fail. */
    CHECK(compare(a, b, 128, 0));
    b[127] = a[127];
    b[0] = NAN;
    CHECK(compare(a, b, 128, 0));
    b[0] = INFINITY;
    CHECK(compare(a, b, 128, 0));
    b[0] = a[1];
    b[1] = a[0];
    size_t at[64], bt[64];
    top64(a, 128, at);
    top64(b, 128, bt);
    CHECK(at[0] == 0 && bt[0] == 1 && intersection(at, bt, 5) == 5);
    CHECK(compare(a, b, 128, 0));
    b[0] = a[0];
    b[1] = a[1];
    b[64] = 200;
    top64(b, 128, bt);
    CHECK(intersection(at, bt, 5) == 4 && intersection(at, bt, 20) == 19 &&
        intersection(at, bt, 64) == 63);
    /* Independent double rank oracle, including ties. */
    for (int i = 0; i < 128; i++) {
        b[i] = (float)((i * 17) % 13);
    }
    top64(b, 128, bt);
    for (size_t i = 0; i < 128; i++) {
        size_t rank = 0;
        for (size_t j = 0; j < 128; j++) {
            rank += (double)b[j] > (double)b[i] || (b[j] == b[i] && j < i);
        }
        CHECK(rank >= 64 || bt[rank] == i);
    }
    /* Always-active format tests, also run under sanitizers and NDEBUG. */
    int64_t prompt[6] = {1, 2, 3, 4, 5, 6};
    int64_t forced[STEPS] = {7, 8, 9, 10, 11, 12, 13, 14};
    FILE *temporary = tmpfile();
    CHECK(temporary);
    CHECK(!header(temporary, 0, 6, prompt, forced, 1));
    rewind(temporary);
    CHECK(!header(temporary, 0, 6, prompt, forced, 0));
    rewind(temporary);
    CHECK(header(temporary, 1, 6, prompt, forced, 0));
    prompt[0]++;
    rewind(temporary);
    CHECK(header(temporary, 0, 6, prompt, forced, 0));
    prompt[0]--;
    /* Corrupt context, step count, shape, and format magic separately. */
    const long offsets[] = {0, 12, 20, 24};
    const uint32_t originals[] = {0x324e554c, 256, STEPS, VOCAB};
    for (size_t i = 0; i < 4; i++) {
        CHECK(!fseek(temporary, offsets[i], SEEK_SET));
        CHECK(!write_u32(temporary, originals[i] + 1));
        rewind(temporary);
        CHECK(header(temporary, 0, 6, prompt, forced, 0));
        CHECK(!fseek(temporary, offsets[i], SEEK_SET));
        CHECK(!write_u32(temporary, originals[i]));
    }
    CHECK(!fclose(temporary));
    float *full = calloc(VOCAB, sizeof(float));
    CHECK(full);
    temporary = tmpfile();
    CHECK(temporary);
    CHECK(vector(temporary, full, 0)); /* Missing step / truncated vector. */
    rewind(temporary);
    CHECK(!vector(temporary, full, 1));
    rewind(temporary);
    CHECK(!vector(temporary, full, 0));
    full[100] = NAN;
    rewind(temporary);
    CHECK(vector(temporary, full, 1));
    rewind(temporary);
    CHECK(vector(temporary, full, 0));
    full[100] = INFINITY;
    rewind(temporary);
    CHECK(vector(temporary, full, 1));
    rewind(temporary);
    CHECK(vector(temporary, full, 0));
    CHECK(!fclose(temporary));
    free(full);
    puts("PASS native metrics/format: exact, nonfinite, same-argmax tail, swaps, double rank/tie oracle, mode/token/context/steps/shape/magic/truncation");
    return 0;
}

#ifndef NATIVE_OFFLINE
static int model(int argc, char **argv)
{
    /* capture|check MODEL MODE DIRECTORY IDS FIXTURE (four pairs) */
    CHECK(argc == 13);
    int writing = strcmp(argv[1], "capture") == 0;
    CHECK(writing || strcmp(argv[1], "check") == 0);
    int mode = strcmp(argv[3], "custom") == 0;
    CHECK(mode || strcmp(argv[3], "mps") == 0);
    int64_t ids[4][256], forced[4][STEPS];
    const size_t counts[] = {6, 24, 150, 248};
    char paths[4][4096], error[512] = "invalid native fixture";
    FILE *outputs[4] = {0};
    lunelette_metal *runtime = NULL;
    float *expected = malloc(VOCAB * 4);
    int status = 1;
    if (!expected) { goto cleanup; }
    for (int c = 0; c < 4; c++) {
        FILE *input = fopen(argv[5 + c * 2], "r");
        if (!input) { goto cleanup; }
        int bad = 0;
        for (size_t i = 0; i < counts[c]; i++) {
            long long token;
            if (fscanf(input, "%lld", &token) != 1 || token < 0 || token >= VOCAB) {
                bad = 1;
                break;
            }
            ids[c][i] = token;
        }
        char extra[2];
        bad |= fscanf(input, "%1s", extra) != EOF || ferror(input);
        fclose(input);
        if (bad) { goto cleanup; }
        st_file fixture = {0};
        if (st_open(&fixture, argv[6 + c * 2], error, sizeof error)) { goto cleanup; }
        const st_tensor *generated = st_find(&fixture, "generated");
        unsigned char bytes[64];
        bad = !generated || generated->dtype != ST_I64 || generated->rank != 1 ||
            generated->shape[0] != STEPS;
        if (!bad) { bad = st_read(&fixture, generated, 0, bytes, sizeof bytes); }
        if (!bad) {
            for (int i = 0; i < STEPS; i++) {
                uint64_t token = 0;
                for (int j = 0; j < 8; j++) { token |= (uint64_t)bytes[i * 8 + j] << (j * 8); }
                if (token >= VOCAB) { bad = 1; }
                forced[c][i] = (int64_t)token;
            }
        }
        st_close(&fixture);
        if (bad) { goto cleanup; }
        int length = snprintf(paths[c], sizeof paths[c], "%s/prefill%zu.bin", argv[4], counts[c]);
        if (length < 0 || (size_t)length >= sizeof paths[c]) { goto cleanup; }
        if (!writing && files(paths[c], paths[c], mode, counts[c])) { goto cleanup; }
        outputs[c] = fopen(paths[c], writing ? "wbx" : "rb");
        if (!outputs[c] || header(outputs[c], mode, counts[c], ids[c], forced[c], writing)) {
            goto cleanup;
        }
    }
    lunelette_metal_options options = {
        LUNELETTE_METAL_PREFILL_MPS, LUNELETTE_METAL_RESIDENT_FP32
    };
    options.strategy = selected_strategy;
    options.prefill = mode ? LUNELETTE_METAL_PREFILL_CUSTOM : LUNELETTE_METAL_PREFILL_MPS;
    runtime = lunelette_metal_create_with_options(argv[2], &options, error, sizeof error);
    if (!runtime || comparator_strategy_actual(runtime, selected_strategy)) { goto cleanup; }
    size_t compared = 0;
    for (int cycle = 0; cycle < (writing ? 1 : 2); cycle++) {
        for (int c = 0; c < 4; c++) {
            if (!writing) {
                rewind(outputs[c]);
                if (header(outputs[c], mode, counts[c], ids[c], forced[c], 0)) { goto cleanup; }
            }
            if (lunelette_metal_reset(runtime, error, sizeof error)) { goto cleanup; }
            for (int step = 0; step < STEPS; step++) {
                const float *logits = NULL;
                int result = step == 0 ? lunelette_metal_prefill(runtime, ids[c], counts[c],
                    &logits, error, sizeof error) : lunelette_metal_decode(runtime,
                    forced[c][step - 1], &logits, error, sizeof error);
                if (result) { goto cleanup; }
                printf("%s mode=%s cycle=%d prefix=%zu step=%d ", argv[1], argv[3], cycle, counts[c], step);
                if (writing) {
                    memcpy(expected, logits, VOCAB * 4);
                    if (compare(logits, expected, VOCAB, 1) || vector(outputs[c], expected, 1)) {
                        goto cleanup;
                    }
                } else if (vector(outputs[c], expected, 0) || compare(logits, expected, VOCAB, 1)) {
                    goto cleanup;
                }
                compared++;
                fflush(stdout);
            }
        }
    }
    printf("PASS native %s steps=%zu cycles=%d\n", argv[1], compared, writing ? 1 : 2);
    status = 0;
cleanup:
    if (status) { fprintf(stderr, "FAIL native: %s\n", error); }
    lunelette_metal_destroy(runtime);
    for (int c = 0; c < 4; c++) {
        if (outputs[c] && fclose(outputs[c])) { status = 1; }
    }
    free(expected);
    return status;
}
#endif

int main(int argc, char **argv)
{
    int original_argc = argc;
    if (comparator_strategy_args(&argc, argv, &selected_strategy)) { return 1; }
    legacy_capture = argc == original_argc;
    if (argc == 2 && !strcmp(argv[1], "--self-test")) { return self_test(); }
    if (argc == 6 && !strcmp(argv[1], "--files")) {
        int mode = !strcmp(argv[4], "custom");
        CHECK(mode || !strcmp(argv[4], "mps"));
        size_t count = 0;
        if (!strcmp(argv[5], "6")) { count = 6; }
        if (!strcmp(argv[5], "24")) { count = 24; }
        if (!strcmp(argv[5], "150")) { count = 150; }
        if (!strcmp(argv[5], "248")) { count = 248; }
        CHECK(count);
        return files(argv[2], argv[3], mode, count);
    }
#ifndef NATIVE_OFFLINE
    if (argc > 1 && (!strcmp(argv[1], "capture") || !strcmp(argv[1], "check"))) {
        return model(argc, argv);
    }
#endif
    fputs("usage: compare_native --self-test | --files REF ACTUAL mps|custom 6|24|150|248 [--strategy=NAME]\n"
        "or capture|check MODEL mps|custom DIRECTORY IDS FIXTURE (four pairs: 6/24/150/248) [--strategy=NAME]\n", stderr);
    return 1;
}
