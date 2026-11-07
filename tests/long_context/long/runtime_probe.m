/* LONG-only diagnostic interpreter. Private owner read-only binding is source-pinned. */
#include "metal_internal.h"
#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *api, *file;
    size_t start, count;
    int64_t ids[256];
} operation;
#include "b_schedule.h"

static int finite_vector(const float *v)
{
    if (!v) return 0;
    for (size_t i = 0; i < 131072; i++) {
        if (!isfinite(v[i])) return 0;
    }
    return 1;
}

/* Each of 52 hashes covers only the retained prefix, never stale tails. */
static int hashes(LuneletteMetalOwner *owner, size_t length, unsigned char out[52][32])
{
    if (!owner->valid || owner->length < length) return -1;
    size_t bytes = length * 1024 * sizeof(float);
    for (size_t layer = 0; layer < 26; layer++) {
        id<MTLBuffer> pair[] = {owner->keys[layer], owner->values[layer]};
        for (size_t k = 0; k < 2; k++) {
            if (!pair[k].contents || pair[k].length < bytes) return -1;
            CC_SHA256(pair[k].contents, (CC_LONG)bytes, out[layer * 2 + k]);
        }
    }
    return 0;
}

static int point(lunelette_metal *r, const char *label)
{
    lunelette_metal_memory m;
    struct task_vm_info info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (lunelette_metal_get_memory(r, &m) ||
        task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return -1;
    printf("RESOURCE %s footprint_point=%llu weights=%llu conversion=%llu scratch=%llu kv=%llu total=%llu\n",
        label, info.phys_footprint, (unsigned long long)m.weight_bytes,
        (unsigned long long)m.conversion_stage_bytes, (unsigned long long)m.scratch_bytes,
        (unsigned long long)m.kv_bytes, (unsigned long long)m.total_bytes);
    return 0;
}

/* Rejections execute real API validation. Check active-handle identity and restore usability. */
static int rejections(lunelette_metal *r, size_t block, const float **v, char *error)
{
    LuneletteMetalOwner *owner = (__bridge LuneletteMetalOwner *)r->owner;
    lunelette_metal_checkpoint *cp = NULL;
    unsigned char before[52][32], after[52][32];
    float *copy = malloc(524288);
    int64_t ids[4353];
    int result = -1;
    size_t length = owner->length;
    uint64_t epoch = owner->checkpoint_epoch;
    if (!copy || !finite_vector(*v)) goto cleanup;
    memcpy(copy, *v, 524288);
    for (size_t i = 0; i < 4353; i++) ids[i] = 1000 + (37 * i) % 8000;
    if (hashes(owner, length, before) ||
        lunelette_metal_checkpoint_create(r, &cp, error, 1024)) goto cleanup;
    epoch = owner->checkpoint_epoch;
    for (size_t test = 0; test < (length == 4352 ? 5U : 2U); test++) {
        const float *rejected = *v;
        size_t n = test == 0 ? 0 : test == 1 ? block + 1 : (test == 2 || test == 3) ? 1 : 4353;
        int status = test == 3 ? lunelette_metal_decode(r, 1000, &rejected, error, 1024) :
            lunelette_metal_append(r, ids, n, &rejected, error, 1024);
        printf("REJECT test=%zu count=%zu status=%d diagnostic=%s\n", test, n, status, error);
        if (!status || rejected || !owner->valid || !owner->endpoint_logits_valid ||
            owner->length != length || owner->active_checkpoint != cp ||
            owner->checkpoint_epoch != epoch || hashes(owner, length, after) ||
            memcmp(before, after, sizeof before) ||
            memcmp(copy, owner->buffers[METAL_LOGITS].contents, 524288)) goto cleanup;
        if (lunelette_metal_checkpoint_restore(r, cp, v, error, 1024) ||
            !finite_vector(*v) || memcmp(copy, *v, 524288) ||
            owner->length != length || hashes(owner, length, after) ||
            memcmp(before, after, sizeof before)) goto cleanup;
    }
    result = 0;
cleanup:
    lunelette_metal_checkpoint_destroy(cp);
    free(copy);
    return result;
}

static int probe_main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "--badargs")) {
        char error[1024] = {0};
        const float *v = (const float *)(uintptr_t)1;
        int64_t ids[4353] = {0};
        if (!lunelette_metal_append(NULL, ids, 0, &v, error, sizeof error) || v ||
            !lunelette_metal_append(NULL, ids, 257, &v, error, sizeof error) || v ||
            !lunelette_metal_prefill(NULL, ids, 4353, &v, error, sizeof error) || v ||
            !lunelette_metal_decode(NULL, 1000, &v, error, sizeof error) || v) return 3;
        lunelette_metal_limits invalid = {4353, 256};
        lunelette_metal *r = lunelette_metal_create_configured("/DO_NOT_OPEN", NULL,
            &invalid, error, sizeof error);
        if (r) {
            lunelette_metal_destroy(r);
            return 3;
        }
        puts("BADARGS null_runtime_rejected invalid_profile_rejected no_weights");
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "--planner")) {
        metal_allocation_plan a, b;
        if (metal_plan_create((lunelette_metal_limits){4352, 128}, &a) ||
            metal_plan_create((lunelette_metal_limits){4352, 256}, &b) ||
            !metal_plan_create((lunelette_metal_limits){4353, 256}, &a) ||
            a.kv_bytes != b.kv_bytes || a.scratch_bytes >= b.scratch_bytes) return 3;
        printf("PLANNER kv=%llu scratch128=%llu scratch256=%llu invalid4353=rejected\n",
            (unsigned long long)a.kv_bytes, (unsigned long long)a.scratch_bytes,
            (unsigned long long)b.scratch_bytes);
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "--dry-run")) {
        printf("DRYRUN block128=%zu block256=%zu no_runtime_owner_no_weights\n",
            sizeof ops128 / sizeof *ops128, sizeof ops256 / sizeof *ops256);
        return 0;
    }
    /* No weights path is inspected without the explicit future execution gate. */
    if (argc != 8 || strcmp(argv[1], "--execute-authorized-LONG") ||
        (strcmp(argv[3], "128") && strcmp(argv[3], "256")) ||
        strcmp(argv[4], "mps") ||
        (strcmp(argv[5], "resident-fp32") && strcmp(argv[5], "compact-fp8-arithmetic")) ||
        strcmp(argv[7], "--ack-model-load") ||
        (!strcmp(argv[3], "128") && strcmp(argv[5], "compact-fp8-arithmetic"))) {
        fprintf(stderr, "REFUSED: explicit LONG execution arguments required\n");
        return 2;
    }
    size_t block = (size_t)atoi(argv[3]);
    const operation *ops = block == 128 ? ops128 : ops256;
    size_t count = block == 128 ? sizeof ops128 / sizeof *ops128 : sizeof ops256 / sizeof *ops256;
    lunelette_metal_options options = { !strcmp(argv[4], "custom"), !strcmp(argv[5], "compact-fp8-arithmetic") };
    lunelette_metal_limits limits = {4352, block};
    char error[1024] = {0};
    lunelette_metal *r = NULL;
    lunelette_metal_checkpoint *cp = NULL;
    const float *v = NULL;
    unsigned char saved[52][32], current[52][32];
    float *saved_logits = malloc(524288);
    size_t prefix = 0;
    int result = 3;
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "STOP line=%d diagnostic=%s\n", __LINE__, error); goto cleanup; } } while (0)
    REQUIRE(saved_logits);
    puts("GUARD create_configured: production admission unmodified; refusal retained on stderr");
    r = lunelette_metal_create_configured(argv[2], &options, &limits, error, sizeof error);
    REQUIRE(r);
    REQUIRE(!point(r, "load-ready"));
    for (size_t i = 0; i < count; i++) {
        const operation *op = &ops[i];
        LuneletteMetalOwner *owner = (__bridge LuneletteMetalOwner *)r->owner;
        int status = 0;
        printf("CALL index=%zu api=%s start=%zu count=%zu ids=", i, op->api, op->start, op->count);
        for (size_t j = 0; j < op->count; j++) printf("%s%lld", j ? "," : "", (long long)op->ids[j]);
        puts("");
        if (!strcmp(op->api, "reset")) {
            lunelette_metal_checkpoint_destroy(cp); cp = NULL; prefix = 0;
            status = lunelette_metal_reset(r, error, sizeof error); v = NULL;
        } else if (!strcmp(op->api, "prefill") || !strcmp(op->api, "append")) {
            REQUIRE(owner->length == op->start);
            status = !strcmp(op->api, "prefill") ?
                lunelette_metal_prefill(r, op->ids, op->count, &v, error, sizeof error) :
                lunelette_metal_append(r, op->ids, op->count, &v, error, sizeof error);
            REQUIRE(!status && finite_vector(v));
            REQUIRE(owner->length == op->start + op->count);
            if (prefix) REQUIRE(!hashes(owner, prefix, current) && !memcmp(saved, current, sizeof saved));
        } else if (!strcmp(op->api, "save")) {
            REQUIRE(!cp && finite_vector(v));
            prefix = owner->length;
            memcpy(saved_logits, v, 524288);
            REQUIRE(!hashes(owner, prefix, saved));
            for (size_t k = 0; k < 52; k++) {
                printf("KV prefix=%zu buffer=%zu sha256=", prefix, k);
                for (size_t h = 0; h < 32; h++) printf("%02x", saved[k][h]);
                puts("");
            }
            status = lunelette_metal_checkpoint_create(r, &cp, error, sizeof error);
            REQUIRE(!point(r, "save"));
        } else if (!strcmp(op->api, "restore")) {
            status = lunelette_metal_checkpoint_restore(r, cp, &v, error, sizeof error);
            REQUIRE(!status && owner->length == prefix && finite_vector(v));
            REQUIRE(!memcmp(saved_logits, v, 524288));
            REQUIRE(!hashes(owner, prefix, current) && !memcmp(saved, current, sizeof saved));
            REQUIRE(!point(r, "restore"));
        } else if (!strcmp(op->api, "snapshot")) {
            REQUIRE(finite_vector(v));
            size_t retained = prefix ? prefix : owner->length;
            REQUIRE(!hashes(owner, retained, current));
            for (size_t k = 0; k < 52; k++) {
                printf("ENDKV file=%s prefix=%zu buffer=%zu sha256=", op->file, retained, k);
                for (size_t h = 0; h < 32; h++) printf("%02x", current[k][h]);
                puts("");
            }
            char path[4096];
            REQUIRE(snprintf(path, sizeof path, "%s/%s", argv[6], op->file) < (int)sizeof path);
            FILE *f = fopen(path, "wbx");
            REQUIRE(f);
            size_t wrote = fwrite(v, 1, 524288, f);
            int closed = fclose(f);
            REQUIRE(wrote == 524288 && !closed);
            REQUIRE(!point(r, "ingestion-endpoint"));
            /* Only fresh endpoints: replacement would invalidate the B2 handle. */
            if (!cp) REQUIRE(!rejections(r, block, &v, error));
        } else REQUIRE(0);
        printf("RESULT index=%zu status=%d length=%zu\n", i, status, owner->length);
        REQUIRE(!status);
        fflush(stdout);
    }
    result = 0;
cleanup:
    /* Preserve the full current endpoint even on nonfinite/restore mismatch stops. */
    if (result && v) {
        char failure_path[4096];
        int length = snprintf(failure_path, sizeof failure_path, "%s/failure-logits.f32", argv[6]);
        if (length > 0 && length < (int)sizeof failure_path) {
            FILE *failure = fopen(failure_path, "wbx");
            if (failure) {
                size_t wrote = fwrite(v, 1, 524288, failure);
                int closed = fclose(failure);
                fprintf(stderr, "FAILURE_VECTOR bytes=%zu closed=%d\n", wrote, closed);
            } else {
                fprintf(stderr, "FAILURE_VECTOR preservation failed\n");
            }
        }
    }
    lunelette_metal_checkpoint_destroy(cp);
    lunelette_metal_destroy(r);
    free(saved_logits);
    printf("CLEANUP checkpoint_destroyed runtime_destroyed host_copy_freed status=%d\n", result);
    return result;
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        return probe_main(argc, argv);
    }
}
