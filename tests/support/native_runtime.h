#ifndef LUNELETTE_TEST_SUPPORT_NATIVE_RUNTIME_H
#define LUNELETTE_TEST_SUPPORT_NATIVE_RUNTIME_H

/* Single-translation-unit runtime stub implementation. Included rather than linked
 * separately so tools-native can rename create_configured and capture its owner.
 * Mutable counters, logits and private runtime layout are deliberate test seams.
 * No scenario assertions or runner belong here. */

#include "native.h"
#include "metal.h"
#include "tokenizer_json.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

/* CPU-only adapter contract test. Link-time runtime stubs, never production.
 * argv[1] is a tokenizer, NOT weights; create does not open a checkpoint. */
struct lunelette_metal {
    int64_t ids[4352];
    unsigned count, capacity, block;
    lunelette_metal_checkpoint *active;
    lunelette_metal_strategy strategy;
};
static unsigned creates, destroys, prefills, decodes, resets;
static int fail, fail_memory, wrong_strategy;
static lunelette_metal_prefill_mode expected_prefill = LUNELETTE_METAL_PREFILL_CUSTOM;
static float logits[131072];
lunelette_metal *lunelette_metal_create_with_options(const char *path,
    const lunelette_metal_options *options, char *error, size_t capacity)
{
    (void)path;
    (void)error;
    (void)capacity;
    CHECK(options->prefill == expected_prefill);
    creates++;
    lunelette_metal *runtime = calloc(1, sizeof(lunelette_metal));
    CHECK(runtime);
    runtime->strategy = wrong_strategy ? LUNELETTE_METAL_RESIDENT_FP32 : options->strategy;
    return runtime;
}
lunelette_metal *lunelette_metal_create_configured(const char *path,
    const lunelette_metal_options *options, const lunelette_metal_limits *limits,
    char *error, size_t capacity)
{
    CHECK(limits && (limits->context == 256 || limits->context == 4352));
    CHECK(limits->block == 256 || (limits->context == 4352 && limits->block == 128));
    lunelette_metal *runtime = lunelette_metal_create_with_options(path, options, error, capacity);
    runtime->capacity = (unsigned)limits->context;
    runtime->block = (unsigned)limits->block;
    return runtime;
}
int lunelette_metal_get_memory(const lunelette_metal *runtime, lunelette_metal_memory *memory)
{
    CHECK(runtime && memory);
    if (fail_memory) {
        return -1;
    }
    *memory = (lunelette_metal_memory){runtime->strategy,
        runtime->strategy ? 4637454336ULL : 13716025344ULL,
        runtime->strategy ? 113246208ULL : 0, 87819264, 54525952,
        runtime->strategy ? 4893045760ULL : 13858370560ULL};
    return 0;
}
void lunelette_metal_destroy(lunelette_metal *runtime)
{
    destroys++;
    free(runtime);
}
int lunelette_metal_reset(lunelette_metal *runtime, char *error, size_t capacity)
{
    (void)error;
    (void)capacity;
    resets++;
    runtime->count = 0;
    runtime->active = NULL;
    return 0;
}
static int result(const float **out, char *error, size_t capacity)
{
    if (fail) {
        *out = NULL;
        snprintf(error, capacity, "deliberate test failure");
        return -1;
    }
    memset(logits, 0, sizeof(logits));
    /* Equal maxima must choose the lower ID using maintained policy. */
    logits[22177] = logits[22178] = 1;
    *out = logits;
    return 0;
}
int lunelette_metal_prefill(lunelette_metal *runtime, const int64_t *ids, size_t count,
                           const float **out, char *error, size_t capacity)
{
    CHECK(!runtime->count && count && count <= runtime->block);
    memcpy(runtime->ids, ids, count * sizeof(*ids));
    runtime->count = (unsigned)count;
    prefills++;
    int status = result(out, error, capacity);
    if (status) {
        runtime->active = NULL;
    }
    return status;
}
int lunelette_metal_decode(lunelette_metal *runtime, int64_t token,
                          const float **out, char *error, size_t capacity)
{
    CHECK(runtime->count && runtime->count < runtime->capacity);
    runtime->ids[runtime->count++] = token;
    decodes++;
    int status = result(out, error, capacity);
    if (status) {
        runtime->active = NULL;
    }
    return status;
}
struct lunelette_metal_checkpoint {
    lunelette_metal *owner;
    unsigned count;
    int64_t ids[4352];
    float *saved;
};
static unsigned appends, snapshots, restores;
static int fail_snapshot;
int lunelette_metal_append(lunelette_metal *runtime, const int64_t *ids, size_t count,
                          const float **out, char *error, size_t capacity)
{
    CHECK(runtime->count && count && runtime->count + count <= runtime->capacity && count <= runtime->block);
    memcpy(runtime->ids + runtime->count, ids, count * sizeof(*ids));
    runtime->count += (unsigned)count;
    appends++;
    int status = result(out, error, capacity);
    if (status) {
        runtime->active = NULL;
    }
    return status;
}
int lunelette_metal_checkpoint_create(lunelette_metal *runtime,
    lunelette_metal_checkpoint **out, char *error, size_t capacity)
{
    (void)error;
    (void)capacity;
    *out = NULL;
    if (fail_snapshot) {
        return -1;
    }
    CHECK(runtime->count);
    lunelette_metal_checkpoint *c = calloc(1, sizeof(*c));
    CHECK(c);
    c->saved = malloc(sizeof(logits));
    CHECK(c->saved);
    memcpy(c->saved, logits, sizeof(logits));
    memcpy(c->ids, runtime->ids, sizeof(c->ids));
    c->owner = runtime;
    c->count = runtime->count;
    runtime->active = c;
    *out = c;
    snapshots++;
    return 0;
}
int lunelette_metal_checkpoint_restore(lunelette_metal *runtime,
    const lunelette_metal_checkpoint *c, const float **out, char *error, size_t capacity)
{
    (void)error;
    (void)capacity;
    *out = NULL;
    if (!c || c->owner != runtime || runtime->active != c ||
        runtime->count < c->count ||
        memcmp(runtime->ids, c->ids, c->count * sizeof(*c->ids))) {
        snprintf(error, capacity, "inactive or incompatible checkpoint");
        return -1;
    }
    runtime->count = c->count;
    memcpy(logits, c->saved, sizeof(logits));
    *out = logits;
    restores++;
    return 0;
}
void lunelette_metal_checkpoint_destroy(lunelette_metal_checkpoint *c)
{
    if (c) {
        if (c->owner->active == c) {
            c->owner->active = NULL;
        }
        free(c->saved);
        free(c);
    }
}


#endif
