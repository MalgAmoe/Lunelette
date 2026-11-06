#include "metal_internal.h"
#include "metal_shader.h"
#include <mach/mach.h>
#include <errno.h>
#include <time.h>
#include "metal_memory_guard.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

/* Saturating process-wide identity allocation: address reuse cannot revive handles.
 * init is used by production and synthetic owners alike. Zero means exhausted. */
static _Atomic uint64_t last_identity;

/* Only the test-specific runtime object overrides this allocation call. */
#ifndef LUNELETTE_METAL_CHECKPOINT_ALLOC
#define LUNELETTE_METAL_CHECKPOINT_ALLOC malloc
#else
extern void *LUNELETTE_METAL_CHECKPOINT_ALLOC(size_t);
#endif

struct lunelette_metal_checkpoint {
    void *owner; /* Borrowed; handles must be destroyed before the context. */
    uint64_t identity, epoch;
    size_t length;
    float logits[LUNELETTE_MODEL_VOCAB];
};

@implementation LuneletteMetalOwner
- (instancetype)init
{
    self = [super init];
    if (self) {
        uint64_t previous = atomic_load(&last_identity);
        while (previous != UINT64_MAX) {
            if (atomic_compare_exchange_weak(&last_identity, &previous, previous + 1)) {
                identity = previous + 1;
                return self;
            }
        }
        return nil;
    }
    return self;
}
@end

/* malloc owns only the wrapper. bridge_retained/bridge_transfer explicitly move
 * one ARC owner reference across the C API; no object pointers live in C arrays. */
static LuneletteMetalOwner *get_owner(lunelette_metal *context)
{
    return context ? (__bridge LuneletteMetalOwner *)context->owner : nil;
}

static int memory_available(void *context, uint64_t *available)
{
    (void)context;
    vm_statistics64_data_t statistics;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page_size;
    host_t host = mach_host_self();
    kern_return_t statistics_status = host_statistics64(host, HOST_VM_INFO64,
        (host_info64_t)&statistics, &count);
    kern_return_t page_status = host_page_size(host, &page_size);
    if (host != MACH_PORT_NULL) {
        mach_port_deallocate(mach_task_self(), host);
    }
    if (statistics_status != KERN_SUCCESS || page_status != KERN_SUCCESS) {
        return -1;
    }
    uint64_t pages = (uint64_t)statistics.free_count + statistics.inactive_count +
        statistics.speculative_count;
    if (page_size == 0 || pages > UINT64_MAX / page_size) {
        return -1;
    }
    *available = pages * page_size;
    return 0;
}

static int memory_now(void *context, uint64_t *now)
{
    (void)context;
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0 || time.tv_sec < 0 ||
        time.tv_nsec < 0 || time.tv_nsec >= 1000000000L ||
        (uint64_t)time.tv_sec > (UINT64_MAX - (uint64_t)time.tv_nsec) / 1000000000ULL) {
        return -1;
    }
    *now = (uint64_t)time.tv_sec * 1000000000ULL + (uint64_t)time.tv_nsec;
    return 0;
}

static int memory_sleep(void *context, uint64_t nanoseconds)
{
    (void)context;
    struct timespec delay = {0, (long)nanoseconds};
    /* EINTR returns to the policy to remeasure remaining time, never restarting
     * a full interval or resetting the deadline. */
    return nanosleep(&delay, NULL) == 0 || errno == EINTR ? 0 : -1;
}

static int memory_guard(uint64_t weights_bytes, char *error, size_t capacity)
{
    metal_memory_probe probe = {NULL, memory_available, memory_now, memory_sleep};
    return metal_memory_guard(weights_bytes, probe, error, capacity);
}

int lunelette_metal_create_pipelines(LuneletteMetalOwner *owner, char *error, size_t capacity)
{
    NSError *failure = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (!options) {
        snprintf(error, capacity, "Metal compile options allocation failed");
        return -1;
    }
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    options.languageVersion = MTLLanguageVersion3_2;
    NSString *source = [[NSString alloc] initWithBytes:lunelette_metal_shader
        length:sizeof lunelette_metal_shader - 1 encoding:NSUTF8StringEncoding];
    if (!source) {
        snprintf(error, capacity, "Metal embedded source allocation failed");
        return -1;
    }
    id<MTLLibrary> library = [owner->device newLibraryWithSource:source
        options:options error:&failure];
    if (!library) {
        snprintf(error, capacity, "Metal shader compilation: %s",
            failure ? failure.localizedDescription.UTF8String : "no library");
        return -1;
    }
    NSArray<NSString *> *names = @[@"project", @"norm", @"element", @"rope", @"scores", @"attend",
        @"convert_fp8", @"project_fp8"];
    for (size_t i = 0; i < METAL_PIPELINE_COUNT; i++) {
        id<MTLFunction> function = [library newFunctionWithName:names[i]];
        if (function) {
            owner->pipelines[i] = [owner->device newComputePipelineStateWithFunction:function
                error:&failure];
        }
        if (!owner->pipelines[i] || owner->pipelines[i].threadExecutionWidth != 32) {
            snprintf(error, capacity, "Metal pipeline %s needs SIMD width 32: %s",
                names[i].UTF8String, failure ? failure.localizedDescription.UTF8String : "unavailable");
            return -1;
        }
    }
    owner->queue = [owner->device newCommandQueue];
    if (!owner->queue) {
        snprintf(error, capacity, "Metal command queue allocation failed");
        return -1;
    }
    return 0;
}

static int allocate_buffers(LuneletteMetalOwner *owner, char *error, size_t capacity)
{
    const size_t widths[METAL_BUFFER_COUNT] = {
        LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_QUERY,
        LUNELETTE_MODEL_KV, LUNELETTE_MODEL_KV, LUNELETTE_MODEL_QUERY,
        LUNELETTE_MODEL_KV, LUNELETTE_MODEL_QUERY, LUNELETTE_MODEL_HIDDEN,
        LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_INNER,
        LUNELETTE_MODEL_INNER, LUNELETTE_MODEL_INNER, LUNELETTE_MODEL_INNER,
        LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_HIDDEN,
        0, LUNELETTE_MODEL_HEAD_DIM, LUNELETTE_MODEL_HEAD_DIM, 1,
        LUNELETTE_MODEL_QUERY_HEADS * LUNELETTE_CONTEXT_LIMIT
    };
    for (size_t i = 0; i < METAL_BUFFER_COUNT; i++) {
        size_t elements = i == METAL_LOGITS ? LUNELETTE_MODEL_VOCAB :
            LUNELETTE_CONTEXT_LIMIT * widths[i];
        owner->buffers[i] = [owner->device newBufferWithLength:elements * sizeof(float)
            options:MTLResourceStorageModeShared];
        if (!owner->buffers[i]) {
            snprintf(error, capacity, "Metal scratch allocation failed at slot %zu", i);
            return -1;
        }
    }
    for (size_t i = 0; i < LUNELETTE_MODEL_LAYERS; i++) {
        size_t bytes = LUNELETTE_CONTEXT_LIMIT * LUNELETTE_MODEL_KV * sizeof(float);
        owner->keys[i] = [owner->device newBufferWithLength:bytes
            options:MTLResourceStorageModeShared];
        owner->values[i] = [owner->device newBufferWithLength:bytes
            options:MTLResourceStorageModeShared];
        if (!owner->keys[i] || !owner->values[i]) {
            snprintf(error, capacity, "Metal KV allocation failed at layer %zu", i);
            return -1;
        }
    }
    return 0;
}

lunelette_metal *lunelette_metal_create(const char *path,
    char *error, size_t capacity)
{
    return lunelette_metal_create_with_options(path, NULL, error, capacity);
}

lunelette_metal *lunelette_metal_create_with_options(const char *path,
    const lunelette_metal_options *options, char *error, size_t capacity)
{
    char fallback[512];
    if (!error || !capacity) {
        error = fallback;
        capacity = sizeof fallback;
    }
    lunelette_metal_prefill_mode mode = options ? options->prefill : LUNELETTE_METAL_PREFILL_MPS;
    if (mode != LUNELETTE_METAL_PREFILL_MPS && mode != LUNELETTE_METAL_PREFILL_CUSTOM) {
        snprintf(error, capacity, "Metal invalid prefill option; select MPS or CUSTOM");
        return NULL;
    }
    lunelette_metal_strategy strategy = options ? options->strategy : LUNELETTE_METAL_RESIDENT_FP32;
    if (strategy != LUNELETTE_METAL_RESIDENT_FP32 &&
        strategy != LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC) {
        snprintf(error, capacity, "Metal invalid strategy; select resident-fp32 or compact-fp8-arithmetic");
        return NULL;
    }
    @autoreleasepool {
        lunelette_metal_weights staged;
        if (lunelette_metal_weights_open(&staged, path, error, capacity) != 0) {
            return NULL;
        }
        lunelette_metal *context = NULL;
        LuneletteMetalOwner *owner = [LuneletteMetalOwner new];
        if (!owner) {
            snprintf(error, capacity, "Metal owner allocation failed");
            lunelette_metal_weights_close(&staged);
            return NULL;
        }
        owner->prefill_mode = mode;
        owner->device = MTLCreateSystemDefaultDevice();
        if (!owner->device || ![owner->device supportsFamily:MTLGPUFamilyApple7]) {
            snprintf(error, capacity, "Metal requires a supported Apple Silicon GPU (Apple7+)");
            goto cleanup;
        }
        if (lunelette_metal_weights_account(&staged, strategy, &owner->memory)) {
            snprintf(error, capacity, "Metal strategy allocation size preflight failed");
            goto cleanup;
        }
        /* Stage is explicitly charged in the base, not hidden in guard overhead. */
        if (memory_guard(owner->memory.weight_bytes + owner->memory.conversion_stage_bytes,
                error, capacity) != 0 ||
            lunelette_metal_create_pipelines(owner, error, capacity) != 0) {
            goto cleanup;
        }
        if (owner->memory.conversion_stage_bytes) {
            owner->conversion_stage = [owner->device
                newBufferWithLength:owner->memory.conversion_stage_bytes
                options:MTLResourceStorageModeShared];
            if (!owner->conversion_stage) {
                snprintf(error, capacity, "Metal compact conversion stage allocation failed");
                goto cleanup;
            }
        }
        for (size_t i = 0; i < LUNELETTE_METAL_WEIGHT_COUNT; i++) {
            bool compact = strategy == LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC &&
                staged.weights[i].tensor->dtype == ST_F8_E4M3;
            owner->scales[i] = staged.weights[i].scale;
            owner->weights[i] = [owner->device
                newBufferWithLength:staged.weights[i].elements * (compact ? 1 : sizeof(float))
                options:MTLResourceStorageModeShared];
            if (!owner->weights[i]) {
                snprintf(error, capacity, "Metal weight allocation failed: %s",
                    staged.weights[i].tensor->name);
                goto cleanup;
            }
            if (compact) {
                if (lunelette_metal_weights_load_raw(&staged, i, owner->weights[i].contents,
                        staged.weights[i].elements, error, capacity)) {
                    goto cleanup;
                }
            } else if (lunelette_metal_weights_load(&staged, i,
                    owner->weights[i].contents, staged.weights[i].elements) != 0) {
                snprintf(error, capacity, "Metal weight decode/read failed: %s",
                    staged.weights[i].tensor->name);
                goto cleanup;
            }
        }
        if (allocate_buffers(owner, error, capacity) != 0) {
            goto cleanup;
        }
        context = calloc(1, sizeof *context);
        if (!context) {
            snprintf(error, capacity, "Metal context allocation failed");
            goto cleanup;
        }
        owner->valid = true;
        context->owner = (__bridge_retained void *)owner;
cleanup:
        lunelette_metal_weights_close(&staged);
        return context;
    }
}

void lunelette_metal_destroy(lunelette_metal *context)
{
    if (context) {
        @autoreleasepool {
            LuneletteMetalOwner *owner = (__bridge_transfer LuneletteMetalOwner *)context->owner;
            (void)owner;
            free(context);
        }
    }
}

int lunelette_metal_reset(lunelette_metal *context, char *error, size_t capacity)
{
    LuneletteMetalOwner *owner = get_owner(context);
    if (!owner) {
        if (error && capacity) {
            snprintf(error, capacity, "Metal reset requires a context");
        }
        return -1;
    }
    owner->active_checkpoint = NULL;
    owner->endpoint_logits_valid = false;
    owner->length = 0;
    owner->valid = true;
    return 0;
}

static int execute(lunelette_metal *context, const int64_t *ids, size_t count,
    bool prefill, const float **logits, char *error, size_t capacity)
{
    char fallback[512];
    if (!error || !capacity) {
        error = fallback;
        capacity = sizeof fallback;
    }
    if (logits) {
        *logits = NULL;
    }
    LuneletteMetalOwner *owner = get_owner(context);
    if (!owner || !logits || !ids || !count || count > LUNELETTE_CONTEXT_LIMIT ||
        !owner->valid || owner->length > LUNELETTE_CONTEXT_LIMIT - count ||
        (prefill ? owner->length != 0 : owner->length == 0)) {
        snprintf(error, capacity, "Metal sequence rejected: require reset/valid IDs and capacity <=256");
        return -1;
    }
    for (size_t i = 0; i < count; i++) {
        if (ids[i] < 0 || ids[i] >= LUNELETTE_MODEL_VOCAB) {
            snprintf(error, capacity, "Metal invalid token ID at position %zu", i);
            return -1;
        }
    }
    @autoreleasepool {
        /* No state publication until all encoders, completion and finite checks succeed. */
        /* Clear endpoint publication before any scratch/model work. A failed
         * execution must also invalidate the retained checkpoint, but a successful
         * suffix must preserve it. Save its borrowed identity until success. */
        lunelette_metal_checkpoint *checkpoint = owner->active_checkpoint;
        owner->active_checkpoint = NULL;
        owner->endpoint_logits_valid = false;
        owner->valid = false;
        if (count > 1 && owner->prefill_mode == LUNELETTE_METAL_PREFILL_MPS &&
            lunelette_metal_mps_prepare(owner, count, error, capacity) != 0) {
            return -1;
        }
        float *input = owner->buffers[METAL_X].contents;
        const float *embedding = owner->weights[0].contents;
        uint32_t *positions = owner->buffers[METAL_POSITIONS].contents;
        for (size_t i = 0; i < count; i++) {
            memcpy(input + i * LUNELETTE_MODEL_HIDDEN,
                embedding + (size_t)ids[i] * LUNELETTE_MODEL_HIDDEN,
                LUNELETTE_MODEL_HIDDEN * sizeof(float));
            positions[i] = (uint32_t)(owner->length + i);
            if (lunelette_metal_yarn(positions[i],
                    (float *)owner->buffers[METAL_COS].contents + i * LUNELETTE_MODEL_HEAD_DIM,
                    (float *)owner->buffers[METAL_SIN].contents + i * LUNELETTE_MODEL_HEAD_DIM)) {
                snprintf(error, capacity, "Metal position out of range; reset required");
                return -1;
            }
        }
        id<MTLCommandBuffer> command = [owner->queue commandBuffer];
        if (!command) {
            snprintf(error, capacity, "Metal command allocation failed; reset required");
            return -1;
        }
        if (lunelette_metal_encode(owner, command, count, error, capacity) != 0) {
            return -1;
        }
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            snprintf(error, capacity, "Metal execution failed; reset required: %s",
                command.error ? command.error.localizedDescription.UTF8String : "unknown command failure");
            return -1;
        }
        const float *output = owner->buffers[METAL_LOGITS].contents;
        for (size_t i = 0; i < LUNELETTE_MODEL_VOCAB; i++) {
            if (!isfinite(output[i])) {
                snprintf(error, capacity, "Metal nonfinite logit %zu; reset required", i);
                return -1;
            }
        }
        owner->length += count;
        owner->valid = true;
        owner->endpoint_logits_valid = true;
        owner->active_checkpoint = checkpoint;
        *logits = output;
        return 0;
    }
}

int lunelette_metal_prefill(lunelette_metal *context, const int64_t *ids, size_t count,
    const float **logits, char *error, size_t capacity)
{
    return execute(context, ids, count, true, logits, error, capacity);
}

int lunelette_metal_decode(lunelette_metal *context, int64_t token,
    const float **logits, char *error, size_t capacity)
{
    return execute(context, &token, 1, false, logits, error, capacity);
}

int lunelette_metal_get_memory(const lunelette_metal *context, lunelette_metal_memory *memory)
{
    if (!context || !context->owner || !memory) {
        return -1;
    }
    LuneletteMetalOwner *owner = (__bridge LuneletteMetalOwner *)context->owner;
    lunelette_metal_memory result = {0};
    result.strategy = owner->memory.strategy;
    result.conversion_stage_bytes = owner->conversion_stage.length;
    for (size_t i = 0; i < LUNELETTE_METAL_WEIGHT_COUNT; i++) {
        uint64_t bytes = owner->weights[i].length;
        if (result.weight_bytes > UINT64_MAX - bytes) {
            return -1;
        }
        result.weight_bytes += bytes;
    }
    for (size_t i = 0; i < METAL_BUFFER_COUNT; i++) {
        uint64_t bytes = owner->buffers[i].length;
        if (result.scratch_bytes > UINT64_MAX - bytes) {
            return -1;
        }
        result.scratch_bytes += bytes;
    }
    for (size_t i = 0; i < LUNELETTE_MODEL_LAYERS; i++) {
        uint64_t keys = owner->keys[i].length, values = owner->values[i].length;
        if (keys > UINT64_MAX - values || result.kv_bytes > UINT64_MAX - keys - values) {
            return -1;
        }
        result.kv_bytes += keys + values;
    }
    const uint64_t parts[] = {result.weight_bytes, result.conversion_stage_bytes,
        result.scratch_bytes, result.kv_bytes};
    for (size_t i = 0; i < sizeof parts / sizeof parts[0]; i++) {
        if (result.total_bytes > UINT64_MAX - parts[i]) {
            return -1;
        }
        result.total_bytes += parts[i];
    }
    *memory = result;
    return 0;
}

int lunelette_metal_append(lunelette_metal *context, const int64_t *ids, size_t count,
    const float **logits, char *error, size_t capacity)
{
    return execute(context, ids, count, false, logits, error, capacity);
}

int lunelette_metal_checkpoint_create(lunelette_metal *context,
    lunelette_metal_checkpoint **checkpoint, char *error, size_t capacity)
{
    if (checkpoint) {
        *checkpoint = NULL;
    }
    LuneletteMetalOwner *owner = get_owner(context);
    if (!owner || !checkpoint || !owner->identity || !owner->valid ||
        !owner->endpoint_logits_valid || !owner->length ||
        owner->length > LUNELETTE_CONTEXT_LIMIT || owner->checkpoint_epoch == UINT64_MAX ||
        owner->buffers[METAL_LOGITS].length < LUNELETTE_MODEL_VOCAB * sizeof(float)) {
        if (error && capacity) {
            snprintf(error, capacity, "Metal checkpoint requires valid endpoint/available epoch");
        }
        return -1;
    }
    const float *output = owner->buffers[METAL_LOGITS].contents;
    for (size_t i = 0; i < LUNELETTE_MODEL_VOCAB; i++) {
        if (!isfinite(output[i])) {
            if (error && capacity) {
                snprintf(error, capacity, "Metal checkpoint nonfinite logit %zu", i);
            }
            owner->valid = false;
            owner->endpoint_logits_valid = false;
            owner->active_checkpoint = NULL;
            return -1;
        }
    }
    lunelette_metal_checkpoint *saved = LUNELETTE_METAL_CHECKPOINT_ALLOC(sizeof *saved);
    if (!saved) {
        if (error && capacity) {
            snprintf(error, capacity, "Metal checkpoint allocation failed");
        }
        return -1;
    }
    saved->owner = (__bridge void *)owner;
    saved->identity = owner->identity;
    saved->epoch = owner->checkpoint_epoch + 1;
    saved->length = owner->length;
    memcpy(saved->logits, output, sizeof saved->logits);
    owner->checkpoint_epoch = saved->epoch;
    owner->active_checkpoint = saved;
    *checkpoint = saved;
    return 0;
}

int lunelette_metal_checkpoint_restore(lunelette_metal *context,
    const lunelette_metal_checkpoint *checkpoint, const float **logits,
    char *error, size_t capacity)
{
    if (logits) {
        *logits = NULL;
    }
    LuneletteMetalOwner *owner = get_owner(context);
    if (!owner || !logits || !checkpoint || owner->active_checkpoint != checkpoint ||
        checkpoint->identity != owner->identity || checkpoint->epoch != owner->checkpoint_epoch ||
        !owner->valid || !checkpoint->length || checkpoint->length > owner->length ||
        owner->buffers[METAL_LOGITS].length < sizeof checkpoint->logits) {
        if (error && capacity) {
            snprintf(error, capacity, "Metal checkpoint stale, invalid or wrong context");
        }
        return -1;
    }
    memcpy(owner->buffers[METAL_LOGITS].contents, checkpoint->logits, sizeof checkpoint->logits);
    owner->length = checkpoint->length;
    owner->endpoint_logits_valid = true;
    *logits = owner->buffers[METAL_LOGITS].contents;
    return 0;
}

void lunelette_metal_checkpoint_destroy(lunelette_metal_checkpoint *checkpoint)
{
    if (checkpoint) {
        LuneletteMetalOwner *owner = (__bridge LuneletteMetalOwner *)checkpoint->owner;
        if (owner->active_checkpoint == checkpoint) {
            owner->active_checkpoint = NULL;
        }
        free(checkpoint);
    }
}
