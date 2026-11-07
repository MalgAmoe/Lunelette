#include "metal_mps.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static id<MTLBuffer> buffer(id<MTLDevice> device, size_t bytes)
{
    id<MTLBuffer> result = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    CHECK(result);
    return result;
}

static int equal_words(id<MTLBuffer> actual, id<MTLBuffer> expected, size_t words)
{
    const float *a = actual.contents, *b = expected.contents;
    for (size_t i = 0; i < words; i++) {
        uint32_t aa, bb;
        memcpy(&aa, a + i, sizeof aa);
        memcpy(&bb, b + i, sizeof bb);
        if (!isfinite(a[i]) || !isfinite(b[i]) || aa != bb) {
            return 0;
        }
    }
    return 1;
}

static void compare(id<MTLBuffer> actual, id<MTLBuffer> expected, size_t words)
{
    CHECK(equal_words(actual, expected, words));
}

static void capture(id<MTLCommandBuffer> command, id<MTLBuffer> source,
    id<MTLBuffer> destination, size_t bytes)
{
    id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
    CHECK(encoder);
    [encoder copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:bytes];
    [encoder endEncoding];
}

static void test_conversion_tail(LuneletteMetalOwner *owner)
{
    const size_t count = 257;
    id<MTLBuffer> packed = buffer(owner->device, count);
    id<MTLBuffer> stage = buffer(owner->device, (count + 4) * sizeof(float));
    id<MTLBuffer> reference = buffer(owner->device, count * sizeof(float));
    unsigned char original[257];
    float scale = st_bf16_to_f32(0x39fb);
    float *expected = reference.contents;
    for (size_t i = 0; i < count; i++) {
        unsigned code = (unsigned)(i % 254);
        code += code >= 127;
        original[i] = (unsigned char)code;
        expected[i] = st_f8_e4m3_to_f32(original[i]) * scale;
    }
    memcpy(packed.contents, original, count);
    memset(stage.contents, 0x5a, stage.length);
    id<MTLCommandBuffer> command = [owner->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:owner->pipelines[METAL_CONVERT_FP8]];
    [encoder setBuffer:packed offset:0 atIndex:0];
    [encoder setBytes:&scale length:sizeof scale atIndex:1];
    [encoder setBuffer:stage offset:0 atIndex:2];
    uint32_t dims[4] = {1, count, 0, 0};
    [encoder setBytes:dims length:sizeof dims atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(count + 4, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted);
    compare(stage, reference, count);
    float saved = expected[1];
    expected[1] = saved + 1;
    CHECK(!equal_words(stage, reference, count));
    expected[1] = NAN;
    CHECK(!equal_words(stage, reference, count));
    expected[1] = saved;
    compare(stage, reference, count);
    CHECK(!memcmp(packed.contents, original, count));
    CHECK(scale == st_bf16_to_f32(0x39fb));
    for (size_t i = count * sizeof(float); i < stage.length; i++) {
        CHECK(((unsigned char *)stage.contents)[i] == 0x5a);
    }
    puts("compact conversion: all 254 finite codes, signed zero, tail, guards PASS");
}

/* Direct production pipelines with awkward dimensions: compare every output bit,
 * not a reduced statistic or a relaxed tolerance. Each finite code appears in
 * both row and column positions, including both signed zeros. */
static void test_scalar_lookup(LuneletteMetalOwner *owner)
{
    const uint32_t columns[] = {1, 7, 31, 33, 257};
    const float scales[] = {1.0f, 0x1.f6p-12f, 0x1.02p+5f, 0x1p-110f, 0x1p+100f};
    for (size_t c = 0; c < sizeof columns / sizeof columns[0]; c++) {
        uint32_t cols = columns[c], rows = 259;
        id<MTLBuffer> packed = buffer(owner->device, rows * cols);
        id<MTLBuffer> input = buffer(owner->device, cols * sizeof(float));
        id<MTLBuffer> outputs[2] = {buffer(owner->device, (rows + 4) * 4),
            buffer(owner->device, (rows + 4) * 4)};
        for (size_t i = 0; i < rows * cols; i++) {
            unsigned code = (unsigned)(i % 254);
            ((unsigned char *)packed.contents)[i] = (unsigned char)(code + (code >= 127));
        }
        for (size_t i = 0; i < cols; i++) {
            ((float *)input.contents)[i] = (float)((int)(i % 17) - 8) / 16.0f;
        }
        for (size_t scale = 0; scale < sizeof scales / sizeof scales[0]; scale++) {
            id<MTLCommandBuffer> command = [owner->queue commandBuffer];
            for (size_t candidate = 0; candidate < 2; candidate++) {
                memset(outputs[candidate].contents, 0x5a, outputs[candidate].length);
                int operation = candidate ? METAL_PROJECT_FP8_LOOKUP : METAL_PROJECT_FP8;
                id<MTLComputePipelineState> pipeline = owner->pipelines[operation];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                CHECK(encoder);
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:packed offset:0 atIndex:0];
                [encoder setBuffer:input offset:0 atIndex:1];
                [encoder setBuffer:outputs[candidate] offset:0 atIndex:2];
                uint32_t dims[4] = {rows, cols, 1, 0};
                [encoder setBytes:dims length:sizeof dims atIndex:3];
                [encoder setBytes:&scales[scale] length:sizeof(float) atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
                [encoder endEncoding];
            }
            [command commit];
            [command waitUntilCompleted];
            CHECK(command.status == MTLCommandBufferStatusCompleted);
            compare(outputs[1], outputs[0], rows);
            for (size_t candidate = 0; candidate < 2; candidate++) {
                for (size_t i = rows * 4; i < outputs[candidate].length; i++) {
                    CHECK(((unsigned char *)outputs[candidate].contents)[i] == 0x5a);
                }
            }
        }
    }
    puts("lookup/arithmetic scalar full outputs bitwise: finite codes/scales/tails PASS");
}

static void run(LuneletteMetalOwner *owner, LuneletteMetalOwner *control,
    size_t count, lunelette_metal_prefill_mode mode)
{
    const size_t indices[] = {3, 19, 3, 22}; /* layer0 Q, layer1 down, Q, layer2 K */
    const size_t rows[] = {4096, 3072, 4096, 1024};
    const size_t cols[] = {3072, 9216, 3072, 3072};
    char error[512];
    owner->prefill_mode = control->prefill_mode = mode;
    if (count > 1 && mode == LUNELETTE_METAL_PREFILL_MPS) {
        CHECK(!lunelette_metal_mps_prepare(owner, count, error, sizeof error));
        CHECK(!lunelette_metal_mps_prepare(control, count, error, sizeof error));
        LuneletteMPSCache *cache = owner->mps_cache;
        for (size_t layer = 0; layer < 26; layer++) {
            const size_t slots[] = {1, 2, 3, 4, 6, 7, 8};
            for (size_t p = 0; p < 7; p++) {
                size_t index = 2 + layer * 9 + slots[p];
                CHECK(cache->weights[index].data == owner->conversion_stage);
                CHECK(cache->weights[index].data != owner->weights[index]);
            }
        }
        CHECK(!lunelette_metal_mps_prepare(owner, count, error, sizeof error));
        CHECK(owner->mps_cache == cache);
    }
    id<MTLBuffer> actual[4], expected[4], stages[4], tails[4];
    const int outputs[] = {METAL_Q, METAL_DOWN, METAL_K};
    for (size_t i = 0; i < 3; i++) {
        memset(owner->buffers[outputs[i]].contents, 0x5a, owner->buffers[outputs[i]].length);
        memset(control->buffers[outputs[i]].contents, 0x5a, control->buffers[outputs[i]].length);
    }
    id<MTLCommandBuffer> command = [owner->queue commandBuffer];
    CHECK(command);
    for (size_t i = 0; i < 4; i++) {
        size_t slot = (indices[i] - 2) % 9;
        int input = slot == 8 ? METAL_GATED : METAL_INPUT_NORM;
        int output = slot == 8 ? METAL_DOWN : (slot == 2 ? METAL_K : METAL_Q);
        size_t bytes = count * rows[i] * sizeof(float);
        actual[i] = buffer(owner->device, bytes + 16);
        expected[i] = buffer(owner->device, bytes + 16);
        /* No CPU writes to stage, inputs or outputs after encoding starts. */
        CHECK(!lunelette_metal_project(owner, command, indices[i], input, output,
            (uint32_t)rows[i], (uint32_t)cols[i], count));
        capture(command, owner->buffers[output], actual[i], bytes + 16);
        CHECK(!lunelette_metal_project(control, command, indices[i], input, output,
            (uint32_t)rows[i], (uint32_t)cols[i], count));
        capture(command, control->buffers[output], expected[i], bytes + 16);
        if (count > 1) {
            stages[i] = buffer(owner->device, rows[i] * cols[i] * sizeof(float));
            capture(command, owner->conversion_stage, stages[i], stages[i].length);
            tails[i] = buffer(owner->device, 16);
            id<MTLBlitCommandEncoder> tail = [command blitCommandEncoder];
            CHECK(tail);
            [tail copyFromBuffer:owner->conversion_stage sourceOffset:stages[i].length
                toBuffer:tails[i] destinationOffset:0 size:16];
            [tail endEncoding];
        }
    }
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted);
    for (size_t i = 0; i < 4; i++) {
        compare(actual[i], expected[i], count * rows[i]);
        for (size_t j = count * rows[i] * 4; j < actual[i].length; j++) {
            CHECK(((unsigned char *)actual[i].contents)[j] == 0x5a);
            CHECK(((unsigned char *)expected[i].contents)[j] == 0x5a);
        }
        if (count > 1) {
            compare(stages[i], control->weights[indices[i]], rows[i] * cols[i]);
        }
    }
    if (count > 1) {
        /* A/B/A: the smaller matrix must leave B's old tail intact. K then leaves A's. */
        CHECK(!memcmp(tails[2].contents,
            (char *)control->weights[19].contents + 4096 * 3072 * 4, 16));
        CHECK(!memcmp(tails[3].contents,
            (char *)control->weights[3].contents + 1024 * 3072 * 4, 16));
    }
    const unsigned char *stage = owner->conversion_stage.contents;
    for (size_t i = 113246208; i < owner->conversion_stage.length; i++) {
        CHECK(stage[i] == 0x5a);
    }
    lunelette_metal context = {(__bridge void *)owner};
    owner->length = 37;
    owner->valid = false;
    __weak id cache = owner->mps_cache;
    CHECK(!lunelette_metal_reset(&context, error, sizeof error));
    CHECK(owner->length == 0 && owner->valid && owner->mps_cache == cache);
    printf("compact production seam strategy=%d count=%zu mode=%s: 4 projections + stage/guards PASS\n",
        owner->memory.strategy, count, mode == LUNELETTE_METAL_PREFILL_MPS ? "mps" : "custom");
}

int main(void)
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyApple7] || !MPSSupportsMTLDevice(device)) {
            puts("SKIP compact GPU test: supported GPU/MPS unavailable");
            return 77;
        }
        LuneletteMetalOwner *owner = [LuneletteMetalOwner new];
        LuneletteMetalOwner *control = [LuneletteMetalOwner new];
        owner->device = control->device = device;
        char error[512];
        CHECK(!lunelette_metal_create_pipelines(owner, error, sizeof error));
        /* Same production kernels, independently decoded FP32 control buffers. */
        for (size_t i = 0; i < METAL_PIPELINE_COUNT; i++) {
            control->pipelines[i] = owner->pipelines[i];
        }
        test_conversion_tail(owner);
        test_scalar_lookup(owner);
        owner->memory.strategy = LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC;
        owner->conversion_stage = buffer(device, 113246208 + 16);
        memset(owner->conversion_stage.contents, 0x5a, owner->conversion_stage.length);
        for (size_t i = 0; i < METAL_BUFFER_COUNT; i++) {
            owner->buffers[i] = buffer(device, 6 * 9216 * sizeof(float) + 16);
            control->buffers[i] = buffer(device, owner->buffers[i].length);
            float *values = owner->buffers[i].contents;
            for (size_t j = 0; j < 6 * 9216; j++) {
                values[j] = (float)((int)(j % 29) - 14) / 32.0f;
            }
            memcpy(control->buffers[i].contents, values, owner->buffers[i].length);
        }
        const size_t indices[] = {3, 19, 22};
        const size_t elements[] = {4096 * 3072, 3072 * 9216, 1024 * 3072};
        for (size_t p = 0; p < 3; p++) {
            size_t index = indices[p];
            owner->weights[index] = buffer(device, elements[p]);
            control->weights[index] = buffer(device, elements[p] * sizeof(float));
            owner->scales[index] = st_bf16_to_f32((uint16_t)(0x39fb + p));
            unsigned char *packed = owner->weights[index].contents;
            float *expanded = control->weights[index].contents;
            for (size_t i = 0; i < elements[p]; i++) {
                unsigned code = (unsigned)((i * 37 + p * 17) % 254);
                code += code >= 127;
                packed[i] = (unsigned char)code;
                expanded[i] = st_f8_e4m3_to_f32(packed[i]) * owner->scales[index];
            }
        }
        /* Unused control views share the largest FP32 matrix, never allocate 182 matrices. */
        for (size_t i = 0; i < LUNELETTE_METAL_WEIGHT_COUNT; i++) {
            if (!control->weights[i]) {
                control->weights[i] = control->weights[19];
            }
        }
        lunelette_metal_options options = LUNELETTE_METAL_OPTIONS_DEFAULT;
        CHECK(options.strategy == LUNELETTE_METAL_RESIDENT_FP32);
        options.strategy = (lunelette_metal_strategy)3;
        CHECK(!lunelette_metal_create_with_options("must-not-open", &options, error, sizeof error));
        CHECK(strstr(error, "invalid strategy"));
        const size_t counts[] = {1, 2, 6, 2};
        for (int strategy = 1; strategy <= 2; strategy++) {
            owner->memory.strategy = (lunelette_metal_strategy)strategy;
            for (size_t mode = 0; mode < 2; mode++) {
                for (size_t i = 0; i < 4; i++) {
                    @autoreleasepool {
                        run(owner, control, counts[i], (lunelette_metal_prefill_mode)mode);
                    }
                }
            }
        }
        id<MTLCommandBuffer> rejected = [owner->queue commandBuffer];
        CHECK(lunelette_metal_project(owner, rejected, 1, METAL_INPUT_NORM, METAL_Q,
            4096, 3072, 1));
        CHECK(lunelette_metal_project(owner, rejected, 3, METAL_INPUT_NORM, METAL_Q,
            4095, 3072, 1));
        CHECK(lunelette_metal_project(owner, rejected, 3, METAL_INPUT_NORM, METAL_Q,
            4096, 3072, 0));
        id<MTLBuffer> saved_input = owner->buffers[METAL_INPUT_NORM];
        owner->buffers[METAL_INPUT_NORM] = nil;
        __weak id complete_cache = owner->mps_cache;
        CHECK(lunelette_metal_mps_prepare(owner, 3, error, sizeof error));
        CHECK(owner->mps_cache == complete_cache);
        owner->buffers[METAL_INPUT_NORM] = saved_input;
        __weak id stage = owner->conversion_stage;
        __weak id old_cache = owner->mps_cache;
        owner->mps_cache = nil;
        CHECK(old_cache == nil && stage == owner->conversion_stage);
        lunelette_metal_memory memory;
        lunelette_metal context = {(__bridge void *)owner};
        CHECK(!lunelette_metal_get_memory(&context, &memory));
        CHECK(memory.strategy == LUNELETTE_METAL_COMPACT_FP8_LOOKUP);
        CHECK(memory.weight_bytes == (uint64_t)elements[0] + elements[1] + elements[2]);
        CHECK(memory.conversion_stage_bytes == 113246208 + 16);
        CHECK(memory.scratch_bytes == METAL_BUFFER_COUNT * (6 * 9216 * sizeof(float) + 16));
        CHECK(memory.kv_bytes == 0);
        CHECK(memory.total_bytes == memory.weight_bytes + memory.conversion_stage_bytes +
            memory.scratch_bytes);
        CHECK(lunelette_metal_get_memory(NULL, &memory));
        CHECK(lunelette_metal_get_memory(&context, NULL));
        CHECK(!lunelette_metal_mps_prepare(owner, 2, error, sizeof error));
        __weak id packed;
        @autoreleasepool {
            /* Never dispatched: no command may legitimately retain this probe. */
            owner->weights[4] = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
            CHECK(owner->weights[4]);
            packed = owner->weights[4];
            owner->mps_cache = nil;
            CHECK(!lunelette_metal_mps_prepare(owner, 2, error, sizeof error));
            CHECK(owner->mps_cache->weights[4].data == owner->conversion_stage);
            owner->weights[4] = nil;
        }
        CHECK(packed == nil); /* Views retain the stage, not the former packed matrix. */
        puts("Compact integration model-free PASS (64 production projections; no model parity claim)");
    }
    return 0;
}
