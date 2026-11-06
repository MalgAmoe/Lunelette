#include "metal_mps.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <objc/runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rope_golden.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

/* Internal Objective-C doubles exercise nil encoders without device fault tricks. */
@interface NilEncoderCommand : NSObject
- (id<MTLComputeCommandEncoder>)computeCommandEncoder;
- (id<MTLBlitCommandEncoder>)blitCommandEncoder;
@end
@implementation NilEncoderCommand
- (id<MTLComputeCommandEncoder>)computeCommandEncoder
{
    return nil;
}
- (id<MTLBlitCommandEncoder>)blitCommandEncoder
{
    return nil;
}
@end

@interface NilEncoderQueue : NSObject
- (id<MTLCommandBuffer>)commandBuffer;
@end
@implementation NilEncoderQueue
- (id<MTLCommandBuffer>)commandBuffer
{
    return (id<MTLCommandBuffer>)[NilEncoderCommand new];
}
@end

/* No-op encoders let the actual host traversal reach each nil-encoder boundary
 * and command/nonfinite publication paths. They do not simulate GPU arithmetic. */
static bool inspect_suffix_copies;
static size_t suffix_copy_count;
static bool fail_checkpoint_allocation;

void *lunelette_test_checkpoint_alloc(size_t bytes)
{
    return fail_checkpoint_allocation ? NULL : malloc(bytes);
}

@interface NoopEncoder : NSObject {
@public
    id<MTLCommandBuffer> kv_command;
}
@end
@implementation NoopEncoder
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline
{
}
- (void)setBuffer:(id<MTLBuffer>)value offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
}
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
}
- (void)dispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads
{
}
- (void)dispatchThreads:(MTLSize)grid threadsPerThreadgroup:(MTLSize)threads
{
}
- (void)copyFromBuffer:(id<MTLBuffer>)source sourceOffset:(NSUInteger)source_offset
    toBuffer:(id<MTLBuffer>)destination destinationOffset:(NSUInteger)destination_offset
    size:(NSUInteger)bytes
{
    /* Observe production KV blits; optionally execute only those copies below.
     * Other copies in this traversal have hidden-width (3072), not KV width. */
    if (inspect_suffix_copies && bytes == 2 * LUNELETTE_MODEL_KV * sizeof(float)) {
        CHECK(source_offset == 0);
        CHECK(destination_offset == LUNELETTE_MODEL_KV * sizeof(float));
        suffix_copy_count++;
    }
    if (kv_command && bytes == 2 * LUNELETTE_MODEL_KV * sizeof(float)) {
        id<MTLBlitCommandEncoder> copy = [kv_command blitCommandEncoder];
        CHECK(copy);
        [copy copyFromBuffer:source sourceOffset:source_offset toBuffer:destination
            destinationOffset:destination_offset size:bytes];
        [copy endEncoding];
    }
}
- (void)endEncoding
{
}
@end

@interface SyntheticCommand : NSObject {
@public
    size_t calls;
    size_t fail_at;
    MTLCommandBufferStatus completion;
    id<MTLCommandBuffer> kv_command;
}
@end
@implementation SyntheticCommand
- (id<MTLComputeCommandEncoder>)computeCommandEncoder
{
    calls++;
    return calls == fail_at ? nil : (id<MTLComputeCommandEncoder>)[NoopEncoder new];
}
- (id<MTLBlitCommandEncoder>)blitCommandEncoder
{
    calls++;
    if (calls == fail_at) {
        return nil;
    }
    NoopEncoder *encoder = [NoopEncoder new];
    encoder->kv_command = kv_command;
    return (id<MTLBlitCommandEncoder>)encoder;
}
- (void)commit
{
    [kv_command commit];
}
- (void)waitUntilCompleted
{
    [kv_command waitUntilCompleted];
}
- (MTLCommandBufferStatus)status
{
    return kv_command ? kv_command.status : completion;
}
- (NSError *)error
{
    return [NSError errorWithDomain:@"synthetic Metal test failure" code:1 userInfo:nil];
}
@end

@interface SyntheticQueue : NSObject {
@public
    SyntheticCommand *command;
}
@end
@implementation SyntheticQueue
- (id<MTLCommandBuffer>)commandBuffer
{
    return (id<MTLCommandBuffer>)command;
}
@end

static id<MTLBuffer> buffer(LuneletteMetalOwner *owner, size_t elements)
{
    id<MTLBuffer> result = [owner->device newBufferWithLength:elements * sizeof(float)
        options:MTLResourceStorageModeShared];
    CHECK(result != nil);
    return result;
}

static void complete(id<MTLCommandBuffer> command)
{
    CHECK(command != nil);
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted);
}

static void test_projection(LuneletteMetalOwner *owner)
{
    /* Multiple tokens, odd columns/rows: SIMD tail and output offset coverage. */
    const size_t rows = 7, columns = 67, tokens = 3;
    id<MTLBuffer> weights = buffer(owner, rows * columns);
    id<MTLBuffer> input = buffer(owner, tokens * columns);
    id<MTLBuffer> output = buffer(owner, rows * tokens);
    float *w = weights.contents;
    float *x = input.contents;
    for (size_t i = 0; i < rows * columns; i++) {
        w[i] = ((int)(i % 19) - 9) * 0.03125f;
    }
    for (size_t i = 0; i < tokens * columns; i++) {
        x[i] = ((int)(i % 13) - 6) * 0.0625f;
    }
    id<MTLCommandBuffer> command = [owner->queue commandBuffer];
    CHECK(lunelette_metal_encode_operator(owner, command, METAL_PROJECT,
        weights, input, output, rows, columns, tokens, tokens, 0) == 0);
    complete(command);
    const float *actual = output.contents;
    for (size_t token = 0; token < tokens; token++) {
        for (size_t row = 0; row < rows; row++) {
            double expected = 0;
            for (size_t column = 0; column < columns; column++) {
                expected += (double)w[row * columns + column] * x[token * columns + column];
            }
            CHECK(fabs(actual[token * rows + row] - expected) < 1e-6);
        }
    }
}

static void test_mps_projection(LuneletteMetalOwner *owner)
{
    const size_t counts[] = {6, 24, 150};
    const size_t rows[] = {7, 31, 65};
    const size_t columns[] = {13, 127, 3073};
    for (size_t t = 0; t < 3; t++) {
        size_t m = counts[t], n = rows[t], k = columns[t];
        id<MTLBuffer> input = buffer(owner, m * k);
        id<MTLBuffer> weights = buffer(owner, n * k);
        id<MTLBuffer> output = buffer(owner, m * n);
        float *x = input.contents;
        float *w = weights.contents;
        for (size_t i = 0; i < m * k; i++) {
            x[i] = ((int)((i * 37 + 11) % 257) - 128) / 127.0f;
        }
        for (size_t i = 0; i < n * k; i++) {
            w[i] = ((int)((i * 53 + 7) % 251) - 125) / 131.0f;
        }
        MPSMatrix *a = lunelette_metal_mps_view(input, m, k);
        MPSMatrix *b = lunelette_metal_mps_view(weights, n, k);
        MPSMatrix *c = lunelette_metal_mps_view(output, m, n);
        CHECK(a && b && c && b.data == weights);
        CHECK(a.rowBytes == k * sizeof(float) && b.rowBytes == k * sizeof(float));
        MPSMatrixMultiplication *kernel = lunelette_metal_mps_kernel(owner->device, m, n, k);
        CHECK(kernel != nil);
        for (size_t repeat = 0; repeat < 2; repeat++) {
            id<MTLCommandBuffer> command = [owner->queue commandBuffer];
            CHECK(command != nil);
            [kernel encodeToCommandBuffer:command leftMatrix:a rightMatrix:b resultMatrix:c];
            complete(command);
            const float *actual = output.contents;
            double maximum = 0;
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    double expected = 0;
                    for (size_t l = 0; l < k; l++) {
                        expected += (double)x[i * k + l] * w[j * k + l];
                    }
                    double delta = fabs(actual[i * n + j] - expected);
                    CHECK(isfinite(actual[i * n + j]));
                    CHECK(delta <= 2e-4 + 1e-4 * fabs(expected));
                    if (delta > maximum) {
                        maximum = delta;
                    }
                }
            }
            printf("MPS synthetic %zu x %zu x %zu repeat %zu maxabs %.9g\n",
                m, n, k, repeat, maximum);
        }
    }
    CHECK(lunelette_metal_mps_kernel(nil, 6, 7, 13) == nil);
    CHECK(lunelette_metal_mps_kernel(owner->device, 0, 7, 13) == nil);
    CHECK(lunelette_metal_mps_kernel(owner->device, 257, 7, 13) == nil);
    CHECK(lunelette_metal_mps_kernel(owner->device, 6, 0, 13) == nil);
    CHECK(lunelette_metal_mps_kernel(owner->device, 6, 7, SIZE_MAX) == nil);
    CHECK(lunelette_metal_mps_view(nil, 6, 13) == nil);
    CHECK(lunelette_metal_mps_view(buffer(owner, 1), SIZE_MAX, 13) == nil);
    CHECK(lunelette_metal_mps_view(buffer(owner, 1), 0, 13) == nil);
    CHECK(lunelette_metal_mps_view(buffer(owner, 1), 6, 13) == nil);
}

/* Swizzle only concrete methods, never inherited NSObject allocation methods.
 * Tests run serially and restore every method before leaving the scope. */
static size_t allocation_index;
static size_t allocation_fail_at;
static bool fail_allocation(void)
{
    return allocation_index++ == allocation_fail_at;
}

@interface MPSMatrixDescriptor (LuneletteFailureTest)
+ (instancetype)test_rows:(NSUInteger)rows columns:(NSUInteger)columns
    rowBytes:(NSUInteger)bytes dataType:(MPSDataType)type;
@end
@implementation MPSMatrixDescriptor (LuneletteFailureTest)
+ (instancetype)test_rows:(NSUInteger)rows columns:(NSUInteger)columns
    rowBytes:(NSUInteger)bytes dataType:(MPSDataType)type
{
    if (fail_allocation()) {
        return nil;
    }
    return [self test_rows:rows columns:columns rowBytes:bytes dataType:type];
}
@end

@interface MPSMatrix (LuneletteFailureTest)
- (instancetype)initTestWithBuffer:(id<MTLBuffer>)buffer descriptor:(MPSMatrixDescriptor *)descriptor;
@end
@implementation MPSMatrix (LuneletteFailureTest)
- (instancetype)initTestWithBuffer:(id<MTLBuffer>)buffer descriptor:(MPSMatrixDescriptor *)descriptor
{
    if (fail_allocation()) {
        return nil;
    }
    return [self initTestWithBuffer:buffer descriptor:descriptor];
}
@end

@interface MPSMatrixMultiplication (LuneletteFailureTest)
- (instancetype)initTestWithDevice:(id<MTLDevice>)device transposeLeft:(BOOL)left
    transposeRight:(BOOL)right resultRows:(NSUInteger)rows resultColumns:(NSUInteger)columns
    interiorColumns:(NSUInteger)interior alpha:(double)alpha beta:(double)beta;
@end
@implementation MPSMatrixMultiplication (LuneletteFailureTest)
- (instancetype)initTestWithDevice:(id<MTLDevice>)device transposeLeft:(BOOL)left
    transposeRight:(BOOL)right resultRows:(NSUInteger)rows resultColumns:(NSUInteger)columns
    interiorColumns:(NSUInteger)interior alpha:(double)alpha beta:(double)beta
{
    if (fail_allocation()) {
        return nil;
    }
    return [self initTestWithDevice:device transposeLeft:left transposeRight:right
        resultRows:rows resultColumns:columns interiorColumns:interior alpha:alpha beta:beta];
}
@end

static void swap_kernel_failure(void)
{
    method_exchangeImplementations(class_getInstanceMethod([MPSMatrixMultiplication class],
        @selector(initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:)),
        class_getInstanceMethod([MPSMatrixMultiplication class],
        @selector(initTestWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:)));
}

static void swap_descriptor_failure(void)
{
    method_exchangeImplementations(class_getClassMethod([MPSMatrixDescriptor class],
        @selector(matrixDescriptorWithRows:columns:rowBytes:dataType:)),
        class_getClassMethod([MPSMatrixDescriptor class],
        @selector(test_rows:columns:rowBytes:dataType:)));
}

static void swap_view_failure(void)
{
    method_exchangeImplementations(class_getInstanceMethod([MPSMatrix class],
        @selector(initWithBuffer:descriptor:)), class_getInstanceMethod([MPSMatrix class],
        @selector(initTestWithBuffer:descriptor:)));
}

@interface LuneletteMPSCache (LuneletteFailureTest)
+ (instancetype)test_new;
@end
@implementation LuneletteMPSCache (LuneletteFailureTest)
+ (instancetype)test_new
{
    return nil;
}
@end

static void test_cache_allocation_failure(LuneletteMetalOwner *owner)
{
    Class type = [LuneletteMPSCache class];
    Method original = class_getClassMethod(type, @selector(new));
    /* Install an override on this class before swapping: never mutate the
     * inherited NSObject +new shared by unrelated runtime objects. */
    CHECK(class_addMethod(object_getClass(type), @selector(new),
        method_getImplementation(original), method_getTypeEncoding(original)));
    Method local = class_getClassMethod(type, @selector(new));
    Method failure = class_getClassMethod(type, @selector(test_new));
    method_exchangeImplementations(local, failure);
    char error[512];
    __weak id old = owner->mps_cache;
    CHECK(lunelette_metal_mps_prepare(owner, 24, error, sizeof error) != 0);
    CHECK(old == owner->mps_cache);
    method_exchangeImplementations(local, failure);
}

static void test_mps_cache(void)
{
    @autoreleasepool {
        LuneletteMetalOwner *owner = [LuneletteMetalOwner new];
        owner->device = MTLCreateSystemDefaultDevice();
        CHECK(owner->device != nil);
        /* Alias buffers deliberately: validate mapping/lifetimes without loading
         * a model or allocating a second resident weight set. No encoding here. */
        id<MTLBuffer> weight = buffer(owner,
            (size_t)LUNELETTE_MODEL_INNER * LUNELETTE_MODEL_HIDDEN);
        id<MTLBuffer> scratch;
        const size_t scratch_elements = (size_t)24 * LUNELETTE_MODEL_INNER;
        for (size_t i = 0; i < LUNELETTE_METAL_WEIGHT_COUNT; i++) {
            owner->weights[i] = weight;
        }
        for (size_t i = 0; i < METAL_BUFFER_COUNT; i++) {
            owner->buffers[i] = buffer(owner, scratch_elements);
        }
        scratch = owner->buffers[METAL_INPUT_NORM];
        char error[512];
        const size_t counts[] = {6, 24, 6};
        for (size_t i = 0; i < 3; i++) {
            __weak id previous = owner->mps_cache;
            CHECK(lunelette_metal_mps_prepare(owner, counts[i], error, sizeof error) == 0);
            CHECK(previous == nil);
            LuneletteMPSCache *cache = owner->mps_cache;
            const size_t slots[] = {1, 2, 3, 4, 6, 7, 8};
            const int inputs[] = {METAL_INPUT_NORM, METAL_INPUT_NORM, METAL_INPUT_NORM,
                METAL_ATTENTION, METAL_POST_NORM, METAL_POST_NORM, METAL_GATED};
            const int outputs[] = {METAL_Q, METAL_K, METAL_V, METAL_O, METAL_GATE,
                METAL_UP, METAL_DOWN};
            for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
                for (size_t p = 0; p < 7; p++) {
                    size_t w = 2 + layer * LUNELETTE_MODEL_BLOCK_WEIGHTS + slots[p];
                    CHECK(cache->inputs[w].data == owner->buffers[inputs[p]]);
                    CHECK(cache->outputs[w].data == owner->buffers[outputs[p]]);
                    CHECK(cache->weights[w].data == weight);
                    CHECK(cache->inputs[w].rows == counts[i]);
                    CHECK(cache->outputs[w].columns == cache->weights[w].rows);
                    CHECK(cache->inputs[w].columns == cache->weights[w].columns);
                    CHECK(cache->weights[w].rowBytes == cache->weights[w].columns * 4);
                    CHECK(cache->kernels[p] != nil);
                }
            }
            cache = nil;
            __weak id current = owner->mps_cache;
            CHECK(current != nil);
            CHECK(lunelette_metal_mps_prepare(owner, counts[i], error, sizeof error) == 0);
            CHECK(current == owner->mps_cache);
            /* Failed preparation retains old complete shape transactionally. */
            owner->buffers[METAL_INPUT_NORM] = nil;
            CHECK(lunelette_metal_mps_prepare(owner, counts[i] + 1,
                error, sizeof error) != 0);
            CHECK(current == owner->mps_cache);
            owner->buffers[METAL_INPUT_NORM] = scratch;
        }
        /* Inject first, interior and final descriptor/view allocation failure.
         * Each must preserve the old cache and release partial new resources. */
        const size_t failures[] = {0, 97, 545};
        for (size_t kind = 0; kind < 2; kind++) {
            if (kind == 0) {
                swap_descriptor_failure();
            } else {
                swap_view_failure();
            }
            for (size_t i = 0; i < 3; i++) {
                allocation_index = 0;
                allocation_fail_at = failures[i];
                __weak id old = owner->mps_cache;
                CHECK(lunelette_metal_mps_prepare(owner, 24, error, sizeof error) != 0);
                CHECK(owner->mps_cache == old);
            }
            if (kind == 0) {
                swap_descriptor_failure();
            } else {
                swap_view_failure();
            }
        }
        test_cache_allocation_failure(owner);
        swap_kernel_failure();
        for (size_t i = 0; i < 7; i++) {
            allocation_index = 0;
            allocation_fail_at = i;
            __weak id old = owner->mps_cache;
            CHECK(lunelette_metal_mps_prepare(owner, 24, error, sizeof error) != 0);
            CHECK(owner->mps_cache == old);
        }
        swap_kernel_failure();
        lunelette_metal context = {(__bridge void *)owner};
        owner->prefill_mode = LUNELETTE_METAL_PREFILL_MPS;
        owner->valid = true;
        owner->buffers[METAL_INPUT_NORM] = nil;
        const float *logits = (const float *)1;
        int64_t ids[24] = {0};
        CHECK(lunelette_metal_prefill(&context, ids, 24, &logits, error, sizeof error) != 0);
        CHECK(logits == NULL && !owner->valid && owner->length == 0);
        CHECK(strstr(error, "custom") != NULL);
        CHECK(lunelette_metal_decode(&context, 0, &logits, error, sizeof error) != 0);
        CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
        CHECK(owner->valid && owner->length == 0);
        __weak id cache = owner->mps_cache;
        owner->mps_cache = nil;
        CHECK(cache == nil);
    }
}

static void test_options(void)
{
    char error[512];
    lunelette_metal_options options = LUNELETTE_METAL_OPTIONS_DEFAULT;
    CHECK(options.prefill == LUNELETTE_METAL_PREFILL_MPS);
    options.prefill = (lunelette_metal_prefill_mode)99;
    CHECK(lunelette_metal_create_with_options(NULL, &options,
        error, sizeof error) == NULL);
    CHECK(strstr(error, "invalid prefill option") != NULL);
    CHECK(lunelette_metal_create_with_options(NULL, &options, NULL, 0) == NULL);
    CHECK(lunelette_metal_create_with_options(NULL, NULL, error, sizeof error) == NULL);
    options.prefill = LUNELETTE_METAL_PREFILL_CUSTOM;
    CHECK(lunelette_metal_create_with_options(NULL, &options,
        error, sizeof error) == NULL);
    CHECK(lunelette_metal_create(NULL, NULL, 0) == NULL);
}

static void test_norm_element(LuneletteMetalOwner *owner)
{
    const size_t width = 67, tokens = 2;
    id<MTLBuffer> weights = buffer(owner, width);
    id<MTLBuffer> input = buffer(owner, width * tokens);
    id<MTLBuffer> output = buffer(owner, width * tokens);
    float *w = weights.contents;
    float *x = input.contents;
    for (size_t i = 0; i < width; i++) {
        w[i] = 0.5f + i * 0.001f;
    }
    for (size_t i = 0; i < width * tokens; i++) {
        x[i] = ((int)(i % 17) - 8) * 0.125f;
    }
    id<MTLCommandBuffer> command = [owner->queue commandBuffer];
    CHECK(lunelette_metal_encode_operator(owner, command, METAL_NORM,
        weights, input, output, width, 0, 0, tokens, 0) == 0);
    complete(command);
    const float *actual = output.contents;
    for (size_t token = 0; token < tokens; token++) {
        double square = 0;
        for (size_t i = 0; i < width; i++) {
            square += (double)x[token * width + i] * x[token * width + i];
        }
        for (size_t i = 0; i < width; i++) {
            double expected = w[i] * x[token * width + i] / sqrt(square / width + 1e-5);
            CHECK(fabs(actual[token * width + i] - expected) < 2e-6);
        }
    }
    for (uint32_t mode = 0; mode < 3; mode++) {
        command = [owner->queue commandBuffer];
        CHECK(lunelette_metal_encode_operator(owner, command, METAL_ELEMENT,
            input, input, output, width * tokens, mode, 0, tokens, 0) == 0);
        complete(command);
        for (size_t i = 0; i < width * tokens; i++) {
            double expected = mode == 0 ? 2 * x[i] :
                (mode == 1 ? x[i] / (1 + exp(-x[i])) : x[i] * x[i]);
            CHECK(fabs(actual[i] - expected) < 2e-6);
        }
    }
}

static void test_attention(LuneletteMetalOwner *owner)
{
    const size_t count = 2, prefix = 3, total = prefix + count;
    owner->length = prefix;
    id<MTLBuffer> query = buffer(owner, count * 4096);
    id<MTLBuffer> keys = buffer(owner, total * 1024);
    id<MTLBuffer> values = buffer(owner, total * 1024);
    id<MTLBuffer> scores = buffer(owner, count * 32 * total);
    id<MTLBuffer> output = buffer(owner, count * 4096);
    float *q = query.contents, *k = keys.contents, *v = values.contents;
    for (size_t i = 0; i < count * 4096; i++) {
        q[i] = ((int)(i % 11) - 5) * 0.03125f;
    }
    for (size_t i = 0; i < total * 1024; i++) {
        k[i] = ((int)(i % 17) - 8) * 0.03125f;
        v[i] = ((int)(i % 23) - 11) * 0.0625f;
    }
    id<MTLCommandBuffer> command = [owner->queue commandBuffer];
    CHECK(lunelette_metal_encode_operator(owner, command, METAL_SCORES,
        query, keys, scores, count, 0, 0, count, 0) == 0);
    CHECK(lunelette_metal_encode_operator(owner, command, METAL_ATTEND,
        scores, values, output, count, 0, 0, count, 0) == 0);
    complete(command);
    const float *actual = output.contents;
    for (size_t token = 0; token < count; token++) {
        for (size_t head = 0; head < 32; head++) {
            double probabilities[5], sum = 0;
            for (size_t position = 0; position <= prefix + token; position++) {
                double dot = 0;
                for (size_t column = 0; column < 128; column++) {
                    dot += (double)q[token * 4096 + head * 128 + column] *
                        k[position * 1024 + (head / 4) * 128 + column];
                }
                probabilities[position] = exp(dot / sqrt(128));
                sum += probabilities[position];
            }
            for (size_t column = 0; column < 128; column++) {
                double expected = 0;
                for (size_t position = 0; position <= prefix + token; position++) {
                    expected += probabilities[position] / sum *
                        v[position * 1024 + (head / 4) * 128 + column];
                }
                CHECK(fabs(actual[token * 4096 + head * 128 + column] - expected) < 2e-6);
            }
        }
    }
}

static void test_rope(LuneletteMetalOwner *owner)
{
    const uint32_t positions[] = {0, 1, 16383, 16384, 16385};
    const size_t indices[] = {1, 24, 40, 63};
    const size_t count = 5, width = 128;
    owner->buffers[METAL_COS] = buffer(owner, count * 128);
    owner->buffers[METAL_SIN] = buffer(owner, count * 128);
    owner->buffers[METAL_POSITIONS] = buffer(owner, count);
    memcpy(owner->buffers[METAL_POSITIONS].contents, positions, sizeof positions);
    id<MTLBuffer> input = buffer(owner, count * width);
    id<MTLBuffer> output = buffer(owner, count * width);
    float *x = input.contents;
    for (size_t i = 0; i < count * width; i++) {
        x[i] = ((int)(i % 23) - 11) * 0.125f;
    }
    float *cosine = owner->buffers[METAL_COS].contents;
    float *sine = owner->buffers[METAL_SIN].contents;
    for (size_t token = 0; token < count; token++) {
        CHECK(lunelette_metal_yarn(positions[token], cosine + token * 128,
            sine + token * 128) == 0);
        for (size_t j = 0; j < 4; j++) {
            CHECK(fabsf(cosine[token * 128 + indices[j]] - rope_golden[token][j][0]) < 2e-6f);
            CHECK(fabsf(sine[token * 128 + indices[j]] - rope_golden[token][j][1]) < 2e-6f);
        }
    }
    float unchanged[128];
    memcpy(unchanged, cosine, sizeof unchanged);
    CHECK(lunelette_metal_yarn(LUNELETTE_MODEL_MAX_POSITION, cosine, sine) != 0);
    CHECK(memcmp(unchanged, cosine, sizeof unchanged) == 0);
    for (uint32_t query = 0; query < 2; query++) {
        id<MTLCommandBuffer> command = [owner->queue commandBuffer];
        CHECK(lunelette_metal_encode_operator(owner, command, METAL_ROPE,
            input, owner->buffers[METAL_COS], output, width, count, query, count, 0) == 0);
        complete(command);
        const float *actual = output.contents;
        for (size_t token = 0; token < count; token++) {
            for (size_t column = 0; column < width; column++) {
                size_t i = token * width + column;
                float rotated = column < 64 ? -x[i + 64] : x[i - 64];
                double expected = (double)x[i] * cosine[i] + (double)rotated * sine[i];
                if (query) {
                    expected *= 1 + 0.1 * log(1 + positions[token] / 16384);
                }
                CHECK(fabs(actual[i] - expected) < 2e-6);
            }
        }
    }
}

static void test_state(LuneletteMetalOwner *owner)
{
    owner->prefill_mode = LUNELETTE_METAL_PREFILL_CUSTOM;
    /* Borrow an owner in the internal wrapper: never pass this wrapper to destroy. */
    lunelette_metal context = {(__bridge void *)owner};
    const float *logits = (const float *)1;
    char error[512];
    int64_t ids[257] = {0};
    owner->valid = true;
    owner->length = 0;
    CHECK(lunelette_metal_prefill(&context, ids, 0, &logits, error, sizeof error) != 0);
    CHECK(logits == NULL && owner->length == 0 && owner->valid);
    CHECK(lunelette_metal_prefill(&context, ids, 257, &logits, error, sizeof error) != 0);
    ids[0] = -1;
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) != 0);
    ids[0] = LUNELETTE_MODEL_VOCAB;
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) != 0);
    CHECK(owner->valid && owner->length == 0);
    CHECK(lunelette_metal_decode(&context, 0, &logits, error, sizeof error) != 0);
    owner->length = 256;
    CHECK(lunelette_metal_decode(&context, 0, &logits, error, sizeof error) != 0);
    CHECK(owner->length == 256 && owner->valid);
    owner->buffers[METAL_X] = buffer(owner, 3072);
    owner->buffers[METAL_COS] = buffer(owner, 128);
    owner->buffers[METAL_SIN] = buffer(owner, 128);
    owner->buffers[METAL_POSITIONS] = buffer(owner, 1);
    owner->weights[0] = buffer(owner, 3072);
    memset(owner->weights[0].contents, 0, 3072 * sizeof(float));
    id<MTLCommandQueue> real_queue = owner->queue;
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    float *scratch = owner->buffers[METAL_X].contents;
    scratch[0] = 123.0f;
    CHECK(lunelette_metal_prefill(&context, ids, 1, NULL, error, sizeof error) != 0);
    CHECK(scratch[0] == 123.0f && owner->valid && owner->length == 0);
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    CHECK(scratch[0] == 123.0f);
    owner->queue = nil;
    ids[0] = 0;
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) != 0);
    CHECK(!owner->valid && owner->length == 0 && logits == NULL);
    owner->queue = real_queue;
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) != 0);
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    owner->queue = (id<MTLCommandQueue>)[NilEncoderQueue new];
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) != 0);
    CHECK(!owner->valid && logits == NULL);
    owner->queue = real_queue;
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    CHECK(owner->valid && owner->length == 0);
    CHECK(lunelette_metal_encode(owner, (id<MTLCommandBuffer>)[NilEncoderCommand new],
        1, error, sizeof error) != 0);
    SyntheticCommand *synthetic = [SyntheticCommand new];
    CHECK(lunelette_metal_encode(owner, (id<MTLCommandBuffer>)synthetic,
        1, error, sizeof error) == 0);
    size_t encoder_count = synthetic->calls;
    for (size_t fail_at = 1; fail_at <= encoder_count; fail_at++) {
        synthetic->calls = 0;
        synthetic->fail_at = fail_at;
        CHECK(lunelette_metal_encode(owner, (id<MTLCommandBuffer>)synthetic,
            1, error, sizeof error) != 0);
    }
    synthetic->fail_at = 0;
    synthetic->completion = MTLCommandBufferStatusError;
    SyntheticQueue *synthetic_queue = [SyntheticQueue new];
    synthetic_queue->command = synthetic;
    owner->queue = (id<MTLCommandQueue>)synthetic_queue;
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) != 0);
    CHECK(!owner->valid && owner->length == 0 && logits == NULL);
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    synthetic->completion = MTLCommandBufferStatusCompleted;
    owner->buffers[METAL_LOGITS] = buffer(owner, LUNELETTE_MODEL_VOCAB);
    float *output = owner->buffers[METAL_LOGITS].contents;
    memset(output, 0, LUNELETTE_MODEL_VOCAB * sizeof(float));
    output[LUNELETTE_MODEL_VOCAB - 1] = INFINITY;
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) != 0);
    CHECK(!owner->valid && owner->length == 0 && logits == NULL);
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    output[LUNELETTE_MODEL_VOCAB - 1] = 0;
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) == 0);
    CHECK(owner->valid && owner->length == 1 && logits == output);
    /* Retired CPU cache tests covered retryable late I/O failure. Here argument
     * rejection preserves a prefix, but execution failure invalidates it until
     * reset. Never import the former CPU rollback contract into this engine. */
    CHECK(lunelette_metal_decode(&context, -1, &logits, error, sizeof error) != 0);
    CHECK(owner->valid && owner->length == 1 && logits == NULL);
    CHECK(lunelette_metal_decode(&context, LUNELETTE_MODEL_VOCAB,
        &logits, error, sizeof error) != 0);
    CHECK(owner->valid && owner->length == 1 && logits == NULL);
    synthetic->completion = MTLCommandBufferStatusError;
    CHECK(lunelette_metal_decode(&context, 0, &logits, error, sizeof error) != 0);
    CHECK(!owner->valid && owner->length == 1 && logits == NULL);
    synthetic->completion = MTLCommandBufferStatusCompleted;
    CHECK(lunelette_metal_decode(&context, 0, &logits, error, sizeof error) != 0);
    CHECK(!owner->valid && logits == NULL);
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    CHECK(owner->valid && owner->length == 0);
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) == 0);
    CHECK(owner->valid && owner->length == 1 && logits == output);
    /* Model arithmetic is mocked; the bounded GPU KV-copy check below separately
     * verifies retained-prefix memory, not model outputs. */
    lunelette_metal_checkpoint *first = NULL, *second = NULL, *failed = NULL;
    CHECK(owner->endpoint_logits_valid);
    output[0] = -0.0f;
    output[LUNELETTE_MODEL_VOCAB - 1] = 17.25f;
    CHECK(lunelette_metal_checkpoint_create(&context, &first, error, sizeof error) == 0);
    CHECK(first != NULL);
    float *saved_output = malloc(LUNELETTE_MODEL_VOCAB * sizeof(float));
    CHECK(saved_output != NULL);
    memcpy(saved_output, output, LUNELETTE_MODEL_VOCAB * sizeof(float));
    size_t before_calls = synthetic->calls;
    CHECK(lunelette_metal_checkpoint_restore(&context, first, &logits, error, sizeof error) == 0);
    CHECK(synthetic->calls == before_calls && signbit(logits[0]));
    CHECK(logits[LUNELETTE_MODEL_VOCAB - 1] == 17.25f);
    owner->buffers[METAL_X] = buffer(owner, 2 * 3072);
    owner->buffers[METAL_COS] = buffer(owner, 2 * 128);
    owner->buffers[METAL_SIN] = buffer(owner, 2 * 128);
    owner->buffers[METAL_POSITIONS] = buffer(owner, 2);
    int64_t suffix[] = {0, 0};
    CHECK(lunelette_metal_append(&context, suffix, 0, &logits, error, sizeof error) != 0);
    suffix[1] = -1;
    CHECK(lunelette_metal_append(&context, suffix, 2, &logits, error, sizeof error) != 0);
    CHECK(owner->length == 1 && owner->endpoint_logits_valid);
    suffix[1] = 0;
    /* Execute production-selected KV blits on a real Metal command while model
     * compute remains no-op. Check all layers' prefix, suffix and tail bytes. */
    owner->buffers[METAL_ROTATED_K] = buffer(owner, 2 * LUNELETTE_MODEL_KV);
    owner->buffers[METAL_V] = buffer(owner, 2 * LUNELETTE_MODEL_KV);
    for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
        owner->keys[layer] = buffer(owner, 4 * LUNELETTE_MODEL_KV + 4);
        owner->values[layer] = buffer(owner, 4 * LUNELETTE_MODEL_KV + 4);
        memset(owner->keys[layer].contents, 0x5a, owner->keys[layer].length);
        memset(owner->values[layer].contents, 0x5a, owner->values[layer].length);
    }
    for (unsigned branch = 0; branch < 2; branch++) {
        CHECK(lunelette_metal_checkpoint_restore(&context, first, &logits, error, sizeof error) == 0);
        memset(owner->buffers[METAL_ROTATED_K].contents, 0x20 + branch,
            owner->buffers[METAL_ROTATED_K].length);
        memset(owner->buffers[METAL_V].contents, 0x40 + branch,
            owner->buffers[METAL_V].length);
        synthetic->kv_command = [real_queue commandBuffer];
        CHECK(synthetic->kv_command);
        CHECK(lunelette_metal_append(&context, suffix, 2, &logits, error, sizeof error) == 0);
        synthetic->kv_command = nil;
        for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
            id<MTLBuffer> caches[] = {owner->keys[layer], owner->values[layer]};
            for (unsigned kind = 0; kind < 2; kind++) {
                unsigned char *bytes = caches[kind].contents;
                size_t token_bytes = LUNELETTE_MODEL_KV * sizeof(float);
                for (size_t i = 0; i < caches[kind].length; i++) {
                    unsigned char expected = i >= token_bytes && i < 3 * token_bytes ?
                        (kind ? 0x40 : 0x20) + branch : 0x5a;
                    CHECK(bytes[i] == expected);
                }
            }
        }
    }
    CHECK(lunelette_metal_checkpoint_restore(&context, first, &logits, error, sizeof error) == 0);
    puts("Checkpoint restore/rebranch real GPU KV blits: 26 layers, prefix/suffix/tail PASS (compute mocked)");
    synthetic->calls = 0;
    suffix_copy_count = 0;
    inspect_suffix_copies = true;
    CHECK(lunelette_metal_append(&context, suffix, 2, &logits, error, sizeof error) == 0);
    inspect_suffix_copies = false;
    CHECK(suffix_copy_count == 2 * LUNELETTE_MODEL_LAYERS);
    CHECK(synthetic->calls == encoder_count); /* One batched traversal, not two decodes. */
    CHECK(owner->length == 3);
    uint32_t *positions = owner->buffers[METAL_POSITIONS].contents;
    CHECK(positions[0] == 1 && positions[1] == 2);
    output[0] = 9;
    CHECK(lunelette_metal_checkpoint_restore(&context, first, &logits, error, sizeof error) == 0);
    CHECK(owner->length == 1 && signbit(logits[0]));
    CHECK(memcmp(logits, saved_output, LUNELETTE_MODEL_VOCAB * sizeof(float)) == 0);
    CHECK(lunelette_metal_decode(&context, 0, &logits, error, sizeof error) == 0);
    CHECK(lunelette_metal_checkpoint_restore(&context, first, &logits, error, sizeof error) == 0);
    CHECK(owner->length == 1 && signbit(logits[0]));
    CHECK(memcmp(logits, saved_output, LUNELETTE_MODEL_VOCAB * sizeof(float)) == 0);
    LuneletteMetalOwner *other = [LuneletteMetalOwner new];
    CHECK(other->identity != 0 && other->identity != owner->identity);
    lunelette_metal wrong = {(__bridge void *)other};
    CHECK(lunelette_metal_checkpoint_restore(&wrong, first, &logits, error, sizeof error) != 0);
    CHECK(lunelette_metal_checkpoint_create(&context, &second, error, sizeof error) == 0);
    CHECK(lunelette_metal_checkpoint_restore(&context, first, &logits, error, sizeof error) != 0);
    CHECK(owner->length == 1 && owner->endpoint_logits_valid);
    uint64_t epoch = owner->checkpoint_epoch;
    fail_checkpoint_allocation = true;
    CHECK(lunelette_metal_checkpoint_create(&context, &failed, error, sizeof error) != 0);
    fail_checkpoint_allocation = false;
    CHECK(failed == NULL && owner->active_checkpoint == second && owner->checkpoint_epoch == epoch);
    CHECK(owner->valid && owner->endpoint_logits_valid && owner->length == 1);
    CHECK(lunelette_metal_checkpoint_restore(&context, second, &logits, error, sizeof error) == 0);
    CHECK(!memcmp(logits, saved_output, LUNELETTE_MODEL_VOCAB * sizeof(float)));
    lunelette_metal_checkpoint_destroy(second);
    CHECK(owner->active_checkpoint == NULL && owner->endpoint_logits_valid);
    CHECK(lunelette_metal_decode(&context, 0, &logits, error, sizeof error) == 0);
    CHECK(lunelette_metal_checkpoint_create(&context, &second, error, sizeof error) == 0);
    epoch = owner->checkpoint_epoch;
    owner->checkpoint_epoch = UINT64_MAX;
    CHECK(lunelette_metal_checkpoint_create(&context, &failed, error, sizeof error) != 0);
    CHECK(failed == NULL && owner->active_checkpoint == second);
    owner->checkpoint_epoch = epoch;
    CHECK(lunelette_metal_checkpoint_restore(&context, second, &logits, error, sizeof error) == 0);
    output[LUNELETTE_MODEL_VOCAB - 1] = NAN;
    CHECK(lunelette_metal_checkpoint_create(&context, &failed, error, sizeof error) != 0);
    CHECK(failed == NULL && !owner->valid && !owner->endpoint_logits_valid);
    CHECK(owner->active_checkpoint == NULL);
    CHECK(lunelette_metal_checkpoint_restore(&context, second, &logits, error, sizeof error) != 0);
    CHECK(logits == NULL);
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    output[LUNELETTE_MODEL_VOCAB - 1] = 17.25f;
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) == 0);
    lunelette_metal_checkpoint_destroy(second);
    CHECK(lunelette_metal_checkpoint_create(&context, &second, error, sizeof error) == 0);
    synthetic->completion = MTLCommandBufferStatusError;
    CHECK(lunelette_metal_append(&context, suffix, 2, &logits, error, sizeof error) != 0);
    CHECK(!owner->valid && !owner->endpoint_logits_valid && owner->active_checkpoint == NULL);
    CHECK(lunelette_metal_checkpoint_restore(&context, second, &logits, error, sizeof error) != 0);
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    CHECK(owner->valid && !owner->endpoint_logits_valid);
    CHECK(lunelette_metal_checkpoint_create(&context, &failed, error, sizeof error) != 0);
    synthetic->completion = MTLCommandBufferStatusCompleted;
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) == 0);
    CHECK(lunelette_metal_checkpoint_create(&context, &failed, error, sizeof error) == 0);
    CHECK(lunelette_metal_reset(&context, error, sizeof error) == 0);
    CHECK(lunelette_metal_checkpoint_restore(&context, failed, &logits, error, sizeof error) != 0);
    /* Capacity validation is host-only; no KV copies execute for these cases. */
    CHECK(lunelette_metal_prefill(&context, ids, 1, &logits, error, sizeof error) == 0);
    owner->length = 254;
    CHECK(lunelette_metal_append(&context, suffix, 2, &logits, error, sizeof error) == 0);
    CHECK(owner->length == 256 && positions[0] == 254 && positions[1] == 255);
    CHECK(lunelette_metal_append(&context, suffix, 1, &logits, error, sizeof error) != 0);
    CHECK(owner->valid && owner->endpoint_logits_valid && owner->length == 256);
    free(saved_output);
    lunelette_metal_checkpoint_destroy(first);
    lunelette_metal_checkpoint_destroy(second);
    lunelette_metal_checkpoint_destroy(failed);
    lunelette_metal_checkpoint_destroy(NULL);
    owner->queue = real_queue;
    lunelette_metal_destroy(NULL);
    CHECK(lunelette_metal_reset(NULL, NULL, 0) != 0);
}

int main(void)
{
    @autoreleasepool {
        char error[512];
        LuneletteMetalOwner *owner = [LuneletteMetalOwner new];
        owner->device = MTLCreateSystemDefaultDevice();
        CHECK(owner->device != nil);
        CHECK(lunelette_metal_create_pipelines(owner, error, sizeof error) == 0);
        test_options();
        test_mps_cache();
        test_projection(owner);
        test_mps_projection(owner);
        test_norm_element(owner);
        test_attention(owner);
        test_rope(owner);
        test_state(owner);
        puts("Metal synthetic kernels/state passed (not full-model validation)");
    }
    return 0;
}
