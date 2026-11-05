/* DS4 frozen-legacy + same-input + analytical-oracle testing method.
 * See frozen/README.md and frozen/DS4-MIT.txt. No DS4 kernel is used.
 * This is native regression evidence, not independent model validation. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "metal_shader.h"

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

static id<MTLLibrary> library(id<MTLDevice> device, NSString *source)
{
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    options.languageVersion = MTLLanguageVersion3_2;
    NSError *error = nil;
    id<MTLLibrary> result = [device newLibraryWithSource:source options:options error:&error];
    if (!result) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
    CHECK(result);
    return result;
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
    id<MTLLibrary> source, NSString *name)
{
    NSError *error = nil;
    id<MTLFunction> function = [source newFunctionWithName:name];
    CHECK(function);
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    CHECK(result);
    CHECK(result.threadExecutionWidth == 32);
    return result;
}

static void encode(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipe,
    id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
    const uint32_t dims[4], size_t groups)
{
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:pipe];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBuffer:output offset:16 atIndex:2];
    [encoder setBytes:dims length:16 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
}

static float value(unsigned pattern, size_t i)
{
    const uint32_t special[] = {0, 0x80000000, 1, 0x80000001,
        0x007fffff, 0x00800000, 0x7f800000, 0xff800000, 0x7fc00125};
    if (pattern == 0) return 1;
    if (pattern <= 9) {
        float result;
        uint32_t bits = special[pattern - 1];
        memcpy(&result, &bits, 4);
        return result;
    }
    if (pattern == 10) return (i % 4 == 0 ? 0x1p24f :
        i % 4 == 2 ? -0x1p24f : 0.25f);
    return (float)((int)((i * 7919 + 313) % 257) - 128) / 64;
}

static void run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> reference, id<MTLComputePipelineState> candidate,
    bool norm, unsigned k, unsigned rows, unsigned tokens, unsigned pattern)
{
    size_t nw = norm ? k : (size_t)k * rows;
    size_t ni = (size_t)k * tokens;
    size_t no = (size_t)(norm ? k : rows) * tokens;
    id<MTLBuffer> weights = [device newBufferWithLength:nw * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> input = [device newBufferWithLength:ni * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> expected = [device newBufferWithLength:(no + 8) * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> actual = [device newBufferWithLength:(no + 8) * 4 options:MTLResourceStorageModeShared];
    CHECK(weights && input && expected && actual);
    float *w = weights.contents, *x = input.contents;
    /* Keep pattern zero as unit weights; other norm cases exercise the scale,
     * including zero, negative and unequal magnitudes even at K=1. */
    const float norm_weights[] = {0, -2, 0.125f, 3, -0.5f, 8, -4};
    for (size_t i = 0; i < nw; ++i) {
        w[i] = pattern == 0 ? 1 : norm ?
            norm_weights[(i + pattern) % 7] : value(11, i);
    }
    for (size_t i = 0; i < ni; ++i) x[i] = value(pattern, i);
    NSData *saved_w = [NSData dataWithBytes:w length:nw * 4];
    NSData *saved_x = [NSData dataWithBytes:x length:ni * 4];
    uint32_t *a = actual.contents, *e = expected.contents;
    const uint32_t canary = 0x7f24d13b;
    for (size_t i = 0; i < no + 8; ++i) a[i] = e[i] = canary;
    uint32_t dims[4] = {norm ? k : rows, k, tokens, 0};
    id<MTLCommandBuffer> command = [queue commandBuffer];
    CHECK(command);
    encode(command, reference, weights, input, expected, dims, norm ? tokens : rows * tokens);
    encode(command, candidate, weights, input, actual, dims, norm ? tokens : rows * tokens);
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted);
    CHECK(!memcmp(w, saved_w.bytes, nw * 4));
    CHECK(!memcmp(x, saved_x.bytes, ni * 4));
    for (size_t i = 0; i < no + 8; ++i) {
        if (i < 4 || i >= no + 4) {
            CHECK(a[i] == canary && e[i] == canary);
            continue;
        }
        bool an = (a[i] & 0x7fffffff) > 0x7f800000;
        bool en = (e[i] & 0x7fffffff) > 0x7f800000;
        if (!(an && en) && a[i] != e[i]) {
            fprintf(stderr, "DIFF norm=%d K=%u rows=%u tokens=%u pattern=%u index=%zu %08x %08x\n",
                norm, k, rows, tokens, pattern, i - 4, e[i], a[i]);
            CHECK(false);
        }
        if (!norm && pattern == 0) {
            float av, ev;
            memcpy(&av, &a[i], 4);
            memcpy(&ev, &e[i], 4);
            CHECK(av == (float)k && ev == (float)k);
        }
    }
}

/* Optional one-operation diagnostic: raw little-endian FP32 rows, exact sizes.
 * No model loading or weight reconstruction occurs in this helper. */
static int norm_diagnostic(int argc, char **argv)
{
    CHECK(argc == 7);
    char *end = NULL;
    unsigned long columns = strtoul(argv[5], &end, 10);
    CHECK(*argv[5] && !*end && columns > 0 && columns <= 65536);
    unsigned long rows = strtoul(argv[6], &end, 10);
    CHECK(*argv[6] && !*end && rows > 0 && rows <= 256);
    NSData *input_data = [NSData dataWithContentsOfFile:@(argv[2])];
    NSData *weight_data = [NSData dataWithContentsOfFile:@(argv[3])];
    CHECK(input_data.length == columns * rows * 4);
    CHECK(weight_data.length == columns * 4);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        puts("SKIP norm diagnostic: no GPU device");
        return 77;
    }
    NSString *source = [[NSString alloc] initWithBytes:lunelette_metal_shader
        length:sizeof lunelette_metal_shader - 1 encoding:NSUTF8StringEncoding];
    CHECK(source);
    id<MTLComputePipelineState> pipe = pipeline(device, library(device, source), @"norm");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    CHECK(queue);
    id<MTLBuffer> input = [device newBufferWithBytes:input_data.bytes
        length:input_data.length options:MTLResourceStorageModeShared];
    id<MTLBuffer> weights = [device newBufferWithBytes:weight_data.bytes
        length:weight_data.length options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:input_data.length + 32
        options:MTLResourceStorageModeShared];
    CHECK(input && weights && output);
    uint32_t *words = output.contents;
    for (size_t i = 0; i < output.length / 4; ++i) words[i] = 0x7f24d13b;
    uint32_t dims[4] = {(uint32_t)columns, (uint32_t)columns, (uint32_t)rows, 0};
    id<MTLCommandBuffer> command = [queue commandBuffer];
    CHECK(command);
    encode(command, pipe, weights, input, output, dims, rows);
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted);
    for (size_t i = 0; i < 4; ++i) {
        CHECK(words[i] == 0x7f24d13b);
        CHECK(words[4 + columns * rows + i] == 0x7f24d13b);
    }
    CHECK(!memcmp(input.contents, input_data.bytes, input_data.length));
    CHECK(!memcmp(weights.contents, weight_data.bytes, weight_data.length));
    FILE *file = fopen(argv[4], "wx");
    CHECK(file);
    CHECK(fwrite(words + 4, 1, input_data.length, file) == input_data.length);
    CHECK(fclose(file) == 0);
    puts("PASS norm diagnostic dispatch (numerical comparison is external)");
    return 0;
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        if (argc > 1 && !strcmp(argv[1], "--norm")) return norm_diagnostic(argc, argv);
        CHECK(argc == 2 || argc == 3);
        NSData *frozen = [NSData dataWithContentsOfFile:@(argv[1])];
        CHECK(frozen && frozen.length <= UINT32_MAX);
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(frozen.bytes, (CC_LONG)frozen.length, digest);
        char hex[65];
        for (unsigned i = 0; i < 32; ++i) snprintf(hex + 2 * i, 3, "%02x", digest[i]);
        CHECK(!strcmp(hex, "f49e63914633c3ac03ca7f232f11d3b27fd0573a5cd9a9c818073c2510b9f256"));
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            puts("SKIP frozen Metal regression: no device; zero GPU cases executed");
            return 0;
        }
        NSString *body = [[NSString alloc] initWithData:frozen encoding:NSUTF8StringEncoding];
        CHECK(body);
        NSString *preamble = @"#define LUNELETTE_MODEL_NORM_EPS 1e-5f\n"
            "#define LUNELETTE_MODEL_QUERY_SCALE 0.1f\n"
            "#define LUNELETTE_MODEL_ORIGINAL_CONTEXT 16384\n";
        id<MTLLibrary> legacy = library(device, [preamble stringByAppendingString:body]);
        NSString *current = [[NSString alloc] initWithBytes:lunelette_metal_shader
            length:sizeof lunelette_metal_shader - 1 encoding:NSUTF8StringEncoding];
        CHECK(current);
        /* Explicit candidate override is for harness fault-injection checks only. */
        if (argc == 3) {
            NSString *override = [NSString stringWithContentsOfFile:@(argv[2])
                encoding:NSUTF8StringEncoding error:NULL];
            CHECK(override);
            current = [preamble stringByAppendingString:override];
        }
        id<MTLLibrary> production = library(device, current);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        CHECK(queue);
        unsigned cases = 0;
        const unsigned sizes[] = {1, 7, 31, 32, 33, 63, 65, 3072};
        for (unsigned family = 0; family < 2; ++family) {
            NSString *name = family ? @"norm" : @"project";
            id<MTLComputePipelineState> ref = pipeline(device, legacy, name);
            id<MTLComputePipelineState> cand = pipeline(device, production, name);
            for (unsigned s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
                for (unsigned pattern = 0; pattern < 12; ++pattern) {
                    run_case(device, queue, ref, cand, family != 0, sizes[s],
                        s % 2 ? 3 : 1, s % 3 + 1, pattern);
                    ++cases;
                }
            }
        }
        printf("PASS frozen project/norm: %u GPU cases; non-NaN bits, NaN class, guards, immutable inputs, exact integer dots\n", cases);
    }
    return 0;
}
