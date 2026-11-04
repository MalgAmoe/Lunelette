#include "metal_internal.h"
#include <stdio.h>

/* All operators encode into one command; no intermediate CPU reads or waits. */
int lunelette_metal_encode_operator(LuneletteMetalOwner *owner, id<MTLCommandBuffer> command,
    int operation, id<MTLBuffer> first, id<MTLBuffer> second, id<MTLBuffer> output,
    uint32_t x, uint32_t y, uint32_t z, size_t count, size_t input_offset)
{
    id<MTLComputePipelineState> pipeline = owner->pipelines[operation];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) {
        return -1;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:first offset:0 atIndex:0];
    [encoder setBuffer:second offset:input_offset atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    uint32_t dimensions[4] = {x, y, z, (uint32_t)owner->length};
    [encoder setBytes:dimensions length:sizeof dimensions atIndex:3];
    if (operation == METAL_ROPE) {
        [encoder setBuffer:owner->buffers[METAL_SIN] offset:0 atIndex:4];
        [encoder setBuffer:owner->buffers[METAL_POSITIONS] offset:0 atIndex:5];
    }
    bool groups = operation == METAL_PROJECT || operation == METAL_NORM ||
        operation == METAL_SCORES;
    NSUInteger threads;
    if (operation == METAL_PROJECT) {
        threads = (NSUInteger)x * z;
    } else if (operation == METAL_NORM) {
        threads = count;
    } else if (operation == METAL_SCORES) {
        threads = count * LUNELETTE_MODEL_QUERY_HEADS;
    } else if (operation == METAL_ROPE) {
        threads = (NSUInteger)x * y;
    } else if (operation == METAL_ATTEND) {
        threads = count * LUNELETTE_MODEL_QUERY;
    } else {
        threads = x;
    }
    if (groups) {
        [encoder dispatchThreadgroups:MTLSizeMake(threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    } else {
        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(pipeline.maxTotalThreadsPerThreadgroup < 256 ?
                pipeline.maxTotalThreadsPerThreadgroup : 256, 1, 1)];
    }
    [encoder endEncoding];
    return 0;
}

static int project(LuneletteMetalOwner *owner, id<MTLCommandBuffer> command,
    size_t weight, int input, int output, uint32_t rows, uint32_t cols, size_t count)
{
    if (count > 1 && owner->prefill_mode == LUNELETTE_METAL_PREFILL_MPS) {
        return lunelette_metal_mps_project(owner, command, weight);
    }
    return lunelette_metal_encode_operator(owner, command, METAL_PROJECT, owner->weights[weight],
        owner->buffers[input], owner->buffers[output], rows, cols, (uint32_t)count,
        count, 0);
}

static int element(LuneletteMetalOwner *owner, id<MTLCommandBuffer> command,
    int first, int second, int output, uint32_t width, uint32_t mode, size_t count)
{
    return lunelette_metal_encode_operator(owner, command, METAL_ELEMENT, owner->buffers[first],
        owner->buffers[second], owner->buffers[output], (uint32_t)count * width,
        mode, 0, count, 0);
}

static int encode_layer(LuneletteMetalOwner *owner, id<MTLCommandBuffer> command,
    size_t layer, size_t count)
{
    size_t weight = 2 + layer * LUNELETTE_MODEL_BLOCK_WEIGHTS;
    id<MTLBuffer> __strong *b = owner->buffers;
    if (lunelette_metal_encode_operator(owner, command, METAL_NORM, owner->weights[weight],
            b[METAL_X], b[METAL_INPUT_NORM], LUNELETTE_MODEL_HIDDEN, 0, 0, count, 0) ||
        project(owner, command, weight + 1, METAL_INPUT_NORM, METAL_Q,
            LUNELETTE_MODEL_QUERY, LUNELETTE_MODEL_HIDDEN, count) ||
        project(owner, command, weight + 2, METAL_INPUT_NORM, METAL_K,
            LUNELETTE_MODEL_KV, LUNELETTE_MODEL_HIDDEN, count) ||
        project(owner, command, weight + 3, METAL_INPUT_NORM, METAL_V,
            LUNELETTE_MODEL_KV, LUNELETTE_MODEL_HIDDEN, count) ||
        lunelette_metal_encode_operator(owner, command, METAL_ROPE, b[METAL_Q], b[METAL_COS],
            b[METAL_ROTATED_Q], LUNELETTE_MODEL_QUERY, (uint32_t)count, 1, count, 0) ||
        lunelette_metal_encode_operator(owner, command, METAL_ROPE, b[METAL_K], b[METAL_COS],
            b[METAL_ROTATED_K], LUNELETTE_MODEL_KV, (uint32_t)count, 0, count, 0)) {
        return -1;
    }
    id<MTLBlitCommandEncoder> copy = [command blitCommandEncoder];
    if (!copy) {
        return -1;
    }
    size_t offset = owner->length * LUNELETTE_MODEL_KV * sizeof(float);
    size_t bytes = count * LUNELETTE_MODEL_KV * sizeof(float);
    [copy copyFromBuffer:b[METAL_ROTATED_K] sourceOffset:0 toBuffer:owner->keys[layer]
        destinationOffset:offset size:bytes];
    [copy copyFromBuffer:b[METAL_V] sourceOffset:0 toBuffer:owner->values[layer]
        destinationOffset:offset size:bytes];
    [copy endEncoding];
    if (lunelette_metal_encode_operator(owner, command, METAL_SCORES, b[METAL_ROTATED_Q],
            owner->keys[layer], b[METAL_SCORE], (uint32_t)count, 0, 0, count, 0) ||
        lunelette_metal_encode_operator(owner, command, METAL_ATTEND, b[METAL_SCORE],
            owner->values[layer], b[METAL_ATTENTION], (uint32_t)count, 0, 0, count, 0) ||
        project(owner, command, weight + 4, METAL_ATTENTION, METAL_O,
            LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_QUERY, count) ||
        element(owner, command, METAL_X, METAL_O, METAL_RESIDUAL,
            LUNELETTE_MODEL_HIDDEN, 0, count) ||
        lunelette_metal_encode_operator(owner, command, METAL_NORM, owner->weights[weight + 5],
            b[METAL_RESIDUAL], b[METAL_POST_NORM], LUNELETTE_MODEL_HIDDEN,
            0, 0, count, 0) ||
        project(owner, command, weight + 6, METAL_POST_NORM, METAL_GATE,
            LUNELETTE_MODEL_INNER, LUNELETTE_MODEL_HIDDEN, count) ||
        project(owner, command, weight + 7, METAL_POST_NORM, METAL_UP,
            LUNELETTE_MODEL_INNER, LUNELETTE_MODEL_HIDDEN, count) ||
        element(owner, command, METAL_GATE, METAL_GATE, METAL_SILU,
            LUNELETTE_MODEL_INNER, 1, count) ||
        element(owner, command, METAL_SILU, METAL_UP, METAL_GATED,
            LUNELETTE_MODEL_INNER, 2, count) ||
        project(owner, command, weight + 8, METAL_GATED, METAL_DOWN,
            LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_INNER, count) ||
        element(owner, command, METAL_RESIDUAL, METAL_DOWN, METAL_HIDDEN,
            LUNELETTE_MODEL_HIDDEN, 0, count)) {
        return -1;
    }
    copy = [command blitCommandEncoder];
    if (!copy) {
        return -1;
    }
    [copy copyFromBuffer:b[METAL_HIDDEN] sourceOffset:0 toBuffer:b[METAL_X]
        destinationOffset:0 size:count * LUNELETTE_MODEL_HIDDEN * sizeof(float)];
    [copy endEncoding];
    return 0;
}

int lunelette_metal_encode(LuneletteMetalOwner *owner, id<MTLCommandBuffer> command,
    size_t count, char *error, size_t capacity)
{
    for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
        if (encode_layer(owner, command, layer, count) != 0) {
            snprintf(error, capacity, "Metal encoder creation failed in layer %zu; reset required",
                layer);
            return -1;
        }
    }
    if (lunelette_metal_encode_operator(owner, command, METAL_NORM, owner->weights[1],
            owner->buffers[METAL_X], owner->buffers[METAL_FINAL],
            LUNELETTE_MODEL_HIDDEN, 0, 0, count, 0) ||
        lunelette_metal_encode_operator(owner, command, METAL_PROJECT, owner->weights[0],
            owner->buffers[METAL_FINAL], owner->buffers[METAL_LOGITS],
            LUNELETTE_MODEL_VOCAB, LUNELETTE_MODEL_HIDDEN, 1, 1,
            (count - 1) * LUNELETTE_MODEL_HIDDEN * sizeof(float))) {
        snprintf(error, capacity, "Metal final encoder creation failed; reset required");
        return -1;
    }
    return 0;
}
