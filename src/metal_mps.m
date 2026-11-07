#include "metal_mps.h"
#include <stdio.h>

/* One shape, seven projections per layer. Views retain only existing buffers;
 * no owner back-reference, repacking, or additional weight storage. */
@implementation LuneletteMPSCache
@end

MPSMatrix *lunelette_metal_mps_view(id<MTLBuffer> buffer, size_t rows, size_t columns)
{
    /* Validate before invoking MPS: malformed descriptors may raise exceptions.
     * Runtime dimensions are bounded constants, but check byte arithmetic too. */
    if (!buffer || !rows || !columns || columns > SIZE_MAX / sizeof(float) ||
        rows > SIZE_MAX / (columns * sizeof(float)) ||
        rows * columns * sizeof(float) > buffer.length) {
        return nil;
    }
    MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
        columns:columns rowBytes:columns * sizeof(float) dataType:MPSDataTypeFloat32];
    if (!descriptor) {
        return nil;
    }
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
}

MPSMatrixMultiplication *lunelette_metal_mps_kernel(id<MTLDevice> device,
    size_t count, size_t rows, size_t columns)
{
    if (!device || !MPSSupportsMTLDevice(device) || !count ||
        count > LUNELETTE_BLOCK_LIMIT || !rows ||
        rows > LUNELETTE_MODEL_INNER || !columns || columns > LUNELETTE_MODEL_INNER) {
        return nil;
    }
    return [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO
        transposeRight:YES resultRows:count resultColumns:rows interiorColumns:columns
        alpha:1.0 beta:0.0];
}

int lunelette_metal_mps_prepare(LuneletteMetalOwner *owner, size_t count,
    char *error, size_t capacity)
{
    LuneletteMPSCache *old = owner->mps_cache;
    if (!owner->device || count < 2 || count > owner->limits.block) {
        snprintf(error, capacity, "MPS invalid device/prefill shape; reset required");
        return -1;
    }
    if (old && old->count == count) {
        return 0;
    }
    const size_t slots[7] = {1, 2, 3, 4, 6, 7, 8};
    const int input[7] = {METAL_INPUT_NORM, METAL_INPUT_NORM, METAL_INPUT_NORM,
        METAL_ATTENTION, METAL_POST_NORM, METAL_POST_NORM, METAL_GATED};
    const int output[7] = {METAL_Q, METAL_K, METAL_V, METAL_O, METAL_GATE,
        METAL_UP, METAL_DOWN};
    const size_t rows[7] = {LUNELETTE_MODEL_QUERY, LUNELETTE_MODEL_KV,
        LUNELETTE_MODEL_KV, LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_INNER,
        LUNELETTE_MODEL_INNER, LUNELETTE_MODEL_HIDDEN};
    const size_t columns[7] = {LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_HIDDEN,
        LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_QUERY, LUNELETTE_MODEL_HIDDEN,
        LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_INNER};
    LuneletteMPSCache *next = [LuneletteMPSCache new];
    if (!next) {
        goto failure;
    }
    next->count = count;
    for (size_t p = 0; p < 7; p++) {
        next->kernels[p] = lunelette_metal_mps_kernel(owner->device,
            count, rows[p], columns[p]);
        if (!next->kernels[p]) {
            goto failure;
        }
        for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
            size_t weight = 2 + layer * LUNELETTE_MODEL_BLOCK_WEIGHTS + slots[p];
            next->inputs[weight] = lunelette_metal_mps_view(owner->buffers[input[p]],
                count, columns[p]);
            next->weights[weight] = lunelette_metal_mps_view(
                owner->memory.strategy == LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC ?
                    owner->conversion_stage : owner->weights[weight],
                rows[p], columns[p]);
            next->outputs[weight] = lunelette_metal_mps_view(owner->buffers[output[p]],
                count, rows[p]);
            if (!next->inputs[weight] || !next->weights[weight] || !next->outputs[weight]) {
                goto failure;
            }
        }
    }
    /* Publish only a complete shape, before any command encoding starts. ARC
     * releases the previous views here; reset can reuse the current shape. */
    owner->mps_cache = next;
    return 0;
failure:
    snprintf(error, capacity, "MPS prefill resource allocation failed; reset required; "
        "select --metal-prefill custom to use precise custom projections");
    return -1;
}

int lunelette_metal_mps_project(LuneletteMetalOwner *owner, id<MTLCommandBuffer> command,
    size_t weight)
{
    LuneletteMPSCache *cache = owner->mps_cache;
    if (!cache || !command || weight >= LUNELETTE_METAL_WEIGHT_COUNT ||
        !cache->inputs[weight]) {
        return -1;
    }
    size_t slot = (weight - 2) % LUNELETTE_MODEL_BLOCK_WEIGHTS;
    size_t p = slot <= 4 ? slot - 1 : slot - 2;
    [cache->kernels[p] encodeToCommandBuffer:command leftMatrix:cache->inputs[weight]
        rightMatrix:cache->weights[weight] resultMatrix:cache->outputs[weight]];
    return 0;
}
