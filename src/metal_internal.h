#ifndef LUNELETTE_METAL_INTERNAL_H
#define LUNELETTE_METAL_INTERNAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal.h"
#include "metal_weights.h"
#include "runtime_limits.h"
#include <stdbool.h>

_Static_assert(LUNELETTE_MODEL_HIDDEN == 3072 && LUNELETTE_MODEL_QUERY == 4096 &&
    LUNELETTE_MODEL_KV == 1024 && LUNELETTE_MODEL_HEAD_DIM == 128 &&
    LUNELETTE_MODEL_QUERY_HEADS == 32 && LUNELETTE_MODEL_KV_HEADS == 8,
    "Update the fixed Metal shader ABI before changing model dimensions");

/* Concrete Metal layout, independent of CPU layer structs. Weight slots follow
 * model_facts: embedding, final norm, then nine weights per layer. */
enum { METAL_PROJECT, METAL_NORM, METAL_ELEMENT, METAL_ROPE, METAL_SCORES,
    METAL_ATTEND, METAL_PIPELINE_COUNT };
enum { METAL_X, METAL_INPUT_NORM, METAL_Q, METAL_K, METAL_V, METAL_ROTATED_Q,
    METAL_ROTATED_K, METAL_ATTENTION, METAL_O, METAL_RESIDUAL, METAL_POST_NORM,
    METAL_GATE, METAL_UP, METAL_SILU, METAL_GATED, METAL_DOWN, METAL_HIDDEN,
    METAL_FINAL, METAL_LOGITS, METAL_COS, METAL_SIN, METAL_POSITIONS, METAL_SCORE,
    METAL_BUFFER_COUNT };

@class LuneletteMPSCache;

@interface LuneletteMetalOwner : NSObject {
@public
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipelines[METAL_PIPELINE_COUNT];
    id<MTLBuffer> weights[LUNELETTE_METAL_WEIGHT_COUNT];
    id<MTLBuffer> buffers[METAL_BUFFER_COUNT];
    id<MTLBuffer> keys[LUNELETTE_MODEL_LAYERS];
    id<MTLBuffer> values[LUNELETTE_MODEL_LAYERS];
    lunelette_metal_prefill_mode prefill_mode;
    LuneletteMPSCache *mps_cache; /* Current shape only; cache never retains this owner. */
    size_t length;
    bool valid;
}
@end

struct lunelette_metal {
    void *owner;
};

int lunelette_metal_create_pipelines(LuneletteMetalOwner *owner, char *error, size_t capacity);
int lunelette_metal_encode_operator(LuneletteMetalOwner *owner, id<MTLCommandBuffer> command,
    int operation, id<MTLBuffer> first, id<MTLBuffer> second, id<MTLBuffer> output,
    uint32_t x, uint32_t y, uint32_t z, size_t count, size_t input_offset);

/* Error strings are caller-owned; no Objective-C exception is used for control flow. */
int lunelette_metal_encode(LuneletteMetalOwner *owner, id<MTLCommandBuffer> command,
    size_t count, char *error, size_t capacity);
int lunelette_metal_mps_prepare(LuneletteMetalOwner *owner, size_t count,
    char *error, size_t capacity);
int lunelette_metal_mps_project(LuneletteMetalOwner *owner, id<MTLCommandBuffer> command,
    size_t weight);
int lunelette_metal_yarn(uint32_t position, float cosine[128], float sine[128]);
#endif
