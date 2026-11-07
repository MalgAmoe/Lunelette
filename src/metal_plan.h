#ifndef LUNELETTE_METAL_PLAN_H
#define LUNELETTE_METAL_PLAN_H

#include "metal.h"
#include "runtime_limits.h"
#include "model_facts.h"
#include <stdint.h>

/* Slot order is the private Metal scratch ABI. Bytes are requested lengths,
 * not a physical peak. Keep the planner shared by accounting and allocation. */
_Static_assert(LUNELETTE_MODEL_HIDDEN == 3072 && LUNELETTE_MODEL_QUERY == 4096 &&
    LUNELETTE_MODEL_KV == 1024 && LUNELETTE_MODEL_INNER == 9216 &&
    LUNELETTE_MODEL_HEAD_DIM == 128 && LUNELETTE_MODEL_QUERY_HEADS == 32 &&
    LUNELETTE_MODEL_LAYERS == 26 && LUNELETTE_MODEL_VOCAB == 131072,
    "Update allocation geometry before changing model dimensions");
#define METAL_PLAN_SLOTS 23
typedef struct {
    size_t scratch[METAL_PLAN_SLOTS], kv_each;
    uint64_t scratch_bytes, kv_bytes, guard_growth;
} metal_allocation_plan;

static inline int metal_plan_multiply(size_t a, size_t b, size_t *out)
{
    if (b && a > SIZE_MAX / b) {
        return -1;
    }
    *out = a * b;
    return 0;
}

static inline int metal_plan_buffer_fits(size_t elements, size_t width,
    uint64_t maximum, size_t *bytes)
{
    return metal_plan_multiply(elements, width, bytes) || *bytes > maximum ? -1 : 0;
}

static inline int metal_plan_create(lunelette_metal_limits limits, metal_allocation_plan *out)
{
    if (!out || !((limits.context == LUNELETTE_CONTEXT_LIMIT && limits.block == LUNELETTE_BLOCK_LIMIT) ||
            (limits.context == LUNELETTE_LONG_CONTEXT_LIMIT &&
                (limits.block == LUNELETTE_BLOCK_LIMIT || limits.block == 128)))) {
        return -1;
    }
    const size_t widths[METAL_PLAN_SLOTS] = {3072, 3072, 4096, 1024, 1024, 4096,
        1024, 4096, 3072, 3072, 3072, 9216, 9216, 9216, 9216, 3072, 3072, 3072,
        0, 128, 128, 1, 32 * limits.context};
    metal_allocation_plan plan = {0};
    for (size_t i = 0; i < METAL_PLAN_SLOTS; i++) {
        size_t elements = 131072;
        if ((i != 18 && metal_plan_multiply(limits.block, widths[i], &elements)) ||
            metal_plan_multiply(elements, sizeof(float), &plan.scratch[i]) ||
            plan.scratch_bytes > UINT64_MAX - plan.scratch[i]) {
            return -1;
        }
        plan.scratch_bytes += plan.scratch[i];
    }
    if (metal_plan_multiply(limits.context, 1024 * sizeof(float), &plan.kv_each) ||
        plan.kv_each > UINT64_MAX / 52) {
        return -1;
    }
    plan.kv_bytes = (uint64_t)plan.kv_each * 52;
    if (plan.scratch_bytes > UINT64_MAX - plan.kv_bytes) {
        return -1;
    }
    uint64_t named = plan.scratch_bytes + plan.kv_bytes;
    plan.guard_growth = named > 142345216 ? named - 142345216 : 0;
    if (limits.context != 256) {
        plan.guard_growth += 8ULL * 1024 * 1024;
    }
    *out = plan;
    return 0;
}

/* Preserve the old guard baseline; charge only named growth and the explicit
 * long-profile host allowance on top of weights plus conversion stage. */
static inline int metal_plan_guard_base(const metal_allocation_plan *plan,
    const lunelette_metal_memory *memory, uint64_t *out)
{
    if (!plan || !memory || !out ||
        memory->weight_bytes > UINT64_MAX - memory->conversion_stage_bytes) {
        return -1;
    }
    uint64_t base = memory->weight_bytes + memory->conversion_stage_bytes;
    if (base > UINT64_MAX - plan->guard_growth) {
        return -1;
    }
    *out = base + plan->guard_growth;
    return 0;
}

static inline int metal_plan_fits(const metal_allocation_plan *plan, uint64_t maximum)
{
    if (!plan || !maximum || plan->kv_each > maximum) {
        return -1;
    }
    for (size_t i = 0; i < METAL_PLAN_SLOTS; i++) {
        if (plan->scratch[i] > maximum) {
            return -1;
        }
    }
    return 0;
}
#endif
