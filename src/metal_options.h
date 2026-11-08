#ifndef LUNELETTE_METAL_OPTIONS_INTERNAL_H
#define LUNELETTE_METAL_OPTIONS_INTERNAL_H

#include "metal.h"
#include <stdio.h>

/* Constructor admission seam: no I/O or GPU work; output unchanged on failure.
 * The constructor supplies a writable nonempty diagnostic buffer. */
static inline int metal_options_resolve(const lunelette_metal_options *options,
    lunelette_metal_options *out, char *error, size_t capacity)
{
    lunelette_metal_options selected = LUNELETTE_METAL_OPTIONS_DEFAULT;
    if (options) {
        selected = *options;
    }
    lunelette_metal_prefill_mode mode = selected.prefill;
    if (mode != LUNELETTE_METAL_PREFILL_MPS && mode != LUNELETTE_METAL_PREFILL_CUSTOM) {
        snprintf(error, capacity, "Metal invalid prefill option; select MPS or CUSTOM");
        return -1;
    }
    lunelette_metal_strategy strategy = selected.strategy;
    if (strategy != LUNELETTE_METAL_RESIDENT_FP32 &&
        strategy != LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC &&
        strategy != LUNELETTE_METAL_COMPACT_FP8_LOOKUP) {
        snprintf(error, capacity, "Metal invalid strategy; select resident-fp32, compact-fp8-arithmetic or compact-fp8-lookup");
        return -1;
    }
    *out = selected;
    return 0;
}

#endif
