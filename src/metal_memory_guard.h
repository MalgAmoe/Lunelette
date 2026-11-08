#ifndef LUNELETTE_METAL_MEMORY_GUARD_H
#define LUNELETTE_METAL_MEMORY_GUARD_H

#include <stdint.h>
#include <stdio.h>

/* Mach free_count already includes speculative pages (vm_statistics.h).
 * Output is unchanged on failure; retain inactive pages in the existing estimate. */
static inline int metal_memory_available_bytes(uint64_t free_pages, uint64_t inactive_pages,
                                               uint64_t page_size, uint64_t *available)
{
    if (!available || !page_size || free_pages > UINT64_MAX - inactive_pages) {
        return -1;
    }
    uint64_t pages = free_pages + inactive_pages;
    if (pages > UINT64_MAX / page_size) {
        return -1;
    }
    *available = pages * page_size;
    return 0;
}

/* Private, model-free seam. Sleep may return early (including EINTR); every
 * iteration checks the original monotonic budget before another query. */
typedef struct {
    void *context;
    int (*available)(void *, uint64_t *);
    int (*now)(void *, uint64_t *);
    int (*sleep)(void *, uint64_t);
} metal_memory_probe;

static int metal_memory_guard(uint64_t weights, metal_memory_probe probe,
                              char *error, size_t capacity)
{
    const uint64_t overhead = 310ULL * 1024 * 1024 + 2ULL * 1024 * 1024 * 1024;
    const uint64_t budget = 3000000000ULL;
    const uint64_t interval = 50000000ULL;
    uint64_t available, start, previous, now;
    if (weights > UINT64_MAX - overhead) {
        snprintf(error, capacity, "Metal memory guard: estimated peak overflow");
        return -1;
    }
    uint64_t peak = weights + overhead;
    if (peak > 24ULL * 1024 * 1024 * 1024) {
        snprintf(error, capacity, "Metal memory guard: estimated peak exceeds 24GiB");
        return -1;
    }
    if (probe.available(probe.context, &available) != 0) {
        goto query_error;
    }
    if (available > peak) {
        return 0;
    }
    if (probe.now(probe.context, &start) != 0) {
        goto clock_error;
    }
    previous = start;
    for (;;) {
        if (probe.now(probe.context, &now) != 0 || now < previous) {
            goto clock_error;
        }
        if (now - start >= budget) {
            break;
        }
        uint64_t remaining = budget - (now - start);
        previous = now;
        if (probe.sleep(probe.context, remaining < interval ? remaining : interval) != 0) {
            snprintf(error, capacity, "Metal memory guard: availability wait failed");
            return -1;
        }
        if (probe.now(probe.context, &now) != 0 || now <= previous) {
            goto clock_error;
        }
        previous = now;
        if (now - start >= budget) {
            break;
        }
        if (probe.available(probe.context, &available) != 0) {
            goto query_error;
        }
        /* A query can itself consume the remaining budget. Never accept a
         * late result, nor reuse an earlier sufficient estimate. */
        if (probe.now(probe.context, &now) != 0 || now < previous) {
            goto clock_error;
        }
        previous = now;
        if (now - start >= budget) {
            break;
        }
        if (available > peak) {
            return 0;
        }
    }
    snprintf(error, capacity, "Metal memory guard: availability wait budget expired "
        "after 3s (required >%.2f GiB; last estimate %.2f GiB)",
        peak / 1073741824.0, available / 1073741824.0);
    return -1;
query_error:
    snprintf(error, capacity, "Metal memory guard: headroom query failed");
    return -1;
clock_error:
    snprintf(error, capacity, "Metal memory guard: monotonic clock failed or did not advance");
    return -1;
}
#endif
