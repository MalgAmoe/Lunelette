#include "../src/metal_memory_guard.h"
#include "../src/metal_plan.h"
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

typedef struct {
    uint64_t time, step, last_sleep, query_cost;
    int reads, clocks, sleeps, sufficient_at, query_error_at, clock_error_at;
    int backwards, sleep_error, stalled;
    uint64_t low;
} fake;

static int available(void *context, uint64_t *value)
{
    fake *f = context;
    ++f->reads;
    f->time += f->query_cost;
    *value = f->sufficient_at && f->reads >= f->sufficient_at ? UINT64_MAX : f->low;
    return f->reads == f->query_error_at ? -1 : 0;
}

static int now(void *context, uint64_t *value)
{
    fake *f = context;
    ++f->clocks;
    *value = f->time;
    return f->clocks == f->clock_error_at ? -1 : 0;
}

static int sleep_for(void *context, uint64_t duration)
{
    fake *f = context;
    ++f->sleeps;
    CHECK(f->sleeps <= 100);
    CHECK(duration <= 50000000ULL);
    CHECK(duration <= 3000000000ULL - (f->time - f->query_cost));
    f->last_sleep = duration;
    if (f->backwards) {
        --f->time;
    } else if (!f->stalled) {
        f->time += f->step ? f->step : duration;
    }
    return f->sleep_error;
}

static int run(fake *f, uint64_t weights, const char *message)
{
    char error[256] = "";
    metal_memory_probe probe = {f, available, now, sleep_for};
    int result = metal_memory_guard(weights, probe, error, sizeof error);
    if (message) {
        CHECK(result == -1);
        CHECK(strstr(error, message));
    } else {
        CHECK(result == 0);
    }
    return result;
}

int main(void)
{
    const uint64_t overhead = 310ULL * 1024 * 1024 + 2ULL * 1024 * 1024 * 1024;
    const size_t contexts[] = {256, 4352, 4352};
    const size_t blocks[] = {256, 256, 128};
    const uint64_t scratch[] = {87819264, 222036992, 111280640};
    const uint64_t requirements[] = {16188567552ULL, 17203589120ULL, 17092832768ULL};
    for (size_t i = 0; i < 3; i++) {
        metal_allocation_plan plan;
        CHECK(!metal_plan_create((lunelette_metal_limits){contexts[i], blocks[i]}, &plan));
        CHECK(plan.scratch_bytes == scratch[i]);
        CHECK(plan.kv_bytes == (i ? 926941184 : 54525952));
        CHECK(plan.scratch[22] == 4 * 32 * contexts[i] * blocks[i]);
        size_t maximum = plan.kv_each;
        for (size_t slot = 0; slot < METAL_PLAN_SLOTS; slot++) {
            if (plan.scratch[slot] > maximum) {
                maximum = plan.scratch[slot];
            }
        }
        CHECK(!metal_plan_fits(&plan, maximum));
        CHECK(metal_plan_fits(&plan, maximum - 1));
        CHECK(metal_plan_fits(&plan, 0));
        lunelette_metal_memory memory = {.weight_bytes = 13716025344ULL};
        uint64_t base;
        CHECK(!metal_plan_guard_base(&plan, &memory, &base));
        CHECK(base + overhead == requirements[i]);
        fake threshold = {.low = base + overhead};
        run(&threshold, base, "after 3s");
        threshold = (fake){.low = base + overhead + 1};
        run(&threshold, base, NULL);
        CHECK(!threshold.sleeps);
        memory.weight_bytes = 4637454336ULL;
        memory.conversion_stage_bytes = 113246208ULL;
        CHECK(!metal_plan_guard_base(&plan, &memory, &base));
        CHECK(base + overhead == requirements[i] - 8965324800ULL);
        memory.weight_bytes = UINT64_MAX;
        CHECK(metal_plan_guard_base(&plan, &memory, &base));
    }
    metal_allocation_plan rejected;
    const lunelette_metal_limits invalid[] = {{0, 0}, {256, 128}, {4352, 64},
        {4352, 4352}, {4353, 256}, {SIZE_MAX, SIZE_MAX}};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; i++) {
        CHECK(metal_plan_create(invalid[i], &rejected));
    }
    size_t product;
    CHECK(metal_plan_multiply(SIZE_MAX, 2, &product));
    CHECK(!metal_plan_multiply(SIZE_MAX, 1, &product) && product == SIZE_MAX);
    CHECK(!metal_plan_buffer_fits(1024, 4, 4096, &product) && product == 4096);
    CHECK(metal_plan_buffer_fits(1024, 4, 4095, &product));
    CHECK(metal_plan_buffer_fits(SIZE_MAX, 4, UINT64_MAX, &product));
    fake f = {.sufficient_at = 1};
    run(&f, 0, NULL);
    CHECK(f.reads == 1 && f.clocks == 0 && f.sleeps == 0);
    f = (fake){.sufficient_at = 3};
    run(&f, 0, NULL);
    CHECK(f.reads == 3 && f.sleeps == 2);
    f = (fake){0};
    run(&f, 0, "after 3s");
    CHECK(f.time == 3000000000ULL && f.reads == 60 && f.sleeps == 60);
    f = (fake){.low = 310ULL * 1024 * 1024 + 2ULL * 1024 * 1024 * 1024};
    run(&f, 0, "after 3s"); /* Equality still rejects. */
    f = (fake){.query_error_at = 1};
    run(&f, 0, "query failed");
    CHECK(!f.sleeps && !f.clocks);
    f = (fake){.query_error_at = 2};
    run(&f, 0, "query failed");
    CHECK(f.sleeps == 1 && f.reads == 2);
    f = (fake){.clock_error_at = 1};
    run(&f, 0, "clock");
    CHECK(!f.sleeps);
    f = (fake){.stalled = 1};
    run(&f, 0, "clock");
    CHECK(f.sleeps == 1);
    f = (fake){.clock_error_at = 3};
    run(&f, 0, "clock");
    f = (fake){.time = 100, .backwards = 1};
    run(&f, 0, "clock");
    f = (fake){.step = 10000000000ULL, .sufficient_at = 2};
    run(&f, 0, "after 3s"); /* Oversleep: no late query or acceptance. */
    CHECK(f.reads == 1 && f.sleeps == 1);
    f = (fake){.step = 49000000ULL}; /* Early wake/EINTR, capped final sleep. */
    run(&f, 0, "after 3s");
    CHECK(f.last_sleep < 50000000ULL);
    f = (fake){.sufficient_at = 2, .query_cost = 3000000000ULL};
    run(&f, 0, "after 3s"); /* A sufficient but late query cannot pass. */
    f = (fake){.sleep_error = -1};
    run(&f, 0, "wait failed");
    f = (fake){0};
    run(&f, UINT64_MAX, "overflow");
    CHECK(!f.reads && !f.clocks && !f.sleeps);
    run(&f, 24ULL * 1024 * 1024 * 1024, "24GiB");
    CHECK(!f.reads && !f.clocks && !f.sleeps);
    f = (fake){.sufficient_at = 1};
    run(&f, 24ULL * 1024 * 1024 * 1024 -
        (310ULL * 1024 * 1024 + 2ULL * 1024 * 1024 * 1024), NULL);
    puts("memory guard tests passed");
    return 0;
}
