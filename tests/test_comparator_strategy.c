/* Host-only policy tests: synthetic memory query, no runtime or model linkage. */
#include "comparator_strategy.h"
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static lunelette_metal_strategy actual;
static int query_error;

int lunelette_metal_get_memory(const lunelette_metal *runtime, lunelette_metal_memory *memory)
{
    (void)runtime;
    if (query_error) {
        return -1;
    }
    *memory = (lunelette_metal_memory){.strategy = actual};
    return 0;
}

int main(void)
{
    lunelette_metal_strategy requested;
    char *legacy[] = {"comparator", "model", "ids", "fixture"};
    int count = 4;
    CHECK(!comparator_strategy_args(&count, legacy, &requested));
    CHECK(count == 4 && requested == LUNELETTE_METAL_RESIDENT_FP32);
    for (int i = 0; i < 3; i++) {
        char selector[128];
        snprintf(selector, sizeof selector, "--strategy=%s",
            comparator_strategy_name((lunelette_metal_strategy)i));
        char *args[] = {"comparator", "model", "ids", "fixture", selector};
        count = 5;
        CHECK(!comparator_strategy_args(&count, args, &requested));
        CHECK(count == 4 && (int)requested == i);
        actual = requested;
        CHECK(!comparator_strategy_actual(NULL, requested));
        actual = (lunelette_metal_strategy)((i + 1) % 3);
        CHECK(comparator_strategy_actual(NULL, requested));
    }
    char *bad[][4] = {
        {"comparator", "--strategy=", NULL, NULL},
        {"comparator", "--strategy=invalid", NULL, NULL},
        {"comparator", "--strategy", NULL, NULL},
        {"comparator", "--strategy", "resident-fp32", NULL},
        {"comparator", "--strategy=resident-fp32", "fixture", NULL},
        {"comparator", "--strategy=resident-fp32", "--strategy=compact-fp8-lookup", NULL}
    };
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        count = 0;
        while (bad[i][count]) {
            count++;
        }
        CHECK(comparator_strategy_args(&count, bad[i], &requested));
    }
    actual = (lunelette_metal_strategy)99;
    CHECK(comparator_strategy_actual(NULL, LUNELETTE_METAL_RESIDENT_FP32));
    query_error = 1;
    CHECK(comparator_strategy_actual(NULL, LUNELETTE_METAL_RESIDENT_FP32));
    puts("comparator strategy parsing and synthetic runtime identity PASS");
    return 0;
}
