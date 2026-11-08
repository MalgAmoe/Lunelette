/* Test-tool policy, deliberately independent of product defaults. */
#ifndef COMPARATOR_STRATEGY_H
#define COMPARATOR_STRATEGY_H
#include "metal.h"
#include <stdio.h>
#include <string.h>

static const char *comparator_strategy_name(lunelette_metal_strategy strategy)
{
    switch (strategy) {
    case LUNELETTE_METAL_RESIDENT_FP32:
        return "resident-fp32";
    case LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC:
        return "compact-fp8-arithmetic";
    case LUNELETTE_METAL_COMPACT_FP8_LOOKUP:
        return "compact-fp8-lookup";
    }
    return NULL;
}

/* Optional trailing selector. Legacy omission is FIXED resident, never DEFAULT. */
static int comparator_strategy_args(int *argc, char **argv,
    lunelette_metal_strategy *strategy)
{
    *strategy = LUNELETTE_METAL_RESIDENT_FP32;
    for (int i = 1; i < *argc; i++) {
        if (!strncmp(argv[i], "--strategy", 10) &&
            (i != *argc - 1 || strncmp(argv[i], "--strategy=", 11))) {
            fputs("Comparator strategy must occur once, as a final --strategy=NAME\n", stderr);
            return 1;
        }
    }
    if (*argc > 1 && !strncmp(argv[*argc - 1], "--strategy=", 11)) {
        const char *name = argv[*argc - 1] + 11;
        for (int i = 0; i < 3; i++) {
            if (!strcmp(name, comparator_strategy_name((lunelette_metal_strategy)i))) {
                *strategy = (lunelette_metal_strategy)i;
                (*argc)--;
                return 0;
            }
        }
        fputs("Invalid comparator strategy\n", stderr);
        return 1;
    }
    return 0;
}

#ifndef NATIVE_OFFLINE
static int comparator_strategy_actual(lunelette_metal *runtime,
    lunelette_metal_strategy requested)
{
    lunelette_metal_memory memory;
    if (lunelette_metal_get_memory(runtime, &memory) ||
        !comparator_strategy_name(memory.strategy)) {
        fputs("Cannot query comparator runtime strategy\n", stderr);
        return 1;
    }
    printf("strategy=%s\n", comparator_strategy_name(memory.strategy));
    if (memory.strategy != requested) {
        fputs("Runtime strategy differs from requested strategy\n", stderr);
        return 1;
    }
    return 0;
}
#endif
#endif
