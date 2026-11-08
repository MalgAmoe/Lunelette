#include "metal_options.h"
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)

static void check(const lunelette_metal_options *input, int prefill, int strategy)
{
    char error[256];
    lunelette_metal_options output;
    CHECK(metal_options_resolve(input, &output, error, sizeof error) == 0);
    CHECK((int)output.prefill == prefill);
    CHECK((int)output.strategy == strategy);
}

int main(void)
{
    lunelette_metal_options official = LUNELETTE_METAL_OPTIONS_DEFAULT;
    lunelette_metal_options zero = {0};
    lunelette_metal_options partial = {.prefill = LUNELETTE_METAL_PREFILL_CUSTOM};
    CHECK(LUNELETTE_METAL_RESIDENT_FP32 == 0);
    CHECK(LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC == 1);
    CHECK(LUNELETTE_METAL_COMPACT_FP8_LOOKUP == 2);
    check(NULL, 0, 1);
    check(&official, 0, 1);
    check(&zero, 0, 0);
    check(&partial, 1, 0);
    for (int strategy = 0; strategy <= 2; strategy++) {
        lunelette_metal_options omitted_prefill = {.strategy = strategy};
        check(&omitted_prefill, 0, strategy);
        for (int prefill = 0; prefill <= 1; prefill++) {
            lunelette_metal_options explicit = {prefill, strategy};
            check(&explicit, prefill, strategy);
        }
    }
    for (int field = 0; field < 2; field++) {
        for (int value = -1; value <= 99; value += 100) {
            lunelette_metal_options invalid = official;
            lunelette_metal_options output = zero;
            char error[256] = "";
            if (field) {
                invalid.strategy = value;
            } else {
                invalid.prefill = value;
            }
            CHECK(metal_options_resolve(&invalid, &output, error, sizeof error) == -1);
            CHECK(output.prefill == 0 && output.strategy == 0);
            CHECK(error[0]);
        }
    }
    puts("Constructor option compatibility matrix PASS");
    return 0;
}
