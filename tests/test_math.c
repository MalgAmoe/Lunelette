#include "lunelette.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main(void)
{
    float input[] = {1.0f, 2.0f, 3.0f};
    float output[3];
    const float expected[] = {0.09003057f, 0.24472847f, 0.66524096f};
    CHECK(lunelette_softmax(input, output, 3) == 0);
    for (size_t i = 0; i < 3; ++i)
        CHECK(fabsf(output[i] - expected[i]) < 1e-6f);

    float shifted[] = {1001.0f, 1002.0f, 1003.0f};
    CHECK(lunelette_softmax(shifted, shifted, 3) == 0);
    for (size_t i = 0; i < 3; ++i)
        CHECK(fabsf(shifted[i] - expected[i]) < 1e-6f);

    float equal[] = {0.0f, 0.0f};
    CHECK(lunelette_softmax(equal, equal, 2) == 0);
    CHECK(equal[0] == 0.5f && equal[1] == 0.5f);
    float single = -1000.0f;
    CHECK(lunelette_softmax(&single, &single, 1) == 0);
    CHECK(single == 1.0f);

    float invalid[] = {0.0f, NAN};
    float sentinel[] = {42.0f, 43.0f};
    CHECK(lunelette_softmax(invalid, sentinel, 2) == -1);
    CHECK(sentinel[0] == 42.0f && sentinel[1] == 43.0f);
    invalid[1] = INFINITY;
    CHECK(lunelette_softmax(invalid, sentinel, 2) == -1);
    CHECK(lunelette_softmax(NULL, output, 3) == -1);
    CHECK(lunelette_softmax(input, NULL, 3) == -1);
    CHECK(lunelette_softmax(input, output, 0) == -1);
    puts("math tests passed");
    return 0;
}
