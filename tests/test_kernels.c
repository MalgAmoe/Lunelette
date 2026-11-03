#include "lunelette.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/* Small scalar examples allow a few float rounding steps, not model accuracy.
 * Combined absolute/relative tolerance also handles expected zero. */
static int close_float(float actual, float expected)
{
    float absolute = fabsf(actual - expected);
    return isfinite(actual) && absolute <= 1e-6f + 2e-6f * fabsf(expected);
}

static void check_close(float actual, float expected)
{
    if (!close_float(actual, expected)) {
        float absolute = fabsf(actual - expected);
        float relative = expected == 0.0f ? INFINITY : absolute / fabsf(expected);
        fprintf(stderr, "actual %.9g expected %.9g abs %.9g rel %.9g\n",
                actual, expected, absolute, relative);
        exit(EXIT_FAILURE);
    }
}

static void test_rmsnorm(void)
{
    float input[] = {1.0f, 3.0f};
    float weight[] = {2.0f, -1.0f};
    float output[2];
    /* mean square = 5; epsilon = 4 gives denominator exactly 3. */
    CHECK(lunelette_rmsnorm(input, weight, output, 2, 4.0f) == 0);
    check_close(output[0], 2.0f / 3.0f);
    check_close(output[1], -1.0f);
    CHECK(lunelette_rmsnorm(input, weight, input, 2, 4.0f) == 0);
    check_close(input[0], output[0]);
    check_close(input[1], output[1]);
    float original[] = {1.0f, 3.0f};
    CHECK(lunelette_rmsnorm(original, weight, weight, 2, 4.0f) == 0);
    check_close(weight[0], output[0]);
    check_close(weight[1], output[1]);
    float same[] = {1.0f, 3.0f};
    CHECK(lunelette_rmsnorm(same, same, same, 2, 4.0f) == 0);
    check_close(same[0], 1.0f / 3.0f);
    check_close(same[1], 3.0f);

    float zero[] = {0.0f, 0.0f};
    CHECK(lunelette_rmsnorm(zero, original, zero, 2, 1e-5f) == 0);
    CHECK(zero[0] == 0.0f && zero[1] == 0.0f);
    float single = 2.0f;
    float unit = 1.0f;
    CHECK(lunelette_rmsnorm(&single, &unit, output, 1, 5.0f) == 0);
    check_close(output[0], 2.0f / 3.0f);
}

static void test_matvec(void)
{
    float weight[] = {1, 2, 3, -1, 0, 4};
    float input[] = {2, -1, 3};
    float output[3];
    CHECK(lunelette_matvec(weight, input, output, 2, 3) == 0);
    check_close(output[0], 9);
    check_close(output[1], 10);
    /* The same storage interpreted as 3x2 verifies both dimension directions. */
    CHECK(lunelette_matvec(weight, input, output, 3, 2) == 0);
    check_close(output[0], 0);
    check_close(output[1], 7);
    check_close(output[2], -4);
    CHECK(lunelette_matvec(weight, input, output, 1, 1) == 0);
    check_close(output[0], 2);
}

static void test_silu(void)
{
    float gate[] = {-1, 0, 1, -FLT_MAX, FLT_MAX, -100, 100};
    float up[] = {2, 3, -2, 1, 1, 1, 1};
    float output[7];
    const float expected[] = {-0.537882843f, 0, -1.462117157f, 0, FLT_MAX, 0, 100};
    CHECK(lunelette_silu_gate(gate, up, output, 7) == 0);
    for (size_t i = 0; i < 7; ++i) {
        check_close(output[i], expected[i]);
    }
    CHECK(lunelette_silu_gate(gate, up, up, 7) == 0);
    for (size_t i = 0; i < 7; ++i) {
        check_close(up[i], output[i]);
    }
    float ones[] = {1, 1, 1, 1, 1, 1, 1};
    CHECK(lunelette_silu_gate(gate, ones, gate, 7) == 0);
    check_close(gate[0], -0.268941421f);
    check_close(gate[2], 0.731058579f);
    float same = 1;
    CHECK(lunelette_silu_gate(&same, &same, &same, 1) == 0);
    check_close(same, 0.731058579f);
}

static void test_invalid(void)
{
    float input = 1;
    float output = 42;
    const size_t too_large = SIZE_MAX / sizeof(float) + 1;
    const float invalid_epsilon[] = {0, -1, NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < sizeof(invalid_epsilon) / sizeof(invalid_epsilon[0]); ++i) {
        CHECK(lunelette_rmsnorm(&input, &input, &output, 1, invalid_epsilon[i]) == -1);
        CHECK(output == 42);
    }
    CHECK(lunelette_rmsnorm(NULL, &input, &output, 1, 1) == -1);
    CHECK(lunelette_rmsnorm(&input, NULL, &output, 1, 1) == -1);
    CHECK(lunelette_rmsnorm(&input, &input, NULL, 1, 1) == -1);
    CHECK(lunelette_rmsnorm(&input, &input, &output, 0, 1) == -1);
    CHECK(lunelette_rmsnorm(&input, &input, &output, too_large, 1) == -1);
    CHECK(lunelette_matvec(NULL, &input, &output, 1, 1) == -1);
    CHECK(lunelette_matvec(&input, NULL, &output, 1, 1) == -1);
    CHECK(lunelette_matvec(&input, &input, NULL, 1, 1) == -1);
    CHECK(lunelette_matvec(&input, &input, &output, 0, 1) == -1);
    CHECK(lunelette_matvec(&input, &input, &output, 1, 0) == -1);
    CHECK(lunelette_matvec(&input, &input, &output, too_large, 1) == -1);
    CHECK(lunelette_matvec(&input, &input, &output, 1, too_large) == -1);
    CHECK(lunelette_matvec(&input, &input, &output, SIZE_MAX / sizeof(float), 2) == -1);
    CHECK(lunelette_matvec(&input, &input, &output, SIZE_MAX / 2 + 1, 2) == -1);
    CHECK(lunelette_silu_gate(NULL, &input, &output, 1) == -1);
    CHECK(lunelette_silu_gate(&input, NULL, &output, 1) == -1);
    CHECK(lunelette_silu_gate(&input, &input, NULL, 1) == -1);
    CHECK(lunelette_silu_gate(&input, &input, &output, 0) == -1);
    CHECK(lunelette_silu_gate(&input, &input, &output, too_large) == -1);
    CHECK(output == 42);
}

int main(void)
{
    CHECK(close_float(1, 1));
    CHECK(!close_float(1, 2));
    CHECK(!close_float(NAN, 1));
    test_rmsnorm();
    test_matvec();
    test_silu();
    test_invalid();
    puts("scalar kernel tests passed");
    return 0;
}
