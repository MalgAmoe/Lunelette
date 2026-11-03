#ifndef LUNELETTE_TEST_FULL_DEPTH_H
#define LUNELETTE_TEST_FULL_DEPTH_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>

/* Frozen PyTorch-only calibration; see docs/full-depth-validation.md.
 * Test-only acceptance: never used by isolated-layer comparisons or execution.
 */
#define DEPTH_RELATIVE_RMS 0.0013427489303392518
#define DEPTH_COSINE_DISTANCE 2.2074261751825475e-7
#define DEPTH_MAX_ABS 0.016815185546875
#define DEPTH_RMS_FLOOR 1e-6

static int compare_full_depth(const float *actual, const float *expected,
                              size_t rows, size_t width)
{
    size_t failed_rows = 0;
    double worst_relative = 0, worst_cosine = 0, worst_absolute = 0;
    if (!rows || rows > 256 || !width || width > 3072) {
        return 1;
    }
    for (size_t row = 0; row < rows; row++) {
        double norm_actual = 0, norm_expected = 0, dot = 0, squared_error = 0;
        double max_absolute = 0;
        int failed = 0;
        for (size_t col = 0; col < width; col++) {
            double a = actual[row * width + col];
            double b = expected[row * width + col];
            if (!isfinite(a) || !isfinite(b)) {
                failed = 1;
                continue;
            }
            double error = a - b;
            norm_actual += a * a;
            norm_expected += b * b;
            dot += a * b;
            squared_error += error * error;
            max_absolute = fmax(max_absolute, fabs(error));
        }
        double relative = sqrt(squared_error / width) /
                          fmax(sqrt(norm_expected / width), DEPTH_RMS_FLOOR);
        double cosine = 0;
        if (norm_actual == 0 || norm_expected == 0) {
            if (norm_actual != norm_expected) {
                failed = 1;
                cosine = 1;
            }
        } else {
            cosine = fmax(0, 1 - dot / (sqrt(norm_actual) * sqrt(norm_expected)));
        }
        if (relative > DEPTH_RELATIVE_RMS || cosine > DEPTH_COSINE_DISTANCE ||
            max_absolute > DEPTH_MAX_ABS) {
            failed = 1;
        }
        failed_rows += failed;
        worst_relative = fmax(worst_relative, relative);
        worst_cosine = fmax(worst_cosine, cosine);
        worst_absolute = fmax(worst_absolute, max_absolute);
    }
    printf("  full_depth relative_rms=%.12g cosine_distance=%.12g max_abs=%.12g "
           "failed_rows=%zu/%zu %s\n", worst_relative, worst_cosine, worst_absolute,
           failed_rows, rows, failed_rows ? "FAIL" : "PASS");
    return failed_rows != 0;
}

#endif
