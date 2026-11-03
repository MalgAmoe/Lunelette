#include "lunelette.h"

#include <math.h>
#include <stdint.h>

int lunelette_softmax(const float *input, float *output, size_t count)
{
    if (!input || !output || count == 0) return -1;

    float maximum = input[0];
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(input[i])) return -1;
        if (input[i] > maximum) maximum = input[i];
    }

    /* Subtracting the maximum avoids exponential overflow. Accumulate in
     * double precision to keep this simple CPU baseline numerically stable. */
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        output[i] = (float)exp((double)input[i] - (double)maximum);
        sum += output[i];
    }
    for (size_t i = 0; i < count; ++i)
        output[i] = (float)(output[i] / sum);
    return 0;
}

int lunelette_rmsnorm(const float *input, const float *weight, float *output,
                     size_t count, float epsilon)
{
    if (!input || !weight || !output || count == 0 ||
        count > SIZE_MAX / sizeof(float) || !isfinite(epsilon) || epsilon <= 0.0f) {
        return -1;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += input[i] * input[i];
    }
    float scale = 1.0f / sqrtf(sum / (float)count + epsilon);
    for (size_t i = 0; i < count; ++i) {
        output[i] = (input[i] * scale) * weight[i];
    }
    return 0;
}

int lunelette_matvec(const float *weight, const float *input, float *output,
                    size_t rows, size_t cols)
{
    if (!weight || !input || !output || rows == 0 || cols == 0 ||
        rows > SIZE_MAX / sizeof(float) || cols > SIZE_MAX / sizeof(float) ||
        rows > (SIZE_MAX / sizeof(float)) / cols) {
        return -1;
    }

    for (size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            sum += weight[row * cols + col] * input[col];
        }
        output[row] = sum;
    }
    return 0;
}

int lunelette_silu_gate(const float *gate, const float *up, float *output, size_t count)
{
    if (!gate || !up || !output || count == 0 || count > SIZE_MAX / sizeof(float)) {
        return -1;
    }

    for (size_t i = 0; i < count; ++i) {
        float value = gate[i];
        float sigmoid;
        if (value >= 0.0f) {
            sigmoid = 1.0f / (1.0f + expf(-value));
        } else {
            /* expf(value) is bounded by one on this branch. */
            float exponential = expf(value);
            sigmoid = exponential / (1.0f + exponential);
        }
        output[i] = (value * sigmoid) * up[i];
    }
    return 0;
}
