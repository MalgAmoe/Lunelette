#include "lunelette.h"

#include <math.h>

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
