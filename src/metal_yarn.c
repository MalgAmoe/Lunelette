#include <math.h>
#include <stdint.h>
#include "model_facts.h"

/* Independently transcribed Transformers YaRN, not a call into CPU attention.
 * mscale=mscale_all_dim=1: attention scaling is exactly one. See NOTICE.
 * Preserve FP32 operation order and disable contraction. */
int lunelette_metal_yarn(uint32_t position, float cosine[128], float sine[128])
{
    if (position >= LUNELETTE_MODEL_MAX_POSITION) {
        return -1;
    }
    const double pi = 3.14159265358979323846;
    double dimension = LUNELETTE_MODEL_HEAD_DIM;
    double original = LUNELETTE_MODEL_ORIGINAL_CONTEXT;
    float low = (float)floor(dimension * log(original /
        (LUNELETTE_MODEL_YARN_BETA_FAST * 2.0 * pi)) /
        (2.0 * log(LUNELETTE_MODEL_ROPE_THETA)));
    float high = (float)ceil(dimension * log(original / (2.0 * pi)) /
        (2.0 * log(LUNELETTE_MODEL_ROPE_THETA)));
    for (int j = 0; j < LUNELETTE_MODEL_HEAD_DIM / 2; j++) {
        float frequency = powf(LUNELETTE_MODEL_ROPE_THETA_F,
            (float)(2 * j) / (float)LUNELETTE_MODEL_HEAD_DIM);
        float ramp = fminf(1.0f, fmaxf(0.0f, ((float)j - low) / (high - low)));
        float extrapolation = 1.0f - ramp;
        float inverse = (1.0f / (LUNELETTE_MODEL_YARN_FACTOR * frequency)) *
            (1.0f - extrapolation) + (1.0f / frequency) * extrapolation;
        float angle = inverse * (float)position;
        cosine[j] = cosine[j + LUNELETTE_MODEL_HEAD_DIM / 2] = cosf(angle);
        sine[j] = sine[j + LUNELETTE_MODEL_HEAD_DIM / 2] = sinf(angle);
    }
    return 0;
}
