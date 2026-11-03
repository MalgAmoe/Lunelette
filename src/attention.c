/* YaRN formulas adapted from Hugging Face Transformers v5.0.0
 * modeling_rope_utils.py, Copyright 2024 The HuggingFace Team.
 * Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt and NOTICE.
 * Split-half rotation and query scaling follow modeling_ministral3.py.
 */
#include "attention.h"

#include <math.h>
#include <stdint.h>

static int valid_shape(size_t tokens, size_t heads, size_t dim)
{
    size_t limit = SIZE_MAX / sizeof(float);
    return tokens && heads && dim && heads <= limit / dim &&
           tokens <= limit / (heads * dim);
}

int lunelette_ministral3_yarn(float *query, float *key, const uint32_t *positions,
                            size_t tokens)
{
    if (!query || !key || !positions || !valid_shape(tokens, 32, 128)) {
        return -1;
    }
    for (size_t t = 0; t < tokens; t++) {
        if (positions[t] >= 262144) {
            return -1;
        }
    }
    /* theta=1e6, factor=16, original context=16384, beta_fast/slow=32/1.
     * mscale=mscale_all_dim=1: their ratio is exactly 1, NOT 1+0.1*log(16).
     * Default truncate=True floors/ceils the correction range in half-dims.
     */
    const double pi = 3.14159265358979323846;
    float low = (float)floor(128 * log(16384 / (32 * 2 * pi)) / (2 * log(1e6)));
    float high = (float)ceil(128 * log(16384 / (2 * pi)) / (2 * log(1e6)));
    float frequencies[64];
    for (size_t i = 0; i < 64; i++) {
        float base = powf(1000000.0f, (float)(2 * i) / 128.0f);
        float ramp = fminf(1.0f, fmaxf(0.0f, ((float)i - low) / (high - low)));
        float extrapolation = 1.0f - ramp;
        frequencies[i] = (1.0f / (16.0f * base)) * (1.0f - extrapolation) +
                         (1.0f / base) * extrapolation;
    }
    for (size_t t = 0; t < tokens; t++) {
        float query_scale = 1.0f + 0.1f * logf(1.0f + (float)(positions[t] / 16384));
        for (size_t i = 0; i < 64; i++) {
            float angle = (float)positions[t] * frequencies[i];
            float cosine = cosf(angle);
            float sine = sinf(angle);
            for (size_t h = 0; h < 40; h++) {
                float *head = h < 32 ? query + (t * 32 + h) * 128 :
                                      key + (t * 8 + h - 32) * 128;
                float first = head[i];
                float second = head[i + 64];
                float scale = h < 32 ? query_scale : 1.0f;
                head[i] = (first * cosine - second * sine) * scale;
                head[i + 64] = (second * cosine + first * sine) * scale;
            }
        }
    }
    return 0;
}

int lunelette_causal_gqa(const float *query, const float *key, const float *value,
                         float *output, float *scratch, size_t tokens,
                         size_t query_heads, size_t kv_heads, size_t head_dim)
{
    if (!query || !key || !value || !output || !scratch ||
        !valid_shape(tokens, query_heads, head_dim) ||
        !valid_shape(tokens, kv_heads, head_dim) || query_heads % kv_heads) {
        return -1;
    }
    size_t group = query_heads / kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < query_heads; h++) {
            const float *q = query + (t * query_heads + h) * head_dim;
            float maximum = -INFINITY;
            for (size_t s = 0; s <= t; s++) {
                const float *k = key + (s * kv_heads + h / group) * head_dim;
                float score = 0;
                for (size_t d = 0; d < head_dim; d++) {
                    score += q[d] * k[d];
                }
                score *= scale;
                if (!isfinite(score)) {
                    return -1;
                }
                scratch[s] = score;
                maximum = fmaxf(maximum, score);
            }
            float sum = 0;
            for (size_t s = 0; s <= t; s++) {
                scratch[s] = expf(scratch[s] - maximum);
                sum += scratch[s];
            }
            for (size_t s = 0; s <= t; s++) {
                scratch[s] /= sum;
            }
            for (size_t d = 0; d < head_dim; d++) {
                float result = 0;
                for (size_t s = 0; s <= t; s++) {
                    result += scratch[s] * value[(s * kv_heads + h / group) * head_dim + d];
                }
                output[(t * query_heads + h) * head_dim + d] = result;
            }
        }
    }
    return 0;
}
