/* YaRN formulas adapted from Hugging Face Transformers v5.0.0
 * modeling_rope_utils.py, Copyright 2024 The HuggingFace Team.
 * Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt and NOTICE.
 * Split-half rotation and query scaling follow modeling_ministral3.py.
 */
#include "attention.h"
#include "forward_internal.h"

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
        if (positions[t] >= LUNELETTE_MODEL_MAX_POSITION) {
            return -1;
        }
    }
    /* theta=1e6, factor=16, original context=16384, beta_fast/slow=32/1.
     * mscale=mscale_all_dim=1: their ratio is exactly 1, NOT 1+0.1*log(16).
     * Default truncate=True floors/ceils the correction range in half-dims.
     */
    const double pi = 3.14159265358979323846;
    float low = (float)floor(LUNELETTE_MODEL_HEAD_DIM *
        log(LUNELETTE_MODEL_ORIGINAL_CONTEXT / (LUNELETTE_MODEL_YARN_BETA_FAST * 2 * pi)) /
        (2 * log(LUNELETTE_MODEL_ROPE_THETA)));
    float high = (float)ceil(LUNELETTE_MODEL_HEAD_DIM *
        log(LUNELETTE_MODEL_ORIGINAL_CONTEXT / (2 * pi)) / (2 * log(LUNELETTE_MODEL_ROPE_THETA)));
    float frequencies[64];
    for (size_t i = 0; i < 64; i++) {
        float base = powf(LUNELETTE_MODEL_ROPE_THETA_F, (float)(2 * i) / 128.0f);
        float ramp = fminf(1.0f, fmaxf(0.0f, ((float)i - low) / (high - low)));
        float extrapolation = 1.0f - ramp;
        frequencies[i] = (1.0f / (LUNELETTE_MODEL_YARN_FACTOR * base)) * (1.0f - extrapolation) +
                         (1.0f / base) * extrapolation;
    }
    for (size_t t = 0; t < tokens; t++) {
        float query_scale = 1.0f + LUNELETTE_MODEL_QUERY_SCALE *
            logf(1.0f + (float)(positions[t] / LUNELETTE_MODEL_ORIGINAL_CONTEXT));
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

int lunelette_offset_gqa(const float *query, const float *key, const float *value,
                         float *output, float *scratch, size_t tokens,
                         size_t query_heads, size_t kv_heads, size_t head_dim, size_t offset)
{
    if (!query || !key || !value || !output || !scratch ||
        !valid_shape(tokens, query_heads, head_dim) ||
        offset > SIZE_MAX - tokens ||
        !valid_shape(tokens + offset, kv_heads, head_dim) || query_heads % kv_heads) {
        return -1;
    }
    size_t group = query_heads / kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < query_heads; h++) {
            const float *q = query + (t * query_heads + h) * head_dim;
            float maximum = -INFINITY;
            for (size_t s = 0; s <= t + offset; s++) {
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
            for (size_t s = 0; s <= t + offset; s++) {
                scratch[s] = expf(scratch[s] - maximum);
                sum += scratch[s];
            }
            for (size_t s = 0; s <= t + offset; s++) {
                scratch[s] /= sum;
            }
            for (size_t d = 0; d < head_dim; d++) {
                float result = 0;
                for (size_t s = 0; s <= t + offset; s++) {
                    result += scratch[s] * value[(s * kv_heads + h / group) * head_dim + d];
                }
                output[(t * query_heads + h) * head_dim + d] = result;
            }
        }
    }
    return 0;
}

int lunelette_causal_gqa(const float *query, const float *key, const float *value,
                         float *output, float *scratch, size_t tokens,
                         size_t query_heads, size_t kv_heads, size_t head_dim)
{
    return lunelette_offset_gqa(query, key, value, output, scratch, tokens,
                                query_heads, kv_heads, head_dim, 0);
}
