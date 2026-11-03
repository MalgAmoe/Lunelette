#include "attention.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rope_golden.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static void rotation(void)
{
    float q[5 * 4096] = {0}, k[5 * 1024] = {0};
    const uint32_t positions[] = {0, 1, 16383, 16384, 16385};
    for (size_t t = 0; t < 5; t++) {
        for (size_t h = 0; h < 32; h++) {
            q[t * 4096 + h * 128] = 1;
            q[t * 4096 + h * 128 + 64] = 2;
        }
        for (size_t h = 0; h < 8; h++) {
            k[t * 1024 + h * 128] = 1;
            k[t * 1024 + h * 128 + 64] = 2;
        }
    }
    CHECK(!lunelette_ministral3_yarn(q, k, positions, 5));
    for (size_t t = 0; t < 5; t++) {
        float c = cosf((float)positions[t]), s = sinf((float)positions[t]);
        float scale = 1 + 0.1f * logf(1 + (float)(positions[t] / 16384));
        for (size_t h = 0; h < 32; h++) {
            CHECK(fabsf(q[t * 4096 + h * 128] - (c - 2 * s) * scale) < 1e-6f);
            CHECK(fabsf(q[t * 4096 + h * 128 + 64] - (2 * c + s) * scale) < 1e-6f);
            CHECK(q[t * 4096 + h * 128 + 1] == 0);
        }
        for (size_t h = 0; h < 8; h++) {
            CHECK(fabsf(k[t * 1024 + h * 128] - (c - 2 * s)) < 1e-6f);
            CHECK(fabsf(k[t * 1024 + h * 128 + 64] - (2 * c + s)) < 1e-6f);
        }
    }
    /* Every head uses the same split-half layout; exercise transition/tail
     * frequencies independently of the first-dimension trigonometric check. */
    memset(q, 0, sizeof q);
    memset(k, 0, sizeof k);
    const size_t indices[] = {1, 24, 40, 63};
    for (size_t t = 0; t < 5; t++) {
        for (size_t j = 0; j < 4; j++) {
            q[t * 4096 + indices[j]] = 1;
            k[t * 1024 + indices[j]] = 1;
        }
    }
    CHECK(!lunelette_ministral3_yarn(q, k, positions, 5));
    for (size_t t = 0; t < 5; t++) {
        float scale = 1 + 0.1f * logf(1 + (float)(positions[t] / 16384));
        for (size_t j = 0; j < 4; j++) {
            size_t i = indices[j];
            CHECK(fabsf(k[t * 1024 + i] - rope_golden[t][j][0]) < 2e-5f);
            CHECK(fabsf(k[t * 1024 + i + 64] - rope_golden[t][j][1]) < 2e-5f);
            CHECK(fabsf(q[t * 4096 + i] - rope_golden[t][j][0] * scale) < 2e-5f);
            CHECK(fabsf(q[t * 4096 + i + 64] - rope_golden[t][j][1] * scale) < 2e-5f);
        }
    }
    float before = q[0];
    uint32_t bad[] = {0, 262144};
    CHECK(lunelette_ministral3_yarn(q, k, bad, 2) == -1);
    CHECK(q[0] == before);
    CHECK(lunelette_ministral3_yarn(NULL, k, positions, 1) == -1);
    CHECK(lunelette_ministral3_yarn(q, NULL, positions, 1) == -1);
    CHECK(lunelette_ministral3_yarn(q, k, NULL, 1) == -1);
    CHECK(lunelette_ministral3_yarn(q, k, positions, 0) == -1);
    CHECK(lunelette_ministral3_yarn(q, k, positions, SIZE_MAX) == -1);
}

static void attention(void)
{
    float q[3 * 32 * 128] = {0}, k[3 * 8 * 128] = {0};
    float v[3 * 8 * 128], out[3 * 32 * 128], scratch[3];
    for (size_t t = 0; t < 3; t++) {
        for (size_t h = 0; h < 8; h++) {
            for (size_t d = 0; d < 128; d++) {
                v[(t * 8 + h) * 128 + d] = (float)(100 * t + 10 * h + d);
            }
        }
    }
    CHECK(!lunelette_causal_gqa(q, k, v, out, scratch, 3, 32, 8, 128));
    for (size_t t = 0; t < 3; t++) {
        for (size_t h = 0; h < 32; h++) {
            for (size_t d = 0; d < 128; d++) {
                CHECK(fabsf(out[(t * 32 + h) * 128 + d] -
                      (float)(50 * t + 10 * (h / 4) + d)) < 4e-5f);
            }
        }
    }
    /* Finite scores far beyond naive exp range; row 0 ignores future keys. */
    float sq[] = {100, 100}, sk[] = {100, -100}, sv[] = {3, 99}, so[2];
    CHECK(!lunelette_causal_gqa(sq, sk, sv, so, scratch, 2, 1, 1, 1));
    CHECK(so[0] == 3 && so[1] == 3);
    sk[1] = 100;
    CHECK(!lunelette_causal_gqa(sq, sk, sv, so, scratch, 2, 1, 1, 1));
    CHECK(so[0] == 3 && so[1] == 51);
    /* Distinct queries and KV groups: neither modulo head mapping nor using
     * one query for all heads can satisfy these nonuniform probabilities. */
    float gq[] = {0, 0, 0, 0, 1, -1, 2, -2};
    float gk[] = {0, 1, 1, 0}, gv[] = {0, 10, 4, 20}, go[8];
    CHECK(!lunelette_causal_gqa(gq, gk, gv, go, scratch, 2, 4, 2, 1));
    for (size_t h = 0; h < 4; h++) {
        float delta = h < 2 ? gq[4 + h] : -gq[4 + h];
        float probability = 1 / (1 + expf(-delta));
        float expected = h < 2 ? 4 * probability : 10 + 10 * probability;
        CHECK(fabsf(go[4 + h] - expected) < 2e-6f);
        CHECK(go[h] == (h < 2 ? 0 : 10));
    }
    sk[0] = NAN;
    CHECK(lunelette_causal_gqa(sq, sk, sv, so, scratch, 2, 1, 1, 1) == -1);
    out[0] = 123;
    CHECK(lunelette_causal_gqa(q, k, v, out, scratch, 3, 3, 2, 128) == -1);
    CHECK(lunelette_causal_gqa(q, k, v, out, scratch, SIZE_MAX, 32, 8, 128) == -1);
    CHECK(lunelette_causal_gqa(q, k, v, out, scratch, 1, SIZE_MAX, 1, 128) == -1);
    CHECK(lunelette_causal_gqa(q, k, v, out, scratch, 1, 1, 1, SIZE_MAX) == -1);
    CHECK(lunelette_causal_gqa(q, k, v, out, scratch, 0, 32, 8, 128) == -1);
    CHECK(lunelette_causal_gqa(q, k, v, out, scratch, 1, 0, 8, 128) == -1);
    CHECK(lunelette_causal_gqa(q, k, v, out, scratch, 1, 32, 0, 128) == -1);
    CHECK(lunelette_causal_gqa(q, k, v, out, scratch, 1, 32, 8, 0) == -1);
    CHECK(lunelette_causal_gqa(NULL, k, v, out, scratch, 1, 32, 8, 128) == -1);
    CHECK(lunelette_causal_gqa(q, NULL, v, out, scratch, 1, 32, 8, 128) == -1);
    CHECK(lunelette_causal_gqa(q, k, NULL, out, scratch, 1, 32, 8, 128) == -1);
    CHECK(lunelette_causal_gqa(q, k, v, NULL, scratch, 1, 32, 8, 128) == -1);
    CHECK(lunelette_causal_gqa(q, k, v, out, NULL, 1, 32, 8, 128) == -1);
    CHECK(out[0] == 123);
}

int main(void)
{
    rotation();
    attention();
    puts("rotation and causal GQA tests passed");
    return 0;
}
