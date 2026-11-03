#include "cache.h"
#include "../src/forward_internal.h"
#include "attention.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "CHECK line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
int main(void)
{
    float q[256 * 4 * 8], k[256 * 2 * 8], v[256 * 2 * 8];
    float full[256 * 4 * 8], step[4 * 8], scratch[256];
    for (size_t i = 0; i < sizeof q / sizeof *q; i++) {
        q[i] = sinf((float)i) * 0.2f;
    }
    for (size_t i = 0; i < sizeof k / sizeof *k; i++) {
        k[i] = cosf((float)i) * 0.3f;
        v[i] = sinf((float)i * 0.1f);
    }
    CHECK(!lunelette_causal_gqa(q, k, v, full, scratch, 256, 4, 2, 8));
    for (size_t t = 0; t < 256; t++) {
        CHECK(!lunelette_offset_gqa(q + t * 32, k, v, step, scratch, 1, 4, 2, 8, t));
        for (size_t d = 0; d < 32; d++) {
            CHECK(step[d] == full[t * 32 + d]);
        }
    }
    CHECK(lunelette_offset_gqa(q, k, v, step, scratch, 1, 4, 2, 8, SIZE_MAX));
    CHECK(lunelette_offset_gqa(q, k, v, step, scratch, 1, 3, 2, 8, 0));
    char error[256];
    float logits[1];
    int64_t id = 1;
    CHECK(!lunelette_cache_init(NULL, NULL, 256, error, sizeof error));
    CHECK(lunelette_cache_forward(NULL, &id, 1, logits, error, sizeof error));
    CHECK(lunelette_cache_length(NULL) == 0);
    lunelette_cache_reset(NULL);
    lunelette_cache_free(NULL);
    /* Shape-only synthetic shards: deliberately absent payload exercises I/O
     * rollback without allocating or bundling any model weights. */
    st_tensor tensors[236] = {0};
    st_file first = {0}, second = {0};
    first.file = tmpfile();
    second.file = tmpfile();
    CHECK(first.file && second.file);
    first.tensors = tensors;
    first.count = 236;
    const char *suffix[] = {
        "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
    };
    size_t rows[] = {3072, 4096, 1024, 1024, 3072, 3072, 9216, 9216, 3072};
    size_t cols[] = {0, 3072, 3072, 3072, 4096, 0, 3072, 3072, 9216};
    strcpy(tensors[0].name, "language_model.model.embed_tokens.weight");
    tensors[0].shape[0] = 131072;
    tensors[0].shape[1] = 3072;
    strcpy(tensors[1].name, "language_model.model.norm.weight");
    tensors[1].shape[0] = 3072;
    for (size_t layer = 0; layer < 26; layer++) {
        for (size_t j = 0; j < 9; j++) {
            st_tensor *tensor = tensors + 2 + layer * 9 + j;
            snprintf(tensor->name, sizeof tensor->name,
                "language_model.model.layers.%zu.%s", layer, suffix[j]);
            tensor->shape[0] = rows[j];
            tensor->shape[1] = cols[j];
        }
    }
    for (size_t i = 0; i < 236; i++) {
        tensors[i].dtype = ST_BF16;
        tensors[i].rank = tensors[i].shape[1] ? 2 : 1;
    }
    CHECK(!lunelette_cache_init(&first, &first, 1, error, sizeof error));
    CHECK(!lunelette_cache_init(&first, &second, 0, error, sizeof error));
    CHECK(!lunelette_cache_init(&first, &second, 257, error, sizeof error));
    for (size_t bound = 1; bound <= 256; bound += 255) {
        lunelette_cache *cache = lunelette_cache_init(&first, &second, bound, error, sizeof error);
        CHECK(cache);
        float *scores = malloc(131072 * sizeof(float));
        CHECK(scores);
        scores[0] = 123.0f;
        int64_t bad = -1;
        CHECK(lunelette_cache_forward(cache, &bad, 1, scores, error, sizeof error));
        CHECK(lunelette_cache_forward(cache, &id, bound + 1, scores, error, sizeof error));
        CHECK(lunelette_cache_forward(cache, &id, 1, scores, error, sizeof error));
        CHECK(scores[0] == 123.0f && lunelette_cache_length(cache) == 0);
        lunelette_cache_reset(cache);
        CHECK(lunelette_cache_forward(cache, &id, 1, scores, error, sizeof error));
        CHECK(lunelette_cache_length(cache) == 0);
        lunelette_generation_config config = {0, 2};
        lunelette_generation_result result = {0};
        if (bound == 256) {
            CHECK(!lunelette_generate_cached(cache, &id, 1, &config, &result, error, sizeof error));
            CHECK(result.count == 0);
            config.eos_token = -1;
            CHECK(lunelette_generate_cached(cache, &id, 1, &config, &result, error, sizeof error));
        } else {
            CHECK(lunelette_generate_cached(cache, &id, 1, &config, &result, error, sizeof error));
        }
        free(scores);
        lunelette_cache_free(cache);
    }
    second.count = 1;
    second.tensors = tensors;
    CHECK(!lunelette_cache_init(&first, &second, 256, error, sizeof error));
    fclose(first.file);
    fclose(second.file);
    puts("cache synthetic boundary attention: exact match through 256");
    return 0;
}
