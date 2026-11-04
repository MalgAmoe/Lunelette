#include "forward.h"
#include "../src/forward_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

/* Metadata-only handles intentionally stop at I/O. This lets every supported
 * shape be checked without allocating multi-gigabyte synthetic checkpoint files.
 * File parsing, overlapping byte ranges and duplicate JSON keys are separately
 * covered by test_safetensors. */
static void descriptor(st_tensor *tensor, const char *name, size_t rows, size_t cols)
{
    memset(tensor, 0, sizeof *tensor);
    snprintf(tensor->name, sizeof tensor->name, "%s", name);
    tensor->dtype = ST_BF16;
    tensor->rank = cols ? 2 : 1;
    tensor->shape[0] = rows;
    tensor->shape[1] = cols;
    tensor->end = rows * (cols ? cols : 1) * 2;
}

int main(void)
{
    /* Exact squares total 100003071, unlike sequential float accumulation
     * after the 1e8 leading term. Check the explicit float-mean rounding point. */
    float norm_input[3072], norm_weight[3072], norm_output[3072];
    for (size_t i = 0; i < 3072; i++) {
        norm_input[i] = i ? 1.0f : 10000.0f;
        norm_weight[i] = 1.0f;
    }
    float scale = 1.0f / sqrtf((float)(100003071.0 / 3072) + 1e-5f);
    lunelette_hidden_norm(norm_input, norm_weight, norm_output);
    CHECK(norm_output[0] == 10000.0f * scale);
    CHECK(norm_output[3071] == scale);
    lunelette_hidden_norm(norm_input, norm_weight, norm_input);
    CHECK(memcmp(norm_input, norm_output, sizeof norm_input) == 0);
    st_tensor tensors[238];
    descriptor(tensors, "language_model.model.embed_tokens.weight", 131072, 3072);
    descriptor(tensors + 1, "language_model.model.norm.weight", 3072, 0);
    const char *suffix[] = {
        "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
    };
    const size_t rows[] = {3072, 4096, 1024, 1024, 3072, 3072, 9216, 9216, 3072};
    const size_t cols[] = {0, 3072, 3072, 3072, 4096, 0, 3072, 3072, 9216};
    for (size_t layer = 0; layer < 26; layer++) {
        for (size_t i = 0; i < 9; i++) {
            char name[128];
            snprintf(name, sizeof name, "language_model.model.layers.%zu.%s", layer, suffix[i]);
            descriptor(tensors + 2 + layer * 9 + i, name, rows[i], cols[i]);
        }
    }
    st_file first = {0}, second = {0};
    first.file = tmpfile();
    second.file = tmpfile();
    CHECK(first.file && second.file);
    first.count = 120;
    first.tensors = tensors;
    second.count = 116;
    second.tensors = tensors + 120;
    /* Independent literal fixtures above pin the checkpoint contract rather than
     * manufacturing expected shapes from the descriptor being tested. */
    CHECK(LUNELETTE_MODEL_HIDDEN == 3072 && LUNELETTE_MODEL_LAYERS == 26);
    CHECK(LUNELETTE_MODEL_VOCAB == 131072 && LUNELETTE_MODEL_INNER == 9216);
    CHECK(LUNELETTE_MODEL_QUERY == LUNELETTE_MODEL_QUERY_HEADS * LUNELETTE_MODEL_HEAD_DIM);
    CHECK(LUNELETTE_MODEL_KV == LUNELETTE_MODEL_KV_HEADS * LUNELETTE_MODEL_HEAD_DIM);
    CHECK(LUNELETTE_MODEL_QUERY_HEADS == 32 && LUNELETTE_MODEL_KV_HEADS == 8);
    CHECK(LUNELETTE_MODEL_HEAD_DIM == 128 && LUNELETTE_MODEL_TIED_EMBEDDINGS == 1);
    CHECK(LUNELETTE_MODEL_NORM_EPS == 1e-5f);
    CHECK(LUNELETTE_CONTEXT_LIMIT == 256 && LUNELETTE_MODEL_MAX_POSITION == 262144);
    CHECK(LUNELETTE_CONTEXT_LIMIT < LUNELETTE_MODEL_MAX_POSITION);
    for (size_t i = 0; i < 9; i++) {
        CHECK(strcmp(lunelette_model_block_weights[i].suffix, suffix[i]) == 0);
        CHECK(lunelette_model_block_weights[i].rows == rows[i]);
        CHECK(lunelette_model_block_weights[i].cols == cols[i]);
    }
    st_file *layout_files[] = {&first, &second};
    char layout_error[512];
    CHECK(lunelette_cpu_validate_layout(layout_files, layout_error, sizeof layout_error) == 0);
    /* Even malformed/duplicate unrelated vision metadata is outside this text contract. */
    descriptor(tensors + 236, "vision_tower.ignored.weight", 7, 3);
    tensors[236].dtype = ST_F32;
    tensors[237] = tensors[236];
    second.count += 2;
    CHECK(lunelette_cpu_validate_layout(layout_files, layout_error, sizeof layout_error) == 0);
    tensors[237] = tensors[0];
    CHECK(lunelette_cpu_validate_layout(layout_files, layout_error, sizeof layout_error) == -1);
    CHECK(strstr(layout_error, "duplicate"));
    second.count -= 2;
    int64_t ids[256] = {0};
    static float hidden[256 * 3072], logits[131072], checkpoint[256 * 3072];
    for (size_t i = 0; i < 256 * 3072; i++) {
        hidden[i] = checkpoint[i] = 123;
    }
    for (size_t i = 0; i < 131072; i++) {
        logits[i] = 123;
    }
    char error[512];
#define REJECT(a,b,c,n,h,l,k) CHECK(lunelette_forward(a,b,c,n,h,l,k,error,sizeof error) == -1)
    REJECT(NULL, &second, ids, 1, hidden, logits, checkpoint);
    REJECT(&first, NULL, ids, 1, hidden, logits, checkpoint);
    REJECT(&first, &first, ids, 1, hidden, logits, checkpoint);
    REJECT(&first, &second, NULL, 1, hidden, logits, checkpoint);
    REJECT(&first, &second, ids, 1, NULL, logits, checkpoint);
    REJECT(&first, &second, ids, 1, hidden, NULL, checkpoint);
    REJECT(&first, &second, ids, 0, hidden, logits, checkpoint);
    REJECT(&first, &second, ids, 257, hidden, logits, checkpoint);
    REJECT(&first, &second, ids, SIZE_MAX, hidden, logits, checkpoint);
    ids[0] = -1;
    REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
    ids[0] = 131072;
    REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
    ids[0] = 131071;
    REJECT(&first, &second, ids, 256, hidden, logits, checkpoint);
    CHECK(strstr(error, "embedding: read failed")); /* Valid upper bounds reach I/O. */
    ids[0] = 0;
    REJECT(&second, &first, ids, 1, hidden, logits, checkpoint);
    CHECK(strstr(error, "embedding: read failed"));
    CHECK(lunelette_forward(NULL, &second, ids, 1, hidden, logits, NULL, NULL, 0) == -1);
    REJECT(&first, &second, ids, 1, hidden, logits, NULL);
    CHECK(strstr(error, "embedding: read failed"));
    /* Check every required tensor's dimensions and dtype, including layer 25.
     * Validation must finish before attempting even the first embedding read. */
    for (size_t i = 0; i < 236; i++) {
        tensors[i].shape[0]--;
        REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
        CHECK(strstr(error, tensors[i].name) && strstr(error, "expected BF16"));
        tensors[i].shape[0]++;
        tensors[i].dtype = ST_F32;
        REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
        CHECK(strstr(error, tensors[i].name));
        tensors[i].dtype = ST_BF16;
        tensors[i].rank++;
        REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
        CHECK(strstr(error, tensors[i].name));
        tensors[i].rank--;
        if (tensors[i].rank == 2) {
            tensors[i].shape[1]--;
            REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
            CHECK(strstr(error, tensors[i].name));
            tensors[i].shape[1]++;
        }
    }
    second.count--;
    REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
    CHECK(strstr(error, "layers.25.mlp.down_proj.weight"));
    second.count++;
    /* Duplicate a required tensor between shards, and within a shard. */
    st_tensor saved = tensors[235];
    tensors[235] = tensors[0];
    REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
    CHECK(strstr(error, "duplicate"));
    tensors[235] = tensors[234];
    REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
    CHECK(strstr(error, "duplicate"));
    tensors[235] = saved;
    /* A sparse all-zero matrix backing store exercises successful reusable
     * execution at both endpoint layer indices, with a nonzero residual.
     * These private synthetic descriptors intentionally share zero storage. */
    CHECK(fseek(first.file, (long)(9216 * 3072 * 2 - 1), SEEK_SET) == 0);
    CHECK(fputc(0, first.file) == 0);
    CHECK(fflush(first.file) == 0);
    st_file *files[] = {&first};
    first.count = 236;
    float input[3072], output[3072];
    for (size_t i = 0; i < 3072; i++) {
        input[i] = (float)((int)(i % 7) - 3) * 0.125f;
        output[i] = 123;
    }
    CHECK(lunelette_layer(files, 1, 0, input, 1, output, error, sizeof error) == 0);
    CHECK(memcmp(input, output, sizeof input) == 0);
    CHECK(lunelette_layer(files, 1, 25, output, 1, output, error, sizeof error) == 0);
    CHECK(memcmp(input, output, sizeof input) == 0);
    CHECK(lunelette_layer(files, 1, 26, input, 1, output, error, sizeof error) == -1);
    tensors[235].begin = 9216 * 3072 * 2;
    tensors[235].end += tensors[235].begin;
    CHECK(lunelette_layer(files, 1, 25, input, 1, output, error, sizeof error) == -1);
    CHECK(strstr(error, "layers.25.mlp.down_proj.weight"));
    CHECK(memcmp(input, output, sizeof input) == 0);
    tensors[235] = saved;
    second.count = 0;
    /* Full execution reaches tied projection, then short backing storage forces
     * a late failure. All three public outputs must still be untouched. */
    REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
    CHECK(strstr(error, "tied logits: embedding row"));
    FILE *second_stream = second.file;
    second.file = NULL;
    REJECT(&first, &second, ids, 1, hidden, logits, checkpoint);
    /* Restore the owned stream without st_close on stack-owned descriptors. */
    for (size_t i = 0; i < 256 * 3072; i++) {
        CHECK(hidden[i] == 123 && checkpoint[i] == 123);
    }
    for (size_t i = 0; i < 131072; i++) {
        CHECK(logits[i] == 123);
    }
    fclose(first.file);
    fclose(second_stream);
    puts("forward argument, layout, duplicate and failure-atomicity tests passed");
    return 0;
}
