#include "generation.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int lunelette_generate_with_logits(lunelette_logits_fn next, void *context,
    const int64_t *prompt, size_t count, const lunelette_generation_config *config,
    lunelette_generation_result *result, char *error, size_t capacity)
{
    if (error && capacity) {
        snprintf(error, capacity, "generation: require callback, prompt 1..256, config budget 0..256, valid EOS, result");
    }
    if (!next || !prompt || !config || !result || !count || count > 256 ||
        config->max_new_tokens > 256 || config->eos_token < 0 ||
        config->eos_token >= LUNELETTE_VOCAB_SIZE) {
        return -1;
    }
    int64_t ids[256];
    for (size_t i = 0; i < count; i++) {
        if (prompt[i] < 0 || prompt[i] >= LUNELETTE_VOCAB_SIZE) {
            if (error && capacity) {
                snprintf(error, capacity, "generation: invalid prompt token at %zu", i);
            }
            return -1;
        }
        ids[i] = prompt[i];
    }
    lunelette_generation_result pending = {0};
    pending.stop = LUNELETTE_STOP_LIMIT;
    float *logits = NULL;
    if (config->max_new_tokens && count < 256) {
        logits = malloc(LUNELETTE_VOCAB_SIZE * sizeof(float));
        if (!logits) {
            if (error && capacity) {
                snprintf(error, capacity, "generation: logits allocation failed");
            }
            return -1;
        }
    }
    while (pending.count < config->max_new_tokens) {
        if (count == 256) {
            pending.stop = LUNELETTE_STOP_CONTEXT;
            break;
        }
        if (error && capacity) {
            snprintf(error, capacity, "generation: logits callback failed at step %zu", pending.count);
        }
        if (next(context, ids, count, logits, error, capacity)) {
            free(logits);
            return -1;
        }
        size_t best = 0;
        for (size_t i = 0; i < LUNELETTE_VOCAB_SIZE; i++) {
            if (!isfinite(logits[i])) {
                if (error && capacity) {
                    snprintf(error, capacity, "generation: nonfinite logit at step %zu token %zu",
                             pending.count, i);
                }
                free(logits);
                return -1;
            }
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        ids[count++] = (int64_t)best;
        pending.tokens[pending.count++] = (int64_t)best;
        if ((int64_t)best == config->eos_token) {
            pending.stop = LUNELETTE_STOP_EOS;
            break;
        }
    }
    free(logits);
    *result = pending;
    if (error && capacity) {
        error[0] = '\0';
    }
    return 0;
}

struct forward_context {
    st_file *first;
    st_file *second;
    float *hidden;
};

static int next_forward(void *opaque, const int64_t *ids, size_t count,
                        float *logits, char *error, size_t capacity)
{
    struct forward_context *context = opaque;
    return lunelette_forward(context->first, context->second, ids, count,
                             context->hidden, logits, NULL, error, capacity);
}

int lunelette_generate(st_file *shard1, st_file *shard2,
    const int64_t *prompt, size_t count, const lunelette_generation_config *config,
    lunelette_generation_result *result, char *error, size_t capacity)
{
    if (!shard1 || !shard2 || shard1 == shard2 || !shard1->file || !shard2->file ||
        shard1->file == shard2->file) {
        if (error && capacity) {
            snprintf(error, capacity, "generation: require two distinct open shards");
        }
        return -1;
    }
    struct forward_context context = {shard1, shard2, NULL};
    context.hidden = malloc(256 * 3072 * sizeof(float));
    if (!context.hidden) {
        if (error && capacity) {
            snprintf(error, capacity, "generation: hidden allocation failed");
        }
        return -1;
    }
    int rc = lunelette_generate_with_logits(next_forward, &context, prompt, count,
                                            config, result, error, capacity);
    free(context.hidden);
    return rc;
}
