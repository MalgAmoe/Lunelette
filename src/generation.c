#include "generation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int lunelette_generate_with_logits_capacity(lunelette_logits_fn next, void *context,
    const int64_t *prompt, size_t count, size_t context_limit,
    const lunelette_generation_config *config,
    lunelette_generation_result *result, char *error, size_t capacity)
{
    if (error && capacity) {
        snprintf(error, capacity, context_limit == LUNELETTE_CONTEXT_LIMIT ?
            "generation: require callback, prompt 1..256, config budget 0..256, valid EOS, result" :
            "generation: require callback, context 256 or 4352, prompt within context, config budget 0..256, valid EOS, result");
    }
    if (!next || !prompt || !config || !result || !count || count > context_limit ||
        (context_limit != LUNELETTE_CONTEXT_LIMIT &&
         context_limit != LUNELETTE_LONG_CONTEXT_LIMIT) ||
        config->max_new_tokens > LUNELETTE_GENERATION_LIMIT || config->eos_token < 0 ||
        config->eos_token >= LUNELETTE_VOCAB_SIZE) {
        return -1;
    }

    for (size_t i = 0; i < count; i++) {
        if (prompt[i] < 0 || prompt[i] >= LUNELETTE_VOCAB_SIZE) {
            if (error && capacity) {
                snprintf(error, capacity, "generation: invalid prompt token at %zu", i);
            }
            return -1;
        }
    }
    int64_t local_ids[LUNELETTE_CONTEXT_LIMIT];
    int64_t *ids = local_ids;
    if (context_limit > LUNELETTE_CONTEXT_LIMIT) {
        ids = malloc(context_limit * sizeof(*ids));
    }
    if (!ids) {
        if (error && capacity) {
            snprintf(error, capacity, "generation: prefix allocation failed");
        }
        return -1;
    }
    memcpy(ids, prompt, count * sizeof(*ids));
    lunelette_generation_result pending = {0};
    pending.stop = LUNELETTE_STOP_LIMIT;
    float *logits = NULL;
    if (config->max_new_tokens && count < context_limit) {
        logits = malloc(LUNELETTE_VOCAB_SIZE * sizeof(float));
        if (!logits) {
            if (error && capacity) {
                snprintf(error, capacity, "generation: logits allocation failed");
            }
            if (ids != local_ids) {
                free(ids);
            }
            return -1;
        }
    }
    while (pending.count < config->max_new_tokens) {
        if (count == context_limit) {
            pending.stop = LUNELETTE_STOP_CONTEXT;
            break;
        }
        if (error && capacity) {
            snprintf(error, capacity, "generation: logits callback failed at step %zu", pending.count);
        }
        if (next(context, ids, count, logits, error, capacity)) {
            if (ids != local_ids) {
                free(ids);
            }
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
                if (ids != local_ids) {
                    free(ids);
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
    if (ids != local_ids) {
        free(ids);
    }
    free(logits);
    *result = pending;
    if (error && capacity) {
        error[0] = '\0';
    }
    return 0;
}

int lunelette_generate_with_logits(lunelette_logits_fn next, void *context,
    const int64_t *prompt, size_t count, const lunelette_generation_config *config,
    lunelette_generation_result *result, char *error, size_t capacity)
{
    return lunelette_generate_with_logits_capacity(next, context, prompt, count,
        LUNELETTE_CONTEXT_LIMIT, config, result, error, capacity);
}
