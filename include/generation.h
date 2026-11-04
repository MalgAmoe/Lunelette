#ifndef LUNELETTE_GENERATION_H
#define LUNELETTE_GENERATION_H

#include "forward.h"

#include "model_facts.h"
#include "runtime_limits.h"

#define LUNELETTE_VOCAB_SIZE LUNELETTE_MODEL_VOCAB

typedef struct {
    size_t max_new_tokens; /* 0..256 */
    int64_t eos_token;     /* 0..131071; default 2 */
} lunelette_generation_config;

#define LUNELETTE_GENERATION_DEFAULT {32, 2}

typedef enum {
    LUNELETTE_STOP_LIMIT,
    LUNELETTE_STOP_EOS,
    LUNELETTE_STOP_CONTEXT
} lunelette_stop_reason;

typedef struct {
    int64_t tokens[LUNELETTE_CONTEXT_LIMIT]; /* New tokens only; includes generated EOS. */
    size_t count;
    lunelette_stop_reason stop;
} lunelette_generation_result;

/* Callback writes all 131072 last-token logits on success, returns 0 or -1.
 * IDs and scratch logits are borrowed only during the call. No retained pointers.
 * Also useful for observing/comparing each step; callback side effects cannot be
 * rolled back on failure. It must not modify config, prompt, or result.
 */
typedef int (*lunelette_logits_fn)(void *context, const int64_t *ids, size_t count,
                                  float *logits, char *error, size_t capacity);

/* Recompute the full prefix for each new token. Prompt length 1..256; all IDs
 * validated even for zero budget. Prompt EOS is not a stop condition. Never
 * returns more than 256 total tokens (prompt + new); a full prompt needs no call.
 * EOS takes precedence over budget/context when simultaneously reached, then
 * token budget takes precedence over context. Exact ties choose lowest ID;
 * any NaN/Inf anywhere in logits rejects the step.
 * All pointers required except context and error. Config is explicit (use the
 * default initializer above). Result is wholly unchanged on failure, including
 * after earlier successful steps; caller owns it, no close/free needed. Inputs,
 * result and error must be disjoint. Scratch is internal; reentrant subject to
 * callback/context safety. Returns 0 or -1 with diagnostic when supplied.
 */
int lunelette_generate_with_logits(lunelette_logits_fn next, void *context,
    const int64_t *prompt, size_t count, const lunelette_generation_config *config,
    lunelette_generation_result *result, char *error, size_t capacity);

/* Same contract using existing uncached forward. Shards borrowed, not concurrent.
 * No tokenizer, cache, or sampling. */
int lunelette_generate(st_file *shard1, st_file *shard2,
    const int64_t *prompt, size_t count, const lunelette_generation_config *config,
    lunelette_generation_result *result, char *error, size_t capacity);

#endif
