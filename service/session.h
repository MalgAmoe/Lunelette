#ifndef LUNELETTE_SERVICE_SESSION_H
#define LUNELETTE_SERVICE_SESSION_H
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include "tokenizer_json.h"
#define SERVICE_REPLAY_TOKENS 16
#define SERVICE_IDS 256
#define SERVICE_CONTEXT_MAX 4352
#define SERVICE_EVENTS 16
#define SERVICE_EVENT_BYTES 8192
#define SERVICE_METRIC_BYTES 7200
/* Runtime callbacks are worker-only. next returns an authoritative nonnegative ID,
 * or a negative error. Without piece, IDs are raw synthetic bytes/EOS=256.
 * materialize must validate the entire committed prefix, not just its length. */
/* Work v2: full_* names retain source compatibility but mean API prefill;
 * suffix_* mean API append, including rebuild chunks. The three disjoint API
 * categories prefill/append/decode alone sum model tokens. Recompute and pending
 * are subsets (recompute repeats a previously committed prefix after reset).
 * Receipts count actual calls and commits, including failed work. */
typedef struct {
    unsigned reused, full_attempted, full_committed, suffix_attempted, suffix_committed;
    unsigned pending_attempted, pending_committed, prefill_calls, append_calls, decode_calls;
    unsigned decode_attempted, decode_committed, logit_restores;
    unsigned recompute_attempted, recompute_committed;
} service_work;

enum {
    SERVICE_MATERIALIZE_ERROR = 0,
    SERVICE_MATERIALIZE_READY = 1,
    SERVICE_MATERIALIZE_MORE = 2
};

typedef struct {
    void *context;
    int (*next)(void *context);
    /* Opt-in native diagnostic selection; worker-only, authoritative forced ID. */
    int (*replay_next)(void *context, unsigned token);
    int (*materialize)(void *context, const unsigned *tokens, unsigned length);
    void (*reset)(void *context);
    void (*destroy)(void *context);
    /* Optional native framing policy. Immutable CPU callbacks; frame/tool_frame
     * run idle under the session lock, piece runs on the worker. EOS defaults to 256. */
    int (*frame)(void *, const unsigned char *, size_t, int, unsigned *, unsigned *);
    int (*piece)(void *, unsigned, unsigned char *, size_t *);
    /* Opt-in V13 profile: 1 calculator, 2 investigation, 3 observation. tool_frame appends 7 text 8;
     * it never executes a model.
     * output/capacity unchanged on failure, capacity is exact count on success. */
    unsigned native_tools;
    int (*tool_frame)(void *, const unsigned char *, size_t, unsigned *, unsigned *);
    unsigned eos;
    const char *info;
    /* Immutable policy: 0 legacy, 1 reuse on, 2 strict recompute off.
     * Remaining callbacks run only on the worker. begin marks a generation
     * boundary, not a scalar decode. work consumes the last operation receipt.
     * save/restore: 1 success, 0 pre-mutation rejection, -1 recovery required.
     * invalidate releases the owned checkpoint, never executes the model. */
    unsigned prefix_reuse;
    void (*begin)(void *);
    int (*save)(void *);
    int (*restore)(void *);
    void (*invalidate)(void *);
    void (*work)(void *, service_work *);
    unsigned context_capacity, block_capacity;
    /* One completed runtime block only: 0 ERROR, 1 READY, 2 MORE.
     * cursor is the actual committed prefix, including on failure. */
    int (*materialize_step)(void *, const unsigned *, unsigned, unsigned *);
} service_generator;
typedef struct {
    unsigned tool_kind, tool_document; /* 1 calculator, 2 lookup; document 1/2 stock, 3 pump-readings. */
    int tool_a, tool_b;
    unsigned tool_operation; /* 0 absent, 1 add, 2 subtract, 3 multiply; terminal only. */
    char operation[16];
    uint64_t checkpoint_id;
    int cache_restored;
    service_work work;
    char id[65];
    char status[16];
    char text[4097];
    unsigned generated, logical, committed, pending;
    unsigned replay_count, replay_tokens[SERVICE_REPLAY_TOKENS];
    unsigned token_ids[256], prompt, reused, newly_processed;
    double materialize_ms, snapshot_ms, restore_ms;
    double framing_ms, prefill_ms, decode_ms, first_token_ms, total_ms;
    uint64_t decode_us[256], decode_total_us, prefill_us, memory_before, memory_after;
    uint64_t suffix_us;
    unsigned suffix_steps;
    unsigned decode_steps, prefill_steps, session_id, before_logical, before_committed, before_pending;
    double cancel_received_ms, cancel_latency_ms;
    char before_state[16], cancel_source[16];
    char stop_reason[32], state[16];
} service_request;
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t wake;
    pthread_t worker;
    service_generator generator;
    unsigned session, logical, committed, pending, tokens[SERVICE_CONTEXT_MAX], budget;
    unsigned requests, active;
    unsigned replay_reserved; /* Process lifetime: reset/close never reclaim. */
    unsigned assistant_started, outstanding_tool; /* history index + 1; zero absent */
    char state[16];
    service_request history[SERVICE_IDS];
    char events[SERVICE_EVENTS][SERVICE_EVENT_BYTES];
    unsigned head, count;
    unsigned char utf8[4];
    unsigned utf8_length;
    uint64_t startup_footprint;
    double startup_sample_ms;
    double framing_ms;
    int cancel, stop, job, reset_job;
    unsigned checkpoint_tokens[SERVICE_CONTEXT_MAX], checkpoint_length;
    char checkpoint_phase[16];
    uint64_t checkpoint_serial, checkpoint_id;
} service_session;
/* Zero denotes unavailable internally; JSON exposes null, never a fake footprint. */
uint64_t service_process_footprint(void);
int service_init(service_session *s, service_generator generator);
void service_destroy(service_session *s);
/* Called by socket owner only. Reply always JSON, status is HTTP status.
 * stream=1 means admitted generation; caller drains service_event. */
int service_route(service_session *s, const char *method, const char *path,
                  const void *body, size_t length, char *reply, size_t size, int *stream);
/* Transport cannot safely infer request identity from unvalidated input: request_id
 * is the empty string. Error names are internal constant strings, not user input.
 * Takes the session mutex; do not call while holding it. */
void service_transport_error(service_session *s, const char *error, char *reply, size_t size);
int service_event(service_session *s, char *event);
void service_disconnect(service_session *s);
/* Transport captures the admitted generation ID, never targets a later job. */
void service_disconnect_request(service_session *s, const char *id);
/* Strict flat object schema, rejecting NUL even in escaped strings. */
int service_json(const void *body, size_t length, const char *fields, tj_document *d);
#endif
