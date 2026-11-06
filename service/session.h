#ifndef LUNELETTE_SERVICE_SESSION_H
#define LUNELETTE_SERVICE_SESSION_H
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include "tokenizer_json.h"
#define SERVICE_IDS 256
#define SERVICE_EVENTS 16
#define SERVICE_EVENT_BYTES 8192
#define SERVICE_METRIC_BYTES 7200
/* Runtime callbacks are worker-only. next returns an authoritative nonnegative ID,
 * or a negative error. Without piece, IDs are raw synthetic bytes/EOS=256.
 * materialize must validate the entire committed prefix, not just its length. */
/* Receipts count actual calls/commits, not plans. pending_* is a subset of
 * executed full/suffix work, or the single scalar decode token; do not add it
 * again to full/suffix totals. Total model tokens are full + suffix + decode
 * (each with attempted/committed variants). A logit restore is an actual runtime copy, not
 * a model decode or a sample. Missing work callback denotes diagnostic legacy
 * instrumentation only. */
typedef struct {
    unsigned reused, full_attempted, full_committed, suffix_attempted, suffix_committed;
    unsigned pending_attempted, pending_committed, prefill_calls, append_calls, decode_calls;
    unsigned decode_attempted, decode_committed, logit_restores;
} service_work;

typedef struct {
    void *context;
    int (*next)(void *context);
    int (*materialize)(void *context, const unsigned *tokens, unsigned length);
    void (*reset)(void *context);
    void (*destroy)(void *context);
    /* Optional native text-only policy. Immutable CPU callbacks; frame runs idle
     * under the session lock, piece runs on the worker. EOS defaults to 256. */
    int (*frame)(void *, const unsigned char *, size_t, int, unsigned *, unsigned *);
    int (*piece)(void *, unsigned, unsigned char *, size_t *);
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
} service_generator;
typedef struct {
    char operation[16];
    uint64_t checkpoint_id;
    int cache_restored;
    service_work work;
    char id[65];
    char status[16];
    char text[4097];
    unsigned generated, logical, committed, pending;
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
    unsigned session, logical, committed, pending, tokens[256], budget;
    unsigned requests, active;
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
    unsigned checkpoint_tokens[256], checkpoint_length;
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
