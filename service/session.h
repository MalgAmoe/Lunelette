#ifndef LUNELETTE_SERVICE_SESSION_H
#define LUNELETTE_SERVICE_SESSION_H
#include <pthread.h>
#include <stddef.h>
#include "tokenizer_json.h"
#define SERVICE_IDS 256
#define SERVICE_EVENTS 16
#define SERVICE_EVENT_BYTES 1024
/* Worker-only callbacks. next returns one raw synthetic byte token, or EOS=256.
 * materialize must validate the entire committed prefix, not just its length. */
typedef struct {
    void *context;
    int (*next)(void *context);
    int (*materialize)(void *context, const unsigned *tokens, unsigned length);
    void (*reset)(void *context);
    void (*destroy)(void *context);
} service_generator;
typedef struct {
    char id[65];
    char status[16];
    char text[4097];
    unsigned generated, logical, committed, pending;
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
    int cancel, stop, job, reset_job;
} service_session;
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
/* Strict flat object schema, rejecting NUL even in escaped strings. */
int service_json(const void *body, size_t length, const char *fields, tj_document *d);
#endif
