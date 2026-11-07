#define _POSIX_C_SOURCE 200809L
#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t wake;
    unsigned cursor, calls, samples, saves, barrier;
    int entered, released, failed;
    service_work work;
} stub;
static int step(void *context, const unsigned *ids, unsigned length, unsigned *cursor)
{
    stub *f = context;
    CHECK(ids[0] == 42);
    unsigned count = length - f->cursor;
    if (count > 128) {
        count = 128;
    }
    memset(&f->work, 0, sizeof(f->work));
    if (!f->cursor) {
        f->work.prefill_calls = 1;
        f->work.full_attempted = count;
    } else {
        f->work.append_calls = 1;
        f->work.suffix_attempted = count;
    }
    f->calls++;
    if (!f->failed || f->calls != f->barrier) {
        f->cursor += count;
        f->work.full_committed = f->work.full_attempted;
        f->work.suffix_committed = f->work.suffix_attempted;
    }
    *cursor = f->cursor;
    pthread_mutex_lock(&f->lock);
    if (f->calls == f->barrier) {
        f->entered = 1;
        pthread_cond_broadcast(&f->wake);
        while (!f->released) {
            pthread_cond_wait(&f->wake, &f->lock);
        }
    }
    pthread_mutex_unlock(&f->lock);
    return f->failed && f->calls == f->barrier ? 0 : f->cursor == length ? 1 : 2;
}
static int next(void *context)
{
    ((stub *)context)->samples++;
    return 'X';
}
static int save(void *context)
{
    ((stub *)context)->saves++;
    return 1;
}
static void reset(void *context)
{
    ((stub *)context)->cursor = 0;
}
static void destroy(void *context)
{
    (void)context;
}
static void work(void *context, service_work *out)
{
    *out = ((stub *)context)->work;
}
static void exercise(unsigned barrier, int saving, int failed)
{
    stub f = {.barrier = barrier, .failed = failed};
    CHECK(!pthread_mutex_init(&f.lock, NULL));
    CHECK(!pthread_cond_init(&f.wake, NULL));
    service_session *s = calloc(1, sizeof(*s));
    CHECK(s);
    service_generator g = {.context = &f, .next = next, .reset = reset, .destroy = destroy,
        .materialize_step = step, .work = work, .context_capacity = 4352, .block_capacity = 128,
        .prefix_reuse = 1, .save = save, .restore = save};
    CHECK(service_init(s, g));
    char reply[65536];
    int stream;
    CHECK(service_route(s, "POST", "/sessions", "{}", 2, reply, sizeof(reply), &stream) == 200);
    pthread_mutex_lock(&s->lock);
    s->logical = 257;
    for (unsigned i = 0; i < 257; i++) {
        s->tokens[i] = 42;
    }
    pthread_mutex_unlock(&s->lock);
    const char *body = "{\"id\":\"blocked\",\"budget\":1}";
    if (saving) {
        body = "{\"id\":\"blocked\"}";
    }
    CHECK(service_route(s, "POST", saving ? "/sessions/1/checkpoint" : "/sessions/1/generate",
        body, strlen(body), reply, sizeof(reply), &stream) == (saving ? 202 : 200));
    pthread_mutex_lock(&f.lock);
    struct timespec deadline;
    CHECK(!clock_gettime(CLOCK_REALTIME, &deadline));
    deadline.tv_sec += 5;
    while (!f.entered) {
        CHECK(!pthread_cond_timedwait(&f.wake, &f.lock, &deadline));
    }
    pthread_mutex_unlock(&f.lock);
    CHECK(service_route(s, "POST", "/requests/blocked/cancel", "{}", 2,
                        reply, sizeof(reply), &stream) == 200);
    pthread_mutex_lock(&f.lock);
    f.released = 1;
    pthread_cond_broadcast(&f.wake);
    pthread_mutex_unlock(&f.lock);
    for (unsigned i = 0; i < 5000; i++) {
        pthread_mutex_lock(&s->lock);
        int done = !s->job;
        pthread_mutex_unlock(&s->lock);
        if (done) {
            break;
        }
        struct timespec delay = {0, 1000000};
        nanosleep(&delay, NULL);
    }
    pthread_mutex_lock(&s->lock);
    CHECK(!s->job && !strcmp(s->state, "needs_reset"));
    CHECK(f.calls == barrier && !f.samples && !f.saves);
    service_request *r = &s->history[s->active];
    CHECK(r->work.prefill_calls == 1 && r->work.append_calls == barrier - 1);
    CHECK(r->work.full_attempted + r->work.suffix_attempted == (barrier == 3 ? 257 : 128 * barrier));
    CHECK(r->work.full_committed + r->work.suffix_committed ==
        (failed ? 128 * (barrier - 1) : barrier == 3 ? 257 : 128 * barrier));
    pthread_mutex_unlock(&s->lock);
    service_destroy(s);
    free(s);
    pthread_cond_destroy(&f.wake);
    pthread_mutex_destroy(&f.lock);
}
int main(void)
{
    for (unsigned b = 1; b <= 3; b++) {
        for (int saving = 0; saving < 2; saving++) {
            exercise(b, saving, 0);
            exercise(b, saving, 1);
        }
    }
    puts("worker mid/last-block cancellation, checkpoint and failed-work barriers passed");
    return 0;
}
