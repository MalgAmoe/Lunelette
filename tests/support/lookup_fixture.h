#ifndef LUNELETTE_TEST_SUPPORT_LOOKUP_FIXTURE_H
#define LUNELETTE_TEST_SUPPORT_LOOKUP_FIXTURE_H

/* Header-local tool fixture shared by lookup/observation tests. Callers explicitly
 * include service/session.c for private grammar/state checks; these callbacks and
 * bounded event helpers assert fixture invariants, not scenario outcomes. */

#include <stdatomic.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
_Static_assert(sizeof(service_session) + 65536 < 8 * 1024 * 1024,
               "session plus bounded request scratch must fit host allowance");
static service_session session;
static char reply[65536];
typedef struct {
    unsigned ids[256], count, position, length, prefix[SERVICE_CONTEXT_MAX];
    unsigned pending_commits, resets, pause_after;
    const char *arguments, *name;
    atomic_int hold, entered;
} fixture;
static fixture f;
static int frame(void *context, const unsigned char *text, size_t length,
                 int initial, unsigned *out, unsigned *capacity)
{
    (void)context;
    (void)text;
    (void)length;
    unsigned ids[] = {1, 5, 1003, 6, 3, 1004, 4};
    unsigned count = initial ? 7 : 3;
    if (*capacity < count) {
        return 0;
    }
    memcpy(out, ids + (initial ? 0 : 4), count * sizeof(*out));
    *capacity = count;
    return 1;
}
static int result_frame(void *context, const unsigned char *text, size_t length,
                        unsigned *out, unsigned *capacity)
{
    (void)context;
    (void)text;
    (void)length;
    if (*capacity < 3) {
        return 0;
    }
    out[0] = 7;
    out[1] = 1005;
    out[2] = 8;
    *capacity = 3;
    return 1;
}
static int materialize_fixture(void *context, const unsigned *ids, unsigned length)
{
    fixture *v = context;
    CHECK(length >= v->length);
    CHECK(!memcmp(ids, v->prefix, v->length * sizeof(*ids)));
    if (v->length && length > v->length && ids[v->length] == 2) {
        v->pending_commits++;
    }
    memcpy(v->prefix, ids, length * sizeof(*ids));
    v->length = length;
    return 1;
}
static int next(void *context)
{
    fixture *v = context;
    if (atomic_load(&v->hold) && v->position >= v->pause_after) {
        atomic_store(&v->entered, 1);
    }
    while (atomic_load(&v->hold) && v->position >= v->pause_after) {
        struct timespec delay = {0, 1000000};
        nanosleep(&delay, NULL);
    }
    return v->position < v->count ? (int)v->ids[v->position++] : -1;
}
static int piece(void *context, unsigned id, unsigned char *out, size_t *capacity)
{
    fixture *v = context;
    CHECK(id >= 1000); /* No structural token may reach the decoder. */
    const char *text = id == 1000 ? v->name : id == 1001 ? v->arguments : "43";
    size_t length = strlen(text);
    CHECK(length <= *capacity);
    memcpy(out, text, length);
    *capacity = length;
    return 1;
}
static void reset(void *context)
{
    fixture *v = context;
    v->length = v->position = 0;
    v->resets++;
}
static void destroy(void *context) { (void)context; }
static int route(const char *method, const char *path, const char *body)
{
    int stream;
    return service_route(&session, method, path, body, strlen(body), reply, sizeof(reply), &stream);
}
static void wait_done(int expect_call)
{
    char event[SERVICE_EVENT_BYTES];
    int call = 0, delta = 0;
    for (unsigned i = 0; i < 10000; i++) {
        while (service_event(&session, event)) {
            call += strstr(event, "\"tool_call\":{") != NULL;
            delta += strstr(event, "\"event\":\"text_delta\"") != NULL;
        }
        pthread_mutex_lock(&session.lock);
        int done = !session.job;
        pthread_mutex_unlock(&session.lock);
        if (done) {
            while (service_event(&session, event)) {
                call += strstr(event, "\"tool_call\":{") != NULL;
                delta += strstr(event, "\"event\":\"text_delta\"") != NULL;
            }
            CHECK(call == expect_call);
            if (expect_call) {
                CHECK(!delta);
            }
            return;
        }
        struct timespec delay = {0, 1000000};
        nanosleep(&delay, NULL);
    }
    CHECK(0);
}
static void prepare(const unsigned *ids, unsigned count, const char *arguments)
{
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    memcpy(f.ids, ids, count * sizeof(*ids));
    f.count = count;
    f.arguments = arguments;
    f.name = "calculator";
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"calculate\"}") == 200);
}

#endif
