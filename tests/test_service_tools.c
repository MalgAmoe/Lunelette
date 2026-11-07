/* White-box bounds checks supplement the public deterministic service flow. */
#include "../service/session.c"
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
int main(void)
{
    service_generator g = {.context = &f, .next = next, .materialize = materialize_fixture,
        .reset = reset, .destroy = destroy, .frame = frame, .piece = piece, .eos = 2,
        .native_tools = 1, .tool_frame = result_frame};
    CHECK(service_init(&session, g));
    CHECK(route("POST", "/sessions", "{}") == 200);
    const unsigned call[] = {9, 1000, 32, 1001, 2, 1002, 2};
    const char *args = "{\"operation\":\"add\",\"a\":19,\"b\":24}";
    prepare(call, 7, args);
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"snapshot\"}") == 501);
    CHECK(route("POST", "/sessions/1/restore", "{\"id\":\"restore\",\"checkpoint_id\":\"1\"}") == 501);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"call\",\"budget\":5}") == 200);
    wait_done(1);
    CHECK(!strcmp(session.state, "awaiting_tool"));
    CHECK(session.pending == 1 && session.tokens[session.logical - 1] == 2);
    CHECK(route("GET", "/requests/call", "") == 200);
    CHECK(strstr(reply, "\"tool_call\":{\"id\":\"call\",\"name\":\"calculator\""));
    CHECK(strstr(reply, "\"text\":\"\""));
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"blocked\",\"budget\":1}") == 409);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"x\"}") == 409);
    unsigned logical = session.logical, committed = session.committed;
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"tool\",\"tool_call_id\":\"wrong\",\"content\":\"43\"}") == 409);
    const char *bad_results[] = {"01", "-0", "+1", "1.0", "1e2", "1000000000001", "-1000000000001", "", " 43"};
    for (unsigned i = 0; i < sizeof(bad_results) / sizeof(*bad_results); i++) {
        char body[256];
        snprintf(body, sizeof(body), "{\"role\":\"tool\",\"tool_call_id\":\"call\",\"content\":\"%s\"}", bad_results[i]);
        CHECK(route("POST", "/sessions/1/messages", body) == 400);
    }
    CHECK(session.logical == logical && session.committed == committed);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"tool\",\"tool_call_id\":\"call\",\"content\":\"43\"}") == 200);
    CHECK(session.pending == 1 && session.committed == committed);
    CHECK(session.tokens[logical] == 7 && session.tokens[session.logical - 1] == 8);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"tool\",\"tool_call_id\":\"call\",\"content\":\"43\"}") == 409);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"final\",\"budget\":2}") == 200);
    wait_done(0);
    CHECK(!strcmp(session.state, "ready") && f.pending_commits == 1);
    CHECK(route("GET", "/requests/final", "") == 200 && strstr(reply, "\"text\":\"43\""));
    const char *invalid[] = {
        "{\"operation\":\"divide\",\"a\":1,\"b\":2}",
        "{\"operation\":\"add\",\"a\":1000001,\"b\":2}",
        "{\"operation\":\"add\",\"a\":-1000001,\"b\":2}",
        "{\"operation\":\"add\",\"a\":1.0,\"b\":2}",
        "{\"operation\":\"add\",\"a\":1e0,\"b\":2}",
        "{\"operation\":\"add\",\"a\":1,\"a\":2,\"b\":2}",
        "{\"operation\":\"add\",\"a\":{},\"b\":2}",
        "{\"operation\":\"add\",\"a\":1}",
        "{\"operation\":\"add\",\"a\":1,\"b\":2,\"extra\":0}",
        "{\"operation\":\"add\",\"a\":1,\"b\":2", "{}{}", "\xff"
    };
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
        prepare(call, 5, invalid[i]);
        char body[128];
        snprintf(body, sizeof(body), "{\"id\":\"bad%u\",\"budget\":5}", i);
        CHECK(route("POST", "/sessions/1/generate", body) == 200);
        wait_done(0);
        CHECK(!strcmp(session.state, "needs_reset") && !session.outstanding_tool);
    }
    prepare(call, 5, args);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"short\",\"budget\":4}") == 200);
    wait_done(0);
    CHECK(!strcmp(session.state, "needs_reset"));
    CHECK(route("GET", "/requests/short", "") == 200 && !strstr(reply, "\"tool_call\":{"));
    const unsigned mixed[] = {1002, 9};
    prepare(mixed, 2, args);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"mixed\",\"budget\":2}") == 200);
    wait_done(0);
    CHECK(!strcmp(session.state, "needs_reset"));
    prepare(mixed, 2, args);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"prose\",\"budget\":1}") == 200);
    wait_done(0);
    CHECK(!strcmp(session.state, "partial"));
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"latecall\",\"budget\":1}") == 200);
    wait_done(0);
    CHECK(!strcmp(session.state, "needs_reset"));
    prepare(call, 5, args);
    f.name = "not_calculator";
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"wrongname\",\"budget\":5}") == 200);
    wait_done(0);
    CHECK(!strcmp(session.state, "needs_reset"));
    const unsigned multiple[] = {9, 1000, 32, 1001, 9};
    prepare(multiple, 5, args);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"multiple\",\"budget\":5}") == 200);
    wait_done(0);
    CHECK(!strcmp(session.state, "needs_reset"));
    prepare(call, 5, args);
    f.pause_after = 4;
    atomic_store(&f.entered, 0);
    atomic_store(&f.hold, 1);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"cancel\",\"budget\":5}") == 200);
    for (unsigned wait = 0; wait < 10000 && !atomic_load(&f.entered); wait++) {
        struct timespec delay = {0, 1000000};
        nanosleep(&delay, NULL);
    }
    CHECK(atomic_load(&f.entered));
    CHECK(route("GET", "/requests/cancel", "") == 200);
    CHECK(strstr(reply, "\"text\":\"\"") && !strstr(reply, "\"tool_call\":{"));
    CHECK(route("POST", "/requests/cancel/cancel", "{}") == 200);
    atomic_store(&f.hold, 0);
    wait_done(0);
    CHECK(!strcmp(session.state, "needs_reset") && !session.outstanding_tool);
    prepare(call, 5, "{\"operation\":\"multiply\",\"a\":-1000000,\"b\":1000000}");
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"bounds\",\"budget\":5}") == 200);
    wait_done(1);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"tool\",\"tool_call_id\":\"bounds\",\"content\":\"-1000000000000\"}") == 200);
    prepare(call, 5, args);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"resetcall\",\"budget\":5}") == 200);
    wait_done(1);
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(!session.outstanding_tool && !session.pending && !session.logical);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"tool\",\"tool_call_id\":\"resetcall\",\"content\":\"43\"}") == 409);
    CHECK(route("DELETE", "/sessions/1", "{}") == 200);
    /* Maximum retained timing/ID arrays plus bounded tool metadata still fit
     * unchanged event and metrics envelopes. */
    pthread_mutex_lock(&session.lock);
    service_request *r = &session.history[0];
    memset(r, 0, sizeof(*r));
    memset(r->id, 'w', 64);
    r->id[64] = 0;
    r->tool_operation = 2;
    r->tool_a = r->tool_b = -1000000;
    r->decode_steps = r->generated = 256;
    r->memory_before = r->memory_after = UINT64_MAX;
    r->prefill_us = r->decode_total_us = r->suffix_us = UINT64_MAX;
    r->prefill_steps = r->suffix_steps = r->session_id = UINT_MAX;
    r->logical = r->committed = r->pending = UINT_MAX;
    r->before_logical = r->before_committed = r->before_pending = UINT_MAX;
    r->prompt = r->reused = r->newly_processed = UINT_MAX;
    r->prefill_ms = r->decode_ms = r->total_ms = r->framing_ms = 9223372036854.0;
    r->first_token_ms = r->cancel_received_ms = r->cancel_latency_ms = 9223372036854.0;
    strcpy(r->operation, "generate");
    strcpy(r->stop_reason, "tool_call");
    strcpy(r->state, "awaiting_tool");
    strcpy(r->before_state, "awaiting_tool");
    strcpy(r->status, "completed");
    strcpy(r->cancel_source, "disconnect");
    r->work = (service_work){.full_attempted = UINT_MAX, .full_committed = UINT_MAX,
        .suffix_attempted = UINT_MAX, .suffix_committed = UINT_MAX,
        .pending_attempted = UINT_MAX, .pending_committed = UINT_MAX,
        .prefill_calls = UINT_MAX, .append_calls = UINT_MAX, .decode_calls = UINT_MAX,
        .decode_attempted = UINT_MAX, .decode_committed = UINT_MAX,
        .logit_restores = UINT_MAX, .recompute_attempted = UINT_MAX,
        .recompute_committed = UINT_MAX};
    for (unsigned i = 0; i < 256; i++) {
        r->decode_us[i] = UINT64_MAX;
        r->token_ids[i] = UINT_MAX;
    }
    session.active = session.head = session.count = 0;
    strcpy(session.state, "awaiting_tool");
    char measured[SERVICE_METRIC_BYTES];
    CHECK(metrics(r, 1, measured, sizeof(measured)));
    event_locked(&session, "completed", "", "tool_call");
    CHECK(session.count == 1);
    CHECK(strlen(session.events[0]) < SERVICE_EVENT_BYTES);
    CHECK(strstr(session.events[0], "\"tool_call\":{"));
    printf("tool envelope bounds: metrics=%zu event=%zu session=%zu bytes\n",
           strlen(measured), strlen(session.events[0]), sizeof(session));
    tj_document parsed;
    CHECK(tj_read_memory_limits(session.events[0], strlen(session.events[0]), &parsed,
                               SERVICE_EVENT_BYTES, 4096, 1024, 8));
    tj_free(&parsed);
    pthread_mutex_unlock(&session.lock);
    char path[80];
    snprintf(path, sizeof(path), "/requests/%s", r->id);
    CHECK(route("GET", path, "") == 200 && strstr(reply, "\"tool_call\":{"));
    service_destroy(&session);
    puts("service tools tests passed");
    return 0;
}
