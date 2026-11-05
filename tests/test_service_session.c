#define _POSIX_C_SOURCE 200809L
#include <limits.h>
#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
typedef struct {
    unsigned position, length, prefix[256], resets, destroys;
    int invalid;
    pthread_mutex_t lock;
    pthread_cond_t wake;
    unsigned block_at;
    int entered, released;
} fake;
static int next(void *context)
{
    fake *f = context;
    pthread_mutex_lock(&f->lock);
    if (f->block_at && f->position == f->block_at) {
        f->entered = 1;
        pthread_cond_broadcast(&f->wake);
        while (!f->released) {
            pthread_cond_wait(&f->wake, &f->lock);
        }
    }
    pthread_mutex_unlock(&f->lock);
    static const unsigned pieces[] = {0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80, 256};
    if (f->invalid == 2) {
        return 'x';
    }
    return f->invalid ? 0xff : (int)pieces[f->position++ % 8];
}
static int commit(void *context, const unsigned *tokens, unsigned length)
{
    fake *f = context;
    CHECK(length >= f->length);
    CHECK(!memcmp(tokens, f->prefix, f->length * sizeof(*tokens)));
    memcpy(f->prefix, tokens, length * sizeof(*tokens));
    f->length = length;
    return 1;
}
static void reset(void *context)
{
    fake *f = context;
    f->position = f->length = 0;
    f->resets++;
}
static void destroy(void *context)
{
    ((fake *)context)->destroys++;
}
static service_session session;
static char reply[4096];
static int route(const char *method, const char *path, const char *body)
{
    int stream;
    return service_route(&session, method, path, body, strlen(body), reply, sizeof(reply), &stream);
}
static void drain(char *text, const char *terminal)
{
    struct timespec delay = {0, 1000000};
    text[0] = 0;
    for (unsigned tries = 0; tries < 3000; tries++) {
        char event[SERVICE_EVENT_BYTES];
        if (service_event(&session, event)) {
            CHECK(strchr(event, '\n'));
            if (strstr(event, "text_delta")) {
                tj_document d;
                CHECK(tj_read_memory(event, strlen(event), &d));
                unsigned n = tj_field(&d, 1, "text");
                size_t old = strlen(text);
                memcpy(text + old, d.nodes[n].text, d.nodes[n].length);
                text[old + d.nodes[n].length] = 0;
                tj_free(&d);
            }
            if (strstr(event, terminal)) {
                return;
            }
        } else {
            nanosleep(&delay, NULL);
        }
    }
    CHECK(0);
}
int main(void)
{
    tj_document d;
    const char *bad[] = {"{\"id\":1,\"id\":2}", "{\"x\":1}", "{\"id\":\"\\u0000\"}",
        "{\"id\":\"\\ud800\"}", "{\"id\":[]}", "{\"id\":01}", "{\"id\":1e}",
        "{\"id\":\"\xff\"}", "{} true", "{\"id\":\"\\udc00\"}"};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        CHECK(!service_json(bad[i], strlen(bad[i]), "|id|", &d));
    }
    const char *valid = "{\"id\":\"\\ud83d\\ude00\\u20ac\"}";
    CHECK(service_json(valid, strlen(valid), "|id|", &d));
    CHECK(d.nodes[tj_field(&d, 1, "id")].length == 7);
    tj_free(&d);
    const char *smuggled_field = "{\"id\":\"x\",\"budget\":1,\"id||budget\":2}";
    CHECK(!service_json(smuggled_field, strlen(smuggled_field), "|id||budget|", &d));
    const char *empty_field = "{\"id\":\"x\",\"budget\":1,\"\":2}";
    CHECK(!service_json(empty_field, strlen(empty_field), "|id||budget|", &d));
    char deep[256];
    memset(deep, '[', 40);
    memset(deep + 40, ']', 40);
    CHECK(!tj_read_memory(deep, 80, &d));
    fake f = {0};
    CHECK(!pthread_mutex_init(&f.lock, NULL));
    CHECK(!pthread_cond_init(&f.wake, NULL));
    service_generator generator = {&f, next, commit, reset, destroy};
    CHECK(service_init(&session, generator));
    CHECK(route("POST", "/sessions", "{}") == 200);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"assistant\",\"content\":\"x\"}") == 400);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"tool_result\",\"content\":\"x\"}") == 501);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"hi\"}") == 200);
    char assembled[128] = "", part[128];
    /* Every byte boundary of both UTF-8 scalars, including pending across budgets. */
    for (unsigned i = 0; i < 8; i++) {
        char body[128];
        snprintf(body, sizeof(body), "{\"id\":\"r%u\",\"budget\":1}", i);
        CHECK(route("POST", "/sessions/1/generate", body) == 200);
        drain(part, "\"event\":\"completed\"");
        strcat(assembled, part);
        CHECK(route("GET", "/sessions/1", "") == 200);
        CHECK(strstr(reply, "\"pending_tokens\":1"));
        CHECK(strstr(reply, i == 7 ? "\"state\":\"ready\"" : "\"state\":\"partial\""));
    }
    CHECK(!strcmp(assembled, "€😀"));
    CHECK(f.length == 9); /* two input bytes + seven committed output bytes */
    const unsigned expected_prefix[] = {'h', 'i', 0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80};
    CHECK(!memcmp(f.prefix, expected_prefix, sizeof(expected_prefix)));
    CHECK(route("GET", "/requests/r7", "") == 200);
    CHECK(strstr(reply, "\"stop_reason\":\"eos\""));
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"r0\",\"budget\":1}") == 409);
    /* EOS remains pending until the next input/generation materializes it once. */
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"q\"}") == 200);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"after-eos\",\"budget\":1}") == 200);
    drain(part, "\"event\":\"completed\"");
    CHECK(f.length == 11 && f.prefix[9] == 256 && f.prefix[10] == 'q');
    CHECK(route("GET", "/requests/r7", "") == 200);
    CHECK(strstr(reply, "\"logical_tokens\":10"));
    CHECK(strstr(reply, "\"state\":\"ready\""));
    CHECK(route("POST", "/sessions/1/checkpoint", "{}") == 501);
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"r0\",\"budget\":1}") == 409);
    /* Block inside next after emitting a full scalar, not merely at job startup. */
    pthread_mutex_lock(&f.lock);
    f.block_at = 3;
    f.entered = f.released = 0;
    pthread_mutex_unlock(&f.lock);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"blocked\",\"budget\":8}") == 200);
    pthread_mutex_lock(&f.lock);
    struct timespec deadline;
    CHECK(!clock_gettime(CLOCK_REALTIME, &deadline));
    deadline.tv_sec += 3;
    while (!f.entered) {
        CHECK(!pthread_cond_timedwait(&f.wake, &f.lock, &deadline));
    }
    CHECK(!f.released);
    pthread_mutex_unlock(&f.lock);
    CHECK(route("GET", "/requests/blocked", "") == 200);
    CHECK(strstr(reply, "\"id\":\"blocked\""));
    CHECK(strstr(reply, "\"status\":\"generating\""));
    CHECK(strstr(reply, "\"state\":\"generating\""));
    CHECK(strstr(reply, "\"generated_tokens\":3"));
    CHECK(strstr(reply, "\"logical_tokens\":3"));
    CHECK(strstr(reply, "\"committed_tokens\":2"));
    CHECK(strstr(reply, "\"pending_tokens\":1"));
    CHECK(strstr(reply, "€"));
    CHECK(route("POST", "/requests/r7/cancel", "{}") == 200);
    CHECK(strstr(reply, "\"cancel_requested\":false"));
    CHECK(route("GET", "/requests/blocked", "") == 200);
    CHECK(strstr(reply, "\"cancel_requested\":false"));
    CHECK(route("POST", "/requests/blocked/cancel", "{}") == 200);
    CHECK(strstr(reply, "\"cancel_requested\":true"));
    pthread_mutex_lock(&f.lock);
    CHECK(f.entered && !f.released);
    f.released = 1;
    pthread_cond_broadcast(&f.wake);
    pthread_mutex_unlock(&f.lock);
    drain(part, "\"event\":\"cancelled\"");
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(route("GET", "/requests/blocked", "") == 200);
    CHECK(strstr(reply, "\"state\":\"needs_reset\""));
    CHECK(strstr(reply, "\"logical_tokens\":3"));
    f.block_at = 0;
    f.invalid = 1;
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"bad\",\"budget\":1}") == 200);
    drain(part, "\"event\":\"error\"");
    CHECK(route("GET", "/sessions/1", "") == 200);
    CHECK(strstr(reply, "needs_reset"));
    f.invalid = 0;
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    /* No consumer: the bounded ring cancels instead of blocking the worker. */
    f.invalid = 2;
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"slow\",\"budget\":64}") == 200);
    struct timespec delay = {0, 1000000};
    int done = 0;
    for (unsigned tries = 0; tries < 3000; tries++) {
        pthread_mutex_lock(&session.lock);
        done = !session.job;
        pthread_mutex_unlock(&session.lock);
        if (done) {
            break;
        }
        nanosleep(&delay, NULL);
    }
    CHECK(done);
    drain(part, "\"event\":\"cancelled\"");
    CHECK(route("GET", "/sessions/1", "") == 200);
    CHECK(strstr(reply, "needs_reset"));
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    /* Exactly started + 15 deltas fills the queue, even without a 17th event.
     * Terminal reservation must not discard text and claim completion. */
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"exact-full\",\"budget\":15}") == 200);
    done = 0;
    for (unsigned tries = 0; tries < 3000; tries++) {
        pthread_mutex_lock(&session.lock);
        done = !session.job;
        pthread_mutex_unlock(&session.lock);
        if (done) {
            break;
        }
        nanosleep(&delay, NULL);
    }
    CHECK(done);
    CHECK(route("GET", "/requests/exact-full", "") == 200);
    CHECK(strstr(reply, "\"status\":\"cancelled\""));
    CHECK(strstr(reply, "\"state\":\"needs_reset\""));
    drain(part, "\"event\":\"cancelled\"");
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    /* One fewer delta leaves the terminal slot: no text may be lost. */
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"fits-queue\",\"budget\":14}") == 200);
    done = 0;
    for (unsigned tries = 0; tries < 3000; tries++) {
        pthread_mutex_lock(&session.lock);
        done = !session.job;
        pthread_mutex_unlock(&session.lock);
        if (done) {
            break;
        }
        nanosleep(&delay, NULL);
    }
    CHECK(done);
    drain(part, "\"event\":\"completed\"");
    CHECK(!strcmp(part, "xxxxxxxxxxxxxx"));
    CHECK(route("GET", "/requests/fits-queue", "") == 200);
    CHECK(strstr(reply, "\"text\":\"xxxxxxxxxxxxxx\""));
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    f.invalid = 0;
    char message[320], input[251];
    memset(input, 'x', sizeof(input) - 1);
    input[sizeof(input) - 1] = 0;
    snprintf(message, sizeof(message), "{\"role\":\"user\",\"content\":\"%s\"}", input);
    CHECK(route("POST", "/sessions/1/messages", message) == 200);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"near\",\"budget\":7}") == 422);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"near\",\"budget\":6}") == 200);
    drain(part, "\"event\":\"completed\"");
    CHECK(!strcmp(part, "€"));
    CHECK(route("GET", "/sessions/1", "") == 200);
    CHECK(strstr(reply, "\"logical_tokens\":256"));
    CHECK(strstr(reply, "\"pending_tokens\":1"));
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"near-overflow\",\"budget\":1}") == 422);
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    /* Request identity cap survives resets, with rejected IDs reusable. */
    for (unsigned i = session.requests; i < SERVICE_IDS; i++) {
        char body[128];
        snprintf(body, sizeof(body), "{\"id\":\"cap%u\",\"budget\":1}", i);
        CHECK(route("POST", "/sessions/1/generate", body) == 200);
        drain(part, "\"event\":\"completed\"");
        CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    }
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"overflow\",\"budget\":1}") == 503);
    CHECK(strstr(reply, "\"recovery\":\"restart_service\""));
    CHECK(route("DELETE", "/sessions/1", "{}") == 200);
    CHECK(route("GET", "/sessions/1", "") == 410);
    CHECK(strstr(reply, "\"session_valid\":false"));
    CHECK(route("POST", "/sessions", "{}") == 200);
    CHECK(strstr(reply, "\"session_id\":2"));
    CHECK(route("DELETE", "/sessions/2", "{}") == 200);
    pthread_mutex_lock(&session.lock);
    session.session = UINT_MAX;
    pthread_mutex_unlock(&session.lock);
    CHECK(route("POST", "/sessions", "{}") == 503);
    CHECK(strstr(reply, "session_id_capacity_restart_required"));
    CHECK(strstr(reply, "\"recovery\":\"restart_service\""));
    service_destroy(&session);
    CHECK(f.destroys == 1);
    CHECK(!pthread_cond_destroy(&f.wake));
    CHECK(!pthread_mutex_destroy(&f.lock));
    puts("service session/parser/UTF-8/identity tests passed");
    return 0;
}
