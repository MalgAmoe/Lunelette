#define _POSIX_C_SOURCE 200809L
#include "support/session_fixture.h"

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
                CHECK(tj_read_memory_limits(event, strlen(event), &d,
                    8192, 4096, TJ_MAX_NODES, 32));
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
static void *shutdown_service(void *unused)
{
    (void)unused;
    service_destroy(&session);
    return NULL;
}
static void checkpoint_invalidate(void *context)
{
    fake *f = context;
    CHECK(pthread_equal(pthread_self(), f->callback_thread));
    f->saved_length = 0;
}
static int checkpoint_save(void *context)
{
    fake *f = context;
    CHECK(pthread_equal(pthread_self(), f->callback_thread));
    pthread_mutex_lock(&f->lock);
    if (f->block_save) {
        f->entered = 1;
        pthread_cond_broadcast(&f->wake);
        while (!f->released) {
            pthread_cond_wait(&f->wake, &f->lock);
        }
    }
    pthread_mutex_unlock(&f->lock);
    if (f->save_failure) {
        return 0;
    }
    f->saved_length = f->length;
    f->saved_position = f->position;
    memcpy(f->saved_prefix, f->prefix, sizeof(f->prefix));
    return 1;
}
static int checkpoint_restore(void *context)
{
    fake *f = context;
    CHECK(pthread_equal(pthread_self(), f->callback_thread));
    pthread_mutex_lock(&f->lock);
    if (f->block_restore) {
        f->entered = 1;
        pthread_cond_broadcast(&f->wake);
        while (!f->released) {
            pthread_cond_wait(&f->wake, &f->lock);
        }
    }
    pthread_mutex_unlock(&f->lock);
    if (f->restore_failure) {
        return f->restore_failure == 2 ? 0 : -1;
    }
    CHECK(f->saved_length);
    f->length = f->policy == 2 ? 0 : f->saved_length;
    f->position = f->saved_position;
    memcpy(f->prefix, f->saved_prefix, sizeof(f->prefix));
    return 1;
}
static void wait_control(const char *id, const char *status)
{
    struct timespec delay = {0, 1000000};
    char path[128];
    snprintf(path, sizeof(path), "/requests/%s", id);
    for (unsigned i = 0; i < 3000; i++) {
        CHECK(route("GET", path, "") == 200);
        pthread_mutex_lock(&session.lock);
        int done = !session.job;
        pthread_mutex_unlock(&session.lock);
        if (done) {
            /* Completion may race the preceding GET snapshot. */
            CHECK(route("GET", path, "") == 200);
            CHECK(strstr(reply, status));
            return;
        }
        nanosleep(&delay, NULL);
    }
    CHECK(0);
}
static void off_begin(void *context)
{
    fake *f = context;
    CHECK(pthread_equal(pthread_self(), f->callback_thread));
    f->length = 0;
}
static void off_controller_test(void)
{
    fake f = {.policy = 2, .invalid = 2};
    CHECK(!pthread_mutex_init(&f.lock, NULL));
    CHECK(!pthread_cond_init(&f.wake, NULL));
    service_generator g = {.context = &f, .next = next, .materialize = commit,
        .reset = reset, .destroy = destroy, .prefix_reuse = 2, .begin = off_begin,
        .save = checkpoint_save, .restore = checkpoint_restore,
        .invalidate = checkpoint_invalidate};
    CHECK(service_init(&session, g));
    CHECK(route("POST", "/sessions", "{}") == 200);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"hi\"}") == 200);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"first\",\"budget\":2}") == 200);
    char text[128];
    drain(text, "\"event\":\"completed\"");
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"offsave\"}") == 202);
    wait_control("offsave", "\"status\":\"completed\"");
    CHECK(session.logical == 4 && session.committed == 4);
    for (unsigned i = 0; i < 3; i++) {
        char id[32], body[128];
        snprintf(id, sizeof(id), "offrestore%u", i);
        snprintf(body, sizeof(body), "{\"id\":\"%s\",\"checkpoint_id\":\"1\"}", id);
        CHECK(route("POST", "/sessions/1/restore", body) == 202);
        wait_control(id, "\"status\":\"completed\"");
        CHECK(strstr(reply, "\"cache_restored\":false"));
        CHECK(session.logical == 4 && !session.committed && !session.pending);
        snprintf(body, sizeof(body), "{\"id\":\"offgen%u\",\"budget\":2}", i);
        CHECK(route("POST", "/sessions/1/generate", body) == 200);
        drain(text, "\"event\":\"completed\"");
        CHECK(session.logical == 6 && session.checkpoint_id == 1);
    }
    service_destroy(&session);
    CHECK(!pthread_cond_destroy(&f.wake));
    CHECK(!pthread_mutex_destroy(&f.lock));
}

static void checkpoint_tests(void)
{
    fake f = {0};
    CHECK(!pthread_mutex_init(&f.lock, NULL));
    CHECK(!pthread_cond_init(&f.wake, NULL));
    service_generator g = {.context = &f, .next = next, .materialize = commit,
        .reset = reset, .destroy = destroy, .prefix_reuse = 1,
        .save = checkpoint_save, .restore = checkpoint_restore,
        .invalidate = checkpoint_invalidate};
    CHECK(service_init(&session, g));
    CHECK(route("POST", "/sessions", "{}") == 200);
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"empty\"}") == 409);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"hi\"}") == 200);
    for (size_t size = 0; size < 512; size = size ? size * 2 : 1) {
        char tiny[512];
        int stream = -1;
        const char *body = "{\"id\":\"tiny\"}";
        unsigned before = session.requests;
        CHECK(service_route(&session, "POST", "/sessions/1/checkpoint", body,
            strlen(body), tiny, size, &stream) == 500);
        CHECK(!stream && session.requests == before && !session.job);
    }
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"utf8\",\"budget\":1}") == 200);
    char text[128];
    drain(text, "\"event\":\"completed\"");
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"carry\"}") == 409);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"partial\",\"budget\":2}") == 200);
    drain(text, "\"event\":\"completed\"");
    service_disconnect_request(&session, "partial");
    service_disconnect_request(&session, "missing");
    CHECK(route("GET", "/requests/partial", "") == 200);
    CHECK(strstr(reply, "\"status\":\"completed\""));
    CHECK(strstr(reply, "\"cancel_requested\":false"));
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"save\"}") == 202);
    /* Admission disconnect must not cancel a control. */
    service_disconnect_request(&session, "save");
    service_disconnect_request(&session, "partial");
    wait_control("save", "\"status\":\"completed\"");
    CHECK(strstr(reply, "\"operation\":\"checkpoint\""));
    CHECK(strstr(reply, "\"checkpoint_id\":\"1\""));
    CHECK(session.logical == 5 && session.committed == 5 && !session.pending);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"branch\",\"budget\":5}") == 200);
    drain(text, "\"event\":\"completed\"");
    CHECK(session.pending && !strcmp(session.state, "ready"));
    CHECK(route("POST", "/sessions/1/restore", "{\"id\":\"beforecarry\",\"checkpoint_id\":\"1\"}") == 202);
    wait_control("beforecarry", "\"status\":\"completed\"");
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"branchcarry\",\"budget\":1}") == 200);
    drain(text, "\"event\":\"completed\"");
    CHECK(session.utf8_length == 1);
    for (unsigned i = 0; i < 3; i++) {
        char body[128], id[32];
        snprintf(id, sizeof(id), "restore%u", i);
        snprintf(body, sizeof(body), "{\"id\":\"%s\",\"checkpoint_id\":\"1\"}", id);
        CHECK(route("POST", "/sessions/1/restore", body) == 202);
        wait_control(id, "\"status\":\"completed\"");
        CHECK(session.logical == 5 && !session.utf8_length && !strcmp(session.state, "partial"));
        CHECK(strstr(reply, "\"cache_restored\":true"));
    }
    /* Save failure preserves old checkpoint and newly materialized bookkeeping. */
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"branch2\",\"budget\":5}") == 200);
    drain(text, "\"event\":\"completed\"");
    f.save_failure = 1;
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"allocfail\"}") == 202);
    wait_control("allocfail", "\"status\":\"error\"");
    CHECK(session.checkpoint_id == 1 && session.committed == session.logical && !session.pending);
    f.save_failure = 0;
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"replace\"}") == 202);
    wait_control("replace", "\"status\":\"completed\"");
    CHECK(session.checkpoint_id == 2 && session.tokens[session.logical - 1] == 256);
    CHECK(f.prefix[f.length - 1] == 256); /* pending EOS materialized exactly once */
    CHECK(route("POST", "/sessions/1/restore", "{\"id\":\"stale\",\"checkpoint_id\":\"1\"}") == 409);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"new\"}") == 200);
    pthread_mutex_lock(&session.lock);
    session.framing_ms = 12.5;
    pthread_mutex_unlock(&session.lock);
    CHECK(route("POST", "/sessions/1/restore", "{\"id\":\"readyrestore\",\"checkpoint_id\":\"2\"}") == 202);
    wait_control("readyrestore", "\"status\":\"completed\"");
    CHECK(strstr(reply, "\"framing_ms\":12.500"));
    CHECK(session.framing_ms == 0);
    CHECK(!strcmp(session.state, "ready") && !session.pending);
    CHECK(route("POST", "/requests/readyrestore/cancel", "{}") == 200);
    CHECK(strstr(reply, "\"cancel_requested\":false"));
    CHECK(session.checkpoint_id == 2 && !strcmp(session.state, "ready") && !session.pending);
    f.restore_failure = 2;
    CHECK(route("POST", "/sessions/1/restore", "{\"id\":\"restore-reject\",\"checkpoint_id\":\"2\"}") == 202);
    wait_control("restore-reject", "\"status\":\"error\"");
    CHECK(!strcmp(session.state, "ready") && session.checkpoint_id == 2);
    f.restore_failure = -1;
    CHECK(route("POST", "/sessions/1/restore", "{\"id\":\"restorefail\",\"checkpoint_id\":\"2\"}") == 202);
    wait_control("restorefail", "\"status\":\"error\"");
    CHECK(!strcmp(session.state, "needs_reset") && !session.checkpoint_id && !f.saved_length);
    f.restore_failure = 0;
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"x\"}") == 200);
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"restore-cancel-save\"}") == 202);
    wait_control("restore-cancel-save", "\"status\":\"completed\"");
    CHECK(route("POST", "/requests/restore-cancel-save/cancel", "{}") == 200);
    CHECK(strstr(reply, "\"cancel_requested\":false"));
    CHECK(session.checkpoint_id == 3 && f.saved_length);
    f.block_restore = 1;
    f.entered = f.released = 0;
    CHECK(route("POST", "/sessions/1/restore", "{\"id\":\"blocked-restore\",\"checkpoint_id\":\"3\"}") == 202);
    pthread_mutex_lock(&f.lock);
    struct timespec deadline;
    CHECK(!clock_gettime(CLOCK_REALTIME, &deadline));
    deadline.tv_sec += 3;
    while (!f.entered) {
        CHECK(!pthread_cond_timedwait(&f.wake, &f.lock, &deadline));
    }
    pthread_mutex_unlock(&f.lock);
    CHECK(route("POST", "/requests/blocked-restore/cancel", "{}") == 200);
    CHECK(strstr(reply, "\"cancel_requested\":true"));
    pthread_mutex_lock(&f.lock);
    f.released = 1;
    pthread_cond_broadcast(&f.wake);
    pthread_mutex_unlock(&f.lock);
    wait_control("blocked-restore", "\"status\":\"cancelled\"");
    CHECK(!strcmp(session.state, "needs_reset") && !session.checkpoint_id && !f.saved_length);
    f.block_restore = 0;
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"x\"}") == 200);
    f.block_save = 1;
    f.entered = f.released = 0;
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"blocked-save\"}") == 202);
    pthread_mutex_lock(&f.lock);
    while (!f.entered) {
        pthread_cond_wait(&f.wake, &f.lock);
    }
    pthread_mutex_unlock(&f.lock);
    CHECK(route("POST", "/sessions/1/reset", "{}") == 409);
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"busy\"}") == 409);
    service_disconnect_request(&session, "branch2");
    service_disconnect_request(&session, "blocked-save");
    service_disconnect_request(&session, "");
    CHECK(route("GET", "/requests/blocked-save", "") == 200);
    CHECK(strstr(reply, "\"cancel_requested\":false"));
    CHECK(route("POST", "/requests/blocked-save/cancel", "{}") == 200);
    pthread_mutex_lock(&f.lock);
    f.released = 1;
    pthread_cond_broadcast(&f.wake);
    pthread_mutex_unlock(&f.lock);
    wait_control("blocked-save", "\"status\":\"cancelled\"");
    CHECK(!strcmp(session.state, "needs_reset") && !session.checkpoint_id && !f.saved_length);
    f.block_save = 0;
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"x\"}") == 200);
    f.commit_failure = 1;
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"materializefail\"}") == 202);
    wait_control("materializefail", "\"status\":\"error\"");
    CHECK(!strcmp(session.state, "needs_reset"));
    f.commit_failure = 0;
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"x\"}") == 200);
    pthread_mutex_lock(&session.lock);
    session.checkpoint_serial = UINT64_MAX;
    pthread_mutex_unlock(&session.lock);
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"overflow\"}") == 503);
    pthread_mutex_lock(&session.lock);
    session.checkpoint_serial = 2;
    pthread_mutex_unlock(&session.lock);
    CHECK(route("POST", "/sessions/1/checkpoint", "{\"id\":\"delete-save\"}") == 202);
    wait_control("delete-save", "\"status\":\"completed\"");
    CHECK(route("DELETE", "/sessions/1", "{}") == 200);
    CHECK(!session.checkpoint_id && !f.saved_length);
    CHECK(route("POST", "/sessions", "{}") == 200);
    CHECK(route("POST", "/sessions/2/restore", "{\"id\":\"foreign\",\"checkpoint_id\":\"3\"}") == 409);
    CHECK(route("POST", "/sessions/2/messages", "{\"role\":\"user\",\"content\":\"x\"}") == 200);
    CHECK(route("POST", "/sessions/2/checkpoint", "{\"id\":\"save\"}") == 409);
    for (unsigned i = session.requests; i < SERVICE_IDS; i++) {
        char id[32], body[96];
        snprintf(id, sizeof(id), "controlcap%u", i);
        snprintf(body, sizeof(body), "{\"id\":\"%s\"}", id);
        CHECK(route("POST", "/sessions/2/checkpoint", body) == 202);
        wait_control(id, "\"status\":\"completed\"");
    }
    CHECK(route("POST", "/sessions/2/checkpoint", "{\"id\":\"full\"}") == 503);
    service_destroy(&session);
    CHECK(!pthread_cond_destroy(&f.wake));
    CHECK(!pthread_mutex_destroy(&f.lock));
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
    CHECK(!tj_read_memory_limits(deep, 80, &d, 8192, 4096, TJ_MAX_NODES, 32));
    fake f = {0};
    CHECK(!pthread_mutex_init(&f.lock, NULL));
    CHECK(!pthread_cond_init(&f.wake, NULL));
    service_generator generator = {.context = &f, .next = next, .materialize = commit,
        .reset = reset, .destroy = destroy};
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
    CHECK(strstr(reply, "\"cancel_latency_ms\":null")); /* Not stopped yet. */
    pthread_mutex_lock(&session.lock);
    double received = session.history[session.active].cancel_received_ms;
    CHECK(received > 0);
    pthread_mutex_unlock(&session.lock);
    CHECK(route("POST", "/requests/blocked/cancel", "{}") == 200);
    pthread_mutex_lock(&session.lock);
    CHECK(session.history[session.active].cancel_received_ms == received);
    pthread_mutex_unlock(&session.lock);
    pthread_mutex_lock(&f.lock);
    CHECK(f.entered && !f.released);
    f.released = 1;
    pthread_cond_broadcast(&f.wake);
    pthread_mutex_unlock(&f.lock);
    drain(part, "\"event\":\"cancelled\"");
    pthread_mutex_lock(&session.lock);
    service_request *cancelled = &session.history[session.active];
    CHECK(cancelled->generated == 3 && cancelled->decode_steps == 3);
    CHECK(cancelled->cancel_latency_ms >= 0);
    CHECK(!strcmp(cancelled->cancel_source, "explicit"));
    CHECK(cancelled->memory_before == 0 && cancelled->memory_after == 0);
    pthread_mutex_unlock(&session.lock);
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(route("GET", "/requests/blocked", "") == 200);
    CHECK(strstr(reply, "\"state\":\"needs_reset\""));
    CHECK(strstr(reply, "\"logical_tokens\":3"));
    f.block_at = 0;
    f.invalid = 1;
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"bad\",\"budget\":1}") == 200);
    drain(part, "\"event\":\"error\"");
    CHECK(route("GET", "/requests/blocked", "") == 200);
    CHECK(strstr(reply, "\"cancel_requested\":true")); /* History survives newer work. */
    CHECK(!strstr(reply, "\"cancel_latency_ms\":null"));
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
    CHECK(!strcmp(session.history[session.active].cancel_source, "overflow"));
    CHECK(session.history[session.active].cancel_latency_ms >= 0);
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
    /* Receipt during a blocked callback: disconnect and shutdown share the same
     * safe-boundary accounting, including work whose sampled ID is discarded. */
    for (unsigned shutdown = 0; shutdown < 2; shutdown++) {
        CHECK(service_init(&session, generator));
        CHECK(route("POST", "/sessions", "{}") == 200);
        f.position = 3;
        f.block_at = 3;
        f.entered = f.released = 0;
        CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"stop\",\"budget\":1}") == 200);
        pthread_mutex_lock(&f.lock);
        while (!f.entered) {
            pthread_cond_wait(&f.wake, &f.lock);
        }
        pthread_mutex_unlock(&f.lock);
        pthread_t thread;
        if (shutdown) {
            CHECK(!pthread_create(&thread, NULL, shutdown_service, NULL));
            for (;;) {
                pthread_mutex_lock(&session.lock);
                int received_stop = session.stop;
                pthread_mutex_unlock(&session.lock);
                if (received_stop) {
                    break;
                }
                nanosleep(&delay, NULL);
            }
        } else {
            /* Absent and nonmatching identities must leave the blocked job intact. */
            service_disconnect_request(&session, "");
            service_disconnect_request(&session, "missing");
            pthread_mutex_lock(&session.lock);
            CHECK(session.job && !session.cancel);
            CHECK(!session.history[session.active].cancel_received_ms);
            pthread_mutex_unlock(&session.lock);
            service_disconnect_request(&session, "stop");
        }
        pthread_mutex_lock(&f.lock);
        f.released = 1;
        pthread_cond_broadcast(&f.wake);
        pthread_mutex_unlock(&f.lock);
        if (shutdown) {
            CHECK(!pthread_join(thread, NULL));
        } else {
            drain(part, "\"event\":\"cancelled\"");
        }
        service_request *stopped = &session.history[0];
        CHECK(stopped->cancel_received_ms > 0 && stopped->cancel_latency_ms >= 0);
        CHECK(!strcmp(stopped->cancel_source, shutdown ? "shutdown" : "disconnect"));
        CHECK(stopped->generated == 0 && stopped->prefill_steps == 1);
        CHECK(stopped->first_token_ms >= 0);
        CHECK(!strcmp(stopped->state, "needs_reset"));
        if (!shutdown) {
            service_destroy(&session);
        }
    }
    CHECK(!pthread_cond_destroy(&f.wake));
    CHECK(!pthread_mutex_destroy(&f.lock));
    checkpoint_tests();
    off_controller_test();
    puts("service session/parser/UTF-8/identity tests passed");
    return 0;
}
