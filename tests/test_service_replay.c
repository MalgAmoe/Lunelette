#include "support/session_fixture.h"

static int replay_frame(void *context, const unsigned char *text, size_t length,
                        int initial, unsigned *out, unsigned *capacity)
{
    (void)context;
    (void)text;
    (void)length;
    (void)initial;
    if (*capacity < 1) {
        return 0;
    }
    out[0] = 1000;
    *capacity = 1;
    return 1;
}

static int replay_piece(void *context, unsigned id, unsigned char *out, size_t *capacity)
{
    (void)context;
    CHECK(id == 53156 && *capacity >= 1);
    out[0] = 'x';
    *capacity = 1;
    return 1;
}

static int replay_next(void *context, unsigned id)
{
    fake *f = context;
    /* Reuse the existing deterministic cancellation gate. */
    next(context);
    CHECK(f->length && f->prefix[f->length - 1] == (f->position == 1 ? 1000u : 53156u));
    return (int)id;
}

static void wait_terminal(const char *terminal)
{
    struct timespec delay = {0, 1000000};
    for (unsigned i = 0; i < 3000; i++) {
        char event[SERVICE_EVENT_BYTES];
        if (!service_event(&session, event)) {
            nanosleep(&delay, NULL);
            continue;
        }
        CHECK(strstr(event, "\"evaluation_mode\":\"controlled_replay\""));
        CHECK(strstr(event, "\"replay_tokens\":["));
        tj_document d;
        CHECK(tj_read_memory(event, strlen(event), &d));
        tj_free(&d);
        if (strstr(event, terminal)) {
            return;
        }
    }
    CHECK(0);
}

int main(void)
{
    fake f = {0};
    CHECK(!pthread_mutex_init(&f.lock, NULL));
    CHECK(!pthread_cond_init(&f.wake, NULL));
    service_generator g = {.context = &f, .next = next, .materialize = commit,
        .reset = reset, .destroy = destroy, .frame = replay_frame, .piece = replay_piece,
        .eos = 2, .replay_next = replay_next};
    g.native_tools = 1;
    CHECK(!service_init(&session, g));
    g.native_tools = 0;
    g.eos = 256;
    CHECK(!service_init(&session, g));
    g.eos = 2;
    CHECK(service_init(&session, g));
    CHECK(route("POST", "/sessions", "{}") == 200);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"hi\"}") == 200);
    const char *invalid[] = {
        "{\"id\":\"bad\",\"budget\":2}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":53156}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":\"053156,2\"}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":\"53156, 2\"}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":\"2,53156\"}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":\"999,2\"}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":\"131072,2\"}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":\"42949672960,2\"}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":\"+53156,2\"}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":\"53156,2,\"}",
        "{\"id\":\"bad\",\"budget\":1,\"replay_tokens\":\"53156,2\"}",
        "{\"id\":\"bad\",\"budget\":2,\"replay_tokens\":\"53156,2\",\"replay_tokens\":\"53156,2\"}",
        "{\"id\":\"bad\",\"budget\":17,\"replay_tokens\":\"53156,2\"}"
    };
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
        CHECK(route("POST", "/sessions/1/generate", invalid[i]) == 400);
        CHECK(!session.requests && !session.replay_reserved && !f.position);
    }
    CHECK(route("POST", "/sessions/1/generate",
        "{\"id\":\"forced\",\"budget\":2,\"replay_tokens\":\"53156,2\"}") == 200);
    wait_terminal("\"event\":\"completed\"");
    CHECK(session.replay_reserved == 2);
    CHECK(route("GET", "/requests/forced", "") == 200);
    CHECK(strstr(reply, "\"token_ids\":[53156,2]"));
    CHECK(strstr(reply, "\"replay_tokens\":[53156,2]"));
    CHECK(strstr(reply, "controlled_replay"));
    CHECK(session.pending == 1 && session.tokens[session.logical - 1] == 2);
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(session.replay_reserved == 2);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"hi\"}") == 200);
    /* History/context/reply-capacity failures cannot reserve capture. */
    char small[128];
    int stream = 0;
    const char *capacity_body = "{\"id\":\"reply\",\"budget\":2,\"replay_tokens\":\"53156,2\"}";
    CHECK(service_route(&session, "POST", "/sessions/1/generate", capacity_body,
        strlen(capacity_body), small, sizeof(small), &stream) == 500);
    CHECK(!stream && session.replay_reserved == 2);
    unsigned history = session.requests;
    session.requests = SERVICE_IDS;
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"capacity\",\"budget\":2,\"replay_tokens\":\"53156,2\"}") == 503);
    session.requests = history;
    unsigned logical = session.logical;
    session.logical = 256;
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"context\",\"budget\":2,\"replay_tokens\":\"53156,2\"}") == 422);
    session.logical = logical;
    CHECK(session.replay_reserved == 2);
    CHECK(route("POST", "/sessions/1/generate",
        "{\"id\":\"partial\",\"budget\":1,\"replay_tokens\":\"53156\"}") == 200);
    wait_terminal("\"event\":\"completed\"");
    CHECK(!strcmp(session.state, "partial") && session.pending == 1);
    CHECK(route("POST", "/sessions/1/generate",
        "{\"id\":\"continuation\",\"budget\":1,\"replay_tokens\":\"2\"}") == 200);
    wait_terminal("\"event\":\"completed\"");
    CHECK(session.tokens[session.logical - 2] == 53156 && session.pending == 1);
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"user\",\"content\":\"hi\"}") == 200);
    /* Gate first selection using the existing position gate (1-based). */
    f.position = 1;
    f.block_at = 1;
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"cancelled\",\"budget\":2,\"replay_tokens\":\"53156,2\"}") == 200);
    pthread_mutex_lock(&f.lock);
    while (!f.entered) {
        pthread_cond_wait(&f.wake, &f.lock);
    }
    pthread_mutex_unlock(&f.lock);
    service_disconnect_request(&session, "cancelled");
    pthread_mutex_lock(&f.lock);
    f.position = 0; /* Correct first materialized prefix assertion after gate. */
    f.released = 1;
    pthread_cond_broadcast(&f.wake);
    pthread_mutex_unlock(&f.lock);
    wait_terminal("\"event\":\"cancelled\"");
    CHECK(session.replay_reserved == 6);
    CHECK(route("GET", "/requests/cancelled", "") == 200);
    CHECK(strstr(reply, "\"token_ids\":[]"));
    CHECK(strstr(reply, "\"replay_tokens\":[53156,2]"));
    session.replay_reserved = 16;
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"limit\",\"budget\":2,\"replay_tokens\":\"53156,2\"}") == 503);
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(session.replay_reserved == 16);
    service_destroy(&session);
    g.replay_next = NULL;
    CHECK(service_init(&session, g));
    CHECK(route("POST", "/sessions", "{}") == 200);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"default\",\"budget\":2,\"replay_tokens\":\"53156,2\"}") == 400);
    service_destroy(&session);
    CHECK(!pthread_cond_destroy(&f.wake));
    CHECK(!pthread_mutex_destroy(&f.lock));
    puts("C session controlled replay admission/metadata/cancellation tests passed");
    return 0;
}
