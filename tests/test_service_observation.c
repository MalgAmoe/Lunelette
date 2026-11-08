/* Intentional private grammar/state checks; do not link session.c separately. */
#include "../service/session.c"
/* Reuse the established deterministic session fixture, not a second tool engine. */
#include "support/lookup_fixture.h"

int main(void)
{
    service_generator g = {.context = &f, .next = next, .materialize = materialize_fixture,
        .reset = reset, .destroy = destroy, .frame = frame, .piece = piece, .eos = 2,
        .native_tools = 4, .tool_frame = result_frame};
    CHECK(!service_init(&session, g));
    g.native_tools = 3;
    CHECK(service_init(&session, g));
    CHECK(route("POST", "/sessions", "{}") == 200);
    const unsigned calls[] = {9, 1000, 32, 1001, 2, 1002, 2};
    prepare(calls, 7, "{\"document_id\":\"pump-readings\"}");
    f.name = "lookup";
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"obs\",\"budget\":5}") == 200);
    wait_done(1);
    CHECK(session.history[0].tool_document == 3 && session.pending == 1);
    const char *record = "{\"document_id\":\"pump-readings\",\"passage_id\":\"PUMP-READINGS:P1\",\"text\":\"Synthetic recorded observation (not live): pump-A temperature 84 C; pump-B temperature 72 C.\"}";
    char quoted[12289], body[16384];
    quote_text(record, quoted);
    snprintf(body, sizeof(body), "{\"role\":\"tool\",\"tool_call_id\":\"stale\",\"content\":\"%s\"}", quoted);
    CHECK(route("POST", "/sessions/1/messages", body) == 409);
    snprintf(body, sizeof(body), "{\"role\":\"tool\",\"tool_call_id\":\"obs\",\"content\":\"%s\"}", quoted);
    CHECK(route("POST", "/sessions/1/messages", body) == 200);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"final\",\"budget\":2}") == 200);
    wait_done(0);
    CHECK(!strcmp(session.state, "ready") && f.pending_commits == 1);
    CHECK(route("POST", "/sessions/1/messages", body) == 409);
    const char *bad[] = {"{\"document_id\":\"larch-stock\"}",
        "{\"document_id\":\"cedar-stock\"}", "{\"document_id\":\"../pump-readings\"}",
        "{\"document_id\":\"pump-readings\",\"extra\":1}",
        "{\"operation\":\"add\",\"a\":1,\"b\":2}"};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        prepare(calls, 5, bad[i]);
        f.name = i == 4 ? "calculator" : "lookup";
        snprintf(body, sizeof(body), "{\"id\":\"bad%u\",\"budget\":5}", i);
        CHECK(route("POST", "/sessions/1/generate", body) == 200);
        wait_done(0);
        CHECK(!strcmp(session.state, "needs_reset") && !session.outstanding_tool);
    }
    for (unsigned profile = 1; profile <= 2; profile++) {
        session.generator.native_tools = profile;
        prepare(calls, 5, "{\"document_id\":\"pump-readings\"}");
        f.name = "lookup";
        snprintf(body, sizeof(body), "{\"id\":\"profile%u\",\"budget\":5}", profile);
        CHECK(route("POST", "/sessions/1/generate", body) == 200);
        wait_done(0);
        CHECK(!strcmp(session.state, "needs_reset"));
    }
    session.generator.native_tools = 3;
    prepare(calls, 5, "{\"document_id\":\"pump-readings\"}");
    f.name = "lookup";
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"partial\",\"budget\":4}") == 200);
    wait_done(0);
    CHECK(!strcmp(session.state, "needs_reset") && !session.outstanding_tool);
    prepare(calls, 5, "{\"document_id\":\"pump-readings\"}");
    f.name = "lookup";
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"reset\",\"budget\":5}") == 200);
    wait_done(1);
    CHECK(route("POST", "/sessions/1/reset", "{}") == 200);
    CHECK(!session.outstanding_tool && !session.logical && !session.pending);
    CHECK(route("POST", "/sessions/1/messages", "{\"role\":\"tool\",\"tool_call_id\":\"reset\",\"content\":\"x\"}") == 409);
    prepare(calls, 5, "{\"document_id\":\"pump-readings\"}");
    f.name = "lookup";
    f.pause_after = 4;
    atomic_store(&f.hold, 1);
    CHECK(route("POST", "/sessions/1/generate", "{\"id\":\"cancel\",\"budget\":5}") == 200);
    for (unsigned i = 0; i < 10000 && !atomic_load(&f.entered); i++) {
        struct timespec delay = {0, 1000000};
        nanosleep(&delay, NULL);
    }
    CHECK(atomic_load(&f.entered));
    CHECK(route("POST", "/requests/cancel/cancel", "{}") == 200);
    atomic_store(&f.hold, 0);
    wait_done(0);
    CHECK(!strcmp(session.state, "needs_reset") && !session.outstanding_tool);
    service_request lookup = {.tool_kind = 2, .tool_document = 3};
    tj_node content = {.text = (unsigned char *)record, .length = strlen(record)};
    CHECK(lookup_result(&lookup, &content));
    lookup.tool_document = 1;
    CHECK(!lookup_result(&lookup, &content));
    lookup.tool_document = 3;
    const char *wrong = "{\"document_id\":\"pump-readings\",\"passage_id\":\"LARCH-STOCK:P1\",\"text\":\"x\"}";
    content.text = (unsigned char *)wrong;
    content.length = strlen(wrong);
    CHECK(!lookup_result(&lookup, &content));
    char buffer[4096];
    for (unsigned length = 0; length <= 1025; length++) {
        if (length > 1 && length < 1024) {
            continue;
        }
        int prefix = snprintf(buffer, sizeof(buffer),
            "{\"document_id\":\"pump-readings\",\"passage_id\":\"PUMP-READINGS:P1\",\"text\":\"");
        memset(buffer + prefix, 'x', length);
        strcpy(buffer + prefix + length, "\"}");
        content.text = (unsigned char *)buffer;
        content.length = strlen(buffer);
        CHECK(lookup_result(&lookup, &content) == (length >= 1 && length <= 1024));
    }
    memcpy(buffer, record, strlen(record));
    memset(buffer + strlen(record), ' ', 2049 - strlen(record));
    content.text = (unsigned char *)buffer;
    content.length = 2048;
    CHECK(lookup_result(&lookup, &content));
    content.length = 2049;
    CHECK(!lookup_result(&lookup, &content));
    const char *invalid[] = {
        "{\"document_id\":\"pump-readings\",\"passage_id\":\"PUMP-READINGS:P1\",\"text\":\"\xff\"}",
        "{\"document_id\":\"pump-readings\",\"passage_id\":\"PUMP-READINGS:P1\",\"text\":\"x\",\"extra\":1}",
        "{\"document_id\":\"pump-readings\",\"passage_id\":\"PUMP-READINGS:P1\",\"text\":\"x\",\"text\":\"y\"}"
    };
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
        content.text = (unsigned char *)invalid[i];
        content.length = strlen(invalid[i]);
        CHECK(!lookup_result(&lookup, &content));
    }
    service_destroy(&session);
    puts("observation model-free tests passed");
    return 0;
}
