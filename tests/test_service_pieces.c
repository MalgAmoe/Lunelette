#define _POSIX_C_SOURCE 200809L
#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

typedef struct {
    unsigned position, mode, length, prefix[256], resets, destroys;
} fixture;
static int frame(void *context, const unsigned char *text, size_t length,
                 int initial, unsigned *output, unsigned *capacity)
{
    (void)context;
    (void)text;
    (void)length;
    unsigned count = initial ? 4 : 3;
    if (*capacity < count) {
        return 0;
    }
    unsigned used = 0;
    if (initial) {
        output[used++] = 1;
    }
    output[used++] = 3;
    output[used++] = initial ? 37133 : 50975;
    output[used++] = 4;
    *capacity = used;
    return 1;
}
static int materialize(void *context, const unsigned *tokens, unsigned length)
{
    fixture *f = context;
    CHECK(length >= f->length);
    CHECK(!memcmp(f->prefix, tokens, f->length * sizeof(*tokens)));
    memcpy(f->prefix, tokens, length * sizeof(*tokens));
    f->length = length;
    return 1;
}
static int next(void *context)
{
    fixture *f = context;
    unsigned position = f->position++;
    if (f->mode == 1) {
        return 1000;
    }
    if (f->mode == 2) {
        return 17;
    }
    if (f->mode >= 3) {
        return 1000;
    }
    return position < 2 ? 1000 + (int)position : 2;
}
static int piece(void *context, unsigned token, unsigned char *bytes, size_t *length)
{
    fixture *f = context;
    CHECK(*length == 1024);
    if (f->mode == 1) {
        memset(bytes, 1, 1024);
        *length = 1024;
        return 1;
    }
    if (f->mode == 2) {
        return 0;
    }
    if (f->mode == 3) {
        memcpy(bytes, "x\0y", 3);
        *length = 3;
        return 1;
    }
    if (f->mode == 4) {
        *length = 1025;
        return 1;
    }
    if (token == 1000) {
        memcpy(bytes, "hello\xe2", 6);
        *length = 6;
    } else {
        memcpy(bytes, "\x82\xac\n\"\\", 5);
        *length = 5;
    }
    return 1;
}
static void reset(void *context)
{
    fixture *f = context;
    f->position = f->length = 0;
    f->resets++;
}
static void destroy(void *context)
{
    ((fixture *)context)->destroys++;
}
static service_session session;
static char reply[30000];
static int route(const char *path, const char *body)
{
    int stream;
    return service_route(&session, "POST", path, body, strlen(body), reply, sizeof(reply), &stream);
}
static void finish(char *assembled)
{
    struct timespec delay = {0, 1000000};
    assembled[0] = 0;
    for (unsigned tries = 0; tries < 3000; tries++) {
        char event[SERVICE_EVENT_BYTES];
        while (service_event(&session, event)) {
            tj_document d;
            CHECK(strlen(event) < SERVICE_EVENT_BYTES);
            CHECK(tj_read_memory_limits(event, strlen(event), &d, 8192, 4096, TJ_MAX_NODES, 32));
            unsigned text = tj_field(&d, 1, "text");
            CHECK(strlen(assembled) + d.nodes[text].length < 8192);
            strncat(assembled, (const char *)d.nodes[text].text, d.nodes[text].length);
            int terminal = strstr(event, "\"event\":\"completed\"") ||
                           strstr(event, "\"event\":\"error\"") ||
                           strstr(event, "\"event\":\"cancelled\"");
            tj_free(&d);
            if (terminal) {
                return;
            }
        }
        nanosleep(&delay, NULL);
    }
    CHECK(0);
}
int main(void)
{
    fixture f = {0};
    service_generator generator = {.context = &f, .next = next, .materialize = materialize,
        .reset = reset, .destroy = destroy, .frame = frame, .piece = piece, .eos = 2, .info = "{}"};
    CHECK(service_init(&session, generator));
    int info_stream;
    CHECK(service_route(&session, "GET", "/info", "", 0, reply, sizeof(reply), &info_stream) == 200);
    CHECK(strstr(reply, "\"phase\":\"after_resident_load\""));
    CHECK(strstr(reply, "\"units\":\"bytes\""));
#ifndef __APPLE__
    CHECK(session.startup_footprint == 0);
#endif
    CHECK(route("/sessions", "{}") == 200);
    CHECK(route("/sessions/1/generate", "{\"id\":\"empty\",\"budget\":1}") == 409);
    CHECK(route("/sessions/1/messages", "{\"role\":\"system\",\"content\":\"x\"}") == 501);
    CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Hi\"}") == 200);
    CHECK(session.logical == 4 && session.tokens[2] == 37133);
    CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Hi\"}") == 409);
    CHECK(route("/sessions/1/generate", "{\"id\":\"one\",\"budget\":1}") == 200);
    char text[8192];
    finish(text);
    CHECK(!strcmp(text, "hello"));
    CHECK(session.history[0].decode_steps == 0);
    CHECK(session.history[0].prefill_ms == session.history[0].prefill_us / 1000.0);
    CHECK(session.history[0].prompt == 4 && session.history[0].newly_processed == 4);
    CHECK(session.history[0].first_token_ms >= 0);
    CHECK(session.pending == 1 && session.tokens[4] == 1000 && session.utf8_length == 1);
    CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Again\"}") == 409);
    CHECK(route("/sessions/1/generate", "{\"id\":\"two\",\"budget\":2}") == 200);
    finish(text);
    CHECK(!strcmp(text, "€\n\"\\"));
    CHECK(!strcmp(session.state, "ready") && session.tokens[6] == 2);
    service_request *continued = &session.history[1];
    CHECK(continued->decode_steps == 2 && continued->generated == 2);
    CHECK(continued->prefill_us == 0 && continued->reused == 4);
    CHECK(continued->newly_processed == 1 && continued->prompt == 5);
    CHECK(continued->decode_total_us == continued->decode_us[0] + continued->decode_us[1]);
    CHECK(continued->decode_ms == continued->decode_total_us / 1000.0);
    CHECK(session.committed == 6 && session.logical == 7 && session.pending == 1);
    CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Again\"}") == 200);
    CHECK(session.logical == 10 && session.tokens[7] == 3 && session.tokens[8] == 50975);
    CHECK(session.tokens[4] == 1000 && session.tokens[5] == 1001);
    CHECK(route("/sessions/1/reset", "{}") == 200);
    f.mode = 1;
    CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Hi\"}") == 200);
    CHECK(route("/sessions/1/generate", "{\"id\":\"large\",\"budget\":5}") == 200);
    finish(text);
    CHECK(!strcmp(session.state, "needs_reset"));
    CHECK(strlen(session.history[2].text) == 4096);
    CHECK(!strcmp(session.history[2].stop_reason, "result_text_overflow"));
    int stream;
    CHECK(service_route(&session, "GET", "/requests/large", "", 0, reply, sizeof(reply), &stream) == 200);
    CHECK(strlen(reply) > 24576 && strlen(reply) < sizeof(reply));
    tj_document document;
    char path[] = "/tmp/lunelette-pieces-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    FILE *file = fdopen(fd, "w");
    CHECK(file);
    CHECK(fwrite(reply, 1, strlen(reply), file) == strlen(reply));
    CHECK(!fclose(file));
    CHECK(tj_read(path, &document));
    CHECK(!unlink(path));
    unsigned field = tj_field(&document, 1, "text");
    CHECK(document.nodes[field].length == 4096);
    tj_free(&document);
    CHECK(route("/sessions/1/reset", "{}") == 200);
    f.mode = 2;
    CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Hi\"}") == 200);
    CHECK(route("/sessions/1/generate", "{\"id\":\"control\",\"budget\":1}") == 200);
    finish(text);
    CHECK(!strcmp(session.state, "needs_reset"));
    CHECK(!strcmp(session.history[3].stop_reason, "unsupported_output_token"));
    for (unsigned mode = 3; mode <= 4; mode++) {
        CHECK(route("/sessions/1/reset", "{}") == 200);
        f.mode = mode;
        CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Hi\"}") == 200);
        char body[128];
        snprintf(body, sizeof(body), "{\"id\":\"invalid-%u\",\"budget\":1}", mode);
        CHECK(route("/sessions/1/generate", body) == 200);
        finish(text);
        CHECK(!strcmp(session.state, "needs_reset"));
        CHECK(!*text);
    }
    CHECK(route("/sessions/1/reset", "{}") == 200);
    f.mode = 0;
    f.position = 2;
    for (unsigned i = 0; i < 63; i++) {
        CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Hi\"}") == 200);
        char body[128];
        snprintf(body, sizeof(body), "{\"id\":\"capacity-%u\",\"budget\":1}", i);
        CHECK(route("/sessions/1/generate", body) == 200);
        finish(text);
    }
    CHECK(session.logical == 253);
    CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Hi\"}") == 200);
    CHECK(session.logical == 256);
    CHECK(route("/sessions/1/generate", "{\"id\":\"overflow\",\"budget\":1}") == 422);
    CHECK(session.logical == 256 && session.requests == 69);
    CHECK(route("/sessions/1/messages", "{\"role\":\"user\",\"content\":\"Hi\"}") == 409);
    service_destroy(&session);
    CHECK(f.resets >= 4 && f.destroys == 1);
    puts("native-ID piece/framing-policy/large-result tests passed");
    return 0;
}
