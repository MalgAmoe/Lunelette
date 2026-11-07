/* Include the implementation to exercise its private bounded serializer directly. */
#include "../service/session.c"
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
_Static_assert(sizeof(((service_session *)0)->history) / sizeof(service_request) == 256,
               "request history must not grow with context");
_Static_assert(sizeof(((service_request *)0)->token_ids) / sizeof(unsigned) == 256,
               "generation IDs remain independently bounded");
_Static_assert(sizeof(((service_request *)0)->decode_us) / sizeof(uint64_t) == 256,
               "generation timing remains independently bounded");
static service_session session;
static int frame_stub(void *c, const unsigned char *t, size_t n, int i, unsigned *o, unsigned *k)
{
    (void)c;
    (void)t;
    (void)n;
    (void)i;
    (void)o;
    (void)k;
    return 0;
}
int main(void)
{
    service_request *r = &session.history[0];
    r->decode_steps = 256;
    r->generated = 256;
    r->memory_before = r->memory_after = UINT64_MAX;
    for (unsigned i = 0; i < 256; i++) {
        r->decode_us[i] = UINT64_MAX;
        r->token_ids[i] = UINT_MAX;
    }
    r->work = (service_work){.full_attempted = UINT_MAX, .full_committed = UINT_MAX,
        .suffix_attempted = UINT_MAX, .suffix_committed = UINT_MAX,
        .pending_attempted = UINT_MAX, .pending_committed = UINT_MAX, .prefill_calls = UINT_MAX,
        .append_calls = UINT_MAX, .decode_calls = UINT_MAX, .decode_attempted = UINT_MAX,
        .decode_committed = UINT_MAX, .logit_restores = UINT_MAX,
        .recompute_attempted = UINT_MAX, .recompute_committed = UINT_MAX};
    r->suffix_us = UINT64_MAX;
    r->suffix_steps = UINT_MAX;
    r->prefill_steps = UINT_MAX;
    r->session_id = r->before_logical = r->before_committed = r->before_pending = UINT_MAX;
    session.logical = session.committed = session.pending = UINT_MAX;
    strcpy(r->operation, "checkpoint");
    r->materialize_ms = r->snapshot_ms = r->restore_ms = 9223372036854.0;
    r->prefill_us = r->decode_total_us = UINT64_MAX;
    r->prefill_ms = r->decode_ms = r->total_ms = r->framing_ms = 9223372036854.0;
    r->first_token_ms = r->cancel_received_ms = r->cancel_latency_ms = 9223372036854.0;
    r->prompt = r->reused = r->newly_processed = UINT_MAX;
    strcpy(r->cancel_source, "disconnect");
    strcpy(r->id, "worst");
    strcpy(r->state, "needs_reset");
    strcpy(r->before_state, "partial");
    strcpy(session.state, "needs_reset");
    session.generator.info = "{}";
    char measured[SERVICE_METRIC_BYTES];
    CHECK(metrics(r, 1, measured, sizeof(measured)));
    char small[10];
    CHECK(!metrics(r, 1, small, sizeof(small)));
    memset(r->id, 'w', 64);
    r->id[64] = 0;
    event_locked(&session, "cancelled", "", "unsupported_output_token");
    strcpy(r->id, "worst");
    CHECK(session.count == 1);
    CHECK(strlen(session.events[0]) < SERVICE_EVENT_BYTES);
    printf("maximum exercised metrics=%zu terminal=%zu\n", strlen(measured), strlen(session.events[0]));
    tj_document d;
    CHECK(tj_read_memory(session.events[0], strlen(session.events[0]), &d));
    CHECK(d.nodes[tj_field(&d, 1, "metrics")].kind == '{');
    tj_free(&d);
    memset(r->text, 1, 4096);
    session.requests = 1;
    CHECK(!pthread_mutex_init(&session.lock, NULL));
    char reply[65536];
    int stream;
    CHECK(service_route(&session, "GET", "/requests/worst", "", 0,
                        reply, sizeof(reply), &stream) == 200);
    CHECK(strlen(reply) > 30000 && strlen(reply) < sizeof(reply));
    CHECK(strstr(reply, measured));
    char path[] = "/tmp/lunelette-metrics-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    FILE *file = fdopen(fd, "w");
    CHECK(file && fwrite(reply, 1, strlen(reply), file) == strlen(reply));
    CHECK(!fclose(file));
    CHECK(tj_read(path, &d));
    CHECK(!unlink(path));
    unsigned metric = tj_field(&d, 1, "metrics");
    unsigned samples = tj_field(&d, metric, "decode_step_duration_us");
    unsigned count = 0;
    for (unsigned node = d.nodes[samples].child; node; node = d.nodes[node].next) {
        count++;
    }
    CHECK(count == 256);
    tj_free(&d);
    memset(r, 0, sizeof(*r));
    r->first_token_ms = -1;
    CHECK(metrics(r, 1, measured, sizeof(measured)));
    CHECK(strstr(measured, "\"process_phys_footprint_before_bytes\":null"));
    CHECK(strstr(measured, "\"process_phys_footprint_after_bytes\":null"));
    CHECK(strstr(measured, "\"first_token_ms\":null"));
    CHECK(strstr(measured, "\"cancel_latency_ms\":null"));
    strcpy(r->id, "worst");
    memset(r->text, 1, 4096);
    CHECK(service_route(&session, "GET", "/requests/worst", "", 0,
                        small, sizeof(small), &stream) == 500);
    session.generator.frame = frame_stub;
    session.generator.context_capacity = 4352;
    session.session = 1;
    session.logical = session.committed = 4352;
    session.pending = 0;
    strcpy(session.state, "ready");
    for (unsigned i = 0; i < 4352; i++) {
        session.tokens[i] = 131071;
    }
    CHECK(service_route(&session, "GET", "/sessions/1", "", 0,
                        reply, sizeof(reply), &stream) == 200);
    CHECK(strlen(reply) < 31000);
    CHECK(tj_read_memory_limits(reply, strlen(reply), &d, 65536, 32768, 10000, 4));
    unsigned ids = tj_field(&d, 1, "token_ids");
    unsigned ledger_count = 0;
    for (unsigned node = d.nodes[ids].child; node; node = d.nodes[node].next) {
        unsigned id;
        CHECK(tj_integer(&d, node, &id) && id == 131071);
        ledger_count++;
    }
    CHECK(ledger_count == 4352);
    tj_free(&d);
    printf("envelopes: metrics<%u event<%u retained<65536 ledger=%u IDs\n",
           SERVICE_METRIC_BYTES, SERVICE_EVENT_BYTES, ledger_count);
    pthread_mutex_destroy(&session.lock);
#ifndef __APPLE__
    CHECK(service_process_footprint() == 0);
#endif
    puts("bounded terminal/retained instrumentation serialization passed");
    return 0;
}
