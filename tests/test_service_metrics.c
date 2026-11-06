/* Include the implementation to exercise its private bounded serializer directly. */
#include "../service/session.c"
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static service_session session;
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
    r->work = (service_work){.full_attempted = 65536, .full_committed = 65536,
        .suffix_attempted = 65536, .suffix_committed = 65536,
        .pending_attempted = 256, .pending_committed = 256, .prefill_calls = 256,
        .append_calls = 256, .decode_calls = 256, .decode_attempted = 256,
        .decode_committed = 256, .logit_restores = 1};
    r->suffix_us = UINT64_MAX;
    r->suffix_steps = 256;
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
    event_locked(&session, "cancelled", "", "cancelled");
    CHECK(session.count == 1);
    CHECK(strlen(session.events[0]) < SERVICE_EVENT_BYTES);
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
    pthread_mutex_destroy(&session.lock);
#ifndef __APPLE__
    CHECK(service_process_footprint() == 0);
#endif
    puts("bounded terminal/retained instrumentation serialization passed");
    return 0;
}
