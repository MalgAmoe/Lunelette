/* Observation-only inclusion: native computation and HTTP routing are forwarded unchanged. */
#define _POSIX_C_SOURCE 200809L
#include "metal.h"
#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
static service_session *observed_session;
static FILE *trace;
static size_t trace_bytes;
static int observed_prefill(lunelette_metal *, const int64_t *, size_t, const float **, char *, size_t);
static int observed_append(lunelette_metal *, const int64_t *, size_t, const float **, char *, size_t);
static int observed_decode(lunelette_metal *, int64_t, const float **, char *, size_t);
static int observed_restore(lunelette_metal *, const lunelette_metal_checkpoint *, const float **, char *, size_t);
#define lunelette_metal_prefill observed_prefill
#define lunelette_metal_append observed_append
#define lunelette_metal_decode observed_decode
#define lunelette_metal_checkpoint_restore observed_restore
#include "native.c"
#undef lunelette_metal_prefill
#undef lunelette_metal_append
#undef lunelette_metal_decode
#undef lunelette_metal_checkpoint_restore
static void emit(const char *line)
{
    size_t length = strlen(line);
    /* 4MiB trace reservation inside the aggregate 64MiB preparation budget. */
    if (!trace || trace_bytes + length > 4 * 1024 * 1024 ||
        fwrite(line, 1, length, trace) != length || fflush(trace)) {
        abort();
    }
    trace_bytes += length;
}
static native_service *owner(void)
{
    if (!observed_session) {
        abort();
    }
    return observed_session->generator.context;
}
static void call_record(const char *api, unsigned count, unsigned start, int status,
                        const float *logits)
{
    native_service *n = owner();
    int finite = !status && logits;
    for (unsigned i = 0; finite && i < LUNELETTE_VOCAB_SIZE; i++) {
        finite = isfinite(logits[i]);
    }
    unsigned pending = n->pending && n->pending_position >= start &&
        n->pending_position < start + count;
    unsigned repeated = start < n->recompute_length ? n->recompute_length - start : 0;
    if (repeated > count) {
        repeated = count;
    }
    char line[1024];
    snprintf(line, sizeof(line), "{\"kind\":\"call\",\"request_id\":\"%s\","
        "\"api\":\"%s\",\"attempted\":%u,\"committed\":%u,\"start\":%u,\"end\":%u,"
        "\"finite\":%s,\"status\":%d,\"pending_attempted\":%u,\"pending_committed\":%u,"
        "\"recompute_attempted\":%u,\"recompute_committed\":%u}\n",
        observed_session->history[observed_session->active].id, api, count,
        status ? 0 : count, start, start + (status ? 0 : count), finite ? "true" : "false",
        status, pending, status ? 0 : pending, repeated, status ? 0 : repeated);
    if (!finite) {
        /* Failure-only exact raw FP32 vector, no header/truncation/substitution. */
        if (!status && logits) {
            FILE *failure = fopen("failure-logits.f32", "wx");
            if (!failure || sizeof(float) != 4 ||
                fwrite(logits, sizeof(float), LUNELETTE_VOCAB_SIZE, failure) != LUNELETTE_VOCAB_SIZE ||
                fflush(failure) || fsync(fileno(failure)) || fclose(failure)) {
                abort();
            }
        }
        emit(line);
        abort(); /* Never continue affected validation after nonfinite output. */
    }
    emit(line);
}
#define BLOCK_WRAPPER(NAME, API, LABEL) \
static int NAME(lunelette_metal *r, const int64_t *ids, size_t count, \
                const float **v, char *error, size_t capacity) \
{ \
    unsigned start = owner()->length; \
    int status = API(r, ids, count, v, error, capacity); \
    if (status) { \
        fprintf(stderr, "D_NATIVE_ERROR %.*s\n", (int)(capacity < 512 ? capacity : 512), error ? error : ""); \
    } \
    call_record(LABEL, (unsigned)count, start, status, status ? NULL : *v); \
    return status; \
}
BLOCK_WRAPPER(observed_prefill, lunelette_metal_prefill, "prefill")
BLOCK_WRAPPER(observed_append, lunelette_metal_append, "append")
static int observed_decode(lunelette_metal *r, int64_t id, const float **v, char *e, size_t c)
{
    unsigned start = owner()->length;
    int status = lunelette_metal_decode(r, id, v, e, c);
    if (status) {
        fprintf(stderr, "D_NATIVE_ERROR %.*s\n", (int)(c < 512 ? c : 512), e ? e : "");
    }
    call_record("decode", 1, start, status, status ? NULL : *v);
    return status;
}
static int observed_restore(lunelette_metal *r, const lunelette_metal_checkpoint *cp,
                            const float **v, char *e, size_t c)
{
    int status = lunelette_metal_checkpoint_restore(r, cp, v, e, c);
    if (status) {
        fprintf(stderr, "D_NATIVE_ERROR %.*s\n", (int)(c < 512 ? c : 512), e ? e : "");
    }
    call_record("restore", 0, owner()->length, status, status ? NULL : *v);
    return status;
}
static int observed_init(service_session *s, service_generator g)
{
    observed_session = s;
    return service_init(s, g);
}
static int observed_route(service_session *s, const char *method, const char *path,
                          const void *body, size_t length, char *reply, size_t size, int *stream)
{
    int status = service_route(s, method, path, body, length, reply, size, stream);
    pthread_mutex_lock(&s->lock);
    if (!s->job && !s->reset_job) {
        native_service *n = s->generator.context;
        lunelette_metal_memory m;
        struct timespec t;
        if (lunelette_metal_get_memory(n->runtime, &m) || clock_gettime(CLOCK_MONOTONIC, &t)) {
            abort();
        }
        char line[1536];
        snprintf(line, sizeof(line), "{\"kind\":\"memory\",\"method\":\"task_info_TASK_VM_INFO_phys_footprint\","
            "\"monotonic_ns\":%llu,\"physical_bytes\":%llu,\"named_buffers\":{"
            "\"weights\":%llu,\"conversion_stage\":%llu,\"scratch\":%llu,\"kv\":%llu,"
            "\"checkpoint_logits\":%u},\"guard_diagnostics\":{\"status\":\"unavailable\",\"source\":\"startup stderr captured separately\"},\"native_ledger\":{"
            "\"committed\":%u,\"capacity\":%u,\"pending\":%d}}\n",
            (unsigned long long)t.tv_sec * 1000000000ULL + t.tv_nsec,
            (unsigned long long)service_process_footprint(), (unsigned long long)m.weight_bytes,
            (unsigned long long)m.conversion_stage_bytes, (unsigned long long)m.scratch_bytes,
            (unsigned long long)m.kv_bytes, n->checkpoint ? 524288 : 0, n->length, n->capacity, n->pending);
        emit(line);
    }
    pthread_mutex_unlock(&s->lock);
    return status;
}
#define service_init observed_init
#define service_route observed_route
#define main forwarded_http_main
#include "http.c"
#undef main
int main(int argc, char **argv)
{
    /* Execution authorized, NOT model validated. Reject incomplete or malformed
     * D argv before trace creation, file hashing, tokenizer or model construction. */
    const char *keys[] = {"--model", "--tokenizer", "--context", "--prefill-chunk",
                          "--prefix-reuse", "--strategy", "--prefill", "--port"};
    const char *values[8] = {0};
    if (argc != 17) {
        fprintf(stderr, "missingargs: all eight explicit D arguments required before modelopen\n");
        return 2;
    }
    for (int i = 1; i < argc; i += 2) {
        int index = -1;
        for (int j = 0; j < 8; j++) {
            if (!strcmp(argv[i], keys[j])) {
                index = j;
                break;
            }
        }
        if (index < 0 || values[index] || !*argv[i + 1] || argv[i + 1][0] == '-') {
            fprintf(stderr, "missingargs: invalid or duplicate D argument before modelopen\n");
            return 2;
        }
        values[index] = argv[i + 1];
    }
    if (strcmp(values[2], "4352") ||
        strcmp(values[3], "256") ||
        strcmp(values[4], "on") ||
        strcmp(values[5], "compact-fp8-arithmetic") ||
        strcmp(values[6], "mps") ||
        strspn(values[7], "0123456789") != strlen(values[7]) ||
        strlen(values[7]) > 5 || strtoul(values[7], NULL, 10) == 0 ||
        strtoul(values[7], NULL, 10) > 65535) {
        fprintf(stderr, "missingargs: invalid D argument value before modelopen\n");
        return 2;
    }
    /* Supervisor sets cwd to an exclusive owned cell directory. */
    struct rlimit core_limit = {0, 0};
    if (setrlimit(RLIMIT_CORE, &core_limit)) {
        return 1;
    }
    trace = fopen("observation.jsonl", "wx");
    if (!trace) {
        return 1;
    }
    int status = forwarded_http_main(argc, argv);
    if (fclose(trace)) {
        return 1;
    }
    return status;
}
