/* Source-bound C preparation probe. Synthetic IDs are NOT framed user text.
 * Real build forwards native callbacks to Metal; authorized model entry requires validated arguments. */
#define _POSIX_C_SOURCE 200809L
#ifdef CONTROLLER_STUB
#define main inherited_native_test_main
#include "tests/test_service_native.c"
#undef main
#endif
#include "metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Native callbacks below are wrapped only to observe actual calls. Direct replay
 * uses the original symbols after the macros are undefined. */
typedef struct {
    unsigned calls, tokens;
    unsigned size[128];
    char api[128];
    int64_t ids[4352];
} runtime_journal;
static runtime_journal current_journal, saved_journal;
static unsigned long long adapter_tokens;
#ifdef CONTROLLER_STUB
#define WAIT_SECONDS 10
#else
#define WAIT_SECONDS 1200
#endif
static unsigned actual_prefills, actual_appends, actual_decodes, actual_resets;
static unsigned actual_snapshots, actual_restores;
static void journal_call(char api, const int64_t *ids, size_t count)
{
    if (current_journal.calls >= 128 || current_journal.tokens + count > 4352) {
        abort();
    }
    for (size_t i = 0; i < count; i++) {
        if (ids[i] < 0 || ids[i] >= 131072) abort();
    }
    adapter_tokens += count;
    current_journal.api[current_journal.calls] = api;
    current_journal.size[current_journal.calls++] = (unsigned)count;
    memcpy(current_journal.ids + current_journal.tokens, ids, count * sizeof(*ids));
    current_journal.tokens += (unsigned)count;
    flockfile(stdout);
    printf("{\"type\":\"runtime_call\",\"api\":\"%c\",\"count\":%zu,\"ids\":[", api, count);
    for (size_t i = 0; i < count; i++) {
        printf("%s%lld", i ? "," : "", (long long)ids[i]);
    }
    puts("]}");
    funlockfile(stdout);
}
static int trace_prefill(lunelette_metal *r, const int64_t *ids, size_t count,
                         const float **v, char *error, size_t capacity)
{
    actual_prefills++;
    journal_call('P', ids, count);
    return lunelette_metal_prefill(r, ids, count, v, error, capacity);
}
static int trace_append(lunelette_metal *r, const int64_t *ids, size_t count,
                        const float **v, char *error, size_t capacity)
{
    actual_appends++;
    journal_call('A', ids, count);
    return lunelette_metal_append(r, ids, count, v, error, capacity);
}
static int trace_decode(lunelette_metal *r, int64_t id, const float **v,
                        char *error, size_t capacity)
{
    actual_decodes++;
    journal_call('D', &id, 1);
    return lunelette_metal_decode(r, id, v, error, capacity);
}
static int trace_reset(lunelette_metal *r, char *error, size_t capacity)
{
    actual_resets++;
    memset(&current_journal, 0, sizeof(current_journal));
    return lunelette_metal_reset(r, error, capacity);
}
static int trace_snapshot(lunelette_metal *r, lunelette_metal_checkpoint **c,
                          char *error, size_t capacity)
{
    actual_snapshots++;
    int result = lunelette_metal_checkpoint_create(r, c, error, capacity);
    if (!result) {
        saved_journal = current_journal;
    }
    return result;
}
static int trace_restore(lunelette_metal *r, const lunelette_metal_checkpoint *c,
                         const float **v, char *error, size_t capacity)
{
    actual_restores++;
    int result = lunelette_metal_checkpoint_restore(r, c, v, error, capacity);
    if (!result) {
        current_journal = saved_journal;
    }
    return result;
}
#define lunelette_metal_prefill trace_prefill
#define lunelette_metal_append trace_append
#define lunelette_metal_decode trace_decode
#define lunelette_metal_reset trace_reset
#define lunelette_metal_checkpoint_create trace_snapshot
#define lunelette_metal_checkpoint_restore trace_restore

#include "service/native.c"
#undef lunelette_metal_prefill
#undef lunelette_metal_append
#undef lunelette_metal_decode
#undef lunelette_metal_reset
#undef lunelette_metal_checkpoint_create
#undef lunelette_metal_checkpoint_restore
#include "service/session.c"
#ifndef CONTROLLER_STUB
#include "metal_internal.h"
#endif
#ifndef CHECK
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#endif

static unsigned probe_policy, probe_block, probe_calls;
static const char *probe_case;
static void endpoint(native_service *n, const char *label);
static void prefix_hashes(native_service *n, unsigned length, unsigned char out[52][32])
{
#ifdef CONTROLLER_STUB
    CHECK(n->runtime->count >= length);
    /* Stub has no KV. This is explicitly only a synthetic input-state hash. */
    for (unsigned i = 0; i < 52; i++) {
        CC_SHA256(n->runtime->ids, (CC_LONG)(length * sizeof(int64_t)), out[i]);
    }
#else
    LuneletteMetalOwner *owner = (__bridge LuneletteMetalOwner *)n->runtime->owner;
    CHECK(owner->valid && owner->length >= length);
    size_t bytes = (size_t)length * 1024 * sizeof(float);
    for (unsigned layer = 0; layer < 26; layer++) {
        id<MTLBuffer> pair[] = {owner->keys[layer], owner->values[layer]};
        for (unsigned k = 0; k < 2; k++) {
            CHECK(pair[k].contents && pair[k].length >= bytes);
            CC_SHA256(pair[k].contents, (CC_LONG)bytes, out[layer * 2 + k]);
        }
    }
#endif
}

/* Canonical digest: unsigned token IDs as four little-endian bytes each.
 * Keep full unique ledgers even for targets interrupted before READY. */
#define TARGET_LEDGER_MAX 32
static unsigned target_ledger_count;
static unsigned target_lengths[TARGET_LEDGER_MAX];
static unsigned target_ledgers[TARGET_LEDGER_MAX][4352];
static void target_digest(const unsigned *ids, unsigned length, char hex[65])
{
    unsigned char bytes[4352 * 4], digest[32];
    CHECK(length <= 4352);
    for (unsigned i = 0; i < length; i++) {
        CHECK(ids[i] < 131072);
        for (unsigned j = 0; j < 4; j++) bytes[i * 4 + j] = (unsigned char)(ids[i] >> (8 * j));
    }
    CC_SHA256(bytes, (CC_LONG)(length * 4), digest);
    for (unsigned i = 0; i < 32; i++) snprintf(hex + i * 2, 3, "%02x", digest[i]);
    unsigned entry = 0;
    while (entry < target_ledger_count &&
           (target_lengths[entry] != length ||
            memcmp(target_ledgers[entry], ids, length * sizeof(*ids)))) entry++;
    if (entry == target_ledger_count) {
        CHECK(entry < TARGET_LEDGER_MAX);
        target_lengths[entry] = length;
        memcpy(target_ledgers[entry], ids, length * sizeof(*ids));
        target_ledger_count++;
        printf("{\"type\":\"target_ledger\",\"encoding\":\"u32le\",\"target\":%u,\"target_sha256\":\"%s\",\"ids\":[", length, hex);
        for (unsigned i = 0; i < length; i++) printf("%s%u", i ? "," : "", ids[i]);
        puts("]}");
    }
}

/* This boundary forwards computation without substituting logits or selection. */
static int observed_step(void *context, const unsigned *ids, unsigned length,
                         unsigned *cursor)
{
    native_service *n = context;
    unsigned before = n->length;
    int status = native_materialize_step(context, ids, length, cursor);
    probe_calls++;
    flockfile(stdout);
    char digest[65];
    target_digest(ids, length, digest);
    printf("{\"type\":\"step\",\"policy\":%u,\"block\":%u,\"case\":\"%s\","
           "\"before\":%u,\"cursor\":%u,\"target\":%u,\"status\":%d,\"target_sha256\":\"%s\","
           "\"prefill_calls\":%u,\"append_calls\":%u,\"decode_calls\":%u,"
           "\"recompute_committed\":%u,\"pending_committed\":%u}\n",
           probe_policy, probe_block, probe_case, before, *cursor, length, status, digest,
           n->work.prefill_calls, n->work.append_calls, n->work.decode_calls,
           n->work.recompute_committed, n->work.pending_committed);
    funlockfile(stdout);
    if (status == 1 && n->logits) {
        endpoint(n, probe_case);
    }
    return status;
}

static int observed_restore(void *context)
{
    native_service *n = context;
    int result = native_restore(n);
    if (result == 1 && n->logits) {
        endpoint(n, "restored_endpoint");
    } else if (result == 1) {
        CHECK(n->policy == 2 && !n->length);
        puts("{\"type\":\"OFF_restore_empty\",\"rematerialization_required\":true}");
    }
    return result;
}

/* Runs the production controller's multi-block ingestion loop, not a replica.
 * Caller owns a single live native owner. No worker/thread is launched here. */
int controller_probe_subset(native_service *n, unsigned policy, unsigned block)
{
    service_session *s = calloc(1, sizeof(*s));
    CHECK(s);
    CHECK(!pthread_mutex_init(&s->lock, NULL));
    probe_policy = policy;
    probe_block = block;
    probe_calls = 0;
    s->generator = (service_generator){.context = n, .materialize_step = observed_step,
        .work = native_work, .context_capacity = 4352, .block_capacity = block};
    for (unsigned i = 0; i < 4352; i++) {
        s->tokens[i] = 1000 + (37 * i) % 8000;
    }
    unsigned lengths[] = {257, 4096, 4351};
    for (unsigned row = 0; row < 3; row++) {
        probe_case = row == 0 ? "initial257" : row == 1 ? "prefix4096" : "capacity4351";
        for (unsigned i = 0; i < 4352; i++) {
            s->tokens[i] = 1000 + (37 * i) % 8000;
        }
        native_reset(n);
        native_begin(n);
        s->committed = s->pending = 0;
        s->logical = lengths[row];
        service_work total = {0};
        CHECK(materialize(s, &total) == 1);
        CHECK(total.prefill_calls == 1);
        CHECK(total.append_calls == (s->logical - 1) / block);
        CHECK(n->length == s->logical);
        int token = native_next(n);
        CHECK(token >= 0);
        printf("{\"type\":\"selection\",\"forced\":false,\"logits_unmodified\":true,\"id\":%d}\n", token);
        /* Observe the native selection, then explicitly replace the pending ID
         * with the frozen non-EOS input; native_next sets pending even for EOS. */
        n->emitted = 9000;
        n->pending = 1;
        puts("{\"type\":\"subset_teacher_forcing\",\"forced_id\":9000,\"natural_EOS_ready_claim\":false}");
        s->tokens[s->logical++] = 9000;
        s->pending = 1;
        probe_case = "pending_reconcile";
        memset(&total, 0, sizeof(total));
        CHECK(materialize(s, &total) == 1);
        CHECK(total.pending_committed == 1);
        if (policy) {
            CHECK(native_save(n) == 1);
            CHECK(observed_restore(n) == 1);
            if (policy == 2) {
                /* OFF restore resets runtime; never claim retained KV here. */
                CHECK(n->length == 0 && n->logits == NULL);
                s->committed = 0;
                probe_case = "off_restore_rematerialization";
                memset(&total, 0, sizeof(total));
                CHECK(materialize(s, &total) == 1);
                CHECK(total.recompute_committed == s->logical);
            }
        }
        if (row == 2) {
            unsigned calls = probe_calls;
            CHECK(native_next(n) == -1);
            CHECK(probe_calls == calls);
        }
    }
    printf("{\"type\":\"subset_bound\",\"policy\":%u,\"block\":%u,"
           "\"materialize_step_calls\":%u,\"scope\":\"partial_not_campaign_C\","
           "\"footprint_point\":%llu}\n", policy, block, probe_calls,
           (unsigned long long)service_process_footprint());
    pthread_mutex_destroy(&s->lock);
    free(s);
    return 0;
}

/* Barrier instrumentation is after the native forward, before the controller
 * consumes either status or receipt. Only this probe uses these barriers. */
static pthread_mutex_t gate_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_wake = PTHREAD_COND_INITIALIZER;
static unsigned gate_target, gate_count;
static int gate_held, gate_release;
static unsigned sample_calls, save_calls, forced_id = 9000;
static unsigned route_serial;
static const char *endpoint_dir;
static unsigned endpoint_serial;
typedef struct {
    runtime_journal journal;
    char path[4096];
    unsigned char hashes[52][32];
} endpoint_record;
static endpoint_record endpoint_records[512];
static unsigned endpoint_count;

static void endpoint(native_service *n, const char *label)
{
    CHECK(n->logits);
    char path[4096];
    CHECK(snprintf(path, sizeof(path), "%s/p%u-b%u-%04u.f32", endpoint_dir,
                   probe_policy, probe_block, endpoint_serial++) < (int)sizeof(path));
    FILE *f = fopen(path, "wx");
    CHECK(f);
    CHECK(fwrite(n->logits, sizeof(float), 131072, f) == 131072);
    CHECK(!fclose(f));
    /* Preserve raw failed values before rejecting a nonfinite endpoint. */
    for (unsigned i = 0; i < 131072; i++) {
        CHECK(isfinite(n->logits[i]));
    }
    CHECK(endpoint_count < 512);
    CHECK(current_journal.tokens == n->length);
    endpoint_records[endpoint_count].journal = current_journal;
    prefix_hashes(n, n->length, endpoint_records[endpoint_count].hashes);
    strcpy(endpoint_records[endpoint_count++].path, path);
    lunelette_metal_memory memory;
    CHECK(!lunelette_metal_get_memory(n->runtime, &memory));
    printf("{\"type\":\"resource_point\",\"case\":\"%s\",\"footprint_point\":%llu,"
           "\"weights\":%llu,\"conversion\":%llu,\"scratch\":%llu,\"kv\":%llu,\"total\":%llu}\n",
           label, (unsigned long long)service_process_footprint(),
           (unsigned long long)memory.weight_bytes, (unsigned long long)memory.conversion_stage_bytes,
           (unsigned long long)memory.scratch_bytes, (unsigned long long)memory.kv_bytes,
           (unsigned long long)memory.total_bytes);
    flockfile(stdout);
    printf("{\"type\":\"prefix_hashes\",\"file\":\"%s\",\"kind\":\"%s\",\"sha256\":[", path,
#ifdef CONTROLLER_STUB
           "stub_input_state_not_KV"
#else
           "52_retained_KV_buffers"
#endif
    );
    for (unsigned buffer = 0; buffer < 52; buffer++) {
        printf("%s\"", buffer ? "," : "");
        for (unsigned byte = 0; byte < 32; byte++) {
            printf("%02x", endpoint_records[endpoint_count - 1].hashes[buffer][byte]);
        }
        printf("\"");
    }
    puts("]}");
    funlockfile(stdout);
    printf("{\"type\":\"endpoint\",\"case\":\"%s\",\"file\":\"%s\","
           "\"policy\":%u,\"block\":%u,\"length\":%u,\"elements\":131072}\n",
           label, path, probe_policy, probe_block, n->length);
}

static int gated_step(void *context, const unsigned *ids, unsigned length, unsigned *cursor)
{
    int result = observed_step(context, ids, length, cursor);
    pthread_mutex_lock(&gate_lock);
    gate_count++;
    if (gate_target && gate_count == gate_target) {
        gate_held = 1;
        printf("{\"type\":\"barrier\",\"policy\":%u,\"block\":%u,"
               "\"case\":\"%s\",\"chunk\":%u,\"cursor\":%u,\"status\":%d}\n",
               probe_policy, probe_block, probe_case, gate_count, *cursor, result);
        pthread_cond_broadcast(&gate_wake);
        while (!gate_release) {
            pthread_cond_wait(&gate_wake, &gate_lock);
        }
    }
    pthread_mutex_unlock(&gate_lock);
    return result;
}

static int selected_next(void *context)
{
    native_service *n = context;
    CHECK(n->logits && !n->pending && n->length < n->capacity);
    sample_calls++;
    /* Explicit teacher forcing, not a changed head or a natural EOS claim. */
    n->emitted = forced_id;
    n->pending = 1;
    printf("{\"type\":\"selection\",\"forced\":true,\"logits_unmodified\":true,\"id\":%u}\n",
           forced_id);
    return (int)forced_id;
}

static int counted_save(void *context)
{
    save_calls++;
    return native_save(context);
}

static int route(service_session *s, const char *path, const char *body, int expected)
{
    char *reply = malloc(131072);
    CHECK(reply);
    int stream = 0;
    int status = service_route(s, "POST", path, body, strlen(body), reply, 131072, &stream);
    printf("{\"type\":\"route\",\"policy\":%u,\"block\":%u,\"path\":\"%s\","
           "\"status\":%d,\"reply\":%s}\n", probe_policy, probe_block, path, status, reply);
    free(reply);
    CHECK(status == expected);
    return status;
}

static void idle(service_session *s)
{
    double deadline = seconds() + WAIT_SECONDS;
    for (;;) {
        char event[SERVICE_EVENT_BYTES];
        while (service_event(s, event)) {}
        pthread_mutex_lock(&s->lock);
        int busy = s->job || s->reset_job;
        pthread_mutex_unlock(&s->lock);
        if (!busy) {
            return;
        }
        CHECK(seconds() < deadline);
        struct timespec delay = {0, 1000000};
        nanosleep(&delay, NULL);
    }
}

static void job(service_session *s, const char *operation, uint64_t checkpoint)
{
    char path[128], body[256];
    snprintf(path, sizeof(path), "/sessions/1/%s", operation);
    if (!strcmp(operation, "restore")) {
        snprintf(body, sizeof(body), "{\"id\":\"r%u\",\"checkpoint_id\":\"%llu\"}",
                 route_serial++, (unsigned long long)checkpoint);
    } else if (!strcmp(operation, "generate")) {
        snprintf(body, sizeof(body), "{\"id\":\"r%u\",\"budget\":1}", route_serial++);
    } else {
        snprintf(body, sizeof(body), "{\"id\":\"r%u\"}", route_serial++);
    }
    route(s, path, body, !strcmp(operation, "generate") ? 200 : 202);
    idle(s);
    CHECK(!strcmp(s->history[s->active].status, "completed"));
    service_request *r = &s->history[s->active];
    flockfile(stdout);
    printf("{\"type\":\"job_completed\",\"policy\":%u,\"block\":%u,\"operation\":\"%s\","
           "\"logical\":%u,\"committed\":%u,\"pending\":%u,\"phase\":\"%s\","
           "\"checkpoint\":%llu,\"prefills\":%u,\"appends\":%u,\"decodes\":%u,"
           "\"recompute_committed\":%u,\"pending_committed\":%u,\"ids\":[",
           probe_policy, probe_block, operation, s->logical, s->committed, s->pending,
           s->state, (unsigned long long)s->checkpoint_id, r->work.prefill_calls,
           r->work.append_calls, r->work.decode_calls, r->work.recompute_committed,
           r->work.pending_committed);
    for (unsigned i = 0; i < s->logical; i++) {
        printf("%s%u", i ? "," : "", s->tokens[i]);
    }
    puts("]}");
    funlockfile(stdout);
}

static void digits(service_session *s, unsigned count, unsigned expected_length)
{
    char body[1024];
    const char *prefix = "{\"role\":\"user\",\"content\":\"";
    size_t start = strlen(prefix);
    memcpy(body, prefix, start);
    CHECK(start + count + 3 < sizeof(body));
    for (unsigned i = 0; i < count; i++) {
        body[start + i] = '0' + i % 10;
    }
    strcpy(body + start + count, "\"}");
    unsigned before = s->logical;
    route(s, "/sessions/1/messages", body, 200);
    CHECK(s->logical - before == expected_length);
}

static uint64_t prime(service_session *s, native_service *n)
{
    probe_case = "public_prime128_forcedEOS";
    route(s, "/sessions/1/reset", "{}", 200);
    /* Initial framing3 + 124 digit IDs + forced EOS1 = 128. */
    digits(s, 124, 127);
    forced_id = 2;
    job(s, "generate", 0);
    CHECK(s->logical == 128 && s->pending == 1 && !strcmp(s->state, "ready"));
    unsigned prior = save_calls;
    job(s, "checkpoint", 0);
    CHECK(save_calls == prior + 1 && s->committed == 128 && !s->pending);
    service_work w = s->history[s->active].work;
    CHECK(w.decode_calls == 1 && w.pending_committed == 1);
    endpoint(n, "public_prime128_materializedEOS");
    uint64_t saved = s->checkpoint_id;
    CHECK(saved && s->checkpoint_length == 128);
    job(s, "restore", saved);
    CHECK(s->logical == 128 && !strcmp(s->state, "ready") && !s->pending);
    if (n->policy == 2) {
        CHECK(!n->length && !n->logits && !s->committed);
        /* Checkpoint materialization is a real worker job, not a scalar edit. */
        job(s, "checkpoint", 0);
        w = s->history[s->active].work;
        CHECK(w.prefill_calls == 1 && w.recompute_committed == 128);
        CHECK(n->length == 128 && n->logits);
        saved = s->checkpoint_id;
        endpoint(n, "public_OFF_restore_rematerialized128");
    }
    char body[128];
    snprintf(body, sizeof(body), "{\"id\":\"no-user%u\",\"budget\":1}", route_serial++);
    route(s, "/sessions/1/generate", body, 409);
    return saved;
}

static void cancellation(service_session *s, native_service *n, int replacement, unsigned position)
{
    uint64_t saved = 0;
    if (n->policy) {
        saved = prime(s, n);
        digits(s, 511, 513);
        CHECK(s->logical == 641);
    } else {
        route(s, "/sessions/1/reset", "{}", 200);
        digits(s, 510, 513);
    }
    unsigned work_length = !n->policy || (n->policy == 2 && !replacement) ? s->logical : 513;
    unsigned blocks = (work_length + n->block - 1) / n->block;
    unsigned target = position == 0 ? 1 : position == 1 ? (blocks + 1) / 2 : blocks;
    probe_case = replacement ? "public_replacement_cancel" : "public_generation_cancel";
    pthread_mutex_lock(&gate_lock);
    gate_target = target;
    gate_count = 0;
    gate_release = gate_held = 0;
    pthread_mutex_unlock(&gate_lock);
    unsigned samples = sample_calls, saves = save_calls, calls = probe_calls;
    char path[128], body[128], id[65];
    snprintf(id, sizeof(id), "cancel%u", route_serial++);
    snprintf(path, sizeof(path), "/sessions/1/%s", replacement ? "checkpoint" : "generate");
    snprintf(body, sizeof(body), replacement ? "{\"id\":\"%s\"}" :
             "{\"id\":\"%s\",\"budget\":1}", id);
    route(s, path, body, replacement ? 202 : 200);
    pthread_mutex_lock(&gate_lock);
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += WAIT_SECONDS;
    while (!gate_held) {
        CHECK(!pthread_cond_timedwait(&gate_wake, &gate_lock, &deadline));
    }
    pthread_mutex_unlock(&gate_lock);
    CHECK(probe_calls == calls + target);
    pthread_mutex_lock(&s->lock);
    printf("{\"type\":\"barrier_controller_state\",\"policy\":%u,\"block\":%u,"
           "\"logical\":%u,\"committed\":%u,\"pending\":%u,\"native_cursor\":%u,"
           "\"checkpoint\":%llu}\n", n->policy, n->block, s->logical, s->committed,
           s->pending, n->length, (unsigned long long)s->checkpoint_id);
    pthread_mutex_unlock(&s->lock);
    CHECK(sample_calls == samples && save_calls == saves);
    snprintf(path, sizeof(path), "/requests/%s/cancel", id);
    route(s, path, "{}", 200);
    pthread_mutex_lock(&gate_lock);
    gate_release = 1;
    pthread_cond_broadcast(&gate_wake);
    pthread_mutex_unlock(&gate_lock);
    idle(s);
    pthread_mutex_lock(&gate_lock);
    gate_target = 0;
    pthread_mutex_unlock(&gate_lock);
    CHECK(probe_calls == calls + target && sample_calls == samples && save_calls == saves);
    CHECK(!strcmp(s->state, "needs_reset") && !s->checkpoint_id && !n->checkpoint);
    CHECK(!strcmp(s->history[s->active].status, "cancelled"));
    if (saved) {
        snprintf(body, sizeof(body), "{\"id\":\"old%u\",\"checkpoint_id\":\"%llu\"}",
                 route_serial++, (unsigned long long)saved);
        route(s, "/sessions/1/restore", body, 409);
    }
    CHECK(probe_calls == calls + target && sample_calls == samples && save_calls == saves);
    printf("{\"type\":\"cancel_verified\",\"policy\":%u,\"block\":%u,"
           "\"replacement\":%s,\"position\":%u,\"blocks\":%u,\"completed\":%u,"
           "\"old_checkpoint\":%llu,\"needs_reset\":true}\n", n->policy, n->block,
           replacement ? "true" : "false", position, blocks, target, (unsigned long long)saved);
    route(s, "/sessions/1/reset", "{}", 200);
    digits(s, 1, 4);
    forced_id = 9000;
    job(s, "generate", 0);
    CHECK(sample_calls == samples + 1);
    /* A cancel against the now-published completion cannot undo it. */
    snprintf(path, sizeof(path), "/requests/%s/cancel", s->history[s->active].id);
    unsigned length = s->logical;
    route(s, path, "{}", 200);
    CHECK(!strcmp(s->history[s->active].status, "completed") && s->logical == length);
}

static void retained_owner(void *context)
{
    (void)context; /* Joining worker transfers the still-live owner back to main. */
}

static void public_cases(native_service *n)
{
    service_session *s = calloc(1, sizeof(*s));
    CHECK(s);
    service_generator g = {.context = n, .next = selected_next, .reset = native_reset, .destroy = retained_owner,
        .frame = native_frame, .piece = native_piece, .eos = 2,
        .prefix_reuse = n->policy, .begin = native_begin, .save = counted_save,
        .restore = observed_restore, .invalidate = native_invalidate, .work = native_work,
        .materialize_step = gated_step, .context_capacity = 4352, .block_capacity = n->block};
    CHECK(service_init(s, g));
    route(s, "/sessions", "{}", 200);
    probe_case = "public_initial257_pending_reconciliation";
    digits(s, 254, 257);
    forced_id = 9000;
    job(s, "generate", 0);
    CHECK(s->logical == 258 && s->pending == 1);
    job(s, "generate", 0);
    CHECK(s->logical == 259 && s->pending == 1);
    route(s, "/sessions/1/reset", "{}", 200);
    if (!n->policy) {
        route(s, "/sessions/1/checkpoint", "{\"id\":\"unsupported-save\"}", 501);
        route(s, "/sessions/1/restore", "{\"id\":\"unsupported-restore\",\"checkpoint_id\":\"1\"}", 501);
    }
    for (unsigned replacement = 0; replacement < (n->policy ? 2u : 1u); replacement++) {
        for (unsigned position = 0; position < 3; position++) {
            cancellation(s, n, (int)replacement, position);
        }
    }
    if (n->policy) {
        route(s, "/sessions/1/reset", "{}", 200);
        /* Explicit controlled controller ledger, not a framed public message. */
        pthread_mutex_lock(&s->lock);
        CHECK(!s->job && !s->reset_job);
        for (unsigned i = 0; i < 4351; i++) {
            s->tokens[i] = 1000 + (37 * i) % 8000;
        }
        s->logical = 4351;
        pthread_mutex_unlock(&s->lock);
        probe_case = "controlled_controller_capacity_pending";
        forced_id = 9000;
        job(s, "generate", 0);
        CHECK(s->logical == 4352 && s->pending == 1);
        job(s, "checkpoint", 0);
        CHECK(!s->pending && s->committed == 4352);
        CHECK(s->history[s->active].work.pending_committed == 1);
        uint64_t saved = s->checkpoint_id;
        for (unsigned branch = 0; branch < 2; branch++) {
            job(s, "restore", saved);
            unsigned calls = probe_calls, samples = sample_calls;
            char body[128];
            snprintf(body, sizeof(body), "{\"id\":\"capacity%u\",\"budget\":1}", route_serial++);
            route(s, "/sessions/1/generate", body, 422);
            CHECK(probe_calls == calls && sample_calls == samples);
            if (n->policy == 2) {
                job(s, "checkpoint", 0);
                saved = s->checkpoint_id;
                CHECK(s->history[s->active].work.recompute_committed == 4352);
            }
        }
    }
    service_destroy(s); /* Worker join transfers the owner back to main. */
    free(s);
}

/* These are controlled adapter ledgers, NOT public framed messages. */
static void adapter_ingest(native_service *n, unsigned *ids, unsigned length)
{
    unsigned cursor = n->length;
    int result;
    do {
        result = observed_step(n, ids, length, &cursor);
        service_work receipt;
        native_work(n, &receipt);
        CHECK(result == 1 || result == 2);
    } while (result == 2);
    CHECK(n->length == length);
}

static void adapter_branches(native_service *n)
{
    unsigned ids[4352];
    for (unsigned i = 0; i < 4352; i++) {
        ids[i] = 1000 + (37 * i) % 8000;
    }
    const unsigned tails[] = {0, 1, 127, 128, 255};
    for (unsigned row = 0; row < sizeof(tails) / sizeof(*tails); row++) {
        native_reset(n);
        native_begin(n);
        probe_case = "controlled_adapter_prefix4096";
        adapter_ingest(n, ids, 4096);
        if (n->policy) {
            CHECK(native_save(n) == 1);
        }
        unsigned char retained[52][32], after[52][32];
        prefix_hashes(n, 4096, retained);
        float *first = malloc(131072 * sizeof(float));
        CHECK(first);
        /* Untouched, restore/replay, divergent branch, restore/rebranch. */
        for (unsigned branch = 0; branch < 4; branch++) {
            if (branch) {
                if (n->policy) {
                    CHECK(observed_restore(n) == 1);
                } else {
                    native_reset(n);
                    native_begin(n);
                    adapter_ingest(n, ids, 4096);
                }
            }
            if (n->policy == 1) {
                prefix_hashes(n, 4096, after);
                CHECK(!memcmp(retained, after, sizeof(after)));
            }
            unsigned old = ids[4096];
            if (branch == 2 && tails[row]) {
                ids[4096] = 8999;
            }
            native_begin(n);
            probe_case = branch == 2 ? "controlled_adapter_divergent_tail" :
                "controlled_adapter_original_tail";
            adapter_ingest(n, ids, 4096 + tails[row]);
            prefix_hashes(n, 4096, after);
            CHECK(!memcmp(retained, after, sizeof(after)));
            if (!branch) {
                memcpy(first, n->logits, 131072 * sizeof(float));
            } else if (branch != 2) {
                CHECK(!memcmp(first, n->logits, 131072 * sizeof(float)));
            }
            printf("{\"type\":\"branch_verified\",\"policy\":%u,\"block\":%u,"
                   "\"tail\":%u,\"branch\":%u,\"public_message\":false}\n",
                   n->policy, n->block, tails[row], branch);
            ids[4096] = old;
        }
        free(first);
    }
}

/* Exclusive failed-only sidecars: raw vector bytes survive NaNs unchanged. */
static int compare_direct(const endpoint_record *record, const float *expected,
                          const float *actual, const unsigned char *hashes)
{
    int vector_bad = memcmp(expected, actual, 131072 * sizeof(float)) != 0;
    for (unsigned i = 0; i < 131072; i++) {
        if (!isfinite(expected[i]) || !isfinite(actual[i])) vector_bad = 1;
    }
    int kv_bad = memcmp(hashes, record->hashes, sizeof(record->hashes)) != 0;
    if (vector_bad || kv_bad) {
        char path[4200], base[4200];
        if (endpoint_dir) {
            CHECK(snprintf(base, sizeof(base), "%s/failure", endpoint_dir) < (int)sizeof(base));
            CHECK(snprintf(path, sizeof(path), "%s/failure-source.txt", endpoint_dir) < (int)sizeof(path));
            size_t length = strlen(record->path);
            CHECK(length + 1 <= 4096);
            FILE *source = fopen(path, "wx");
            CHECK(source);
            CHECK(fwrite(record->path, 1, length, source) == length);
            CHECK(fputc('\n', source) != EOF);
            CHECK(!fclose(source));
        } else {
            CHECK(snprintf(base, sizeof(base), "%s", record->path) < (int)sizeof(base));
        }
        CHECK(snprintf(path, sizeof(path), "%s.failed-actual.f32", base) < (int)sizeof(path));
        FILE *f = fopen(path, "wx");
        CHECK(f);
        CHECK(fwrite(actual, sizeof(float), 131072, f) == 131072);
        CHECK(!fclose(f));
        if (kv_bad) {
            CHECK(snprintf(path, sizeof(path), "%s.failed-actual-kv.sha256", base) < (int)sizeof(path));
            f = fopen(path, "wx");
            CHECK(f);
            CHECK(fwrite(hashes, 1, sizeof(record->hashes), f) == sizeof(record->hashes));
            CHECK(!fclose(f));
            CHECK(snprintf(path, sizeof(path), "%s.failed-expected-kv.sha256", base) < (int)sizeof(path));
            f = fopen(path, "wx");
            CHECK(f);
            CHECK(fwrite(record->hashes, 1, sizeof(record->hashes), f) == sizeof(record->hashes));
            CHECK(!fclose(f));
        }
        return 0;
    }
    return 1;
}

/* Only active journal fields define identity; unused slots/padding never do. */
static int same_journal(const runtime_journal *a, const runtime_journal *b)
{
    return a->calls == b->calls && a->tokens == b->tokens &&
        !memcmp(a->api, b->api, a->calls * sizeof(a->api[0])) &&
        !memcmp(a->size, b->size, a->calls * sizeof(a->size[0])) &&
        !memcmp(a->ids, b->ids, a->tokens * sizeof(a->ids[0]));
}

static void read_endpoint(const endpoint_record *record, float *v)
{
    FILE *f = fopen(record->path, "rb");
    CHECK(f);
    CHECK(fread(v, sizeof(float), 131072, f) == 131072);
    CHECK(fgetc(f) == EOF && !ferror(f));
    CHECK(!fclose(f));
}

/* Worker has been joined; no callback or owner is concurrently live. Replays
 * deliberately invalidate checkpoints and are last in each owner's schedule. */
static void direct_replays(native_service *n)
{
    native_reset(n);
    char error[512];
    float *expected = malloc(131072 * sizeof(float));
    CHECK(expected);
    float *representative = malloc(131072 * sizeof(float));
    CHECK(representative);
    unsigned verified[512], unique = 0, duplicates = 0;
    unsigned calls = 0;
    unsigned long long direct_tokens = 0;
    for (unsigned row = 0; row < endpoint_count; row++) {
        endpoint_record *record = &endpoint_records[row];
        runtime_journal *j = &record->journal;
        unsigned match = unique;
        for (unsigned k = 0; k < unique; k++) {
            if (same_journal(j, &endpoint_records[verified[k]].journal)) {
                match = k;
                break;
            }
        }
        read_endpoint(record, expected);
        if (match < unique) {
            endpoint_record *rep = &endpoint_records[verified[match]];
            read_endpoint(rep, representative);
#ifdef CONTROLLER_STUB
            /* Inject only after real duplicate discovery, before the production
             * comparison path. Failed candidate bytes must remain on disk. */
            const char *negative = getenv("C_NEGATIVE_DUPLICATE");
            if (negative) {
                if (!strcmp(negative, "vector")) {
                    expected[131071] = NAN;
                    FILE *f = fopen(record->path, "r+b");
                    CHECK(f);
                    CHECK(fwrite(expected, sizeof(float), 131072, f) == 131072);
                    CHECK(!fclose(f));
                } else {
                    CHECK(!strcmp(negative, "hash"));
                    record->hashes[51][31] ^= 1;
                }
            }
#endif
            CHECK(compare_direct(record, expected, representative, &rep->hashes[0][0]));
            duplicates++;
            printf("{\"type\":\"duplicate_verified\",\"policy\":%u,\"block\":%u,"
                   "\"file\":\"%s\",\"representative\":\"%s\",\"same_owner\":true,"
                   "\"same_observed_partition\":true,\"finite_bitwise\":true}\n",
                   n->policy, n->block, record->path, rep->path);
            continue;
        }
        CHECK(!lunelette_metal_reset(n->runtime, error, sizeof(error)));
        const float *v = NULL;
        unsigned offset = 0;
        for (unsigned i = 0; i < j->calls; i++) {
            int result;
            if (j->api[i] == 'P') {
                result = lunelette_metal_prefill(n->runtime, j->ids + offset, j->size[i],
                                                &v, error, sizeof(error));
            } else if (j->api[i] == 'A') {
                result = lunelette_metal_append(n->runtime, j->ids + offset, j->size[i],
                                               &v, error, sizeof(error));
            } else {
                CHECK(j->api[i] == 'D' && j->size[i] == 1);
                result = lunelette_metal_decode(n->runtime, j->ids[offset], &v,
                                               error, sizeof(error));
            }
            CHECK(!result);
            calls++;
            direct_tokens += j->size[i];
            offset += j->size[i];
        }
        CHECK(offset == j->tokens && v);
        unsigned char hashes[52][32];
        prefix_hashes(n, j->tokens, hashes);
        CHECK(compare_direct(record, expected, v, &hashes[0][0]));
        verified[unique++] = row;
        printf("{\"type\":\"direct_verified\",\"policy\":%u,\"block\":%u,"
               "\"file\":\"%s\",\"calls\":%u,\"tokens\":%u,\"same_owner\":true,"
               "\"same_observed_partition\":true,\"finite_bitwise\":true}\n",
               n->policy, n->block, record->path, j->calls, j->tokens);
    }
    printf("{\"type\":\"owner_schedule\",\"policy\":%u,\"block\":%u,"
           "\"owners\":1,\"adapter_prefills\":%u,\"adapter_appends\":%u,"
           "\"adapter_decodes\":%u,\"adapter_resets\":%u,\"direct_calls\":%u,"
           "\"unique_controls\":%u,\"deduplicated_comparisons\":%u,\"direct_resets\":%u,\"endpoint_files\":%u,\"endpoint_bytes\":%u,"
           "\"runtime_snapshots\":%u,\"runtime_restores\":%u,\"controller_selections\":%u,"
           "\"controller_saves\":%u}\n",
           n->policy, n->block, actual_prefills, actual_appends, actual_decodes,
           actual_resets, calls, unique, duplicates, unique, endpoint_count, endpoint_count * 524288,
           actual_snapshots, actual_restores, sample_calls, save_calls);
    printf("{\"type\":\"token_budget\",\"policy\":%u,\"block\":%u,"
           "\"adapter_model_tokens\":%llu,\"direct_model_tokens\":%llu,"
           "\"total_model_tokens\":%llu,\"diagnostic_static_host_bytes\":%zu,"
           "\"outside_production_8MiB_census\":true,\"deadline_fit_claim\":false}\n",
           n->policy, n->block, adapter_tokens, direct_tokens,
           adapter_tokens + direct_tokens, sizeof(endpoint_records) + sizeof(target_ledgers));
    adapter_tokens = 0;
    free(expected);
    free(representative);
    endpoint_count = 0;
    actual_prefills = actual_appends = actual_decodes = actual_resets = 0;
    actual_snapshots = actual_restores = sample_calls = save_calls = 0;
}

/* Diagnostic-only admission: pinned 3B named weights/stage, not file size.
 * Production constructor independently repeats its unchanged guard afterward. */
#include "metal_plan.h"
#include "metal_memory_guard.h"
#define DIAGNOSTIC_RESERVATION (32ULL * 1024 * 1024)
static int diagnostic_guard_base(unsigned block, int compact, uint64_t *out)
{
    metal_allocation_plan plan;
    lunelette_metal_memory memory = {0};
    memory.weight_bytes = compact ? 4637454336ULL : 13716025344ULL;
    memory.conversion_stage_bytes = compact ? 113246208ULL : 0;
    if (metal_plan_create((lunelette_metal_limits){4352, block}, &plan) ||
        metal_plan_guard_base(&plan, &memory, out) ||
        *out > UINT64_MAX - DIAGNOSTIC_RESERVATION) return -1;
    *out += DIAGNOSTIC_RESERVATION;
    return 0;
}
static int diagnostic_preflight(unsigned block, int compact, metal_memory_probe probe,
                                char *error, size_t capacity)
{
    uint64_t base;
    if (diagnostic_guard_base(block, compact, &base)) {
        snprintf(error, capacity, "diagnostic reservation: invalid plan or overflow");
        return -1;
    }
    return metal_memory_guard(base, probe, error, capacity);
}
#ifndef CONTROLLER_STUB
#include <mach/mach.h>
#include <errno.h>
static int memory_available(void *context, uint64_t *available)
{
    (void)context;
    vm_statistics64_data_t statistics;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page_size;
    host_t host = mach_host_self();
    kern_return_t statistics_status = host_statistics64(host, HOST_VM_INFO64,
        (host_info64_t)&statistics, &count);
    kern_return_t page_status = host_page_size(host, &page_size);
    if (host != MACH_PORT_NULL) {
        mach_port_deallocate(mach_task_self(), host);
    }
    if (statistics_status != KERN_SUCCESS || page_status != KERN_SUCCESS) {
        return -1;
    }
    uint64_t pages = (uint64_t)statistics.free_count + statistics.inactive_count +
        statistics.speculative_count;
    if (page_size == 0 || pages > UINT64_MAX / page_size) {
        return -1;
    }
    *available = pages * page_size;
    return 0;
}

static int memory_now(void *context, uint64_t *now)
{
    (void)context;
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0 || time.tv_sec < 0 ||
        time.tv_nsec < 0 || time.tv_nsec >= 1000000000L ||
        (uint64_t)time.tv_sec > (UINT64_MAX - (uint64_t)time.tv_nsec) / 1000000000ULL) {
        return -1;
    }
    *now = (uint64_t)time.tv_sec * 1000000000ULL + (uint64_t)time.tv_nsec;
    return 0;
}

static int memory_sleep(void *context, uint64_t nanoseconds)
{
    (void)context;
    struct timespec delay = {0, (long)nanoseconds};
    /* EINTR returns to the policy to remeasure remaining time, never restarting
     * a full interval or resetting the deadline. */
    return nanosleep(&delay, NULL) == 0 || errno == EINTR ? 0 : -1;
}

static int official_diagnostic_preflight(unsigned block, int compact,
                                         char *error, size_t capacity)
{
    return diagnostic_preflight(block, compact,
        (metal_memory_probe){NULL, memory_available, memory_now, memory_sleep}, error, capacity);
}
#endif

/* Compile-time approval only. No argument or environment variable opens this gate. */
#ifndef CONTROLLER_MODEL_ENTRY
#define CONTROLLER_MODEL_ENTRY 0
#endif

typedef struct {
    native_service *(*create)(unsigned, unsigned, char **);
    void (*run)(native_service *);
    void (*destroy)(void *);
} owner_lifecycle;

static int sequential_owners(unsigned block, char **args, owner_lifecycle ops)
{
    CHECK(block == 128 || block == 256);
    endpoint_serial = 0;
    for (unsigned policy = 0; policy < 3; policy++) {
        native_service *n = ops.create(policy, block, args);
        if (!n) {
            return 3;
        }
        CHECK(n->policy == policy && n->block == block && n->capacity == 4352);
        ops.run(n);
        ops.destroy(n); /* Includes checkpoint, runtime, tokenizer before next create. */
    }
    return 0;
}

static void owner_cases(native_service *n)
{
    controller_probe_subset(n, n->policy, n->block);
    public_cases(n); /* Joins worker before returning. */
    adapter_branches(n);
    direct_replays(n);
}

#ifndef CONTROLLER_STUB
static native_service *official_owner(unsigned policy, unsigned block, char **args)
{
    native_service *n = calloc(1, sizeof(*n));
    if (!n) {
        return NULL;
    }
    n->policy = policy;
    n->capacity = 4352;
    n->block = block;
    char error[512] = {0};
    lunelette_metal_options options = {
        !strcmp(args[4], "custom") ? LUNELETTE_METAL_PREFILL_CUSTOM : LUNELETTE_METAL_PREFILL_MPS,
        !strcmp(args[5], "compact") ? LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC : LUNELETTE_METAL_RESIDENT_FP32
    };
    lunelette_metal_limits limits = {4352, block};
    if (!lunelette_tokenizer_open(args[2], &n->tokenizer, error, sizeof(error))) {
        fprintf(stderr, "tokenizer: %s\n", error);
        free(n);
        return NULL;
    }
    /* Measure after persistent tokenizer allocation so it cannot consume the
     * separate diagnostic reservation between this check and model admission. */
    if (official_diagnostic_preflight(block, !strcmp(args[5], "compact"), error, sizeof(error))) {
        fprintf(stderr, "diagnostic preflight: %s\n", error);
        lunelette_tokenizer_close(n->tokenizer);
        free(n);
        return NULL; /* No model constructed on extra refusal. */
    }
    n->runtime = lunelette_metal_create_configured(args[1], &options, &limits, error, sizeof(error));
    if (!n->runtime) {
        fprintf(stderr, "load: %s\n", error);
        lunelette_tokenizer_close(n->tokenizer);
        free(n);
        return NULL;
    }
    n->loads = 1;
    return n;
}

/* Kept compiled even while gate is false, so constructor/case linkage is checked. */
static int future_entry(int argc, char **argv)
{
    if (argc != 7 || (strcmp(argv[3], "128") && strcmp(argv[3], "256")) ||
        (strcmp(argv[4], "mps") && strcmp(argv[4], "custom")) ||
        (strcmp(argv[5], "resident") && strcmp(argv[5], "compact")) ||
        !argv[1][0] || argv[1][0] == '-' || !argv[2][0] || argv[2][0] == '-' ||
        !argv[6][0] || argv[6][0] == '-' || strlen(argv[6]) > 256) {
        fputs("REFUSED: invalid C arguments; no model opened.\n", stderr);
        return 2;
    }
    endpoint_dir = argv[6]; /* Caller supplies an exclusive output directory. */
    return sequential_owners((unsigned)atoi(argv[3]), argv,
        (owner_lifecycle){official_owner, owner_cases, native_destroy});
}
#endif

#ifdef CONTROLLER_STUB
typedef struct { uint64_t peak, clock; unsigned queries, sleeps; int sufficient; } admission_mock;
static int admission_available(void *ctx, uint64_t *out)
{
    admission_mock *m = ctx;
    m->queries++;
    *out = m->peak + m->sufficient;
    return 0;
}
static int admission_now(void *ctx, uint64_t *out)
{
    *out = ((admission_mock *)ctx)->clock;
    return 0;
}
static int admission_sleep(void *ctx, uint64_t ns)
{
    admission_mock *m = ctx;
    CHECK(ns == 50000000ULL);
    m->clock += ns;
    m->sleeps++;
    return 0;
}
static int test_diagnostic_admission(void)
{
    for (unsigned block = 128; block <= 256; block += 128) {
        for (int compact = 0; compact <= 1; compact++) {
            uint64_t base;
            CHECK(!diagnostic_guard_base(block, compact, &base));
            uint64_t growth = (block == 128 ? 111280640ULL : 222036992ULL) +
                926941184ULL - 142345216ULL + 8ULL * 1024 * 1024;
            CHECK(base == (compact ? 4750700544ULL : 13716025344ULL) +
                growth + 32ULL * 1024 * 1024);
            for (int sufficient = 0; sufficient <= 1; sufficient++) {
                admission_mock m = {base + 310ULL * 1024 * 1024 +
                    2ULL * 1024 * 1024 * 1024, 0, 0, 0, sufficient};
                char error[512] = {0};
                int result = diagnostic_preflight(block, compact,
                    (metal_memory_probe){&m, admission_available, admission_now, admission_sleep},
                    error, sizeof(error));
                CHECK(result == (sufficient ? 0 : -1));
                CHECK(m.queries == (sufficient ? 1 : 60));
                CHECK(m.sleeps == (sufficient ? 0 : 60));
                CHECK(m.clock == (sufficient ? 0 : 3000000000ULL));
                if (!sufficient) CHECK(strstr(error, "budget expired"));
            }
        }
    }
    uint64_t base;
    CHECK(diagnostic_guard_base(64, 0, &base) == -1);
    return 0;
}
static unsigned mock_live, mock_created, mock_phase, mock_fail;
static native_service mock_owner;
static native_service *mock_create(unsigned policy, unsigned block, char **args)
{
    (void)args;
    CHECK(!mock_live && mock_phase == 0 && policy == mock_created);
    if (mock_fail == policy + 1) {
        return NULL;
    }
    mock_created++;
    mock_live = 1;
    mock_phase = 1;
    memset(&mock_owner, 0, sizeof(mock_owner));
    mock_owner.policy = policy;
    mock_owner.block = block;
    mock_owner.capacity = 4352;
    return &mock_owner;
}
static void mock_run(native_service *n)
{
    CHECK(n == &mock_owner && mock_live == 1 && mock_phase == 1);
    mock_phase = 2;
}
static void mock_destroy(void *n)
{
    CHECK(n == &mock_owner && mock_live == 1 && mock_phase == 2);
    mock_live = mock_phase = 0;
}
static int test_lifecycle(void)
{
    for (unsigned block = 128; block <= 256; block += 128) {
        for (mock_fail = 0; mock_fail <= 3; mock_fail++) {
            mock_created = 0;
            int result = sequential_owners(block, NULL,
                (owner_lifecycle){mock_create, mock_run, mock_destroy});
            CHECK(result == (mock_fail ? 3 : 0));
            CHECK(mock_created == (mock_fail ? mock_fail - 1 : 3));
            CHECK(!mock_live && !mock_phase);
        }
    }
    return 0;
}
#endif

int reviewed_unused_main(int argc, char **argv)
{
    (void)argv;
#ifdef CONTROLLER_STUB
    if (argc == 2 && !strcmp(argv[1], "--test-admission")) return test_diagnostic_admission();
    if (argc == 2 && !strcmp(argv[1], "--test-lifecycle")) {
        return test_lifecycle();
    }
    if (argc == 2 && !strcmp(argv[1], "--test-journal-key")) {
        runtime_journal a = {0}, b;
        a.calls = 2; a.tokens = 3;
        a.api[0] = 'P'; a.api[1] = 'A';
        a.size[0] = 1; a.size[1] = 2;
        a.ids[0] = 7; a.ids[1] = 8; a.ids[2] = 9;
        memset(&b, 0xa5, sizeof(b));
        b.calls = a.calls; b.tokens = a.tokens;
        memcpy(b.api, a.api, 2); memcpy(b.size, a.size, 2 * sizeof(unsigned));
        memcpy(b.ids, a.ids, 3 * sizeof(int64_t));
        CHECK(same_journal(&a, &b));
        b.ids[2]++; CHECK(!same_journal(&a, &b)); b.ids[2]--;
        b.api[1] = 'D'; CHECK(!same_journal(&a, &b)); b.api[1] = 'A';
        b.size[0] = 2; b.size[1] = 1; CHECK(!same_journal(&a, &b));
        b = a; b.calls--; CHECK(!same_journal(&a, &b));
        b = a; b.tokens--; CHECK(!same_journal(&a, &b));
        return 0;
    }
    if (argc == 4 && !strcmp(argv[1], "--negative-vector")) {
        endpoint_record record = {0};
        CHECK(strlen(argv[2]) < sizeof(record.path));
        strcpy(record.path, argv[2]);
        float *expected = calloc(131072, sizeof(float));
        float *actual = calloc(131072, sizeof(float));
        unsigned char hashes[52][32] = {{0}};
        CHECK(expected && actual);
        if (!strcmp(argv[3], "nan")) actual[131071] = NAN;
        else if (!strcmp(argv[3], "bits")) actual[131071] = -0.0f;
        else { CHECK(!strcmp(argv[3], "kv")); hashes[51][31] = 1; }
        CHECK(!compare_direct(&record, expected, actual, &hashes[0][0]));
        free(expected); free(actual);
        return 0;
    }
    CHECK(argc == 4);
    unsigned requested_block = (unsigned)strtoul(argv[3], NULL, 10);
    CHECK(requested_block == 128 || requested_block == 256);
    CHECK(strlen(argv[2]) <= 256);
    endpoint_dir = argv[2];
    for (unsigned block = requested_block; block == requested_block; block = 0) {
        for (unsigned policy = 0; policy < 3; policy++) {
            native_service *n = calloc(1, sizeof(*n));
            CHECK(n);
            n->runtime = calloc(1, sizeof(*n->runtime));
            CHECK(n->runtime);
            n->runtime->capacity = n->capacity = 4352;
            n->runtime->block = n->block = block;
            n->policy = policy;
            char error[512];
            CHECK(lunelette_tokenizer_open(argv[1], &n->tokenizer, error, sizeof(error)));
            owner_cases(n);
            native_invalidate(n);
            lunelette_metal_destroy(n->runtime);
            lunelette_tokenizer_close(n->tokenizer);
            free(n);
        }
    }
    return 0;
#else
    if (CONTROLLER_MODEL_ENTRY) {
        @autoreleasepool {
            return future_entry(argc, argv);
        }
    }
    fputs("REFUSED: C model entry compile gate CLOSED.\n", stderr);
    return 77;
#endif
}
