#define _POSIX_C_SOURCE 200809L
#include "session.h"
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

uint64_t service_process_footprint(void)
{
#ifdef __APPLE__
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS &&
        count >= TASK_VM_INFO_REV1_COUNT) {
        return info.phys_footprint;
    }
#endif
    return 0;
}

static double milliseconds(void)
{
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) || time.tv_sec < 0 ||
        (uint64_t)time.tv_sec > INT64_MAX / 1000000000) {
        abort(); /* Timing is required: never emit fabricated successful measurements. */
    }
    return time.tv_sec * 1000.0 + time.tv_nsec / 1e6;
}

static void quote_text(const char *input, char *output)
{
    /* Callers reserve six output bytes per retained input byte plus NUL. */
    while (*input) {
        unsigned c = (unsigned char)*input++;
        if (c < 32 || c == '"' || c == '\\') {
            snprintf(output, 7, "\\u%04x", c);
            output += 6;
        } else {
            *output++ = (char)c;
        }
    }
    *output = 0;
}

static int equals(const tj_node *n, const char *text)
{
    return n->length == strlen(text) && !memcmp(n->text, text, n->length);
}

static int service_json_profile(const void *body, size_t length, const char *fields,
                                tj_document *d, int extended)
{
    if (!tj_read_memory_limits(body, length, d, extended ? 262144 : 8192,
                               extended ? 32768 : 4096, 64, 4)) {
        return 0;
    }
    if (d->nodes[1].kind != '{') {
        tj_free(d);
        return 0;
    }
    for (unsigned p = d->nodes[1].child; p; p = d->nodes[d->nodes[p].next].next) {
        tj_node *key = &d->nodes[p], *value = &d->nodes[key->next];
        char name[128];
        if (!key->length || key->length > 64 || memchr(key->text, 0, key->length) ||
            memchr(key->text, '|', key->length) ||
            (value->kind != '"' && value->kind != '0') ||
            memchr(value->text, 0, value->length)) {
            tj_free(d);
            return 0;
        }
        snprintf(name, sizeof(name), "|%.*s|", (int)key->length, key->text);
        if (!strstr(fields, name)) {
            tj_free(d);
            return 0;
        }
    }
    return 1;
}

int service_json(const void *body, size_t length, const char *fields, tj_document *d)
{
    return service_json_profile(body, length, fields, d, 0);
}

/* All append operations either fit completely or fail, without advancing past capacity. */
static int append(char *out, size_t size, size_t *used, const char *format, ...)
{
    if (*used >= size) {
        return 0;
    }
    va_list args;
    va_start(args, format);
    int n = vsnprintf(out + *used, size - *used, format, args);
    va_end(args);
    if (n < 0 || (size_t)n >= size - *used) {
        return 0;
    }
    *used += (size_t)n;
    return 1;
}

/* Decimal integers only; canonical results additionally reject -0. */
static int bounded_integer(const unsigned char *text, size_t length, uint64_t bound,
                           int canonical, int64_t *value)
{
    size_t i = 0;
    int negative = length && text[0] == '-';
    if (negative) {
        i++;
    }
    if (i == length || (text[i] == '0' && ((canonical && negative) || length - i != 1))) {
        return 0;
    }
    uint64_t number = 0;
    for (; i < length; i++) {
        unsigned digit = (unsigned)text[i] - '0';
        if (digit > 9 || number > (bound - digit) / 10) {
            return 0;
        }
        number = number * 10 + digit;
    }
    *value = negative ? -(int64_t)number : (int64_t)number;
    return 1;
}

static int calculator_arguments(service_request *r)
{
    tj_document d = {0};
    if (!service_json(r->text, strlen(r->text), "|operation||a||b|", &d)) {
        return 0;
    }
    unsigned op = tj_field(&d, 1, "operation");
    unsigned a = tj_field(&d, 1, "a"), b = tj_field(&d, 1, "b");
    int64_t av = 0, bv = 0;
    unsigned operation = op && d.nodes[op].kind == '"' ?
        (equals(&d.nodes[op], "add") ? 1 : equals(&d.nodes[op], "subtract") ? 2 :
         equals(&d.nodes[op], "multiply") ? 3 : 0) : 0;
    int valid = operation && a && b && d.nodes[a].kind == '0' && d.nodes[b].kind == '0' &&
        bounded_integer(d.nodes[a].text, d.nodes[a].length, 1000000, 0, &av) &&
        bounded_integer(d.nodes[b].text, d.nodes[b].length, 1000000, 0, &bv);
    if (valid) {
        r->tool_operation = operation;
        r->tool_a = (int)av;
        r->tool_b = (int)bv;
    }
    tj_free(&d);
    return valid;
}

static const char *const lookup_documents[] = {"", "larch-stock", "cedar-stock", "pump-readings"};
static const char *const lookup_passages[] = {
    "", "LARCH-STOCK:P1", "CEDAR-STOCK:P1", "PUMP-READINGS:P1"
};

static int lookup_arguments(service_request *r, unsigned profile)
{
    tj_document d = {0};
    if (!service_json(r->text, strlen(r->text), "|document_id|", &d)) {
        return 0;
    }
    unsigned id = tj_field(&d, 1, "document_id");
    unsigned document = id && d.nodes[id].kind == '"' ?
        (equals(&d.nodes[id], lookup_documents[1]) ? 1 :
         equals(&d.nodes[id], lookup_documents[2]) ? 2 :
         equals(&d.nodes[id], lookup_documents[3]) ? 3 : 0) : 0;
    tj_free(&d);
    if (!((profile == 2 && document >= 1 && document <= 2) ||
          (profile == 3 && document == 3))) {
        document = 0;
    }
    r->tool_document = document;
    return document != 0;
}

static int lookup_result(const service_request *r, const tj_node *content)
{
    tj_document d = {0};
    if (content->length > 2048 || !service_json(content->text, content->length,
            "|document_id||passage_id||text|", &d)) {
        return 0;
    }
    unsigned id = tj_field(&d, 1, "document_id");
    unsigned passage = tj_field(&d, 1, "passage_id");
    unsigned text = tj_field(&d, 1, "text");
    int valid = r->tool_document >= 1 && r->tool_document <= 3 && id && passage && text &&
        d.nodes[id].kind == '"' && d.nodes[passage].kind == '"' &&
        d.nodes[text].kind == '"' && d.nodes[text].length && d.nodes[text].length <= 1024 &&
        equals(&d.nodes[id], lookup_documents[r->tool_document]) &&
        equals(&d.nodes[passage], lookup_passages[r->tool_document]);
    tj_free(&d);
    return valid;
}

/* Small terminal metadata, never argument text deltas or unvalidated JSON. */
static void tool_metadata(const service_request *r, char *output, size_t size)
{
    output[0] = 0;
    if (r->tool_kind == 2 && r->tool_document && !strcmp(r->stop_reason, "tool_call")) {
        snprintf(output, size, ",\"tool_call\":{\"id\":\"%s\",\"name\":\"lookup\","
            "\"arguments\":{\"document_id\":\"%s\"}}", r->id,
            lookup_documents[r->tool_document]);
    } else if (r->tool_operation && !strcmp(r->stop_reason, "tool_call")) {
        const char *names[] = {"", "add", "subtract", "multiply"};
        snprintf(output, size, ",\"tool_call\":{\"id\":\"%s\",\"name\":\"calculator\","
            "\"arguments\":{\"operation\":\"%s\",\"a\":%d,\"b\":%d}}",
            r->id, names[r->tool_operation], r->tool_a, r->tool_b);
    }
}

static void cancel_locked(service_session *s, const char *source)
{
    service_request *r = &s->history[s->active];
    if (!s->cancel && s->job) {
        r->cancel_received_ms = milliseconds();
        strcpy(r->cancel_source, source);
    }
    s->cancel = 1;
}

static int metrics(const service_request *r, int native, char *out, size_t size)
{
    size_t n = 0;
    char before[32] = "null", after[32] = "null", first[32] = "null", cancel[32] = "null";
    if (r->memory_before) {
        snprintf(before, sizeof(before), "%llu", (unsigned long long)r->memory_before);
    }
    if (r->memory_after) {
        snprintf(after, sizeof(after), "%llu", (unsigned long long)r->memory_after);
    }
    if (r->first_token_ms >= 0) {
        snprintf(first, sizeof(first), "%.3f", r->first_token_ms);
    }
    if (r->cancel_received_ms && strcmp(r->status, "generating") &&
        strcmp(r->status, "running")) {
        snprintf(cancel, sizeof(cancel), "%.3f", r->cancel_latency_ms);
    }
    if (!append(out, size, &n,
        "{\"test_only\":%s,\"framing_ms\":%.3f,\"prefill_ms\":%.3f,"
        "\"decode_ms\":%.3f,\"first_token_ms\":%s,\"total_ms\":%.3f,"
        "\"cancel_latency_ms\":%s,\"cancel_source\":\"%s\","
        "\"process_phys_footprint_before_bytes\":%s,\"process_phys_footprint_after_bytes\":%s,"
        "\"driver_memory_bytes\":null,\"prompt_tokens\":%u,\"reused_tokens\":%u,"
        "\"newly_processed_tokens\":%u,\"prefill_us\":%llu,\"prefill_steps\":%u,"
        "\"decode_total_us\":%llu,\"decode_steps\":%u,\"cancel_received_monotonic_ms\":%.3f,"
        "\"decode_step_duration_us\":[",
        native ? "false" : "true", r->framing_ms, r->prefill_ms, r->decode_ms,
        first, r->total_ms, cancel, r->cancel_source, before, after,
        r->prompt, r->reused, r->newly_processed, (unsigned long long)r->prefill_us,
        r->prefill_steps, (unsigned long long)r->decode_total_us, r->decode_steps,
        r->cancel_received_ms)) {
        return 0;
    }
    for (unsigned i = 0; i < r->decode_steps; i++) {
        if (!append(out, size, &n, "%s%llu", i ? "," : "",
                    (unsigned long long)r->decode_us[i])) {
            return 0;
        }
    }
    if (!append(out, size, &n, "]")) {
        return 0;
    }
    if ((!strcmp(r->operation, "checkpoint") || !strcmp(r->operation, "restore")) &&
        !append(out, size, &n, ",\"materialize_ms\":%.3f,\"snapshot_ms\":%.3f,"
                "\"restore_ms\":%.3f", r->materialize_ms, r->snapshot_ms, r->restore_ms)) {
        return 0;
    }
    return append(out, size, &n, ",\"metrics_version\":3,\"suffix_us\":%llu,"
        "\"suffix_steps\":%u,\"work_version\":2,\"work\":{"
        "\"prefill_attempted\":%u,\"prefill_committed\":%u,"
        "\"append_attempted\":%u,\"append_committed\":%u,"
        "\"pending_attempted\":%u,\"pending_committed\":%u,"
        "\"prefill_calls\":%u,\"append_calls\":%u,\"decode_calls\":%u,"
        "\"decode_attempted\":%u,\"decode_committed\":%u,\"logit_restores\":%u,"
        "\"recompute_attempted\":%u,\"recompute_committed\":%u}}",
        (unsigned long long)r->suffix_us, r->suffix_steps,
        r->work.full_attempted, r->work.full_committed,
        r->work.suffix_attempted, r->work.suffix_committed, r->work.pending_attempted,
        r->work.pending_committed, r->work.prefill_calls, r->work.append_calls,
        r->work.decode_calls, r->work.decode_attempted, r->work.decode_committed,
        r->work.logit_restores, r->work.recompute_attempted, r->work.recompute_committed);
}

/* Planned diagnostic trajectory, including attempts cancelled before publication. */
static void replay_metadata(const service_request *r, char *output, size_t size)
{
    size_t used = 0;
    output[0] = 0;
    if (!r->replay_count) {
        return;
    }
    append(output, size, &used,
        ",\"evaluation_mode\":\"controlled_replay\",\"replay_tokens\":[");
    for (unsigned i = 0; i < r->replay_count; i++) {
        append(output, size, &used, "%s%u", i ? "," : "", r->replay_tokens[i]);
    }
    append(output, size, &used, "]");
}

static int replay_sequence(const tj_document *d, unsigned node, unsigned budget,
                           unsigned tokens[SERVICE_REPLAY_TOKENS])
{
    if (!node || d->nodes[node].kind != '"' || !budget || budget > SERVICE_REPLAY_TOKENS) {
        return 0;
    }
    const unsigned char *text = (const unsigned char *)d->nodes[node].text;
    size_t length = d->nodes[node].length, at = 0;
    for (unsigned i = 0; i < budget; i++) {
        unsigned value = 0;
        if (at == length || text[at] < '1' || text[at] > '9') {
            return 0;
        }
        while (at < length && text[at] >= '0' && text[at] <= '9') {
            unsigned digit = text[at++] - '0';
            if (value > (131071u - digit) / 10) {
                return 0;
            }
            value = value * 10 + digit;
        }
        if (!(value >= 1000 && value <= 131071) && !(value == 2 && i + 1 == budget)) {
            return 0;
        }
        tokens[i] = value;
        if (i + 1 < budget && (at == length || text[at++] != ',')) {
            return 0;
        }
    }
    return at == length;
}

static void event_locked(service_session *s, const char *type, const char *text,
                         const char *reason)
{
    if (s->count == SERVICE_EVENTS) {
        cancel_locked(s, "overflow");
        return;
    }
    unsigned slot = (s->head + s->count) % SERVICE_EVENTS;
    service_request *r = &s->history[s->active];
    char measured[SERVICE_METRIC_BYTES] = "null", tool[256], replay[256];
    replay_metadata(r, replay, sizeof(replay));
    tool_metadata(r, tool, sizeof(tool));
    int terminal = strcmp(type, "started") && strcmp(type, "text_delta");
    size_t used = 0;
    if ((terminal && !metrics(r, s->generator.info != NULL, measured, sizeof(measured))) ||
        !append(s->events[slot], SERVICE_EVENT_BYTES, &used,
             "{\"event\":\"%s\",\"id\":\"%s\",\"session_id\":%u,\"text\":\"%s\","
             "\"stop_reason\":\"%s\",\"generated_tokens\":%u,"
             "\"logical_tokens\":%u,\"committed_tokens\":%u,\"pending_tokens\":%u,"
             "\"before_state\":\"%s\",\"before_logical_tokens\":%u,"
             "\"before_committed_tokens\":%u,\"before_pending_tokens\":%u,"
             "\"state\":\"%s\",\"session_valid\":%s,\"recovery\":\"%s\",\"metrics\":%s%s%s}\n",
             type, r->id, r->session_id, text, reason, r->generated, s->logical,
             s->committed, s->pending, r->before_state, r->before_logical,
             r->before_committed, r->before_pending, s->state,
             !strcmp(s->state, "needs_reset") ? "false" : "true",
             !strcmp(s->state, "needs_reset") ? "reset" : "none", measured, tool, replay)) {
        cancel_locked(s, "overflow");
        return;
    }
    s->count++;
}

static void add_work(service_work *total, const service_work *work)
{
    /* At most 256 generation boundaries, each bounded by context4352.
     * Token and call aggregates remain below unsigned capacity. */
    total->full_attempted += work->full_attempted;
    total->full_committed += work->full_committed;
    total->suffix_attempted += work->suffix_attempted;
    total->suffix_committed += work->suffix_committed;
    total->pending_attempted += work->pending_attempted;
    total->pending_committed += work->pending_committed;
    total->prefill_calls += work->prefill_calls;
    total->append_calls += work->append_calls;
    total->decode_calls += work->decode_calls;
    total->decode_attempted += work->decode_attempted;
    total->decode_committed += work->decode_committed;
    total->logit_restores += work->logit_restores;
    total->recompute_attempted += work->recompute_attempted;
    total->recompute_committed += work->recompute_committed;
}

/* Worker-owned orchestration, unlocked on entry/exit. No adapter may hide a
 * multi-block loop. Publish progress and observe cancellation after every block,
 * including READY, before any sample or checkpoint publication. */
static int materialize(service_session *s, service_work *total)
{
    int result;
    unsigned pending_position = s->committed;
    unsigned previous = 0;
    int first = 1;
    do {
        unsigned cursor = s->committed;
        if (s->generator.materialize_step) {
            result = s->generator.materialize_step(s->generator.context,
                s->tokens, s->logical, &cursor);
        } else {
            result = s->generator.materialize(s->generator.context, s->tokens, s->logical);
            if (result) {
                cursor = s->logical;
            }
        }
        service_work receipt = {0};
        if (s->generator.work) {
            s->generator.work(s->generator.context, &receipt);
            if (!total->prefill_calls && !total->append_calls && !total->decode_calls) {
                total->reused = receipt.reused;
            }
            add_work(total, &receipt);
        }
        if (result < 0 || result > SERVICE_MATERIALIZE_MORE || cursor > s->logical ||
            (result == 1 && cursor != s->logical) ||
            (result == SERVICE_MATERIALIZE_MORE && (!cursor || cursor >= s->logical || (!first && cursor <= previous)))) {
            cursor = s->committed;
            result = 0;
        }
        previous = cursor;
        first = 0;
        pthread_mutex_lock(&s->lock);
        if (s->generator.materialize_step) {
            s->committed = cursor;
            if (result == 1 || cursor > pending_position) {
                s->pending = 0;
            }
        }
        int cancelled = s->cancel || s->stop;
        pthread_mutex_unlock(&s->lock);
        if (cancelled || !result) {
            return 0;
        }
    } while (result == SERVICE_MATERIALIZE_MORE);
    return result == 1;
}

static unsigned context_capacity(const service_session *s)
{
    return s->generator.context_capacity ? s->generator.context_capacity : 256;
}

/* Called with the lock held; callbacks remain worker-only and unlocked. */
static void control_job(service_session *s, service_request *r)
{
    char phase[16];
    strcpy(phase, r->before_state);
    int saving = !strcmp(r->operation, "checkpoint");
    int result = 0, mutated = 0;
    service_work receipt = {0};
    double start = milliseconds();
    double materialize_ms = 0, snapshot_ms = 0, restore_ms = 0;
    r->memory_before = s->generator.info ? service_process_footprint() : 0;
    r->prompt = s->logical;
    /* Charge pending framing once, including branch input discarded by restore. */
    r->framing_ms = s->framing_ms;
    s->framing_ms = 0;
    pthread_mutex_unlock(&s->lock);
    if (saving) {
        result = materialize(s, &receipt);
        materialize_ms = milliseconds() - start;
        /* Materialization may have changed KV even on failure. */
        mutated = 1;
        if (result) {
            double tick = milliseconds();
            result = s->generator.save(s->generator.context);
            snapshot_ms = milliseconds() - tick;
        } else {
            result = -1;
        }
    } else {
        result = s->generator.restore(s->generator.context);
        restore_ms = milliseconds() - start;
        mutated = result != 0;
        if (s->generator.work) {
            s->generator.work(s->generator.context, &receipt);
        }
    }
    pthread_mutex_lock(&s->lock);
    r->materialize_ms = materialize_ms;
    r->snapshot_ms = snapshot_ms;
    r->restore_ms = restore_ms;
    r->work = receipt;
    r->reused = receipt.reused;
    r->newly_processed = receipt.full_committed + receipt.suffix_committed +
        receipt.decode_committed;
    if (saving && result >= 0) {
        /* A failed snapshot allocation cannot undo successful materialization. */
        s->committed = s->logical;
        s->pending = 0;
    }
    if (result == 1 && !s->cancel && !s->stop) {
        if (saving) {
            memcpy(s->checkpoint_tokens, s->tokens, s->logical * sizeof(*s->tokens));
            s->checkpoint_length = s->logical;
            strcpy(s->checkpoint_phase, phase);
            s->checkpoint_id = ++s->checkpoint_serial;
        } else {
            memcpy(s->tokens, s->checkpoint_tokens,
                   s->checkpoint_length * sizeof(*s->tokens));
            s->logical = s->checkpoint_length;
            s->committed = s->generator.prefix_reuse == 1 ? s->logical : 0;
            s->pending = s->utf8_length = 0;
            strcpy(phase, s->checkpoint_phase);
            r->cache_restored = s->generator.prefix_reuse == 1;
        }
        r->checkpoint_id = s->checkpoint_id;
        strcpy(r->status, "completed");
        strcpy(s->state, phase);
    } else {
        strcpy(r->status, s->cancel || s->stop ? "cancelled" : "error");
        if (result < 0 || ((s->cancel || s->stop) && mutated)) {
            strcpy(s->state, "needs_reset");
            s->checkpoint_id = s->checkpoint_length = 0;
            pthread_mutex_unlock(&s->lock);
            if (s->generator.invalidate) {
                s->generator.invalidate(s->generator.context);
            }
            pthread_mutex_lock(&s->lock);
        } else {
            strcpy(s->state, phase);
        }
    }
    strcpy(r->state, s->state);
    strcpy(r->stop_reason, !strcmp(r->status, "completed") ? "none" : "control_failure");
    r->logical = s->logical;
    r->committed = s->committed;
    r->pending = s->pending;
    r->memory_after = s->generator.info ? service_process_footprint() : 0;
    r->total_ms = milliseconds() - start;
    if (r->cancel_received_ms) {
        r->cancel_latency_ms = milliseconds() - r->cancel_received_ms;
    }
    s->job = 0;
}

static void *worker(void *context)
{
    service_session *s = context;
    pthread_mutex_lock(&s->lock);
    for (;;) {
        while (!s->job && !s->reset_job && !s->stop) {
            pthread_cond_wait(&s->wake, &s->lock);
        }
        if (s->stop && !s->job) {
            break;
        }
        if (s->reset_job) {
            pthread_mutex_unlock(&s->lock);
            s->generator.reset(s->generator.context);
            pthread_mutex_lock(&s->lock);
            s->reset_job = 0;
            pthread_cond_broadcast(&s->wake);
            continue;
        }
        service_request *r = &s->history[s->active];
        if (strcmp(r->operation, "generate")) {
            control_job(s, r);
            continue;
        }
        unsigned budget = s->budget;
        unsigned tool_phase = 0; /* 1 name, 2 arguments; partial calls require reset. */
        double started = milliseconds();
        r->memory_before = s->generator.info ? service_process_footprint() : 0;
        r->prompt = s->logical;
        r->framing_ms = s->framing_ms;
        s->framing_ms = 0;
        r->first_token_ms = -1;
        unsigned planned_reuse = s->committed;
        if (s->generator.begin) {
            pthread_mutex_unlock(&s->lock);
            s->generator.begin(s->generator.context);
            pthread_mutex_lock(&s->lock);
        }
        event_locked(s, "started", "", "none");
        const char *terminal = "completed", *reason = "budget";
        while (budget && !s->cancel && !s->stop) {
            /* No shared state can mutate ledger while generating. Generator calls
             * happen unlocked, so status/cancel remain responsive during a tick. */
            pthread_mutex_unlock(&s->lock);
            double tick = milliseconds();
            service_work receipt = {0};
            int valid = materialize(s, &receipt);
            /* Quantize once; aggregates are sums of these executed-step samples. */
            uint64_t elapsed_us = (uint64_t)((milliseconds() - tick) * 1000.0);
            int token = !valid ? -1 : r->replay_count ?
                s->generator.replay_next(s->generator.context, r->replay_tokens[r->generated]) :
                s->generator.next(s->generator.context);
            double available = milliseconds();
            pthread_mutex_lock(&s->lock);
            if (s->generator.work) {
                add_work(&r->work, &receipt);
                if (!r->generated) {
                    planned_reuse = receipt.reused;
                }
            }
            if ((!s->generator.work && !r->generated && !planned_reuse) ||
                receipt.prefill_calls) {
                r->prefill_steps++;
                r->prefill_us += elapsed_us;
                r->prefill_ms = r->prefill_us / 1000.0;
            } else if (receipt.append_calls) {
                r->suffix_steps++;
                r->suffix_us += elapsed_us;
            } else if (!s->generator.work || receipt.decode_calls) {
                r->decode_us[r->decode_steps++] = elapsed_us;
                r->decode_total_us += elapsed_us;
                r->decode_ms = r->decode_total_us / 1000.0;
            }
            if (token >= 0 && r->first_token_ms < 0) {
                r->first_token_ms = available - started;
            }
            if (valid) {
                if (!r->generated) {
                    r->reused = planned_reuse;
                    r->newly_processed = r->prompt - planned_reuse;
                }
                s->committed = s->logical;
                s->pending = 0;
            }
            if (s->cancel || s->stop) {
                break;
            }
            unsigned eos = s->generator.eos ? s->generator.eos : 256;
            if (token < 0 || (!s->generator.piece && token > 256) || s->logical == context_capacity(s)) {
                terminal = "error";
                reason = "generator_failure";
                break;
            }
            s->tokens[s->logical++] = (unsigned)token;
            s->pending = 1;
            r->token_ids[r->generated++] = (unsigned)token;
            budget--;
            if (s->generator.native_tools) {
                if ((unsigned)token == 9 && !s->assistant_started && r->generated == 1) {
                    s->assistant_started = 1;
                    tool_phase = 1;
                    continue;
                }
                s->assistant_started = 1;
                if (tool_phase == 1 && (unsigned)token == 32 &&
                    !s->utf8_length && ((s->generator.native_tools != 3 &&
                     !strcmp(r->text, "calculator")) ||
                     (s->generator.native_tools >= 2 && !strcmp(r->text, "lookup")))) {
                    r->tool_kind = !strcmp(r->text, "lookup") ? 2 : 1;
                    r->text[0] = 0;
                    tool_phase = 2;
                    continue;
                }
                if ((unsigned)token == eos && tool_phase) {
                    if (tool_phase == 2 && !s->utf8_length &&
                        (r->tool_kind == 2 ? lookup_arguments(r, s->generator.native_tools) :
                         calculator_arguments(r))) {
                        reason = "tool_call";
                    } else {
                        terminal = "error";
                        reason = "invalid_tool_call";
                    }
                    break;
                }
                /* Tekken ordinary vocabulary starts at 1000. Never decode specials. */
                if ((unsigned)token < 1000 && (unsigned)token != eos) {
                    terminal = "error";
                    reason = "unsupported_output_token";
                    break;
                }
            }
            if ((unsigned)token == eos) {
                if (s->utf8_length) {
                    terminal = "error";
                    reason = "incomplete_utf8";
                } else {
                    reason = "eos";
                }
                break;
            }
            unsigned char bytes[1027];
            size_t length = 1;
            memcpy(bytes, s->utf8, s->utf8_length);
            if (s->generator.piece) {
                length = 1024;
                if (!s->generator.piece(s->generator.context, (unsigned)token,
                                        bytes + s->utf8_length, &length) || length > 1024) {
                    terminal = "error";
                    reason = "unsupported_output_token";
                    break;
                }
            } else {
                bytes[s->utf8_length] = (unsigned char)token;
            }
            length += s->utf8_length;
            size_t complete = 0;
            int invalid = 0;
            while (complete < length) {
                size_t width;
                unsigned cp;
                if (tj_utf8(bytes + complete, length - complete, &width, &cp)) {
                    if (!cp) {
                        invalid = 1;
                        break;
                    }
                    complete += width;
                    continue;
                }
                unsigned first = bytes[complete];
                unsigned need = first >= 0xc2 && first <= 0xdf ? 2 :
                                first >= 0xe0 && first <= 0xef ? 3 :
                                first >= 0xf0 && first <= 0xf4 ? 4 : 0;
                size_t tail = length - complete;
                if (!need || tail >= need) {
                    invalid = 1;
                }
                for (size_t i = 1; i < tail; i++) {
                    if ((bytes[complete + i] & 0xc0) != 0x80) {
                        invalid = 1;
                    }
                }
                break;
            }
            if (invalid || length - complete > 3) {
                terminal = "error";
                reason = "invalid_utf8";
                break;
            }
            size_t old = strlen(r->text);
            if (complete > sizeof(r->text) - 1 - old) {
                terminal = "error";
                reason = "result_text_overflow";
                break;
            }
            s->utf8_length = (unsigned)(length - complete);
            memcpy(s->utf8, bytes + complete, s->utf8_length);
            if (complete) {
                char raw[1028], escaped[1027 * 6 + 1];
                memcpy(raw, bytes, complete);
                raw[complete] = 0;
                quote_text(raw, escaped);
                memcpy(r->text + old, raw, complete + 1);
                if (!tool_phase) {
                    event_locked(s, "text_delta", escaped, "none");
                }
            }
        }
        /* A terminal event also needs a slot. Even an exactly full queue must
         * cancel before discarding deltas; otherwise completed would lie about
         * successful delivery of the retained text. */
        if (s->count == SERVICE_EVENTS) {
            cancel_locked(s, "overflow");
        }
        if (s->cancel || s->stop) {
            terminal = "cancelled";
            reason = "cancelled";
        }
        if (tool_phase && !strcmp(reason, "budget")) {
            terminal = "error";
            reason = "truncated_tool_call";
        }
        if (!strcmp(terminal, "completed")) {
            if (!strcmp(reason, "tool_call")) {
                strcpy(s->state, "awaiting_tool");
                s->outstanding_tool = s->active + 1;
            } else {
                strcpy(s->state, !strcmp(reason, "eos") ? "ready" : "partial");
            }
        } else {
            strcpy(s->state, "needs_reset");
            /* Controller invalidity must invalidate the adapter checkpoint too. */
            s->checkpoint_id = s->checkpoint_length = 0;
            pthread_mutex_unlock(&s->lock);
            if (s->generator.invalidate) {
                s->generator.invalidate(s->generator.context);
            }
            pthread_mutex_lock(&s->lock);
        }
        if (tool_phase) {
            r->text[0] = 0;
        }
        r->total_ms = milliseconds() - started;
        if (r->cancel_received_ms) {
            r->cancel_latency_ms = milliseconds() - r->cancel_received_ms;
        }
        r->memory_after = s->generator.info ? service_process_footprint() : 0;
        strcpy(r->status, terminal);
        strcpy(r->stop_reason, reason);
        strcpy(r->state, s->state);
        r->logical = s->logical;
        r->committed = s->committed;
        r->pending = s->pending;
        /* Reserve terminal delivery even on overflow; discard buffered deltas.
         * A slow reader must inspect cancelled, never assume text is complete. */
        if (s->count == SERVICE_EVENTS) {
            s->head = 0;
            s->count = 0;
        }
        event_locked(s, terminal, "", reason);
        s->job = 0;
    }
    pthread_mutex_unlock(&s->lock);
    s->generator.reset(s->generator.context);
    s->generator.destroy(s->generator.context);
    return NULL;
}

int service_init(service_session *s, service_generator generator)
{
    if ((generator.replay_next && (generator.native_tools || !generator.next ||
         !generator.frame || !generator.piece || generator.eos != 2 ||
         !generator.materialize || !generator.reset || !generator.destroy)) ||
        generator.native_tools > 3 ||
        (generator.native_tools && (!generator.tool_frame || !generator.piece ||
         !generator.frame || generator.eos != 2)) ||
        (generator.context_capacity && generator.context_capacity != 256 &&
         generator.context_capacity != SERVICE_CONTEXT_MAX) ||
        (generator.block_capacity && generator.block_capacity != 256 &&
         !(generator.context_capacity == SERVICE_CONTEXT_MAX && generator.block_capacity == 128))) {
        return 0;
    }
    memset(s, 0, sizeof(*s));
    s->generator = generator;
    s->startup_footprint = generator.info ? service_process_footprint() : 0;
    s->startup_sample_ms = milliseconds();
    strcpy(s->state, "closed");
    if (pthread_mutex_init(&s->lock, NULL)) {
        return 0;
    }
    if (pthread_cond_init(&s->wake, NULL)) {
        pthread_mutex_destroy(&s->lock);
        return 0;
    }
    if (pthread_create(&s->worker, NULL, worker, s)) {
        pthread_cond_destroy(&s->wake);
        pthread_mutex_destroy(&s->lock);
        return 0;
    }
    return 1;
}

void service_destroy(service_session *s)
{
    pthread_mutex_lock(&s->lock);
    s->stop = 1;
    cancel_locked(s, "shutdown");
    pthread_cond_signal(&s->wake);
    pthread_mutex_unlock(&s->lock);
    pthread_join(s->worker, NULL);
    pthread_cond_destroy(&s->wake);
    pthread_mutex_destroy(&s->lock);
}

void service_disconnect_request(service_session *s, const char *id)
{
    pthread_mutex_lock(&s->lock);
    if (s->job && !strcmp(s->history[s->active].operation, "generate") &&
        !strcmp(s->history[s->active].id, id)) {
        cancel_locked(s, "disconnect");
    }
    pthread_mutex_unlock(&s->lock);
}

void service_disconnect(service_session *s)
{
    pthread_mutex_lock(&s->lock);
    if (s->job && !strcmp(s->history[s->active].operation, "generate")) {
        cancel_locked(s, "disconnect");
    }
    pthread_mutex_unlock(&s->lock);
}

int service_event(service_session *s, char *event)
{
    pthread_mutex_lock(&s->lock);
    int result = s->count != 0;
    if (result) {
        strcpy(event, s->events[s->head]);
        s->head = (s->head + 1) % SERVICE_EVENTS;
        s->count--;
    }
    pthread_mutex_unlock(&s->lock);
    return result;
}

static void reset_locked(service_session *s)
{
    s->reset_job = 1;
    pthread_cond_signal(&s->wake);
    while (s->reset_job) {
        pthread_cond_wait(&s->wake, &s->lock);
    }
    s->checkpoint_id = s->checkpoint_length = 0;
    s->outstanding_tool = s->assistant_started = 0;
    s->framing_ms = 0;
    s->logical = s->committed = s->pending = s->utf8_length = 0;
    strcpy(s->state, "ready");
}

/* Caller holds lock: route errors and transport errors share one snapshot. */
static void error_locked(service_session *s, const char *error, const char *request_id,
                         char *reply, size_t size)
{
    int valid = s->session && strcmp(s->state, "closed") && strcmp(s->state, "needs_reset");
    const char *recovery = "correct_request";
    if (strstr(error, "capacity_restart_required")) {
        recovery = "restart_service";
    } else if (!strcmp(error, "duplicate_id_existing_request")) {
        recovery = "inspect_existing_request";
    } else if (!strcmp(s->state, "needs_reset")) {
        recovery = "reset";
    } else if (!strcmp(error, "busy_stream") || !strcmp(error, "busy") ||
               (!strcmp(error, "busy_or_invalid_state") && s->job)) {
        recovery = "wait_for_request_completion";
    } else if (!strcmp(s->state, "closed")) {
        recovery = "new_session";
    }
    snprintf(reply, size, "{\"error\":\"%s\",\"request_id\":\"%s\",\"session_id\":%u,"
             "\"session_valid\":%s,\"state\":\"%s\",\"recovery\":\"%s\"}",
             error, request_id, s->session, valid ? "true" : "false", s->state, recovery);
}

void service_transport_error(service_session *s, const char *error, char *reply, size_t size)
{
    pthread_mutex_lock(&s->lock);
    error_locked(s, error, "", reply, size);
    pthread_mutex_unlock(&s->lock);
}

int service_route(service_session *s, const char *method, const char *path,
                  const void *body, size_t length, char *reply, size_t size, int *stream)
{
    int status = 200;
    const char *error = NULL;
    char error_request_id[65] = "", admission_id[65] = "";
    tj_document d = {0};
    *stream = 0;
    pthread_mutex_lock(&s->lock);
    if (!strcmp(method, "GET") && length) {
        status = 400;
        error = "unexpected_body";
        goto done;
    }
    if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        if (s->generator.info) {
            snprintf(reply, size, "{\"model_loaded\":true,\"runtime_ready\":true}");
            goto done;
        }
        snprintf(reply, size, "{\"operational_test_service\":true,\"model_loaded\":false,\"runtime_ready\":false}");
        goto done;
    }
    if (!strcmp(method, "GET") && !strcmp(path, "/info")) {
        if (s->generator.info) {
            size_t n = 0;
            char sample[32] = "null";
            if (s->startup_footprint) {
                snprintf(sample, sizeof(sample), "%llu", (unsigned long long)s->startup_footprint);
            }
            if (!append(reply, size, &n, "%.*s,\"process_memory_sample\":{"
                        "\"method\":\"task_info_TASK_VM_INFO_phys_footprint\","
                        "\"units\":\"bytes\",\"phase\":\"after_resident_load\","
                        "\"monotonic_ms\":%.3f,\"bytes\":%s}}",
                        (int)strlen(s->generator.info) - 1, s->generator.info,
                        s->startup_sample_ms, sample)) {
                status = 500;
                error = "reply_capacity";
            }
            goto done;
        }
        snprintf(reply, size, "{\"generator\":\"diagnostic_test_generator\","
                 "\"conversation_protocol\":\"test-ledger-v1\",\"tokenizer\":\"raw-synthetic-bytes-v1\","
                 "\"event_bytes\":8192,\"model\":null,\"model_loaded\":false,\"runtime_ready\":false,\"context_capacity\":256,"
                 "\"execution_strategy\":\"synthetic_fixed_raw_pieces_v1\",\"precision_policy\":null,"
                 "\"prefill_chunk_size\":null,\"fixture_text\":\"A€😀Z\",\"fixture_eos_id\":256,"
                 "\"tools\":false,\"checkpoints\":false,\"compact\":false,\"context4k\":false,"
                 "\"numerical_validation\":\"not_applicable_test_generator\","
                 "\"loading_duration_ms\":0,\"memory_bytes\":null,\"test_only\":true,\"performance_evidence\":false,"
                 "\"build_commit\":\"%s\",\"build_dirty\":%s}", SERVICE_COMMIT, SERVICE_DIRTY);
        goto done;
    }
    if (!strcmp(method, "POST") && !strcmp(path, "/sessions")) {
        if (!service_json(body, length, "", &d)) {
            status = 400;
            error = "invalid_json";
            goto done;
        }
        if (strcmp(s->state, "closed")) {
            status = 409;
            error = "session_active";
            goto done;
        }
        if (s->session == UINT_MAX) {
            status = 503;
            error = "session_id_capacity_restart_required";
            goto done;
        }
        reset_locked(s);
        s->session++;
        goto snapshot;
    }
    if (!strncmp(path, "/requests/", 10)) {
        char id[65];
        const char *tail = strchr(path + 10, '/');
        size_t n = tail ? (size_t)(tail - path - 10) : strlen(path + 10);
        if (!n || n > 64) {
            status = 404;
            error = "request_not_found";
            goto done;
        }
        memcpy(id, path + 10, n);
        id[n] = 0;
        unsigned r;
        for (r = 0; r < s->requests && strcmp(id, s->history[r].id); r++) {}
        if (r == s->requests) {
            status = 404;
            error = "request_not_found";
            goto done;
        }
        strcpy(error_request_id, s->history[r].id);
        if (tail && !strcmp(tail, "/cancel") && !strcmp(method, "POST")) {
            if (!service_json(body, length, "", &d)) {
                status = 400;
                error = "invalid_json";
                goto done;
            }
            if (s->job && r == s->active) {
                cancel_locked(s, "explicit");
            }
        } else if (tail || strcmp(method, "GET")) {
            status = 501;
            error = "unsupported_operation";
            goto done;
        }
        char text[4096 * 6 + 1];
        service_request *request = &s->history[r];
        /* Active counters and state share this mutex; history retains its terminal snapshot. */
        service_request snapshot = *request;
        if (s->job && r == s->active) {
            strcpy(snapshot.state, s->state);
            snapshot.logical = s->logical;
            snapshot.committed = s->committed;
            snapshot.pending = s->pending;
        }
        request = &snapshot;
        int buffered_call = s->generator.native_tools && request->generated &&
            request->token_ids[0] == 9;
        quote_text(buffered_call ? "" : request->text, text);
        if (strlen(text) + 1000 > size) {
            status = 500;
            error = "reply_capacity";
            goto done;
        }
        size_t offset = 0;
        if (!append(reply, size, &offset, "{\"id\":\"%s\",\"status\":\"%s\",\"text\":\"%s\","
                 "\"generated_tokens\":%u,\"cancel_requested\":%s,\"stop_reason\":\"%s\","
                 "\"state\":\"%s\",\"logical_tokens\":%u,\"committed_tokens\":%u,\"pending_tokens\":%u}",
                 id, request->status, text, request->generated,
                 request->cancel_received_ms ? "true" : "false", request->stop_reason,
                 request->state, request->logical, request->committed, request->pending)) {
            status = 500;
            error = "reply_capacity";
            goto done;
        }
        char measured[SERVICE_METRIC_BYTES];
        offset--;
        if (!metrics(request, s->generator.info != NULL, measured, sizeof(measured)) ||
            !append(reply, size, &offset,
                ",\"operation\":\"%s\",\"checkpoint_id\":\"%llu\",\"cache_restored\":%s,"
                "\"session_id\":%u,\"before_state\":\"%s\","
                "\"before_logical_tokens\":%u,\"before_committed_tokens\":%u,"
                "\"before_pending_tokens\":%u,\"session_valid\":%s,\"recovery\":\"%s\","
                "\"metrics\":%s,\"token_ids\":[",
                request->operation, (unsigned long long)request->checkpoint_id,
                request->cache_restored ? "true" : "false",
                request->session_id, request->before_state, request->before_logical,
                request->before_committed, request->before_pending,
                !strcmp(request->state, "needs_reset") ? "false" : "true",
                !strcmp(request->state, "needs_reset") ? "reset" : "none", measured)) {
            status = 500;
            error = "reply_capacity";
            goto done;
        }
        for (unsigned i = 0; i < request->generated; i++) {
            if (!append(reply, size, &offset, "%s%u", i ? "," : "", request->token_ids[i])) {
                status = 500;
                error = "reply_capacity";
                goto done;
            }
        }
        char tool[256], replay[256];
        tool_metadata(request, tool, sizeof(tool));
        replay_metadata(request, replay, sizeof(replay));
        if (!append(reply, size, &offset, "]%s%s}", tool, replay)) {
            status = 500;
            error = "reply_capacity";
        }
        goto done;
    }
    unsigned session = 0;
    size_t used = 10;
    if (strncmp(path, "/sessions/", 10) || path[10] < '1' || path[10] > '9') {
        status = 501;
        error = "unsupported_operation";
        goto done;
    }
    while (path[used] >= '0' && path[used] <= '9') {
        unsigned digit = (unsigned)(path[used++] - '0');
        if (session > (UINT_MAX - digit) / 10) {
            status = 400;
            error = "invalid_session_id";
            goto done;
        }
        session = session * 10 + digit;
    }
    if (path[used] && path[used] != '/') {
        status = 501;
        error = "unsupported_operation";
        goto done;
    }
    if (!session || session > s->session) {
        status = 404;
        error = "session_not_found";
        goto done;
    }
    if (session != s->session || !strcmp(s->state, "closed")) {
        status = 410;
        error = "session_closed";
        goto done;
    }
    const char *op = path + used;
    if (!*op && !strcmp(method, "GET")) {
        goto snapshot;
    }
    if ((strcmp(method, "POST") && strcmp(method, "DELETE")) ||
        (!strcmp(method, "DELETE") && *op) || (!strcmp(method, "POST") && !*op)) {
        status = 501;
        error = "unsupported_operation";
        goto done;
    }
    if (*op && strcmp(op, "/generate") && strcmp(op, "/messages") && strcmp(op, "/reset") &&
        strcmp(op, "/checkpoint") && strcmp(op, "/restore")) {
        status = 501;
        error = "unsupported_operation";
        goto done;
    }
    const char *fields = !strcmp(op, "/generate") ? (s->generator.replay_next ?
                             "|id||budget||replay_tokens|" : "|id||budget|") :
                         !strcmp(op, "/messages") ? (s->generator.native_tools ?
                             "|role||content||tool_call_id|" : "|role||content|") :
                         !strcmp(op, "/checkpoint") ? "|id|" :
                         !strcmp(op, "/restore") ? "|id||checkpoint_id|" : "";
    if (!service_json_profile(body, length, fields, &d, context_capacity(s) > 256)) {
        status = 400;
        error = "invalid_json_or_fields";
        goto done;
    }
    int control = !strcmp(op, "/checkpoint") || !strcmp(op, "/restore");
    if (control && (s->generator.native_tools || !s->generator.prefix_reuse ||
                    !s->generator.save || !s->generator.restore)) {
        status = 501;
        error = "unsupported_operation";
        goto done;
    }
    if (!strcmp(op, "/generate") || control) {
        unsigned id = tj_field(&d, 1, "id"), budget;
        unsigned b = tj_field(&d, 1, "budget");
        budget = 0;
        if (!id || d.nodes[id].kind != '"' || !d.nodes[id].length ||
            d.nodes[id].length > 64 || (!control &&
            (!tj_integer(&d, b, &budget) || !budget || budget > 256))) {
            status = 400;
            error = "invalid_generation";
            goto done;
        }
        unsigned replay_tokens[SERVICE_REPLAY_TOKENS], replay_count = 0;
        if (!control && s->generator.replay_next) {
            if (!replay_sequence(&d, tj_field(&d, 1, "replay_tokens"), budget, replay_tokens)) {
                status = 400;
                error = "invalid_replay_tokens";
                goto done;
            }
            replay_count = budget;
        }
        char request_id[65];
        for (size_t i = 0; i < d.nodes[id].length; i++) {
            unsigned c = d.nodes[id].text[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '-' || c == '_')) {
                status = 400;
                error = "invalid_id";
                goto done;
            }
            request_id[i] = (char)c;
        }
        request_id[d.nodes[id].length] = 0;
        strcpy(error_request_id, request_id);
        for (unsigned i = 0; i < s->requests; i++) {
            if (!strcmp(request_id, s->history[i].id)) {
                status = 409;
                error = "duplicate_id_existing_request";
                goto done;
            }
        }
        if (replay_count > SERVICE_REPLAY_TOKENS - s->replay_reserved) {
            status = 503;
            error = "replay_capture_capacity_restart_required";
            goto done;
        }
        if (s->job || (strcmp(s->state, "ready") && strcmp(s->state, "partial"))) {
            status = 409;
            error = "busy_or_invalid_state";
            goto done;
        }
        if (!control && s->generator.frame && (!s->logical ||
            (!strcmp(s->state, "ready") && s->tokens[s->logical - 1] == s->generator.eos))) {
            status = 409;
            error = "user_input_required";
            goto done;
        }
        if (control) {
            if (!strcmp(op, "/checkpoint")) {
                if (!s->logical || s->utf8_length) {
                    status = 409;
                    error = "checkpoint_boundary_required";
                    goto done;
                }
                if (s->checkpoint_serial == UINT64_MAX) {
                    status = 503;
                    error = "checkpoint_id_capacity_restart_required";
                    goto done;
                }
            } else {
                unsigned node = tj_field(&d, 1, "checkpoint_id");
                uint64_t value = 0;
                if (!node || d.nodes[node].kind != '"' || !d.nodes[node].length ||
                    d.nodes[node].length > 20) {
                    status = 400;
                    error = "invalid_checkpoint_id";
                    goto done;
                }
                for (size_t i = 0; i < d.nodes[node].length; i++) {
                    unsigned digit = (unsigned)d.nodes[node].text[i] - '0';
                    if (digit > 9 || value > (UINT64_MAX - digit) / 10) {
                        status = 400;
                        error = "invalid_checkpoint_id";
                        goto done;
                    }
                    value = value * 10 + digit;
                }
                if (!value || value != s->checkpoint_id) {
                    status = 409;
                    error = "stale_checkpoint_id";
                    goto done;
                }
            }
        }
        if (budget > context_capacity(s) - s->logical) {
            status = 422;
            error = "context_overflow";
            goto done;
        }
        if (s->requests == SERVICE_IDS) {
            status = 503;
            error = "request_id_capacity_restart_required";
            goto done;
        }
        /* Reserve the bounded control envelope before admitting a job. */
        if ((control || s->generator.frame) &&
            size < 512 + (s->generator.frame ? 7 * (size_t)s->logical : 0)) {
            status = 500;
            error = "reply_capacity";
            goto done;
        }
        s->active = s->requests++;
        service_request *admitted = &s->history[s->active];
        admitted->replay_count = replay_count;
        memcpy(admitted->replay_tokens, replay_tokens, replay_count * sizeof(unsigned));
        s->replay_reserved += replay_count;
        strcpy(admitted->operation, control ? op + 1 : "generate");
        if (!strcmp(op, "/restore")) {
            admitted->checkpoint_id = s->checkpoint_id;
        }
        admitted->session_id = s->session;
        strcpy(admitted->before_state, s->state);
        admitted->before_logical = s->logical;
        admitted->before_committed = s->committed;
        admitted->before_pending = s->pending;
        admitted->first_token_ms = -1;
        strcpy(s->history[s->active].id, request_id);
        strcpy(s->history[s->active].status, control ? "running" : "generating");
        strcpy(s->state, control ? "control" : "generating");
        s->budget = budget;
        s->cancel = 0;
        if (!control) {
            s->head = s->count = 0;
        }
        s->job = 1;
        *stream = !control;
        status = control ? 202 : 200;
        if (control) {
            strcpy(admission_id, request_id);
        }
        pthread_cond_signal(&s->wake);
        goto snapshot;
    }
    if (s->job) {
        status = 409;
        error = "busy";
        goto done;
    }
    if (!strcmp(op, "/reset") || (!*op && !strcmp(method, "DELETE"))) {
        reset_locked(s);
        if (!*op) {
            strcpy(s->state, "closed");
        }
        goto snapshot;
    }
    if (!strcmp(op, "/messages")) {
        unsigned role = tj_field(&d, 1, "role"), content = tj_field(&d, 1, "content");
        if (!role || !content || d.nodes[role].kind != '"' || d.nodes[content].kind != '"') {
            status = 400;
            error = "invalid_message";
            goto done;
        }
        unsigned call_id = tj_field(&d, 1, "tool_call_id");
        if (s->generator.native_tools && equals(&d.nodes[role], "tool")) {
            if (strcmp(s->state, "awaiting_tool") || !s->outstanding_tool) {
                status = 409;
                error = "no_outstanding_tool_call";
                goto done;
            }
            service_request *call = &s->history[s->outstanding_tool - 1];
            if (!call_id || d.nodes[call_id].kind != '"' ||
                !equals(&d.nodes[call_id], call->id)) {
                status = 409;
                error = "tool_call_id_mismatch";
                goto done;
            }
            int64_t result;
            if (!(call->tool_kind == 2 ? lookup_result(call, &d.nodes[content]) :
                    bounded_integer(d.nodes[content].text, d.nodes[content].length,
                                    1000000000000ULL, 1, &result))) {
                status = 400;
                error = "invalid_tool_result";
                goto done;
            }
            unsigned framed[SERVICE_CONTEXT_MAX], count = context_capacity(s) - s->logical;
            double tick = milliseconds();
            if (!s->generator.tool_frame(s->generator.context, d.nodes[content].text,
                    d.nodes[content].length, framed, &count)) {
                status = 422;
                error = "framing_or_context_overflow";
                goto done;
            }
            if (size < 512 + 7 * (size_t)(s->logical + count)) {
                status = 500;
                error = "reply_capacity";
                goto done;
            }
            memcpy(s->tokens + s->logical, framed, count * sizeof(*framed));
            s->logical += count;
            s->framing_ms += milliseconds() - tick;
            s->outstanding_tool = s->assistant_started = 0;
            strcpy(s->state, "ready");
            goto snapshot;
        }
        if (call_id) {
            status = 400;
            error = "unexpected_tool_call_id";
            goto done;
        }
        if (!equals(&d.nodes[role], "user")) {
            status = s->generator.frame || equals(&d.nodes[role], "tool_result") ? 501 : 400;
            error = "unsupported_role"; goto done;
        }
        if (strcmp(s->state, "ready")) {
            status = 409;
            error = "invalid_state";
            goto done;
        }
        if (s->generator.frame) {
            if (s->logical && s->tokens[s->logical - 1] != s->generator.eos) {
                status = 409;
                error = "assistant_generation_required";
                goto done;
            }
            unsigned framed[SERVICE_CONTEXT_MAX], count = context_capacity(s) - s->logical;
            double frame_start = milliseconds();
            if (!s->generator.frame(s->generator.context, d.nodes[content].text,
                                   d.nodes[content].length, !s->logical, framed, &count)) {
                status = 422;
                error = "framing_or_context_overflow";
                goto done;
            }
            /* IDs are at most six decimal digits; reserve the complete ledger
             * plus the bounded snapshot envelope before changing the session. */
            if (size < 512 + 7 * (size_t)(s->logical + count)) {
                status = 500;
                error = "reply_capacity";
                goto done;
            }
            memcpy(s->tokens + s->logical, framed, count * sizeof(*framed));
            s->logical += count;
            s->assistant_started = 0;
            s->framing_ms = milliseconds() - frame_start;
            goto snapshot;
        }
        if (d.nodes[content].length > context_capacity(s) - s->logical) {
            status = 422;
            error = "context_overflow";
            goto done;
        }
        for (size_t i = 0; i < d.nodes[content].length; i++) {
            s->tokens[s->logical++] = d.nodes[content].text[i];
        }
        goto snapshot;
    }
    status = 501; error = "unsupported_operation";
    goto done;
snapshot: {
    size_t offset = 0;
    if (!append(reply, size, &offset,
                "{\"session_id\":%u,\"state\":\"%s\",\"logical_tokens\":%u,"
                "\"committed_tokens\":%u,\"pending_tokens\":%u,\"remaining\":%u}",
                s->session, s->state, s->logical, s->committed,
                s->pending, context_capacity(s) - s->logical)) {
        status = 500;
        error = "reply_capacity";
        goto done;
    }
    if (admission_id[0]) {
        offset--;
        if (!append(reply, size, &offset, ",\"id\":\"%s\",\"operation\":\"%s\","
                    "\"status\":\"running\"}", admission_id, s->history[s->active].operation)) {
            status = 500;
            error = "reply_capacity";
            goto done;
        }
    }
}
    if (s->generator.frame && !error && reply[0]) {
        size_t offset = strlen(reply) - 1;
        int fits = append(reply, size, &offset, ",\"token_ids\":[");
        for (unsigned i = 0; fits && i < s->logical; i++) {
            fits = append(reply, size, &offset, "%s%u", i ? "," : "", s->tokens[i]);
        }
        if (!fits || !append(reply, size, &offset, "]}")) {
            status = 500;
            error = "reply_capacity";
        }
    }
done:
    if (error) {
        error_locked(s, error, error_request_id, reply, size);
    }
    tj_free(&d);
    pthread_mutex_unlock(&s->lock);
    return status;
}
