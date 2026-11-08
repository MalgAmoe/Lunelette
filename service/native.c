#define _POSIX_C_SOURCE 200809L
#include "native.h"
#include "framing.h"
#include "metal.h"
#include "tokenizer.h"
#include "generation.h"
#include <CommonCrypto/CommonDigest.h>
#include <stdio.h>
#include <float.h>
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    lunelette_metal *runtime;
    lunelette_tokenizer *tokenizer;
    unsigned prefix[SERVICE_CONTEXT_MAX], length, emitted, loads;
    unsigned capacity, block, target, pending_position, recompute_length, recompute_hint;
    int ingesting, rebuilding;
    int pending, boundary;
    unsigned tools;
    unsigned policy, saved_length, saved_prefix[SERVICE_CONTEXT_MAX];
    lunelette_metal_checkpoint *checkpoint;
    service_work work;
    const float *logits;
    FILE *replay_file;
    unsigned replay_index;
    int replay_failed;
    char info[4096];
} native_service;

static double seconds(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec / 1e9;
}

static int hash_file(const char *path, char hex[65])
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        return 0;
    }
    CC_SHA256_CTX hash;
    CC_SHA256_Init(&hash);
    unsigned char buffer[65536], digest[CC_SHA256_DIGEST_LENGTH];
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), file))) {
        CC_SHA256_Update(&hash, buffer, (CC_LONG)count);
    }
    int valid = !ferror(file);
    fclose(file);
    CC_SHA256_Final(digest, &hash);
    for (unsigned i = 0; i < sizeof(digest); i++) {
        snprintf(hex + i * 2, 3, "%02x", digest[i]);
    }
    return valid;
}

static void native_invalidate(void *context)
{
    native_service *n = context;
    lunelette_metal_checkpoint_destroy(n->checkpoint);
    n->checkpoint = NULL;
    n->saved_length = 0;
    n->recompute_hint = 0;
    n->ingesting = 0;
    n->logits = NULL;
    n->pending = 0;
    n->emitted = 0;
}

static void native_begin(void *context)
{
    native_service *n = context;
    n->boundary = 1;
}

static void native_work(void *context, service_work *work)
{
    native_service *n = context;
    *work = n->work;
    memset(&n->work, 0, sizeof(n->work));
}

static void native_reset(void *context)
{
    native_service *n = context;
    char error[512];
    native_invalidate(n);
    n->logits = NULL;
    n->length = 0;
    n->boundary = 0;
    n->ingesting = 0;
    n->emitted = 0;
    n->pending = 0;
    if (lunelette_metal_reset(n->runtime, error, sizeof(error))) {
        fprintf(stderr, "reset: %s\n", error);
    }
}

static void native_destroy(void *context)
{
    native_service *n = context;
    native_invalidate(n);
    lunelette_metal_destroy(n->runtime);
    lunelette_tokenizer_close(n->tokenizer);
    if (n->replay_file) {
        fclose(n->replay_file);
    }
    free(n);
}

static int native_frame(void *context, const unsigned char *text, size_t length,
                        int initial, unsigned *output, unsigned *capacity)
{
    native_service *n = context;
    if (n->tools) {
        return (n->tools == 3 ? service_frame_observation :
            n->tools == 2 ? service_frame_investigation : service_frame_tools)(
            n->tokenizer, text, length, n->capacity > 256 ? 32768 : 4096, initial, 0, output, capacity);
    }
    return service_frame_with_limit(n->tokenizer, text, length,
        n->capacity > 256 ? 32768 : 4096, initial, output, capacity);
}

static int native_tool_frame(void *context, const unsigned char *text, size_t length,
                             unsigned *output, unsigned *capacity)
{
    native_service *n = context;
    return (n->tools == 3 ? service_frame_observation :
            n->tools == 2 ? service_frame_investigation : service_frame_tools)(
        n->tokenizer, text, length, 4096, 0, 1, output, capacity);
}

static int native_piece(void *context, unsigned id, unsigned char *output, size_t *capacity)
{
    native_service *n = context;
    unsigned char *bytes = NULL;
    size_t length;
    char error[512];
    uint32_t token = id;
    if (!lunelette_tokenizer_decode(n->tokenizer, &token, 1, &bytes, &length,
                                    error, sizeof(error))) {
        return 0;
    }
    int valid = length <= *capacity;
    if (valid) {
        memcpy(output, bytes, length);
        *capacity = length;
    }
    free(bytes);
    return valid;
}

static int native_materialize(void *context, const unsigned *tokens, unsigned length)
{
    native_service *n = context;
    char error[512];
    memset(&n->work, 0, sizeof(n->work));
    if (!length || length > n->capacity || length < n->length ||
        memcmp(n->prefix, tokens, n->length * sizeof(*tokens))) {
        return 0;
    }
    if (!n->ingesting && n->pending && (length <= n->length || tokens[n->length] != n->emitted)) {
        return 0;
    }
    if (!n->ingesting && length == n->length && !(n->policy == 2 && n->boundary)) {
        n->boundary = 0;
        n->work.reused = length;
        return n->logits != NULL;
    }
    int status;
    unsigned old_length = n->length;
    if (!n->ingesting) {
        n->rebuilding = !old_length || (n->policy == 2 && n->boundary) ||
            (!n->policy && !(n->pending && length == old_length + 1 && n->emitted != 2));
        n->target = length;
        n->pending_position = old_length;
        n->recompute_length = n->rebuilding ? old_length : 0;
        if (n->recompute_hint > n->recompute_length) {
            n->recompute_length = n->recompute_hint;
        }
        n->recompute_hint = 0;
        n->boundary = 0;
        n->work.reused = n->rebuilding ? 0 : old_length;
        if (n->rebuilding) {
            n->logits = NULL;
            if (lunelette_metal_reset(n->runtime, error, sizeof(error))) {
                return 0;
            }
            n->length = 0;
        }
        n->ingesting = 1;
    } else if (n->target != length) {
        return 0;
    }
    unsigned count = length - n->length;
    if (count > n->block) {
        count = n->block;
    }
    int64_t ids[256];
    for (unsigned i = 0; i < count; i++) {
        ids[i] = tokens[n->length + i];
    }
    n->work.pending_attempted = n->pending && n->pending_position >= n->length &&
        n->pending_position < n->length + count;
    if (!n->length) {
        n->work.full_attempted = count;
        n->work.prefill_calls = 1;
        status = lunelette_metal_prefill(n->runtime, ids, count, &n->logits,
                                         error, sizeof(error));
    } else if (!n->rebuilding && n->pending && count == 1 && length == old_length + 1) {
        n->work.decode_calls = 1;
        n->work.decode_attempted = 1;
        status = lunelette_metal_decode(n->runtime, n->emitted, &n->logits,
                                        error, sizeof(error));
    } else {
        n->work.suffix_attempted = count;
        n->work.append_calls = 1;
        status = lunelette_metal_append(n->runtime, ids, count, &n->logits,
                                        error, sizeof(error));
    }
    if (n->length < n->recompute_length) {
        unsigned repeated = n->recompute_length - n->length;
        n->work.recompute_attempted = repeated < count ? repeated : count;
    }
    if (status) {
        fprintf(stderr, "materialize: %s\n", error);
        n->ingesting = 0;
        n->logits = NULL;
        return 0;
    }
    n->work.recompute_committed = n->work.recompute_attempted;
    n->work.full_committed = n->work.full_attempted;
    n->work.suffix_committed = n->work.suffix_attempted;
    n->work.pending_committed = n->work.pending_attempted;
    n->work.decode_committed = n->work.decode_attempted;
    n->length += count;
    memcpy(n->prefix, tokens, n->length * sizeof(*tokens));
    if (n->work.pending_committed) {
        n->pending = 0;
    }
    if (n->length != length) {
        return SERVICE_MATERIALIZE_MORE;
    }
    n->ingesting = 0;
    n->pending = 0;
    return 1;
}

static int native_materialize_step(void *context, const unsigned *tokens,
                                   unsigned length, unsigned *cursor)
{
    native_service *n = context;
    int result = native_materialize(context, tokens, length);
    *cursor = n->length;
    return result;
}

static int native_save(void *context)
{
    native_service *n = context;
    char error[512];
    lunelette_metal_checkpoint *replacement = NULL;
    if (n->tools) {
        return 0;
    }
    if (!n->logits || !n->length || n->pending) {
        return -1;
    }
    if (lunelette_metal_checkpoint_create(n->runtime, &replacement, error, sizeof(error))) {
        /* Nonfinite detection invalidates the runtime; allocation/epoch
         * rejection instead preserves its endpoint and previous handle. */
        for (unsigned i = 0; i < LUNELETTE_VOCAB_SIZE; i++) {
            if (!isfinite(n->logits[i])) {
                return -1;
            }
        }
        return 0;
    }
    lunelette_metal_checkpoint_destroy(n->checkpoint);
    n->checkpoint = replacement;
    n->saved_length = n->length;
    memcpy(n->saved_prefix, n->prefix, n->length * sizeof(*n->prefix));
    return 1;
}

static int native_restore(void *context)
{
    native_service *n = context;
    char error[512];
    memset(&n->work, 0, sizeof(n->work));
    if (n->tools || !n->saved_length) {
        return 0;
    }
    if (n->policy == 2) {
        n->logits = NULL;
        n->pending = 0;
        n->emitted = 0;
        n->boundary = 0;
        n->ingesting = 0;
        memset(n->prefix, 0, sizeof(n->prefix));
        n->length = 0;
        n->recompute_hint = n->saved_length;
        return lunelette_metal_reset(n->runtime, error, sizeof(error)) ? -1 : 1;
    }
    const float *restored = NULL;
    if (lunelette_metal_checkpoint_restore(n->runtime, n->checkpoint, &restored,
                                           error, sizeof(error))) {
        return 0;
    }
    n->logits = restored;
    memcpy(n->prefix, n->saved_prefix, n->saved_length * sizeof(*n->prefix));
    n->length = n->saved_length;
    n->pending = 0;
    n->emitted = 0;
    n->boundary = 0;
    n->ingesting = 0;
    n->work.logit_restores = 1;
    return 1;
}

static int write_u32(FILE *file, uint32_t value)
{
    unsigned char bytes[4] = {value & 255, (value >> 8) & 255,
                              (value >> 16) & 255, (value >> 24) & 255};
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static int capture(native_service *n, unsigned forced, unsigned greedy)
{
    FILE *file = n->replay_file;
    if (n->replay_failed || n->replay_index >= SERVICE_REPLAY_TOKENS ||
        !write_u32(file, n->replay_index) || !write_u32(file, n->length) ||
        !write_u32(file, forced) || !write_u32(file, greedy)) {
        n->replay_failed = 1;
        return 0;
    }
    for (unsigned i = 0; i < n->length; i++) {
        if (!write_u32(file, n->prefix[i])) {
            n->replay_failed = 1;
            return 0;
        }
    }
    for (unsigned i = 0; i < LUNELETTE_VOCAB_SIZE; i++) {
        uint32_t bits;
        memcpy(&bits, &n->logits[i], sizeof(bits));
        if (!isfinite(n->logits[i]) || !write_u32(file, bits)) {
            n->replay_failed = 1;
            return 0;
        }
    }
    if (fflush(file)) {
        n->replay_failed = 1;
        return 0;
    }
    n->replay_index++;
    return 1;
}

static int native_select(void *context, int replay, unsigned forced)
{
    native_service *n = context;
    /* Selection and capture run under the session worker's exclusive ownership.
     * Runtime logits are borrowed: no reset/append/restore may intervene here.
     * Failure returns -1 without publishing emitted/pending; capture may leave a
     * partial record, but latches replay_failed and never advances its index. */
    if (n->pending || (replay && (!n->replay_file || n->replay_failed ||
        n->replay_index >= SERVICE_REPLAY_TOKENS ||
        (!(forced >= 1000 && forced < LUNELETTE_VOCAB_SIZE) && forced != 2)))) {
        return -1;
    }
    /* Bound length before reading even the first prefix ID. Full context has no
     * next-token capacity, but still validate its IDs just as generation does. */
    if ((n->capacity != LUNELETTE_CONTEXT_LIMIT &&
         n->capacity != LUNELETTE_LONG_CONTEXT_LIMIT) ||
        !n->length || n->length > n->capacity) {
        return -1;
    }
    for (unsigned i = 0; i < n->length; i++) {
        if (n->prefix[i] >= LUNELETTE_VOCAB_SIZE) {
            return -1;
        }
    }
    if (n->length == n->capacity) {
        return -1;
    }
    int64_t token;
    char error[512];
    if (lunelette_select_greedy(n->logits, &token, error, sizeof(error))) {
        fprintf(stderr, "generate: %s\n", error);
        return -1;
    }
    if (replay && !capture(n, forced, (unsigned)token)) {
        return -1;
    }
    n->emitted = replay ? forced : (unsigned)token;
    n->pending = 1;
    return (int)n->emitted;
}

static int native_next(void *context)
{
    return native_select(context, 0, 0);
}

static int native_replay_next(void *context, unsigned token)
{
    return native_select(context, 1, token);
}

/* Encode explicitly LE; reject non-IEEE host representations before creating evidence. */
static int replay_float_format(void)
{
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) {
        return 0;
    }
    float one = 1.0f, minus = -2.5f;
    uint32_t a, b;
    memcpy(&a, &one, 4);
    memcpy(&b, &minus, 4);
    return a == UINT32_C(0x3f800000) && b == UINT32_C(0xc0200000);
}

int service_native_open(int argc, char **argv, service_generator *generator, unsigned *port)
{
    const char *model = NULL, *tokenizer = NULL, *prefill = NULL, *strategy_name = NULL;
    const char *evaluation = NULL, *replay_path = NULL;
    int context = 0, chunk = 0, port_seen = 0;
    unsigned policy = 0, tools = 0;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) {
            return 0;
        }
        const char *key = argv[i], *value = argv[i + 1];
        if (!strcmp(key, "--evaluation-mode") && !evaluation &&
            !strcmp(value, "controlled-replay")) {
            evaluation = value;
        } else if (!strcmp(key, "--replay-logits") && !replay_path && *value) {
            replay_path = value;
        } else if (!strcmp(key, "--model") && !model) {
            model = value;
        } else if (!strcmp(key, "--tokenizer") && !tokenizer) {
            tokenizer = value;
        } else if (!strcmp(key, "--strategy") && !strategy_name &&
                   (!strcmp(value, "resident-fp32") || !strcmp(value, "compact-fp8-arithmetic") ||
                    !strcmp(value, "compact-fp8-lookup"))) {
            strategy_name = value;
        } else if (!strcmp(key, "--context") && !context && (!strcmp(value, "256") || !strcmp(value, "4352"))) {
            context = atoi(value);
        } else if (!strcmp(key, "--prefill-chunk") && !chunk && (!strcmp(value, "256") || !strcmp(value, "128"))) {
            chunk = atoi(value);
        } else if (!strcmp(key, "--prefill") && !prefill &&
                   (!strcmp(value, "mps") || !strcmp(value, "custom"))) {
            prefill = value;
        } else if (!strcmp(key, "--prefix-reuse") && !policy &&
                   (!strcmp(value, "on") || !strcmp(value, "off"))) {
            policy = !strcmp(value, "on") ? 1 : 2;
        } else if (!strcmp(key, "--tools") && !tools &&
                   (!strcmp(value, "calculator-v13") || !strcmp(value, "investigation-v13") ||
                    !strcmp(value, "observation-v13"))) {
            tools = !strcmp(value, "observation-v13") ? 3 :
                !strcmp(value, "investigation-v13") ? 2 : 1;
        } else if (!strcmp(key, "--port") && !port_seen) {
            char *end;
            if (!*value || strspn(value, "0123456789") != strlen(value)) {
                return 0;
            }
            unsigned long number = strtoul(value, &end, 10);
            if (!*value || *end || number > 65535) {
                return 0;
            }
            *port = (unsigned)number;
            port_seen = 1;
        } else {
            return 0;
        }
    }
    if (!model || !tokenizer || !strategy_name || !prefill || !port_seen ||
        (!!evaluation != !!replay_path) || (evaluation && (tools || !replay_float_format()))) {
        return 0;
    }
    if (!context) {
        context = 256;
    }
    if (!chunk) {
        chunk = 256;
    }
    native_service *n = calloc(1, sizeof(*n));
    if (!n) {
        return 0;
    }
    n->policy = policy;
    n->tools = tools;
    n->capacity = (unsigned)context;
    n->block = (unsigned)chunk;
    if (context == 256 && chunk != 256) {
        free(n);
        return 0;
    }
    if (evaluation) {
        int fd = open(replay_path, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd < 0) {
            free(n);
            return 0;
        }
        n->replay_file = fdopen(fd, "wb");
        if (!n->replay_file) {
            close(fd);
            free(n);
            return 0;
        }
        if (fwrite("LNRPLY1\0", 1, 8, n->replay_file) != 8 ||
            !write_u32(n->replay_file, LUNELETTE_VOCAB_SIZE) ||
            !write_u32(n->replay_file, SERVICE_REPLAY_TOKENS) || fflush(n->replay_file)) {
            native_destroy(n);
            return 0;
        }
    }
    char error[512], model_hash[65], tokenizer_hash[65];
    double start = seconds();
    if (!hash_file(model, model_hash) || !hash_file(tokenizer, tokenizer_hash)) {
        fprintf(stderr, "load: cannot hash the explicit model/tokenizer files\n");
        native_destroy(n);
        return 0;
    }
    if (!lunelette_tokenizer_open(tokenizer, &n->tokenizer, error, sizeof(error))) {
        fprintf(stderr, "tokenizer: %s\n", error);
        native_destroy(n);
        return 0;
    }
    lunelette_metal_options options = {
        !strcmp(prefill, "mps") ? LUNELETTE_METAL_PREFILL_MPS : LUNELETTE_METAL_PREFILL_CUSTOM,
        !strcmp(strategy_name, "resident-fp32") ? LUNELETTE_METAL_RESIDENT_FP32 :
            (!strcmp(strategy_name, "compact-fp8-lookup") ?
                LUNELETTE_METAL_COMPACT_FP8_LOOKUP : LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC)
    };
    lunelette_metal_limits limits = {(size_t)context, (size_t)chunk};
    n->runtime = lunelette_metal_create_configured(model, &options, &limits, error, sizeof(error));
    if (!n->runtime) {
        fprintf(stderr, "load: %s\n", error);
        native_destroy(n);
        return 0;
    }
    n->loads++;
    lunelette_metal_memory memory;
    if (lunelette_metal_get_memory(n->runtime, &memory) ||
        memory.strategy != options.strategy) {
        fprintf(stderr, "load: runtime strategy/allocation accounting query failed or strategy mismatch\n");
        native_destroy(n);
        return 0;
    }
    const char *actual_strategy = memory.strategy == LUNELETTE_METAL_RESIDENT_FP32 ?
        "resident-fp32" : (memory.strategy == LUNELETTE_METAL_COMPACT_FP8_LOOKUP ?
            "compact-fp8-lookup" : "compact-fp8-arithmetic");
    int written = snprintf(n->info, sizeof(n->info),
        "{\"generator\":\"%s\",%s\"model_loaded\":true,\"runtime_ready\":true,"
        "\"model_sha256\":\"%s\",\"tokenizer_sha256\":\"%s\","
        "\"tokenizer_remote_identity\":\"content_hash_only_see_framing_evidence\","
        "\"model_load_count\":%u,\"loading_duration_ms\":%.3f,"
        "\"execution_strategy\":\"%s\",\"prefill\":\"%s\","
        "\"precision_policy\":\"%s; FP32 activations norms embedding head KV; %s\","
        "\"single_token_block_kernel\":\"%s\",\"head_kernel\":\"precise-FP32\","
        "\"multi_token_block_kernel\":\"%s\","
        "\"context_capacity\":%u,\"prefill_chunk_size\":%u,\"event_bytes\":8192,"
        "\"long_context_validation\":\"%s\","
        "\"long_context_validation_document\":\"docs/limitations.md#long-context-evidence\","
        "\"conversation_protocol\":\"%s\","
        "\"checkpoint_payload_bytes_per_handle\":524288,\"checkpoint_transient_payload_bound_bytes\":1048576,"
        "\"tools\":%s,\"checkpoints\":%s,\"reuse_new_turn\":%s,\"prefix_reuse\":\"%s\","
        "\"prefix_reuse_validation\":\"%s\","
        "\"prefix_reuse_validation_scope\":\"published source-bound evidence; no runtime attestation\","
        "\"checkpoint_validation_document\":\"docs/limitations.md#checkpoint-evidence\","
        "\"numerical_validation\":\"diagnostic_known_reference_mismatches\","
        "\"numerical_validation_scope\":\"%s; strict independent reference gates unresolved\","
        "\"compact_model_validation\":\"%s\","
        "\"numerical_comparison_performed_at_startup\":false,"
        "\"published_validation_document\":\"%s\","
        "\"named_resident_weight_bytes\":%" PRIu64 ",\"named_conversion_stage_bytes\":%" PRIu64 ","
        "\"named_scratch_bytes\":%" PRIu64 ",\"named_kv_bytes\":%" PRIu64 ","
        "\"named_buffer_total_bytes\":%" PRIu64 ","
        "\"driver_memory_bytes\":null,\"build_commit\":\"%s\",\"build_dirty\":%s}",
        evaluation ? "native_controlled_replay" : "native_greedy",
        evaluation ? "\"evaluation_mode\":\"controlled_replay\","
            "\"replay_max_tokens\":16,\"replay_capture_limit\":16," : "",
        model_hash, tokenizer_hash, n->loads, (seconds() - start) * 1000, actual_strategy, prefill,
        memory.strategy ? "FP8 projection storage, FP32 reconstruction" : "FP32 resident weights",
        options.prefill == LUNELETTE_METAL_PREFILL_MPS ? "MPS accumulation unspecified" :
            "precise FP32 custom arithmetic",
        memory.strategy == LUNELETTE_METAL_COMPACT_FP8_LOOKUP ? "compact-FP8-lookup" :
            (memory.strategy ? "compact-FP8-arithmetic" : "precise-FP32"),
        memory.strategy ? (options.prefill == LUNELETTE_METAL_PREFILL_MPS ?
            "FP8-convert-stage-then-MPS" : "FP8-convert-stage-then-precise-FP32") :
            (options.prefill == LUNELETTE_METAL_PREFILL_MPS ? "MPS" : "precise-FP32"),
        n->capacity, n->block,
        n->capacity > 256 ? "external_evidence_only" : "not_applicable",
        tools == 3 ? "native-observation-v13" :
        tools == 2 ? "native-investigation-v13" :
            tools ? "native-calculator-v13" : "native-no-system-nonthinking-text-v1",
        tools ? "true" : "false",
        policy && !tools ? "true" : "false", policy == 1 ? "true" : "false",
        policy == 1 ? "on" : policy == 2 ? "off" : "legacy",
        policy ? "external_evidence_only" : "not_applicable",
        memory.strategy == LUNELETTE_METAL_COMPACT_FP8_LOOKUP ?
            "lookup candidate strategy-specific evidence only; no runtime attestation" :
            (memory.strategy ? "compact published evidence only; no runtime attestation" :
                "resident published evidence only; no runtime attestation"),
        memory.strategy ? "external_evidence_only" : "not_applicable",
        memory.strategy == LUNELETTE_METAL_COMPACT_FP8_LOOKUP ?
            "docs/limitations.md#compact-lookup-evidence" :
            (memory.strategy ? "docs/limitations.md#compact-arithmetic-evidence" :
                "docs/limitations.md#independent-numerical-evidence"),
        memory.weight_bytes, memory.conversion_stage_bytes, memory.scratch_bytes,
        memory.kv_bytes, memory.total_bytes, SERVICE_COMMIT, SERVICE_DIRTY);
    if (written < 0 || (size_t)written >= sizeof(n->info)) {
        fprintf(stderr, "load: native info exceeds fixed capacity\n");
        native_destroy(n);
        return 0;
    }
    *generator = (service_generator){
        .context = n, .next = native_next,
        .replay_next = evaluation ? native_replay_next : NULL,
        .materialize = native_materialize,
        .reset = native_reset, .destroy = native_destroy, .frame = native_frame,
        .native_tools = tools, .tool_frame = tools ? native_tool_frame : NULL,
        .piece = native_piece, .eos = 2, .info = n->info, .prefix_reuse = policy,
        .begin = native_begin, .save = native_save, .restore = native_restore,
        .invalidate = native_invalidate, .work = native_work,
        .context_capacity = n->capacity, .block_capacity = n->block,
        .materialize_step = native_materialize_step
    };
    return 1;
}
