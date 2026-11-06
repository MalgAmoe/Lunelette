#define _POSIX_C_SOURCE 200809L
#include "native.h"
#include "framing.h"
#include "metal.h"
#include "tokenizer.h"
#include "generation.h"
#include <CommonCrypto/CommonDigest.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    lunelette_metal *runtime;
    lunelette_tokenizer *tokenizer;
    unsigned prefix[256], length, emitted, loads;
    int pending, boundary;
    unsigned policy, saved_length, saved_prefix[256];
    lunelette_metal_checkpoint *checkpoint;
    service_work work;
    const float *logits;
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
    free(n);
}

static int native_frame(void *context, const unsigned char *text, size_t length,
                        int initial, unsigned *output, unsigned *capacity)
{
    native_service *n = context;
    return service_frame(n->tokenizer, text, length, initial, output, capacity);
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
    if (!length || length > 256 || length < n->length ||
        memcmp(n->prefix, tokens, n->length * sizeof(*tokens))) {
        return 0;
    }
    if (n->pending && (length <= n->length || tokens[n->length] != n->emitted)) {
        return 0;
    }
    if (length == n->length && !(n->policy == 2 && n->boundary)) {
        n->boundary = 0;
        n->work.reused = length;
        return n->logits != NULL;
    }
    int status;
    unsigned old_length = n->length;
    int rebuild = !old_length || (n->policy == 2 && n->boundary) ||
        (!n->policy && !(n->pending && length == old_length + 1 && n->emitted != 2));
    n->boundary = 0;
    int64_t ids[256];
    for (unsigned i = 0; i < length; i++) {
        ids[i] = tokens[i];
    }
    n->work.reused = rebuild ? 0 : old_length;
    if (rebuild) {
        n->logits = NULL;
        /* Internal recompute invalidates KV, but controller keeps its recipe. */
        status = lunelette_metal_reset(n->runtime, error, sizeof(error));
        if (!status) {
            n->work.pending_attempted = n->pending ? 1 : 0;
            n->work.full_attempted = length;
            n->work.prefill_calls = 1;
            status = lunelette_metal_prefill(n->runtime, ids, length, &n->logits,
                                             error, sizeof(error));
        }
    } else if (n->pending && length == old_length + 1) {
        n->work.pending_attempted = 1;
        n->work.decode_calls = 1;
        n->work.decode_attempted = 1;
        status = lunelette_metal_decode(n->runtime, n->emitted, &n->logits,
                                        error, sizeof(error));
    } else {
        n->work.pending_attempted = n->pending ? 1 : 0;
        n->work.suffix_attempted = length - old_length;
        n->work.append_calls = 1;
        status = lunelette_metal_append(n->runtime, ids + old_length, length - old_length,
                                        &n->logits, error, sizeof(error));
    }
    if (status) {
        fprintf(stderr, "materialize: %s\n", error);
        n->logits = NULL;
        return 0;
    }
    n->work.full_committed = n->work.full_attempted;
    n->work.suffix_committed = n->work.suffix_attempted;
    n->work.pending_committed = n->work.pending_attempted;
    n->work.decode_committed = n->work.decode_attempted;
    memcpy(n->prefix, tokens, length * sizeof(*tokens));
    n->length = length;
    n->pending = 0;
    return 1;
}

static int native_save(void *context)
{
    native_service *n = context;
    char error[512];
    lunelette_metal_checkpoint *replacement = NULL;
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
    if (!n->saved_length) {
        return 0;
    }
    if (n->policy == 2) {
        n->logits = NULL;
        n->pending = 0;
        n->emitted = 0;
        n->boundary = 0;
        memset(n->prefix, 0, sizeof(n->prefix));
        n->length = 0;
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
    n->work.logit_restores = 1;
    return 1;
}

static int copy_logits(void *context, const int64_t *ids, size_t count,
                       float *logits, char *error, size_t capacity)
{
    native_service *n = context;
    (void)ids;
    (void)count;
    (void)error;
    (void)capacity;
    if (!n->logits) {
        return -1;
    }
    memcpy(logits, n->logits, LUNELETTE_VOCAB_SIZE * sizeof(*logits));
    return 0;
}

static int native_next(void *context)
{
    native_service *n = context;
    int64_t ids[256];
    for (unsigned i = 0; i < n->length; i++) {
        ids[i] = n->prefix[i];
    }
    lunelette_generation_config config = {1, 2};
    lunelette_generation_result result;
    char error[512];
    if (n->pending) {
        return -1;
    }
    if (lunelette_generate_with_logits(copy_logits, n, ids, n->length,
                                       &config, &result, error, sizeof(error))) {
        fprintf(stderr, "generate: %s\n", error);
        return -1;
    }
    if (result.count != 1) {
        fprintf(stderr, "generate: expected one token, got %zu\n", result.count);
        return -1;
    }
    n->emitted = (unsigned)result.tokens[0];
    n->pending = 1;
    return (int)n->emitted;
}

int service_native_open(int argc, char **argv, service_generator *generator, unsigned *port)
{
    const char *model = NULL, *tokenizer = NULL, *prefill = NULL, *strategy_name = NULL;
    int context = 0, chunk = 0, port_seen = 0;
    unsigned policy = 0;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) {
            return 0;
        }
        const char *key = argv[i], *value = argv[i + 1];
        if (!strcmp(key, "--model") && !model) {
            model = value;
        } else if (!strcmp(key, "--tokenizer") && !tokenizer) {
            tokenizer = value;
        } else if (!strcmp(key, "--strategy") && !strategy_name &&
                   (!strcmp(value, "resident-fp32") || !strcmp(value, "compact-fp8-arithmetic"))) {
            strategy_name = value;
        } else if (!strcmp(key, "--context") && !context && !strcmp(value, "256")) {
            context = 1;
        } else if (!strcmp(key, "--prefill-chunk") && !chunk && !strcmp(value, "256")) {
            chunk = 1;
        } else if (!strcmp(key, "--prefill") && !prefill &&
                   (!strcmp(value, "mps") || !strcmp(value, "custom"))) {
            prefill = value;
        } else if (!strcmp(key, "--prefix-reuse") && !policy &&
                   (!strcmp(value, "on") || !strcmp(value, "off"))) {
            policy = !strcmp(value, "on") ? 1 : 2;
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
    if (!model || !tokenizer || !strategy_name || !context || !chunk || !prefill || !port_seen) {
        return 0;
    }
    native_service *n = calloc(1, sizeof(*n));
    if (!n) {
        return 0;
    }
    n->policy = policy;
    char error[512], model_hash[65], tokenizer_hash[65];
    double start = seconds();
    if (!hash_file(model, model_hash) || !hash_file(tokenizer, tokenizer_hash)) {
        fprintf(stderr, "load: cannot hash the explicit model/tokenizer files\n");
        free(n);
        return 0;
    }
    if (!lunelette_tokenizer_open(tokenizer, &n->tokenizer, error, sizeof(error))) {
        fprintf(stderr, "tokenizer: %s\n", error);
        free(n);
        return 0;
    }
    lunelette_metal_options options = {
        !strcmp(prefill, "mps") ? LUNELETTE_METAL_PREFILL_MPS : LUNELETTE_METAL_PREFILL_CUSTOM,
        !strcmp(strategy_name, "resident-fp32") ? LUNELETTE_METAL_RESIDENT_FP32 :
            LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC
    };
    n->runtime = lunelette_metal_create_with_options(model, &options, error, sizeof(error));
    if (!n->runtime) {
        fprintf(stderr, "load: %s\n", error);
        lunelette_tokenizer_close(n->tokenizer);
        free(n);
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
        "resident-fp32" : "compact-fp8-arithmetic";
    int written = snprintf(n->info, sizeof(n->info),
        "{\"generator\":\"native_greedy\",\"model_loaded\":true,\"runtime_ready\":true,"
        "\"model_sha256\":\"%s\",\"tokenizer_sha256\":\"%s\","
        "\"tokenizer_remote_identity\":\"content_hash_only_see_framing_evidence\","
        "\"model_load_count\":%u,\"loading_duration_ms\":%.3f,"
        "\"execution_strategy\":\"%s\",\"prefill\":\"%s\","
        "\"precision_policy\":\"%s; FP32 activations norms embedding head KV; %s\","
        "\"single_token_block_kernel\":\"%s\",\"head_kernel\":\"precise-FP32\","
        "\"multi_token_block_kernel\":\"%s\","
        "\"context_capacity\":256,\"prefill_chunk_size\":256,\"event_bytes\":8192,"
        "\"conversation_protocol\":\"native-no-system-nonthinking-text-v1\","
        "\"checkpoint_payload_bytes_per_handle\":524288,\"checkpoint_transient_payload_bound_bytes\":1048576,"
        "\"tools\":false,\"checkpoints\":%s,\"reuse_new_turn\":%s,\"prefix_reuse\":\"%s\","
        "\"prefix_reuse_validation\":\"%s\","
        "\"prefix_reuse_validation_scope\":\"published source-bound evidence; no runtime attestation\","
        "\"checkpoint_validation_document\":\"docs/real-workload-m5.md\","
        "\"numerical_validation\":\"diagnostic_known_reference_mismatches\","
        "\"numerical_validation_scope\":\"%s; strict independent reference gates unresolved\","
        "\"compact_model_validation\":\"%s\","
        "\"numerical_comparison_performed_at_startup\":false,"
        "\"published_validation_document\":\"docs/real-workload-m4.md\","
        "\"named_resident_weight_bytes\":%" PRIu64 ",\"named_conversion_stage_bytes\":%" PRIu64 ","
        "\"named_scratch_bytes\":%" PRIu64 ",\"named_kv_bytes\":%" PRIu64 ","
        "\"named_buffer_total_bytes\":%" PRIu64 ","
        "\"driver_memory_bytes\":null,\"build_commit\":\"%s\",\"build_dirty\":%s}",
        model_hash, tokenizer_hash, n->loads, (seconds() - start) * 1000, actual_strategy, prefill,
        memory.strategy ? "FP8 projection storage, FP32 reconstruction" : "FP32 resident weights",
        options.prefill == LUNELETTE_METAL_PREFILL_MPS ? "MPS accumulation unspecified" :
            "precise FP32 custom arithmetic",
        memory.strategy ? "compact-FP8-arithmetic" : "precise-FP32",
        memory.strategy ? (options.prefill == LUNELETTE_METAL_PREFILL_MPS ?
            "FP8-convert-stage-then-MPS" : "FP8-convert-stage-then-precise-FP32") :
            (options.prefill == LUNELETTE_METAL_PREFILL_MPS ? "MPS" : "precise-FP32"),
        policy ? "true" : "false", policy == 1 ? "true" : "false",
        policy == 1 ? "on" : policy == 2 ? "off" : "legacy",
        policy ? "external_evidence_only" : "not_applicable",
        memory.strategy ? "compact published evidence only; no runtime attestation" :
            "resident published evidence only; no runtime attestation",
        memory.strategy ? "external_evidence_only" : "not_applicable",
        memory.weight_bytes, memory.conversion_stage_bytes, memory.scratch_bytes,
        memory.kv_bytes, memory.total_bytes, SERVICE_COMMIT, SERVICE_DIRTY);
    if (written < 0 || (size_t)written >= sizeof(n->info)) {
        fprintf(stderr, "load: native info exceeds fixed capacity\n");
        native_destroy(n);
        return 0;
    }
    *generator = (service_generator){n, native_next, native_materialize, native_reset,
                                    native_destroy, native_frame, native_piece, 2, n->info,
                                    policy, native_begin, native_save, native_restore,
                                    native_invalidate, native_work};
    return 1;
}
