#include "lunelette.h"
#include "chat.h"
#include "cache.h"
#ifdef LUNELETTE_USE_METAL
#include "metal.h"
#include "metal_weights.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(void)
{
    return puts("Usage: lunelette --tokenizer TEKKEN --prompt TEXT [--system TEXT] [--raw]\n"
         "       [--weights SHARD1 --weights2 SHARD2] [--max-new-tokens N]\n"
         "       [--backend cpu|metal] [--metal-prefill mps|custom] [--uncached]\n"
         "       [--dump-prompt-ids] [--dump-output-ids]\n"
         "Single-turn 3B text chat; no default system message. Raw: BOS + text.\n"
         "Generation requires both shards; prompt-ID mode needs neither.\n"
         "Greedy cached CPU (default) or optional Metal; total context <=256.\n"
         "N=0..256 (default 32).\n"
         "--dump-prompt-ids prints IDs only, without opening weights.\n"
         "--dump-output-ids writes generated IDs (including EOS) to stderr.\n"
         "Also: --help, --version");
}

/* Deliberately decimal digits only: no signs, whitespace, suffix or overflow. */
static int parse_budget(const char *text, size_t *value)
{
    size_t n = 0;
    if (!*text) {
        return -1;
    }
    for (; *text; text++) {
        if (*text < '0' || *text > '9' || n > (256u - (unsigned)(*text - '0')) / 10u) {
            return -1;
        }
        n = n * 10 + (unsigned)(*text - '0');
    }
    *value = n;
    return 0;
}

#ifdef LUNELETTE_USE_METAL
typedef struct {
    lunelette_metal *runtime;
    int started;
} metal_generation;

static int metal_next(void *context, const int64_t *ids, size_t count,
    float *output, char *error, size_t capacity)
{
    metal_generation *generation = context;
    const float *logits = NULL;
    int status;
    if (!generation->started) {
        status = lunelette_metal_prefill(generation->runtime, ids, count,
            &logits, error, capacity);
    } else {
        status = lunelette_metal_decode(generation->runtime, ids[count - 1],
            &logits, error, capacity);
    }
    if (status != 0) {
        return -1;
    }
    generation->started = 1;
    memcpy(output, logits, LUNELETTE_VOCAB_SIZE * sizeof(float));
    return 0;
}
#endif

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        int failed = puts("Lunelette " LUNELETTE_VERSION) == EOF;
        if (fflush(stdout) == EOF || ferror(stdout)) {
            failed = 1;
        }
        if (failed) {
            fputs("Writing version to stdout failed; check output destination\n", stderr);
        }
        return failed;
    }
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        int failed = usage() == EOF;
        if (fflush(stdout) == EOF || ferror(stdout)) {
            failed = 1;
        }
        if (failed) {
            fputs("Writing help to stdout failed; check output destination\n", stderr);
        }
        return failed;
    }
    const char *weights = NULL, *weights2 = NULL, *path = NULL;
    const char *prompt = NULL, *system = NULL;
    const char *backend = "cpu";
    const char *metal_prefill = NULL;
    int raw = 0, uncached = 0, dump = 0, output_ids = 0;
    unsigned seen = 0;
    lunelette_generation_config config = LUNELETTE_GENERATION_DEFAULT;
    const char *names[] = {"--weights", "--weights2", "--tokenizer", "--prompt",
        "--system", "--max-new-tokens", "--raw", "--uncached", "--dump-prompt-ids",
        "--dump-output-ids", "--backend", "--metal-prefill"};
    for (int i = 1; i < argc; i++) {
        size_t option;
        for (option = 0; option < sizeof(names) / sizeof(names[0]); option++) {
            if (strcmp(argv[i], names[option]) == 0) {
                break;
            }
        }
        if (option == sizeof(names) / sizeof(names[0]) || (seen & (1u << option))) {
            fprintf(stderr, "Unknown or duplicate argument: %s\n", argv[i]);
            return 1;
        }
        seen |= 1u << option;
        const char *value = NULL;
        if (option < 6 || option >= 10) {
            if (++i == argc || strncmp(argv[i], "--", 2) == 0) {
                fprintf(stderr, "Missing value for %s\n", names[option]);
                return 1;
            }
            value = argv[i];
        }
        switch (option) {
        case 0: weights = value; break;
        case 1: weights2 = value; break;
        case 2: path = value; break;
        case 3: prompt = value; break;
        case 4: system = value; break;
        case 5:
            if (parse_budget(value, &config.max_new_tokens) != 0) {
                fputs("max-new-tokens must be decimal 0..256\n", stderr);
                return 1;
            }
            break;
        case 6: raw = 1; break;
        case 7: uncached = 1; break;
        case 8: dump = 1; break;
        case 9: output_ids = 1; break;
        case 10: backend = value; break;
        case 11: metal_prefill = value; break;
        }
    }
    int use_metal = strcmp(backend, "metal") == 0;
    if (!use_metal && strcmp(backend, "cpu") != 0) {
        fputs("--backend must be cpu or metal\n", stderr);
        return 1;
    }
    if (metal_prefill && (!use_metal || (strcmp(metal_prefill, "mps") != 0 &&
            strcmp(metal_prefill, "custom") != 0))) {
        fputs("--metal-prefill requires --backend metal and mps or custom\n", stderr);
        return 1;
    }
    if (use_metal && uncached) {
        fputs("Metal does not support --uncached; omit it or use --backend cpu\n", stderr);
        return 1;
    }
#ifndef LUNELETTE_USE_METAL
    if (use_metal) {
        fputs("Metal backend unavailable: rebuild on macOS with METAL=1 BUILD=build/metal\n", stderr);
        return 1;
    }
#endif
    if (!path || !prompt || (raw && system) || (!dump && (!weights || !weights2))) {
        fputs("Require --tokenizer, --prompt and both weights (unless --dump-prompt-ids); "
              "--raw forbids --system. Use --help.\n", stderr);
        return 1;
    }
    char error[512] = "Operation failed";
    lunelette_tokenizer *tokenizer = NULL;
    lunelette_cache *cache = NULL;
#ifdef LUNELETTE_USE_METAL
    lunelette_metal *metal = NULL;
#endif
    st_file first = {0}, second = {0};
    unsigned char *bytes = NULL;
    int status = 1;
    int64_t ids[LUNELETTE_CONTEXT_LIMIT];
    size_t count = 0;
    if (!lunelette_tokenizer_open(path, &tokenizer, error, sizeof error) ||
        lunelette_chat_prompt(tokenizer, prompt, system, raw, ids, &count,
            error, sizeof error) != 0) {
        goto cleanup;
    }
    if (dump) {
        for (size_t i = 0; i < count; i++) {
            printf("%s%lld", i ? " " : "", (long long)ids[i]);
        }
        int failed = puts("") == EOF;
        if (fflush(stdout) == EOF || ferror(stdout)) {
            failed = 1;
        }
        status = failed ? 1 : 0;
        if (failed) {
            snprintf(error, sizeof error, "Writing prompt IDs to stdout failed; check output destination");
        }
        goto cleanup;
    }
    lunelette_generation_result result;
#ifdef LUNELETTE_USE_METAL
    if (use_metal && (!config.max_new_tokens || count == LUNELETTE_CONTEXT_LIMIT)) {
        /* Keep full checkpoint validation, but do not create a GPU context for
         * a request that cannot execute even one logits step. */
        lunelette_metal_weights metadata;
        if (lunelette_metal_weights_open(&metadata, weights, weights2,
                error, sizeof error) != 0) {
            goto cleanup;
        }
        lunelette_metal_weights_close(&metadata);
        metal_generation generation = {NULL, 0};
        if (lunelette_generate_with_logits(metal_next, &generation, ids, count,
                &config, &result, error, sizeof error) != 0) {
            goto cleanup;
        }
    } else if (use_metal) {
        lunelette_metal_options metal_options = LUNELETTE_METAL_OPTIONS_DEFAULT;
        if (metal_prefill && strcmp(metal_prefill, "custom") == 0) {
            metal_options.prefill = LUNELETTE_METAL_PREFILL_CUSTOM;
        }
        metal = lunelette_metal_create_with_options(weights, weights2, &metal_options,
            error, sizeof error);
        metal_generation generation = {metal, 0};
        if (!metal || lunelette_generate_with_logits(metal_next, &generation, ids, count,
                &config, &result, error, sizeof error) != 0) {
            goto cleanup;
        }
    } else {
#endif
        if (st_open(&first, weights, error, sizeof error) != 0 ||
            st_open(&second, weights2, error, sizeof error) != 0) {
            goto cleanup;
        }
        if (uncached) {
            if (lunelette_generate(&first, &second, ids, count, &config, &result,
                    error, sizeof error) != 0) {
                goto cleanup;
            }
        } else {
            cache = lunelette_cache_init(&first, &second, 256, error, sizeof error);
            if (!cache || lunelette_generate_cached(cache, ids, count, &config, &result,
                    error, sizeof error) != 0) {
                goto cleanup;
            }
        }
#ifdef LUNELETTE_USE_METAL
    }
#endif
    uint32_t ordinary[LUNELETTE_CONTEXT_LIMIT];
    size_t n = 0;
    for (size_t i = 0; i < result.count; i++) {
        if (result.tokens[i] == LUNELETTE_TOKEN_EOS && i + 1 == result.count) {
            continue;
        }
        if (result.tokens[i] < LUNELETTE_TOKENIZER_SPECIAL_COUNT) {
            snprintf(error, sizeof error, "Unexpected generated special token ID %lld",
                (long long)result.tokens[i]);
            goto cleanup;
        }
        ordinary[n++] = (uint32_t)result.tokens[i];
    }
    size_t length;
    if (!lunelette_tokenizer_decode(tokenizer, ordinary, n, &bytes, &length,
            error, sizeof error)) {
        goto cleanup;
    }
    if (output_ids) {
        fputs("generated_ids:", stderr);
        for (size_t i = 0; i < result.count; i++) {
            fprintf(stderr, " %lld", (long long)result.tokens[i]);
        }
        int failed = fputc('\n', stderr) == EOF;
        if (fflush(stderr) == EOF || ferror(stderr)) {
            failed = 1;
        }
        if (failed) {
            snprintf(error, sizeof error,
                "Writing generated IDs to stderr failed; check diagnostic destination");
            goto cleanup;
        }
    }
    if (fwrite(bytes, 1, length, stdout) != length || fflush(stdout) == EOF) {
        snprintf(error, sizeof error, "Writing generated text failed");
        goto cleanup;
    }
    status = 0;
cleanup:
    if (status) {
        fprintf(stderr, "%s\n", error);
    }
    free(bytes);
#ifdef LUNELETTE_USE_METAL
    lunelette_metal_destroy(metal);
#endif
    lunelette_cache_free(cache);
    st_close(&second);
    st_close(&first);
    lunelette_tokenizer_close(tokenizer);
    return status;
}
