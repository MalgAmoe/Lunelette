#include "lunelette.h"
#include "chat.h"
#include "generation.h"
#include "metal.h"
#include "metal_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(void)
{
    return puts("Usage: lunelette --tokenizer TEKKEN --prompt TEXT [--system TEXT] [--raw]\n"
         "       [--weights MODEL.safetensors] [--max-new-tokens N]\n"
         "       Storage default: compact-fp8-arithmetic; use resident-fp32 for previous storage behavior.\n"
         "       [--metal-prefill mps|custom] [--metal-strategy resident-fp32|compact-fp8-arithmetic|compact-fp8-lookup]\n"
         "       [--dump-prompt-ids] [--dump-output-ids]\n"
         "Single-turn 3B text chat; no default system message. Raw: BOS + text.\n"
         "Generation requires the FP8 checkpoint; prompt-ID mode needs no weights.\n"
         "Greedy resident Metal/MPS runtime; total context <=256.\n"
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
    const char *weights = NULL, *path = NULL;
    const char *prompt = NULL, *system = NULL;
    const char *metal_prefill = NULL, *metal_strategy = NULL;
    int raw = 0, dump = 0, output_ids = 0;
    unsigned seen = 0;
    lunelette_generation_config config = LUNELETTE_GENERATION_DEFAULT;
    const char *names[] = {"--weights", "--tokenizer", "--prompt",
        "--system", "--max-new-tokens", "--raw", "--dump-prompt-ids",
        "--dump-output-ids", "--metal-prefill", "--metal-strategy"};
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backend") == 0 || strcmp(argv[i], "--uncached") == 0) {
            fputs("--backend and --uncached were retired; use the resident Metal/MPS runtime\n", stderr);
            return 1;
        }
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
        if (option < 5 || option >= 8) {
            if (++i == argc || strncmp(argv[i], "--", 2) == 0) {
                fprintf(stderr, "Missing value for %s\n", names[option]);
                return 1;
            }
            value = argv[i];
        }
        switch (option) {
        case 0: weights = value; break;
        case 1: path = value; break;
        case 2: prompt = value; break;
        case 3: system = value; break;
        case 4:
            if (parse_budget(value, &config.max_new_tokens) != 0) {
                fputs("max-new-tokens must be decimal 0..256\n", stderr);
                return 1;
            }
            break;
        case 5: raw = 1; break;
        case 6: dump = 1; break;
        case 7: output_ids = 1; break;
        case 8: metal_prefill = value; break;
        case 9: metal_strategy = value; break;
        }
    }
    if (metal_prefill && strcmp(metal_prefill, "mps") != 0 &&
            strcmp(metal_prefill, "custom") != 0) {
        fputs("--metal-prefill must be mps or custom\n", stderr);
        return 1;
    }
    if (metal_strategy && strcmp(metal_strategy, "resident-fp32") &&
        strcmp(metal_strategy, "compact-fp8-arithmetic") &&
        strcmp(metal_strategy, "compact-fp8-lookup")) {
        fputs("--metal-strategy must be resident-fp32, compact-fp8-arithmetic or compact-fp8-lookup\n", stderr);
        return 1;
    }
    if (!path || !prompt || (raw && system) || (!dump && !weights)) {
        fputs("Require --tokenizer, --prompt and --weights (unless --dump-prompt-ids); "
              "--raw forbids --system. Use --help.\n", stderr);
        return 1;
    }
    char error[512] = "Operation failed";
    lunelette_tokenizer *tokenizer = NULL;
    lunelette_metal *metal = NULL;
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
    if (!config.max_new_tokens || count == LUNELETTE_CONTEXT_LIMIT) {
        /* Keep full checkpoint validation, but do not create a GPU context for
         * a request that cannot execute even one logits step. */
        lunelette_metal_weights metadata;
        if (lunelette_metal_weights_open(&metadata, weights,
                error, sizeof error) != 0) {
            goto cleanup;
        }
        lunelette_metal_weights_close(&metadata);
        metal_generation generation = {NULL, 0};
        if (lunelette_generate_with_logits(metal_next, &generation, ids, count,
                &config, &result, error, sizeof error) != 0) {
            goto cleanup;
        }
    } else {
        lunelette_metal_options metal_options = LUNELETTE_METAL_OPTIONS_DEFAULT;
        if (metal_prefill && strcmp(metal_prefill, "custom") == 0) {
            metal_options.prefill = LUNELETTE_METAL_PREFILL_CUSTOM;
        }
        if (metal_strategy && !strcmp(metal_strategy, "resident-fp32")) {
            metal_options.strategy = LUNELETTE_METAL_RESIDENT_FP32;
        }
        if (metal_strategy && !strcmp(metal_strategy, "compact-fp8-arithmetic")) {
            metal_options.strategy = LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC;
        }
        if (metal_strategy && !strcmp(metal_strategy, "compact-fp8-lookup")) {
            metal_options.strategy = LUNELETTE_METAL_COMPACT_FP8_LOOKUP;
        }
        metal = lunelette_metal_create_with_options(weights, &metal_options,
            error, sizeof error);
        metal_generation generation = {metal, 0};
        if (!metal || lunelette_generate_with_logits(metal_next, &generation, ids, count,
                &config, &result, error, sizeof error) != 0) {
            goto cleanup;
        }
    }
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
    lunelette_metal_destroy(metal);
    lunelette_tokenizer_close(tokenizer);
    return status;
}
