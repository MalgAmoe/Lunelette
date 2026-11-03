#include "lunelette.h"
#include "chat.h"
#include "cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void)
{
    puts("Usage: lunelette --tokenizer TEKKEN --prompt TEXT [--system TEXT] [--raw]\n"
         "       [--weights SHARD1 --weights2 SHARD2] [--max-new-tokens N]\n"
         "       [--uncached] [--dump-prompt-ids] [--dump-output-ids]\n"
         "Single-turn 3B text chat; no default system message. Raw: BOS + text.\n"
         "Generation requires both shards; prompt-ID mode needs neither.\n"
         "Greedy cached CPU, total context <=256, N=0..256 (default 32).\n"
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

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        puts("Lunelette " LUNELETTE_VERSION);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage();
        return 0;
    }
    const char *weights = NULL, *weights2 = NULL, *path = NULL;
    const char *prompt = NULL, *system = NULL;
    int raw = 0, uncached = 0, dump = 0, output_ids = 0;
    unsigned seen = 0;
    lunelette_generation_config config = LUNELETTE_GENERATION_DEFAULT;
    const char *names[] = {"--weights", "--weights2", "--tokenizer", "--prompt",
        "--system", "--max-new-tokens", "--raw", "--uncached", "--dump-prompt-ids",
        "--dump-output-ids"};
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
        if (option < 6) {
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
        }
    }
    if (!path || !prompt || (raw && system) || (!dump && (!weights || !weights2))) {
        fputs("Require --tokenizer, --prompt and both weights (unless --dump-prompt-ids); "
              "--raw forbids --system. Use --help.\n", stderr);
        return 1;
    }
    char error[512] = "Operation failed";
    lunelette_tokenizer *tokenizer = NULL;
    lunelette_cache *cache = NULL;
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
        status = puts("") == EOF || ferror(stdout) ? 1 : 0;
        goto cleanup;
    }
    if (st_open(&first, weights, error, sizeof error) != 0 ||
        st_open(&second, weights2, error, sizeof error) != 0) {
        goto cleanup;
    }
    lunelette_generation_result result;
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
        fputc('\n', stderr);
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
    lunelette_cache_free(cache);
    st_close(&second);
    st_close(&first);
    lunelette_tokenizer_close(tokenizer);
    return status;
}
