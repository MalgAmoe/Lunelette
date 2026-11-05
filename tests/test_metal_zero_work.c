/* Portable CLI control-flow test: real generation loop, fake tokenizer and
 * Metal boundaries. Metadata policy itself is tested by test_metal_weights. */
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#define TEST_POSIX_FDS 1
#endif
#define main lunelette_cli_main
#define lunelette_metal_weights_open test_metadata_open
#define lunelette_metal_weights_close test_metadata_close
#include "../src/main.c"
#undef main

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static size_t prompt_count;
static int64_t prompt_id = 42;
static int metadata_fails;
static size_t opens, closes, creates, callbacks, decodes;
static int tokenizer_marker;

int lunelette_tokenizer_open(const char *path, lunelette_tokenizer **out,
    char *error, size_t capacity)
{
    (void)path;
    (void)error;
    (void)capacity;
    *out = (lunelette_tokenizer *)&tokenizer_marker;
    return 1;
}

void lunelette_tokenizer_close(lunelette_tokenizer *tokenizer)
{
    (void)tokenizer;
}

int lunelette_chat_prompt(const lunelette_tokenizer *tokenizer, const char *prompt,
    const char *system, int raw, int64_t ids[LUNELETTE_CONTEXT_LIMIT], size_t *count,
    char *error, size_t capacity)
{
    (void)tokenizer;
    (void)prompt;
    (void)system;
    (void)raw;
    (void)error;
    (void)capacity;
    *count = prompt_count;
    for (size_t i = 0; i < *count; i++) {
        ids[i] = prompt_id;
    }
    return 0;
}

int lunelette_tokenizer_decode(const lunelette_tokenizer *tokenizer,
    const uint32_t *ids, size_t count, unsigned char **bytes, size_t *length,
    char *error, size_t capacity)
{
    (void)tokenizer;
    (void)ids;
    (void)error;
    (void)capacity;
    CHECK(count == 0);
    decodes++;
    *bytes = malloc(1);
    CHECK(*bytes);
    *length = 0;
    return 1;
}

int test_metadata_open(lunelette_metal_weights *weights, const char *first,
    char *error, size_t capacity)
{
    (void)weights;
    CHECK(strcmp(first, "first") == 0);
    opens++;
    if (metadata_fails) {
        snprintf(error, capacity, "deliberate metadata rejection");
        return -1;
    }
    return 0;
}

void test_metadata_close(lunelette_metal_weights *weights)
{
    (void)weights;
    closes++;
}

lunelette_metal *lunelette_metal_create_with_options(const char *first,
    const lunelette_metal_options *options, char *error, size_t capacity)
{
    (void)first;
    (void)options;
    creates++;
    snprintf(error, capacity, "deliberate GPU creation rejection");
    return NULL;
}

void lunelette_metal_destroy(lunelette_metal *metal)
{
    CHECK(metal == NULL);
}

int lunelette_metal_prefill(lunelette_metal *metal, const int64_t *ids, size_t count,
    const float **logits, char *error, size_t capacity)
{
    (void)metal;
    (void)ids;
    (void)count;
    (void)logits;
    (void)error;
    (void)capacity;
    callbacks++;
    CHECK(0);
    return -1;
}

int lunelette_metal_decode(lunelette_metal *metal, int64_t token,
    const float **logits, char *error, size_t capacity)
{
    (void)token;
    return lunelette_metal_prefill(metal, NULL, 0, logits, error, capacity);
}

int main(void)
{
    char *args[] = {"test", "--tokenizer", "fake",
        "--prompt", "fake", "--weights", "first",
        "--dump-output-ids", "--max-new-tokens", "0"};
    int argc = (int)(sizeof args / sizeof args[0]);
    for (int full = 0; full < 2; full++) {
        prompt_count = full ? LUNELETTE_CONTEXT_LIMIT : 1;
        for (int budget = 0; budget < 2; budget++) {
            if (!full && budget) {
                continue;
            }
            args[argc - 1] = budget ? "1" : "0";
            for (int fail = 0; fail < 2; fail++) {
                opens = closes = creates = callbacks = decodes = 0;
                metadata_fails = fail;
                CHECK(lunelette_cli_main(argc, args) == fail);
                CHECK(opens == 1 && closes == (size_t)!fail);
                CHECK(creates == 0 && callbacks == 0 && decodes == (size_t)!fail);
            }
        }
    }
    /* Invalid IDs must be diagnosed even when no logits work is possible. */
    prompt_count = 1;
    metadata_fails = 0;
    args[argc - 1] = "0";
    for (int invalid = 0; invalid < 2; invalid++) {
        prompt_id = invalid ? LUNELETTE_MODEL_VOCAB : -1;
        opens = closes = creates = callbacks = decodes = 0;
        CHECK(lunelette_cli_main(argc, args) == 1);
        CHECK(opens == 1 && closes == 1 && creates == 0 && callbacks == 0 && decodes == 0);
    }
    prompt_id = 42;
#ifdef TEST_POSIX_FDS
    /* A read-only descriptor makes generated_ids fail without a pipe race or
     * SIGPIPE. Restore stderr before CHECK so failures remain visible. */
    CHECK(fflush(stderr) == 0);
    int saved_stderr = dup(STDERR_FILENO);
    CHECK(saved_stderr >= 0);
    int read_only = open("/dev/null", O_RDONLY);
    CHECK(read_only >= 0);
    CHECK(dup2(read_only, STDERR_FILENO) == STDERR_FILENO);
    CHECK(close(read_only) == 0);
    opens = closes = creates = callbacks = decodes = 0;
    int status = lunelette_cli_main(argc, args);
    int output_failed = ferror(stderr);
    int restored = dup2(saved_stderr, STDERR_FILENO);
    clearerr(stderr);
    CHECK(restored == STDERR_FILENO);
    CHECK(close(saved_stderr) == 0);
    CHECK(status == 1 && output_failed);
    CHECK(opens == 1 && closes == 1 && creates == 0 && callbacks == 0 && decodes == 1);
#endif
    args[argc - 1] = "1";
    /* Positive work still reaches context creation, not the metadata-only path. */
    prompt_count = 1;
    opens = creates = 0;
    CHECK(lunelette_cli_main(argc, args) == 1);
    CHECK(opens == 0 && creates == 1 && callbacks == 0);
    puts("Metal zero-work CLI: metadata required, no GPU creation or logits callback PASS");
    return 0;
}
