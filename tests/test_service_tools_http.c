#define _POSIX_C_SOURCE 200809L
/* SCRIPTED socket/client plumbing only: no inference or numerical evidence. */
#include "tokenizer.h"
#define main oldmain
#include "test_service_native.c"
#undef main

static unsigned script[256], script_count, script_cursor;
static int (*actual_next)(void *);

static void encode_script(lunelette_tokenizer *tokenizer, const char *text)
{
    uint32_t *ids = NULL;
    size_t count = 0;
    char error[512];
    CHECK(lunelette_tokenizer_encode(tokenizer, (const unsigned char *)text,
                                    strlen(text), &ids, &count, error, sizeof(error)));
    CHECK(count <= 256 - script_count);
    for (size_t i = 0; i < count; i++) {
        script[script_count++] = ids[i];
    }
    free(ids);
}

static int scripted_next(void *context)
{
    CHECK(script_cursor < script_count);
    memset(logits, 0, sizeof(logits));
    logits[script[script_cursor++]] = 2;
    return actual_next(context);
}

static int scripted_open(int argc, char **argv, service_generator *g, unsigned *port)
{
    if (!service_native_open(argc, argv, g, port)) {
        return 0;
    }
    const char *path = NULL;
    for (int i = 1; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--tokenizer")) {
            path = argv[i + 1];
        }
    }
    CHECK(path);
    lunelette_tokenizer *tokenizer = NULL;
    char error[512];
    CHECK(lunelette_tokenizer_open(path, &tokenizer, error, sizeof(error)));
    /* Reset exactly once at launch, never per generation or session reset. */
    script_count = script_cursor = 0;
    script[script_count++] = 9;
    encode_script(tokenizer, "calculator");
    script[script_count++] = 32;
    encode_script(tokenizer, "{\"operation\": \"add\", \"a\": 19, \"b\": 24}");
    script[script_count++] = 2;
    encode_script(tokenizer, "43");
    script[script_count++] = 2;
    lunelette_tokenizer_close(tokenizer);
    actual_next = g->next;
    g->next = scripted_next;
    fprintf(stderr, "SCRIPTED CPU stub: not inference or model evidence\n");
    return 1;
}

#define SERVICE_NATIVE
#define service_native_open scripted_open
#include "../service/http.c"
#undef service_native_open
