/* Reuse link-time CPU runtime stubs, not an inference engine or model file. */
#include "tokenizer.h"
#define lunelette_metal_create_configured stub_create_configured
#include "support/native_runtime.h"
#undef lunelette_metal_create_configured
static lunelette_metal *captured;
lunelette_metal *lunelette_metal_create_configured(const char *path,
    const lunelette_metal_options *options, const lunelette_metal_limits *limits,
    char *error, size_t capacity)
{
    captured = stub_create_configured(path, options, limits, error, capacity);
    return captured;
}

static unsigned walk(service_generator *g, unsigned *ids, unsigned length)
{
    unsigned pending = 0, cursor = 0;
    int status;
    do {
        status = g->materialize_step(g->context, ids, length, &cursor);
        CHECK(status != SERVICE_MATERIALIZE_ERROR);
        CHECK(cursor == captured->count && cursor <= length);
        for (unsigned i = 0; i < cursor; i++) {
            CHECK(captured->ids[i] == ids[i]);
        }
        service_work work;
        g->work(g->context, &work);
        CHECK(work.full_committed + work.suffix_committed + work.decode_committed <= captured->block);
        pending += work.pending_committed;
    } while (status == SERVICE_MATERIALIZE_MORE);
    CHECK(cursor == length);
    return pending;
}

static void continuation(char *tokenizer, char *policy, char *chunk)
{
    char *args[] = {"test", "--model", tokenizer, "--tokenizer", tokenizer,
        "--strategy", "resident-fp32", "--context", "4352", "--prefill-chunk", chunk,
        "--prefill", "custom", "--port", "0", "--tools", "calculator-v13",
        "--prefix-reuse", policy};
    service_generator g;
    unsigned port;
    CHECK(service_native_open(policy ? 19 : 17, args, &g, &port));
    lunelette_tokenizer *encoder = NULL;
    char error[512];
    CHECK(lunelette_tokenizer_open(tokenizer, &encoder, error, sizeof(error)));
    uint32_t *name_ids = NULL, *argument_ids = NULL;
    size_t name_count = 0, argument_count = 0;
    const char *arguments = "{\"operation\": \"add\", \"a\": 19, \"b\": 24}";
    CHECK(lunelette_tokenizer_encode_with_limit(encoder, (const unsigned char *)"calculator",
        10, 4096, &name_ids, &name_count, error, sizeof(error)));
    CHECK(lunelette_tokenizer_encode_with_limit(encoder, (const unsigned char *)arguments,
        strlen(arguments), 4096, &argument_ids, &argument_count, error, sizeof(error)));
    unsigned call[256], call_length = 0;
    CHECK(name_count + argument_count + 3 <= 256);
    call[call_length++] = 9;
    for (size_t i = 0; i < name_count; i++) {
        call[call_length++] = name_ids[i];
    }
    call[call_length++] = 32;
    for (size_t i = 0; i < argument_count; i++) {
        call[call_length++] = argument_ids[i];
    }
    call[call_length++] = 2;
    free(name_ids);
    free(argument_ids);
    lunelette_tokenizer_close(encoder);
    for (unsigned trial = 0; trial < 2; trial++) {
        unsigned ids[4353], length = 4352;
        CHECK(g.frame(g.context, (const unsigned char *)"Calculate 19 + 24.", 18, 1, ids, &length));
        while (length < captured->block + 1) {
            ids[length++] = 42;
        }
        g.begin(g.context);
        CHECK(walk(&g, ids, length) == 0);
        /* Each selected token is authoritative; EOS alone stays pending. */
        for (unsigned i = 0; i < call_length; i++) {
            memset(logits, 0, sizeof(logits));
            logits[call[i]] = 2;
            CHECK(g.next(g.context) == (int)call[i]);
            ids[length++] = call[i];
            if (call[i] != 2) {
                CHECK(walk(&g, ids, length) == 1);
            }
        }
        CHECK(captured->count == length - 1);
        unsigned result_length = 4352 - length;
        CHECK(g.tool_frame(g.context, (const unsigned char *)"43", 2, ids + length, &result_length));
        CHECK(result_length == 4 && ids[length] == 7 && ids[length + 1] == 1052 &&
              ids[length + 2] == 1051 && ids[length + 3] == 8);
        length += result_length;
        g.begin(g.context);
        CHECK(walk(&g, ids, length) == 1);
        unsigned before = prefills + appends + decodes;
        CHECK(walk(&g, ids, length) == 0);
        CHECK(walk(&g, ids, length) == 0);
        CHECK(before == prefills + appends + decodes);
        while (length < 4352) {
            ids[length++] = 42;
        }
        CHECK(walk(&g, ids, length) == 0);
        ids[length] = 42;
        unsigned cursor;
        CHECK(g.materialize_step(g.context, ids, length + 1, &cursor) == SERVICE_MATERIALIZE_ERROR);
        CHECK(captured->count == 4352);
        g.reset(g.context);
        CHECK(captured->count == 0);
    }
    g.destroy(g.context);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    const char *policies[] = {"on", "off", NULL};
    for (unsigned i = 0; i < 3; i++) {
        continuation(argv[1], (char *)policies[i], "128");
        continuation(argv[1], (char *)policies[i], "256");
    }
    /* Tokenizer only; the stub model path is never loaded as weights. */
    char *arguments[] = {"native-tools-test", "--model", argv[1], "--tokenizer", argv[1],
        "--strategy", "resident-fp32", "--context", "4352", "--prefill-chunk", "256",
        "--prefill", "custom", "--port", "0", "--tools", "calculator-v13",
        "--prefix-reuse", "on"};
    service_generator g;
    unsigned port = 123;
    CHECK(service_native_open(19, arguments, &g, &port));
    CHECK(port == 0 && g.native_tools && g.tool_frame);
    CHECK(strstr(g.info, "\"tools\":true,\"checkpoints\":false"));
    CHECK(strstr(g.info, "\"conversation_protocol\":\"native-calculator-v13\""));
    CHECK(!g.save(g.context) && !g.restore(g.context));
    unsigned ids[4352], count = 4352;
    CHECK(g.frame(g.context, (const unsigned char *)"Calculate 19 + 24.", 18,
                  1, ids, &count));
    CHECK(ids[0] == 1 && ids[1] == 5 && ids[count - 1] == 4);
    unsigned result[4352], length = 4352;
    CHECK(g.tool_frame(g.context, (const unsigned char *)"43", 2, result, &length));
    CHECK(length == 4 && result[0] == 7 && result[1] == 1052 &&
          result[2] == 1051 && result[3] == 8);
    g.destroy(g.context);
    arguments[16] = "wrong";
    CHECK(!service_native_open(19, arguments, &g, &port));
    arguments[16] = "calculator-v13";
    char *duplicate[21];
    memcpy(duplicate, arguments, sizeof(arguments));
    duplicate[19] = "--tools";
    duplicate[20] = "calculator-v13";
    CHECK(!service_native_open(21, duplicate, &g, &port));
    CHECK(service_native_open(15, arguments, &g, &port));
    CHECK(!g.native_tools && !g.tool_frame && strstr(g.info, "\"tools\":false"));
    g.destroy(g.context);
    puts("native tools startup/framing/continuation/reset CPU stub contract passed (no inference)");
    return 0;
}
