/* Model-free runtime stubs; optional existing Tekken tokenizer, never model weights. */
#define _POSIX_C_SOURCE 200809L
#include "support/native_runtime.h"
#include "../service/native.c"

static uint32_t read_u32(FILE *file)
{
    unsigned char b[4];
    CHECK(fread(b, 1, 4, file) == 4);
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

static void record(FILE *file, unsigned index, const unsigned *ids, unsigned length,
                   unsigned forced)
{
    CHECK(read_u32(file) == index);
    CHECK(read_u32(file) == length);
    CHECK(read_u32(file) == forced);
    CHECK(read_u32(file) == 22177);
    for (unsigned i = 0; i < length; i++) {
        CHECK(read_u32(file) == ids[i]);
    }
    for (unsigned i = 0; i < 131072; i++) {
        CHECK(read_u32(file) == ((i == 22177 || i == 22178) ? 0x3f800000u : 0));
    }
}

static void policy_replay(const char *tokenizer, unsigned policy)
{
    char path[] = "/tmp/lunelette-replay-native-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(!close(fd));
    char *args[] = {"test", "--model", (char *)tokenizer, "--tokenizer", (char *)tokenizer,
        "--strategy", "resident-fp32", "--prefill", "custom", "--port", "0",
        "--prefix-reuse", policy == 1 ? "on" : "off",
        "--evaluation-mode", "controlled-replay", "--replay-logits", path};
    service_generator g;
    unsigned port, before = creates;
    CHECK(!service_native_open(17, args, &g, &port)); /* Exclusive, even empty. */
    CHECK(creates == before);
    CHECK(!unlink(path));
    CHECK(!service_native_open(15, args, &g, &port)); /* Missing paired option. */
    CHECK(access(path, F_OK));
    char *tool_args[19];
    memcpy(tool_args, args, sizeof(args));
    tool_args[17] = "--tools";
    tool_args[18] = "calculator-v13";
    CHECK(!service_native_open(19, tool_args, &g, &port));
    CHECK(access(path, F_OK));
    args[2] = "/nonexistent/lunelette-replay-test-model";
    CHECK(!service_native_open(17, args, &g, &port));
    FILE *failed = fopen(path, "rb");
    CHECK(failed && !fseek(failed, 0, SEEK_END) && ftell(failed) == 16);
    CHECK(!fclose(failed));
    CHECK(!service_native_open(17, args, &g, &port)); /* Preserve failed evidence. */
    CHECK(!unlink(path));
    args[2] = (char *)tokenizer;
    CHECK(service_native_open(17, args, &g, &port));
    CHECK(g.replay_next && strstr(g.info, "native_controlled_replay"));
    CHECK(strstr(g.info, "\"replay_capture_limit\":16"));
    unsigned ids[256] = {1, 3, 37133, 4};
    g.begin(g.context);
    CHECK(g.materialize(g.context, ids, 4) == 1);
    CHECK(g.save(g.context) == 1);
    CHECK(g.replay_next(g.context, 999) == -1);
    CHECK(g.replay_next(g.context, 131072) == -1);
    CHECK(g.replay_next(g.context, 53156) == 53156);
    CHECK(g.replay_next(g.context, 2) == -1); /* Pending not yet consumed. */
    ids[4] = 53156;
    CHECK(g.materialize(g.context, ids, 5) == 1);
    native_service *n = g.context;
    CHECK(n->runtime->ids[4] == 53156 && n->runtime->count == 5);
    CHECK(g.replay_next(g.context, 2) == 2);
    ids[5] = 2;
    CHECK(g.materialize(g.context, ids, 6) == 1);
    CHECK(n->runtime->ids[5] == 2);
    CHECK(g.restore(g.context) == 1);
    g.begin(g.context);
    CHECK(g.materialize(g.context, ids, 4) == 1);
    CHECK(g.replay_next(g.context, 53156) == 53156);
    CHECK(n->replay_index == 3);
    FILE *file = fopen(path, "rb");
    CHECK(file);
    char magic[8];
    CHECK(fread(magic, 1, 8, file) == 8 && !memcmp(magic, "LNRPLY1\0", 8));
    CHECK(read_u32(file) == 131072 && read_u32(file) == 16);
    record(file, 0, ids, 4, 53156);
    record(file, 1, ids, 5, 2);
    record(file, 2, ids, 4, 53156);
    CHECK(fgetc(file) == EOF && !ferror(file));
    CHECK(!fclose(file));
    g.reset(g.context);
    CHECK(n->replay_index == 3);
    CHECK(g.materialize(g.context, ids, 4));
    logits[131071] = NAN;
    CHECK(g.replay_next(g.context, 53156) == -1 && n->replay_index == 3);
    g.reset(g.context);
    for (unsigned i = 3; i < 16; i++) {
        CHECK(g.materialize(g.context, ids, 4));
        CHECK(g.replay_next(g.context, 2) == 2);
        g.reset(g.context);
    }
    CHECK(g.materialize(g.context, ids, 4));
    CHECK(g.replay_next(g.context, 2) == -1);
    /* A real stdio write failure is sticky across reset; no publication/index advance. */
    CHECK(!fclose(n->replay_file));
    n->replay_file = fopen(path, "rb");
    CHECK(n->replay_file);
    n->replay_index = 0; /* Test-only fault injection after the independent cap check. */
    CHECK(g.replay_next(g.context, 2) == -1);
    CHECK(n->replay_failed && !n->replay_index && !n->pending);
    g.reset(g.context);
    CHECK(g.materialize(g.context, ids, 4));
    CHECK(g.replay_next(g.context, 2) == -1);
    g.destroy(g.context);
    CHECK(!unlink(path));
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    CHECK(sizeof(service_session) + sizeof(native_service) < 8u * 1024 * 1024);
    policy_replay(argv[1], 1);
    policy_replay(argv[1], 2);
    puts("native controlled replay capture/forced-prefix/checkpoint tests passed (stub runtime)");
    return 0;
}
