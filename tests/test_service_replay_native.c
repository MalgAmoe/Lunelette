/* Model-free runtime stubs; optional existing Tekken tokenizer, never model weights. */
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "support/native_runtime.h"
#include "../service/native.c"
#include <limits.h>
#include <errno.h>

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
    unsigned emitted_before = n->emitted;
    long capture_before = ftell(n->replay_file);
    float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (unsigned i = 0; i < 3; i++) {
        logits[131071] = nonfinite[i];
        CHECK(g.replay_next(g.context, 53156) == -1 && n->replay_index == 3);
        CHECK(!n->pending && n->emitted == emitted_before && !n->replay_failed);
        CHECK(ftell(n->replay_file) == capture_before);
    }
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

/* Frozen pre-P5 semantic adapter: full-prefix validation and one-token
 * generation, with an independent strict-greater finite scan as oracle. */
static int baseline_select(const native_service *n)
{
    if (n->pending || !n->length || n->length > n->capacity ||
        (n->capacity != 256 && n->capacity != 4352)) {
        return -1;
    }
    for (unsigned i = 0; i < n->length; i++) {
        if (n->prefix[i] >= 131072) {
            return -1;
        }
    }
    if (n->length == n->capacity || !n->logits) {
        return -1;
    }
    unsigned best = 0;
    for (unsigned i = 0; i < 131072; i++) {
        if (!isfinite(n->logits[i])) {
            return -1;
        }
        if (n->logits[i] > n->logits[best]) {
            best = i;
        }
    }
    return (int)best;
}

typedef struct {
    size_t bytes, limit;
} failing_capture;

static int fail_write(void *context, const char *bytes, int count)
{
    failing_capture *sink = context;
    (void)bytes;
    if ((size_t)count > sink->limit - sink->bytes) {
        errno = EIO;
        return -1;
    }
    sink->bytes += (size_t)count;
    return count;
}

static void capture_failures(void)
{
    native_service *n = calloc(1, sizeof(*n));
    CHECK(n);
    n->capacity = 256;
    n->length = 1;
    n->logits = logits;
    memset(logits, 0, sizeof(logits));
    logits[42] = 1;
    char *buffer = malloc(1024 * 1024);
    CHECK(buffer);
    for (unsigned mode = 0; mode < 2; mode++) {
        failing_capture sink = {0, mode ? 0 : 32};
        n->replay_file = funopen(&sink, NULL, fail_write, NULL, NULL);
        CHECK(n->replay_file);
        if (!mode) {
            CHECK(!setvbuf(n->replay_file, NULL, _IONBF, 0));
        } else {
            CHECK(!setvbuf(n->replay_file, buffer, _IOFBF, 1024 * 1024));
        }
        n->replay_failed = 0;
        n->replay_index = 3;
        n->emitted = 77;
        CHECK(native_replay_next(n, 1000) == -1);
        CHECK(n->replay_failed && n->replay_index == 3 && !n->pending && n->emitted == 77);
        CHECK(mode || sink.bytes > 0); /* A partial record is retained. */
        CHECK(native_replay_next(n, 1000) == -1 && n->replay_index == 3);
        fclose(n->replay_file); /* Deliberately failed stream. */
    }
    free(buffer);
    free(n);
}

static void selection_contract(void)
{
    native_service *n = calloc(1, sizeof(*n));
    CHECK(n);
    n->capacity = 256;
    n->length = 1;
    n->logits = logits;
    for (unsigned step = 0; step < 20; step++) {
        for (unsigned i = 0; i < 131072; i++) {
            logits[i] = -10;
        }
        unsigned winner = step == 19 ? 2 : 1000 + step;
        logits[winner] = step % 2 ? -0.0f : -1.0f;
        logits[winner + 1] = step % 2 ? 0.0f : -1.0f;
        int expected = baseline_select(n);
        CHECK(native_next(n) == expected && expected == (int)winner);
        CHECK(n->pending && n->emitted == winner);
        CHECK(native_next(n) == -1);
        n->prefix[n->length++] = n->emitted; /* Synthetic materialize. */
        n->pending = 0;
    }
    for (unsigned cap = 256; cap <= 4352; cap += 4096) {
        n->capacity = cap;
        unsigned lengths[] = {0, cap, cap + 1, UINT_MAX};
        for (unsigned i = 0; i < 4; i++) {
            n->length = lengths[i];
            CHECK(native_next(n) == -1 && !n->pending && n->emitted == 2);
        }
        n->length = cap - 1;
        int expected = baseline_select(n);
        CHECK(native_next(n) == expected);
        n->pending = 0;
        n->prefix[cap - 2] = 131072;
        CHECK(native_next(n) == -1 && !n->pending);
        n->prefix[cap - 2] = 0;
    }
    n->length = 1;
    n->capacity = UINT_MAX;
    CHECK(native_next(n) == -1);
    n->capacity = 256;
    n->prefix[0] = UINT_MAX;
    CHECK(native_next(n) == -1);
    n->prefix[0] = 0;
    n->logits = NULL;
    CHECK(native_next(n) == -1);
    n->logits = logits;
    float bad[] = {NAN, INFINITY, -INFINITY};
    for (unsigned i = 0; i < 3; i++) {
        logits[131071] = bad[i];
        CHECK(native_next(n) == -1 && !n->pending && n->emitted == 2);
    }
    free(n);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    selection_contract();
    capture_failures();
    CHECK(sizeof(service_session) + sizeof(native_service) < 8u * 1024 * 1024);
    policy_replay(argv[1], 1);
    policy_replay(argv[1], 2);
    puts("native controlled replay capture/forced-prefix/checkpoint tests passed (stub runtime)");
    return 0;
}
