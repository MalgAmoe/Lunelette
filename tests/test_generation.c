#include "generation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

struct fake {
    size_t calls;
    size_t initial;
    int mode;
};

static int next(void *opaque, const int64_t *ids, size_t count, float *logits,
                char *error, size_t capacity)
{
    struct fake *fake = opaque;
    (void)error;
    (void)capacity;
    CHECK(count == fake->initial + fake->calls);
    CHECK(count <= 256);
    if (fake->calls) {
        CHECK(ids[count - 1] == 42);
    }
    fake->calls++;
    if (fake->mode == 1 && fake->calls == 2) {
        return -1;
    }
    for (size_t i = 0; i < LUNELETTE_VOCAB_SIZE; i++) {
        logits[i] = -10;
    }
    logits[42] = -1;
    logits[43] = -1; /* Negative scores and lowest-ID exact tie. */
    if (fake->mode == 2 && fake->calls == 2) {
        logits[2] = 0;
    }
    if (fake->mode >= 3) {
        logits[131071] = fake->mode == 3 ? NAN : fake->mode == 4 ? INFINITY : -INFINITY;
    }
    return 0;
}

int main(void)
{
    int64_t ids[257] = {2}; /* Prompt EOS does not stop generation. */
    char error[256];
    lunelette_generation_config config = LUNELETTE_GENERATION_DEFAULT;
    CHECK(config.eos_token == 2);
    config.max_new_tokens = 3;
    lunelette_generation_result result, sentinel;
    memset(&sentinel, 0x55, sizeof sentinel);
    struct fake fake = {0, 1, 0};
#define RUN(n) lunelette_generate_with_logits(next, &fake, ids, n, &config, &result, error, sizeof error)
    CHECK(!RUN(1));
    CHECK(result.count == 3 && result.tokens[0] == 42 && result.stop == LUNELETTE_STOP_LIMIT);
    fake = (struct fake){0, 1, 2};
    CHECK(!RUN(1));
    CHECK(result.count == 2 && result.tokens[1] == 2 && result.stop == LUNELETTE_STOP_EOS);
    config.max_new_tokens = 2;
    fake = (struct fake){0, 254, 2};
    CHECK(!RUN(254));
    CHECK(result.stop == LUNELETTE_STOP_EOS && result.count == 2);
    config.max_new_tokens = 3;
    fake = (struct fake){0, 255, 0};
    CHECK(!RUN(255));
    CHECK(result.count == 1 && result.stop == LUNELETTE_STOP_CONTEXT && fake.calls == 1);
    fake = (struct fake){0, 256, 0};
    CHECK(!RUN(256));
    CHECK(!result.count && !fake.calls && result.stop == LUNELETTE_STOP_CONTEXT);
    result = sentinel;
    CHECK(RUN(257) == -1 && !memcmp(&result, &sentinel, sizeof result));
    CHECK(RUN(0) == -1);
    for (int mode = 1; mode <= 5; mode++) {
        if (mode == 2) {
            continue;
        }
        fake = (struct fake){0, 1, mode};
        result = sentinel;
        CHECK(RUN(1) == -1 && !memcmp(&result, &sentinel, sizeof result));
        CHECK(error[0]);
    }
    config.max_new_tokens = 256;
    fake = (struct fake){0, 1, 0};
    CHECK(!RUN(1));
    CHECK(result.count == 255 && fake.calls == 255 && result.stop == LUNELETTE_STOP_CONTEXT);
    config.max_new_tokens = 1;
    fake = (struct fake){0, 255, 0};
    CHECK(!RUN(255) && result.count == 1 && result.stop == LUNELETTE_STOP_LIMIT);
    config.max_new_tokens = SIZE_MAX;
    CHECK(RUN(1) == -1);
    config.max_new_tokens = 0;
    fake = (struct fake){0, 1, 0};
    CHECK(!RUN(1) && !result.count && !fake.calls && result.stop == LUNELETTE_STOP_LIMIT);
    ids[0] = -1;
    CHECK(RUN(1) == -1);
    ids[0] = 131072;
    CHECK(RUN(1) == -1);
    ids[0] = 0;
    config.max_new_tokens = 257;
    CHECK(RUN(1) == -1);
    config.max_new_tokens = 1;
    config.eos_token = -1;
    CHECK(RUN(1) == -1);
    config.eos_token = 131072;
    CHECK(RUN(1) == -1);
    config.eos_token = 42;
    CHECK(!RUN(1) && result.count == 1 && result.stop == LUNELETTE_STOP_EOS);
    CHECK(lunelette_generate_with_logits(NULL, NULL, ids, 1, &config, &result, NULL, 0) == -1);
    CHECK(lunelette_generate_with_logits(next, &fake, NULL, 1, &config, &result, NULL, 0) == -1);
    CHECK(lunelette_generate_with_logits(next, &fake, ids, 1, NULL, &result, NULL, 0) == -1);
    CHECK(lunelette_generate_with_logits(next, &fake, ids, 1, &config, NULL, NULL, 0) == -1);
    puts("generation: EOS, limits 256/257, ties, finite logits, atomic errors PASS");
    return 0;
}
