/* Dedicated CPU-only allocation-failure regression; compile with -Iinclude -lm. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static size_t allocation, fail_at, live;
static void *test_malloc(size_t size)
{
    if (++allocation == fail_at) {
        return NULL;
    }
    void *p = malloc(size);
    if (p) {
        live++;
    }
    return p;
}
static void test_free(void *p)
{
    if (p) {
        CHECK(live);
        live--;
    }
    free(p);
}
#define malloc test_malloc
#define free test_free
#include "../src/generation.c"
#undef malloc
#undef free
static int callback(void *ctx, const int64_t *ids, size_t count, float *logits,
                    char *error, size_t capacity)
{
    (void)ctx;
    (void)ids;
    (void)count;
    (void)logits;
    (void)error;
    (void)capacity;
    return -1;
}
int main(void)
{
    int64_t prompt = 1;
    lunelette_generation_config config = LUNELETTE_GENERATION_DEFAULT;
    lunelette_generation_result result, sentinel;
    memset(&sentinel, 0x55, sizeof sentinel);
    for (size_t failure = 1; failure <= 3; failure++) {
        allocation = 0;
        fail_at = failure;
        result = sentinel;
        CHECK(lunelette_generate_with_logits_capacity(callback, NULL, &prompt, 1,
            4352, &config, &result, NULL, 0) == -1);
        CHECK(!memcmp(&result, &sentinel, sizeof result));
        CHECK(!live);
    }
    puts("generation allocation cleanup PASS");
    return 0;
}
