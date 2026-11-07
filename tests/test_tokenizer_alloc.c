/* Dedicated CPU regression: link src/tokenizer_json.c, not src/tokenizer.c. */
#include <stdio.h>
#include <stdlib.h>
static size_t allocation, fail_at, live;
static int tracking;
static void *test_malloc(size_t size)
{
    if (++allocation == fail_at) {
        return NULL;
    }
    void *p = malloc(size);
    if (p && tracking) {
        live++;
    }
    return p;
}
static void test_free(void *p)
{
    if (p && tracking) {
        live--;
    }
    free(p);
}
#define malloc test_malloc
#define free test_free
#include "../src/tokenizer.c"
#undef malloc
#undef free
#define main tokenizer_contract_main
#include "test_tokenizer.c"
#undef main
int main(void)
{
    synthetic_file(0);
    lunelette_tokenizer *t;
    char error[256];
    CHECK(lunelette_tokenizer_open(scratch, &t, error, sizeof error));
    CHECK(!remove(scratch));
    unsigned char text[32768];
    memset(text, 'a', sizeof text);
    tracking = 1;
    size_t baseline = live;
    for (size_t failure = 1; failure <= 3; failure++) {
        allocation = 0;
        fail_at = failure;
        uint32_t *ids = (void *)1;
        size_t count = 99;
        CHECK(!lunelette_tokenizer_encode_with_limit(t, text, sizeof text, 32768,
            &ids, &count, error, sizeof error));
        CHECK(!ids && !count && live == baseline);
        CHECK(!strcmp(error, "tokenizer encode: allocation failed"));
    }
    fail_at = 0;
    uint32_t *ids = NULL;
    size_t count = 0;
    CHECK(lunelette_tokenizer_encode_with_limit(t, text, sizeof text, 32768,
        &ids, &count, error, sizeof error));
    CHECK(ids && count && live == 1);
    test_free(ids);
    CHECK(!live);
    text[sizeof text - 1] = 0x80;
    CHECK(!lunelette_tokenizer_encode_with_limit(t, text, sizeof text, 32768,
        &ids, &count, error, sizeof error));
    CHECK(!ids && !count && !live);
    tracking = 0;
    lunelette_tokenizer_close(t);
    puts("tokenizer long allocation cleanup PASS");
    return 0;
}
