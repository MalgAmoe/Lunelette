/* Private algorithm tests: no model, tokenizer file, Python, or network. */
#include <stdlib.h>
static int reject_allocation;
static void *test_malloc(size_t size)
{
    if (reject_allocation && --reject_allocation == 0) {
        return NULL;
    }
    return malloc(size);
}
#define malloc test_malloc
#include "../src/tokenizer.c"
#undef malloc

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static void add(lunelette_tokenizer *t, unsigned rank, const char *text)
{
    t->tokens[rank].bytes = (unsigned char *)text;
    t->tokens[rank].length = strlen(text);
    unsigned slot = hash_bytes((const unsigned char *)text, strlen(text));
    while (t->hash[slot]) {
        slot = (slot + 1) & (TK_HASH_SIZE - 1);
    }
    t->hash[slot] = rank + 1;
}

/* Frozen pre-optimization merge rule, deliberately independent of heap links. */
static size_t legacy(const lunelette_tokenizer *t, const unsigned char *text,
                     size_t length, uint32_t *ids)
{
    size_t boundaries[65];
    unsigned rank = lookup(t, text, length);
    if (rank != TK_NONE) {
        ids[0] = rank + 1000;
        return 1;
    }
    for (size_t i = 0; i <= length; i++) {
        boundaries[i] = i;
    }
    size_t parts = length;
    while (parts > 1) {
        unsigned best_rank = TK_NONE;
        size_t best = 0;
        for (size_t i = 0; i + 1 < parts; i++) {
            rank = lookup(t, text + boundaries[i], boundaries[i + 2] - boundaries[i]);
            if (rank < best_rank) {
                best_rank = rank;
                best = i;
            }
        }
        if (best_rank == TK_NONE) {
            break;
        }
        memmove(boundaries + best + 1, boundaries + best + 2,
                (parts - best - 1) * sizeof(*boundaries));
        parts--;
    }
    for (size_t i = 0; i < parts; i++) {
        ids[i] = lookup(t, text + boundaries[i], boundaries[i + 1] - boundaries[i]) + 1000;
    }
    return parts;
}

int main(void)
{
    lunelette_tokenizer *t = calloc(1, sizeof(*t));
    CHECK(t);
    add(t, 0, "a");
    add(t, 1, "b");
    add(t, 2, "aa");
    add(t, 3, "ab");
    add(t, 4, "ba");
    add(t, 5, "bb");
    add(t, 6, "aaa");
    add(t, 7, "aab");
    add(t, 8, "abba");
    add(t, 9, "aaaa");
    add(t, 10, " ");
    unsigned char text[64];
    uint32_t expected[64], actual[64];
    /* Exhaustive ties, changing neighbours, and removed/right-edge entries. */
    for (size_t n = 1; n <= 14; n++) {
        for (unsigned bits = 0; bits < (1u << n); bits++) {
            for (size_t i = 0; i < n; i++) {
                text[i] = (bits >> i) & 1 ? 'a' : 'b';
            }
            size_t count = 0;
            size_t want = legacy(t, text, n, expected);
            CHECK(bpe(t, text, n, actual, &count));
            CHECK(count == want && !memcmp(actual, expected, want * sizeof(*actual)));
        }
    }
    /* Non-monotonic ranks exercise repairs in both heap directions. */
    const char *tokens[] = {"a", "b", "aa", "ab", "ba", "bb", "aaa", "aab", "abba", "aaaa", " "};
    unsigned random = 72619;
    for (size_t trial = 0; trial < 32; trial++) {
        unsigned ranks[11];
        for (unsigned i = 0; i < 11; i++) {
            ranks[i] = i;
        }
        for (unsigned i = 10; i; i--) {
            random = random * 1664525u + 1013904223u;
            unsigned j = random % (i + 1);
            unsigned swap = ranks[i];
            ranks[i] = ranks[j];
            ranks[j] = swap;
        }
        memset(t, 0, sizeof(*t));
        for (unsigned i = 0; i < 11; i++) {
            add(t, ranks[i], tokens[i]);
        }
        for (size_t n = 1; n <= 8; n++) {
            for (unsigned bits = 0; bits < (1u << n); bits++) {
                for (size_t i = 0; i < n; i++) {
                    text[i] = (bits >> i) & 1 ? 'a' : 'b';
                }
                size_t count = 1;
                actual[0] = 123456;
                size_t want = legacy(t, text, n, expected);
                CHECK(bpe(t, text, n, actual, &count));
                CHECK(count == want + 1 && actual[0] == 123456);
                CHECK(!memcmp(actual + 1, expected, want * sizeof(*actual)));
            }
        }
    }
    uint32_t *ids = (uint32_t *)1;
    size_t count = 99;
    char error[128];
    reject_allocation = 1;
    CHECK(!bpe(t, (const unsigned char *)"ababab", 6, actual, &count));
    /* Result, first scratch, then later scratch after partial output. */
    for (int failure = 1; failure <= 3; failure++) {
        reject_allocation = failure;
        CHECK(!lunelette_tokenizer_encode(t, (const unsigned char *)"ababab ababab", 13,
                                          &ids, &count, error, sizeof(error)));
        CHECK(!ids && !count);
        CHECK(!strcmp(error, "tokenizer encode: allocation failed"));
    }
    reject_allocation = 0;
    free(t); /* Synthetic token bytes are borrowed literals. */
    puts("synthetic BPE differential tests passed (49086 cases; allocation failure reset)");
    return 0;
}
