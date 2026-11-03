#include "chat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

/* Isolate framing from BPE: tokenizer's independent suite tests actual encoding. */
int lunelette_tokenizer_encode(const lunelette_tokenizer *t, const unsigned char *text,
    size_t length, uint32_t **ids, size_t *count, char *error, size_t capacity)
{
    (void)t;
    (void)error;
    (void)capacity;
    *ids = NULL;
    *count = 0;
    if (length > 4096 || (length && text[0] == 255)) {
        return 0;
    }
    *ids = malloc((length + 1) * sizeof(**ids));
    CHECK(*ids);
    for (size_t i = 0; i < length; i++) {
        (*ids)[i] = 1000 + text[i];
    }
    *count = length;
    return 1;
}

int main(void)
{
    /* Opaque handle is never dereferenced by the stub. */
    char handle;
    const lunelette_tokenizer *t = (const lunelette_tokenizer *)&handle;
    int64_t ids[256];
    size_t count = 999;
    char error[256];
    CHECK(lunelette_chat_prompt(t, "u", "s", 0, ids, &count, error, sizeof error) == 0);
    int64_t expected[] = {1, 17, 1115, 18, 3, 1117, 4};
    CHECK(count == 7 && !memcmp(ids, expected, sizeof expected));
    CHECK(lunelette_chat_prompt(t, "", NULL, 0, ids, &count, error, sizeof error) == 0);
    CHECK(count == 3 && ids[0] == 1 && ids[1] == 3 && ids[2] == 4);
    CHECK(lunelette_chat_prompt(t, "", "", 0, ids, &count, error, sizeof error) == 0);
    int64_t empty[] = {1, 17, 18, 3, 4};
    CHECK(count == 5 && !memcmp(ids, empty, sizeof empty));
    CHECK(lunelette_chat_prompt(t, "x", NULL, 1, ids, &count, error, sizeof error) == 0);
    CHECK(count == 2 && ids[1] == 1120);
    CHECK(lunelette_chat_prompt(t, "x", "", 1, ids, &count, error, sizeof error) == -1);
    CHECK(count == 2 && ids[1] == 1120);
    char text[258];
    memset(text, 'x', 255);
    text[255] = 0;
    CHECK(lunelette_chat_prompt(t, text, NULL, 1, ids, &count, error, sizeof error) == 0);
    CHECK(count == 256);
    int64_t saved[256];
    memcpy(saved, ids, sizeof ids);
    CHECK(lunelette_chat_prompt(t, text, NULL, 0, ids, &count, error, sizeof error) == -1);
    CHECK(count == 256 && !memcmp(saved, ids, sizeof ids));
    CHECK(lunelette_chat_prompt(t, "\xff", NULL, 0, ids, &count, error, sizeof error) == -1);
    CHECK(lunelette_chat_prompt(t, "u", "\xff", 0, ids, &count, error, sizeof error) == -1);
    CHECK(lunelette_chat_prompt(NULL, "u", NULL, 0, ids, &count, error, sizeof error) == -1);
    CHECK(count == 256 && !memcmp(saved, ids, sizeof ids));
    puts("chat framing tests passed");
    return 0;
}
