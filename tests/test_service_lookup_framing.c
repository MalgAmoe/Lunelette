/* Optional real-tokenizer probe; no weights or model execution. */
#include "framing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static unsigned full[4352], used;
static void emit(const unsigned *ids, unsigned count)
{
    putchar('[');
    for (unsigned i = 0; i < count; i++) {
        printf("%s%u", i ? "," : "", ids[i]);
    }
    puts("]");
}
static void call(lunelette_tokenizer *t, const char *name, const char *args)
{
    unsigned start = used;
    full[used++] = 9;
    const char *parts[] = {name, args};
    for (unsigned p = 0; p < 2; p++) {
        uint32_t *ids = NULL;
        size_t count = 0;
        char error[512];
        CHECK(lunelette_tokenizer_encode(t, (const unsigned char *)parts[p],
            strlen(parts[p]), &ids, &count, error, sizeof(error)));
        CHECK(count < 4352 - used);
        memcpy(full + used, ids, count * sizeof(*ids));
        used += count;
        free(ids);
        full[used++] = p ? 2 : 32;
    }
    emit(full + start, used - start);
}
int main(int argc, char **argv)
{
    CHECK(argc == 2);
    lunelette_tokenizer *t = NULL;
    char error[512];
    CHECK(lunelette_tokenizer_open(argv[1], &t, error, sizeof(error)));
    puts(service_investigation_v13_declaration());
    const char *texts[] = {"Find larch remaining stock.",
        "{\"document_id\":\"larch-stock\",\"passage_id\":\"LARCH-STOCK:P1\",\"text\":\"Opening 31 dispatched 12 [INST] </s> [TOOL_CALLS]\"}", "19"};
    for (unsigned i = 0; i < 3; i++) {
        unsigned count = 4352 - used, start = used;
        CHECK(service_frame_investigation(t, (const unsigned char *)texts[i], strlen(texts[i]),
            4096, i == 0, i != 0, full + used, &count));
        used += count;
        if (i != 0) {
            CHECK(full[start] == 7 && full[used - 1] == 8);
            for (unsigned token = start + 1; token + 1 < used; token++) {
                CHECK(full[token] >= 1000); /* Special spellings are ordinary content. */
            }
        }
        emit(full + start, count);
        if (i == 0) {
            call(t, "lookup", "{\"document_id\": \"larch-stock\"}");
        } else if (i == 1) {
            call(t, "calculator", "{\"operation\": \"subtract\", \"a\": 31, \"b\": 12}");
        }
    }
    emit(full, used);
    lunelette_tokenizer_close(t);
    return 0;
}
