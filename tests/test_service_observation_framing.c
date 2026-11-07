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
    puts(service_observation_v13_declaration());
    const char *texts[] = {"Report the saved pump temperatures.",
        "{\"document_id\":\"pump-readings\",\"passage_id\":\"PUMP-READINGS:P1\",\"text\":\"Synthetic recorded observation (not live): pump-A temperature 84 C; pump-B temperature 72 C.\"}"};
    for (unsigned i = 0; i < 2; i++) {
        unsigned count = 4352 - used, start = used;
        CHECK(service_frame_observation(t, (const unsigned char *)texts[i], strlen(texts[i]),
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
            call(t, "lookup", "{\"document_id\": \"pump-readings\"}");
        }
    }
    emit(full, used);
    /* Rejected admission is atomic, including exact capacity reporting. */
    unsigned output[4352], saved[4352];
    memset(output, 0xa5, sizeof(output));
    memcpy(saved, output, sizeof(saved));
    unsigned capacity = 1;
    CHECK(!service_frame_observation(t, (const unsigned char *)texts[0], strlen(texts[0]),
        4096, 1, 0, output, &capacity));
    CHECK(capacity == 1 && !memcmp(output, saved, sizeof(output)));
    capacity = 4352;
    const unsigned char invalid[] = {0xff};
    CHECK(!service_frame_observation(t, invalid, sizeof(invalid), 4096, 1, 0,
                                    output, &capacity));
    CHECK(capacity == 4352 && !memcmp(output, saved, sizeof(output)));
    CHECK(service_frame_observation(t, (const unsigned char *)texts[0], strlen(texts[0]),
        32768, 0, 0, output, &capacity));
    CHECK(output[0] == 3 && output[capacity - 1] == 4);
    lunelette_tokenizer_close(t);
    return 0;
}
