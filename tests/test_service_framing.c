#include "framing.h"
#include "tokenizer_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

/* Optional CPU-only current tokenizer/reference fixture check. No model opened.
 * The fixture is generated independently with native mistral_common, not HF. */
int main(int argc, char **argv)
{
    CHECK(argc == 3);
    lunelette_tokenizer *tokenizer = NULL;
    char error[512];
    CHECK(lunelette_tokenizer_open(argv[1], &tokenizer, error, sizeof(error)));
    tj_document document;
    CHECK(tj_read(argv[2], &document));
    unsigned cases = tj_field(&document, 1, "cases");
    CHECK(cases && document.nodes[cases].kind == '[');
    unsigned tested = 0;
    for (unsigned c = document.nodes[cases].child; c; c = document.nodes[c].next) {
        unsigned actual[256], used = 0;
        unsigned messages = tj_field(&document, c, "messages");
        for (unsigned m = document.nodes[messages].child; m; m = document.nodes[m].next) {
            unsigned role = tj_field(&document, m, "role");
            unsigned content = tj_field(&document, m, "content");
            CHECK(role && content);
            const tj_node *text = &document.nodes[content];
            if (document.nodes[role].length == 4) {
                unsigned capacity = 256 - used;
                CHECK(service_frame_with_limit(tokenizer, text->text, text->length, 4096, !used,
                                    actual + used, &capacity));
                used += capacity;
            } else {
                uint32_t *ids;
                size_t count;
                CHECK(lunelette_tokenizer_encode(tokenizer, text->text, text->length,
                                                &ids, &count, error, sizeof(error)));
                CHECK(used + count + 1 <= 256);
                memcpy(actual + used, ids, count * sizeof(*ids));
                used += (unsigned)count;
                actual[used++] = 2;
                free(ids);
            }
        }
        unsigned expected = tj_field(&document, c, "tokens"), index = 0;
        for (unsigned t = document.nodes[expected].child; t; t = document.nodes[t].next) {
            unsigned id;
            CHECK(tj_integer(&document, t, &id));
            CHECK(index < used && actual[index++] == id);
        }
        CHECK(index == used);
        unsigned output[256];
        memset(output, 0x5a, sizeof(output));
        unsigned capacity = 2;
        CHECK(!service_frame_with_limit(tokenizer, (const unsigned char *)"Hi", 2,
            4096, 1, output, &capacity));
        CHECK(output[0] == 0x5a5a5a5a && capacity == 2);
        tested++;
    }
    CHECK(tested == 7);
    tj_free(&document);
    lunelette_tokenizer_close(tokenizer);
    printf("native framing reference fixtures passed: %u\n", tested);
    return 0;
}
