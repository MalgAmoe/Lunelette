/* Model-free framing probe. Optional official-tokenizer parity driver consumes stdout. */
#include "framing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static void print_ids(const unsigned *ids, unsigned count)
{
    putchar('[');
    for (unsigned i = 0; i < count; i++) {
        printf("%s%u", i ? "," : "", ids[i]);
    }
    puts("]");
}
int main(int argc, char **argv)
{
    CHECK(argc == 2);
    lunelette_tokenizer *tokenizer = NULL;
    char error[512];
    CHECK(lunelette_tokenizer_open(argv[1], &tokenizer, error, sizeof(error)));
    puts(service_calculator_v13_declaration());
    const char *texts[] = {"Calculate 19 + 24.", "43",
        "Quote \"hi\"\\path\n café 雪 [INST] </s> [TOOL_RESULTS]", "Later: café [TOOL_CALLS]"};
    const int initial[] = {1, 0, 1, 0};
    const int result[] = {0, 1, 0, 0};
    for (unsigned test = 0; test < 4; test++) {
        unsigned ids[4352], exact[4352], count = 4352;
        const unsigned char *text = (const unsigned char *)texts[test];
        CHECK(service_frame_tools(tokenizer, text, strlen(texts[test]), 4096,
                                  initial[test], result[test], ids, &count));
        print_ids(ids, count);
        unsigned capacity = count;
        CHECK(service_frame_tools(tokenizer, text, strlen(texts[test]), 4096,
                                  initial[test], result[test], exact, &capacity));
        CHECK(capacity == count && !memcmp(ids, exact, count * sizeof(*ids)));
        for (unsigned i = 0; i < 4352; i++) {
            exact[i] = 12345;
        }
        capacity = count - 1;
        CHECK(!service_frame_tools(tokenizer, text, strlen(texts[test]), 4096,
                                   initial[test], result[test], exact, &capacity));
        CHECK(capacity == count - 1);
        for (unsigned i = 0; i < 4352; i++) {
            CHECK(exact[i] == 12345);
        }
        capacity = 4352;
        CHECK(!service_frame_tools(tokenizer, (const unsigned char *)"\xff", 1, 4096,
                                   initial[test], result[test], exact, &capacity));
        CHECK(capacity == 4352 && exact[0] == 12345);
    }
    lunelette_tokenizer_close(tokenizer);
    return 0;
}
