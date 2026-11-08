#include "tool_grammar.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)

static tj_document parse(const char *json)
{
    tj_document d = {0};
    CHECK(tj_read_memory_limits(json, strlen(json), &d, 8192, 4096, 64, 4));
    return d;
}

int main(void)
{
    const char *operations[] = {"add", "subtract", "multiply", "divide", "ADD"};
    for (unsigned i = 0; i < 5; i++) {
        char json[128];
        snprintf(json, sizeof(json), "{\"operation\":\"%s\",\"a\":-1000000,\"b\":1000000}",
                 operations[i]);
        tj_document d = parse(json);
        service_tool_calculation out = {99, 88, 77};
        CHECK(service_tool_calculator(&d, &out) == (i < 3));
        CHECK(out.operation == (i < 3 ? i + 1 : 99));
        CHECK(out.a == (i < 3 ? -1000000 : 88));
        CHECK(out.b == (i < 3 ? 1000000 : 77));
        tj_free(&d);
    }
    const char *arguments[] = {"-0", "0", "1000001", "-1000001", "1.0", "1e0", "\"1\""};
    for (unsigned i = 0; i < sizeof(arguments) / sizeof(*arguments); i++) {
        for (unsigned side = 0; side < 2; side++) {
            char json[128];
            snprintf(json, sizeof(json), "{\"operation\":\"add\",\"a\":%s,\"b\":%s}",
                     side ? "1" : arguments[i], side ? arguments[i] : "1");
            tj_document d = parse(json);
            service_tool_calculation out = {99, 88, 77};
            CHECK(service_tool_calculator(&d, &out) == (i < 2));
            if (i >= 2) {
                CHECK(out.operation == 99 && out.a == 88 && out.b == 77);
            }
            tj_free(&d);
        }
    }
    const char *missing[] = {"{}", "{\"operation\":\"add\",\"a\":1}",
                            "{\"operation\":1,\"a\":1,\"b\":2}"};
    for (unsigned i = 0; i < sizeof(missing) / sizeof(*missing); i++) {
        tj_document d = parse(missing[i]);
        service_tool_calculation out = {99, 88, 77};
        CHECK(!service_tool_calculator(&d, &out));
        CHECK(out.operation == 99 && out.a == 88 && out.b == 77);
        tj_free(&d);
    }
    const char *integers[] = {"0", "1", "-1", "1000000000000", "-1000000000000",
        "", "-", "-0", "00", "01", "-01", "+1", " 1", "1 ", "1.0", "1e0",
        "1000000000001", "-1000000000001", "999999999999999999999999999999"};
    const int64_t expected[] = {0, 1, -1, 1000000000000LL, -1000000000000LL};
    for (unsigned i = 0; i < sizeof(integers) / sizeof(*integers); i++) {
        int64_t out = 42;
        CHECK(service_tool_calculator_result((const unsigned char *)integers[i],
                                            strlen(integers[i]), &out) == (i < 5));
        CHECK(out == (i < 5 ? expected[i] : 42));
    }
    const unsigned char embedded_nul[] = {'1', 0, '2'};
    int64_t out = 42;
    CHECK(!service_tool_calculator_result(embedded_nul, sizeof(embedded_nul), &out));
    CHECK(out == 42);
    const char *passages[] = {"", "LARCH-STOCK:P1", "CEDAR-STOCK:P1", "PUMP-READINGS:P1"};
    for (unsigned document = 0; document <= 3; document++) {
        char json[2048];
        snprintf(json, sizeof(json), "{\"document_id\":\"%s\"}", service_tool_document(document));
        tj_document d = parse(json);
        for (unsigned profile = 0; profile <= 4; profile++) {
            unsigned expected_document = ((profile == 2 && document >= 1 && document <= 2) ||
                                          (profile == 3 && document == 3)) ? document : 0;
            CHECK(service_tool_lookup(&d, profile) == expected_document);
        }
        tj_free(&d);
        const unsigned lengths[] = {0, 1, 1024, 1025};
        for (unsigned i = 0; i < 4; i++) {
            int prefix = snprintf(json, sizeof(json),
                "{\"document_id\":\"%s\",\"passage_id\":\"%s\",\"text\":\"",
                service_tool_document(document), passages[document]);
            CHECK(prefix > 0);
            memset(json + prefix, 'x', lengths[i]);
            strcpy(json + prefix + lengths[i], "\"}");
            d = parse(json);
            for (unsigned selected = 0; selected <= 4; selected++) {
                CHECK(service_tool_lookup_result(&d, selected) ==
                      (document && selected == document && lengths[i] >= 1 && lengths[i] <= 1024));
            }
            tj_free(&d);
        }
    }
    puts("stateless tool grammar tests passed");
    return 0;
}
