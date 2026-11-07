#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
int main(void)
{
    tj_document d;
    char *body = malloc(262145);
    CHECK(body);
    body[0] = '"';
    memset(body + 1, 'a', 32769);
    body[32769] = '"';
    CHECK(tj_read_memory_limits(body, 32770, &d, 262144, 32768, 64, 4));
    CHECK(d.capacity == 64 && d.nodes[1].length == 32768);
    CHECK(d.nodes[1].text >= d.data && d.nodes[1].text < d.data + d.length);
    tj_free(&d);
    CHECK(!tj_read_memory(body, 32770, &d));
    body[32769] = 'a';
    body[32770] = '"';
    CHECK(!tj_read_memory_limits(body, 32771, &d, 262144, 32768, 64, 4));
    memset(body, ' ', 262145);
    body[0] = '{';
    body[1] = '}';
    CHECK(tj_read_memory_limits(body, 262144, &d, 262144, 32768, 64, 4));
    CHECK(d.capacity == 64);
    tj_free(&d);
    CHECK(!tj_read_memory_limits(body, 262145, &d, 262144, 32768, 64, 4));
    CHECK(tj_read_memory_limits("[[[[0]]]]", 9, &d, 262144, 32768, 64, 4));
    tj_free(&d);
    CHECK(!tj_read_memory_limits("[[[[[0]]]]]", 11, &d, 262144, 32768, 64, 4));
    CHECK(tj_read_memory("[[[[[0]]]]]", 11, &d));
    tj_free(&d);
    size_t used = 0;
    body[used++] = '[';
    for (unsigned i = 0; i < 63; i++) {
        if (i) {
            body[used++] = ',';
        }
        body[used++] = '0';
    }
    body[used++] = ']';
    CHECK(!tj_read_memory_limits(body, used, &d, 262144, 32768, 64, 4));
    CHECK(tj_read_memory(body, used, &d));
    tj_free(&d);
    CHECK(!service_json("{\"x\":\"\\u0000\"}", 14, "|x|", &d));
    CHECK(!service_json("{\"x\":0,\"x\":0}", 13, "|x|", &d));
    free(body);
    puts("service body/string/node/depth limits and legacy defaults passed");
    return 0;
}
