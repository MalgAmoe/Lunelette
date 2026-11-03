#include "safetensors.h"
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Isolated generated files under the selected build directory. */
static const char *path;
static void file(const char *header, size_t bytes) {
    FILE *f = fopen(path, "wb"); CHECK(f);
    uint64_t n = strlen(header);
    for (int i = 0; i < 8; i++) CHECK(fputc((int)((n >> (8*i)) & 255), f) != EOF);
    CHECK(fwrite(header, 1, (size_t)n, f) == n);
    for (size_t i = 0; i < bytes; i++) CHECK(fputc(0, f) != EOF);
    CHECK(!fclose(f));
}
static void bad(const char *h, size_t bytes) {
    st_file s; char error[256]; file(h, bytes);
    CHECK(st_open(&s, path, error, sizeof error) == -1);
    CHECK(error[0] && !s.file && !s.tensors); st_close(&s);
}

static void good(const char *header, size_t bytes)
{
    st_file s;
    char error[256];
    file(header, bytes);
    CHECK(!st_open(&s, path, error, sizeof error));
    st_close(&s);
}

/* Generate boundary cases without large checked-in fixture blobs. */
static void capacity_tests(void)
{
    char *header = malloc(ST_MAX_HEADER + 2);
    CHECK(header);
    for (size_t count = ST_MAX_TENSORS; count <= ST_MAX_TENSORS + 1; count++) {
        size_t used = 0;
        header[used++] = '{';
        for (size_t i = 0; i < count; i++) {
            int n = snprintf(header + used, ST_MAX_HEADER - used,
                "%s\"t%zu\":{\"dtype\":\"BF16\",\"shape\":[0],\"data_offsets\":[0,0]}",
                i ? "," : "", i);
            CHECK(n > 0 && (size_t)n < ST_MAX_HEADER - used);
            used += (size_t)n;
        }
        strcpy(header + used, "}");
        if (count == ST_MAX_TENSORS) good(header, 0);
        else bad(header, 0);
    }
    for (size_t length = ST_MAX_NAME - 1; length <= ST_MAX_NAME; length++) {
        header[0] = '{'; header[1] = '"';
        memset(header + 2, 'a', length);
        strcpy(header + 2 + length, "\":{\"dtype\":\"BF16\",\"shape\":[0],\"data_offsets\":[0,0]}}");
        if (length < ST_MAX_NAME) good(header, 0);
        else bad(header, 0);
    }
    for (size_t length = ST_MAX_NAME - 1; length <= ST_MAX_NAME; length++) {
        size_t used = (size_t)snprintf(header, ST_MAX_HEADER + 2, "{\"__metadata__\":{\"");
        memset(header + used, 'k', length);
        strcpy(header + used + length, "\":\"v\"}}");
        if (length < ST_MAX_NAME) good(header, 0);
        else bad(header, 0);
    }
    for (size_t count = 64; count <= 65; count++) {
        size_t used = (size_t)snprintf(header, ST_MAX_HEADER + 2, "{\"__metadata__\":{");
        for (size_t i = 0; i < count; i++)
            used += (size_t)snprintf(header + used, ST_MAX_HEADER + 2 - used, "%s\"k%zu\":\"v\"", i ? "," : "", i);
        strcpy(header + used, "}}");
        if (count == 64) good(header, 0);
        else bad(header, 0);
    }
    for (size_t length = 4095; length <= 4096; length++) {
        size_t used = (size_t)snprintf(header, ST_MAX_HEADER + 2, "{\"__metadata__\":{\"k\":\"");
        memset(header + used, 'v', length);
        strcpy(header + used + length, "\"}}");
        if (length == 4095) good(header, 0);
        else bad(header, 0);
    }
    good("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1,1,1,1,1,1,1,1],\"data_offsets\":[0,2]}}", 2);
    header[0] = '{'; header[1] = '}';
    memset(header + 2, ' ', ST_MAX_HEADER - 2);
    header[ST_MAX_HEADER] = 0;
    good(header, 0);
    header[ST_MAX_HEADER] = ' '; header[ST_MAX_HEADER + 1] = 0;
    bad(header, 0);
    free(header);
}

/* The Makefile runs the comparator against both outputs, checking its exit code
 * and mismatch diagnostic. All payloads are explicitly little-endian. */
static void comparator_fixture(const char *destination, int kind)
{
    const char *saved = path;
    path = destination;
    const char *weights = "{\"language_model.model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[1,2],\"data_offsets\":[0,4]}}";
    const char *fixture = "{\"input_ids\":{\"dtype\":\"I64\",\"shape\":[1,1],\"data_offsets\":[0,8]},\"hidden_0\":{\"dtype\":\"F32\",\"shape\":[1,1,2],\"data_offsets\":[8,16]}}";
    const char *header = kind == 0 ? weights : fixture;
    file(header, kind == 0 ? 4 : 16);
    FILE *f = fopen(path, "r+b");
    CHECK(f);
    CHECK(!fseek(f, (long)(8 + strlen(header)), SEEK_SET));
    const unsigned char row[] = {0x80, 0x3f, 0, 0xc0};
    unsigned char reference[] = {0,0,0,0,0,0,0,0, 0,0,0x80,0x3f, 0,0,0,0xc0};
    if (kind == 2) reference[8] = 1;
    CHECK(fwrite(kind == 0 ? row : reference, 1, kind == 0 ? 4 : 16, f) == (size_t)(kind == 0 ? 4 : 16));
    CHECK(!fclose(f));
    path = saved;
}

int main(int argc, char **argv) {
    CHECK(argc == 5); path = argv[1];
    const char *valid = "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]},\"f\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[8,12]},\"i\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[12,20]}}";
    st_file s; char error[256]; file(valid, 20);
    /* Write known little-endian embedding rows. */
    FILE *f = fopen(path, "r+b"); CHECK(f);
    CHECK(!fseek(f, (long)(8 + strlen(valid)), SEEK_SET));
    unsigned char values[] = {0x80,0x3f,0,0xc0,0,0x80,0x80,0x7f};
    CHECK(fwrite(values, 1, 8, f) == 8); CHECK(!fclose(f));
    CHECK(!st_open(&s, path, error, sizeof error));
    const st_tensor *w = st_find(&s, "w"); float row[2] = {9,9};
    CHECK(w && !st_find(&s, "missing"));
    CHECK(!st_embedding(&s, w, 0, row, 2) && row[0] == 1 && row[1] == -2);
    CHECK(!st_embedding(&s, w, 1, row, 2) && signbit(row[0]) && isinf(row[1]));
    CHECK(st_embedding(&s, w, -1, row, 2) == -1);
    CHECK(st_embedding(&s, w, 2, row, 2) == -1);
    CHECK(st_embedding(&s, w, 0, row, 1) == -1);
    CHECK(st_embedding(&s, st_find(&s,"f"), 0, row, 2) == -1);
    CHECK(st_read(&s, w, UINT64_MAX, row, 1) == -1);
    CHECK(st_read(&s, w, 7, row, 2) == -1);
    st_tensor foreign = *w; CHECK(st_read(&s, &foreign, 0, row, 1) == -1);
    st_close(&s);
    const uint16_t bits[] = {0,0x8000,0x3f80,0xc000,1,0x7f7f,0x7f80,0xff80,0x7fc1};
    for (size_t i = 0; i < sizeof bits/sizeof bits[0]; i++) {
        float v = st_bf16_to_f32(bits[i]); uint32_t raw;
        memcpy(&raw, &v, 4); CHECK(raw == (uint32_t)bits[i] << 16);
    }
    CHECK(isnan(st_bf16_to_f32(0x7fc1)));
    bad("{", 0); bad("{}junk", 0); bad("{}", 1);
    bad("{\"\" : {}}", 0);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1]}}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1.0],\"data_offsets\":[0,2]}}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"\\u0078\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]}}", 4);
    bad("{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[-1],\"data_offsets\":[0,2]}}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[01],\"data_offsets\":[0,2]}}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[9223372036854775807,3],\"data_offsets\":[0,2]}}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[18446744073709551616],\"data_offsets\":[0,2]}}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,0]}}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[1,3]}}", 3);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", 1);
    bad("{\"x\":{\"dtype\":\"BF16\",\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", 2);
    bad("{\"x\":{},\"x\":{}}", 0);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,2]}}", 2);
    bad("{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"y\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", 2);
    good("{\"__metadata__\":{\"key\":\"value\"}}", 0);
    bad("{\"__metadata__\":{},\"__metadata__\":{}}", 0);
    bad("{\"__metadata__\":{\"k\":\"v\",\"k\":\"w\"}}", 0);
    bad("{\"__metadata__\":{\"k\":1}}", 0);
    bad("{\"__metadata__\":[]}", 0);
    good("{\"a\\\"\\\\\\/\\u0041\":{\"dtype\":\"BF16\",\"shape\":[0],\"data_offsets\":[0,0]}}", 0);
    CHECK(!st_open(&s, path, error, sizeof error));
    CHECK(st_find(&s, "a\"\\/A"));
    st_close(&s);
    bad("{\"__metadata__\":{\"k\":\"\\n\"}}", 0);
    bad("{\"__metadata__\":{\"k\":\"\\u001f\"}}", 0);
    bad("{\"__metadata__\":{\"k\":\"\\u0080\"}}", 0);
    bad("{\"__metadata__\":{\"k\":\"\\u00xz\"}}", 0);
    good("{\"b\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]},\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", 4);
    for (unsigned boundary = 0; boundary <= 4; boundary++) {
        char header[512];
        CHECK(snprintf(header, sizeof header,
            "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},"
            "\"b\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]},"
            "\"e\":{\"dtype\":\"BF16\",\"shape\":[0],\"data_offsets\":[%u,%u]}}",
            boundary, boundary) > 0);
        if (boundary % 2 == 0) good(header, 4);
        else bad(header, 4);
    }
    capacity_tests();
    /* Every truncated prefix/header/data position must fail. */
    for (size_t n = 0; n < 8 + strlen(valid) + 20; n++) {
        f = fopen(path, "wb"); CHECK(f);
        uint64_t length = strlen(valid);
        for (size_t i = 0; i < n; i++) {
            int c = i < 8 ? (int)((length >> (i*8)) & 255) :
                    i < 8 + length ? valid[i-8] : 0;
            CHECK(fputc(c, f) != EOF);
        }
        CHECK(!fclose(f)); CHECK(st_open(&s, path, error, sizeof error) == -1);
    }
    f = fopen(path,"wb"); CHECK(f);
    for (int i = 0; i < 8; i++) CHECK(fputc(255,f) != EOF);
    CHECK(!fclose(f)); CHECK(st_open(&s,path,error,sizeof error) == -1);
    file("{\"empty\":{\"dtype\":\"BF16\",\"shape\":[0,2],\"data_offsets\":[0,0]}}",0);
    CHECK(!st_open(&s,path,error,sizeof error)); st_close(&s);
    comparator_fixture(argv[2], 0);
    comparator_fixture(argv[3], 1);
    comparator_fixture(argv[4], 2);
    CHECK(!remove(path)); puts("safetensors tests passed"); return 0;
}
