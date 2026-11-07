#include "tokenizer.h"
#include "../src/tokenizer_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static const char *scratch = "test_tokenizer_fixture.tmp";

static void write_text(const char *text)
{
    FILE *f = fopen(scratch, "wx");
    CHECK(f != NULL);
    CHECK(fwrite(text, 1, strlen(text), f) == strlen(text));
    CHECK(fclose(f) == 0);
}

static void parser_tests(void)
{
    const char *bad[] = {
        "", "{", "[1,]", "{\"a\":1,}", "{\"a\":1,\"a\":2}",
        "{\"a\":1,\"\\u0061\":2}", "[01]", "[-]", "[1.]", "[1e+]",
        "[true false]", "true x", "\"\\uD800\"", "\"\\uDC00\"", "\"\\x\"",
        "\"\xc0\x80\"", "\"\xed\xa0\x80\"", "\"\xf4\x90\x80\x80\"", "\"\n\""
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        tj_document d;
        write_text(bad[i]);
        CHECK(!tj_read(scratch, &d));
        CHECK(remove(scratch) == 0);
    }
    write_text("{\"s\":\"\\u0000\\u00e9\\u20ac\\uD83D\\uDE00\",\"n\":4294967295,"
               "\"x\":[true,false,null,-0.5e+3,{},[]]}");
    tj_document d;
    CHECK(tj_read(scratch, &d));
    unsigned s = tj_field(&d, 1, "s");
    const unsigned char expected[] = {0, 0xc3, 0xa9, 0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80};
    CHECK(s && d.nodes[s].length == sizeof(expected));
    CHECK(!memcmp(d.nodes[s].text, expected, sizeof(expected)));
    unsigned n;
    CHECK(tj_integer(&d, tj_field(&d, 1, "n"), &n) && n == 0xffffffffu);
    tj_free(&d);
    CHECK(remove(scratch) == 0);
    /* Depth limit and dynamic-node growth. */
    FILE *f = fopen(scratch, "wx");
    CHECK(f != NULL);
    for (unsigned i = 0; i < 40; i++) {
        CHECK(fputc('[', f) != EOF);
    }
    for (unsigned i = 0; i < 40; i++) {
        CHECK(fputc(']', f) != EOF);
    }
    CHECK(fclose(f) == 0);
    CHECK(!tj_read(scratch, &d));
    CHECK(remove(scratch) == 0);
    f = fopen(scratch, "wx");
    CHECK(f != NULL);
    CHECK(fputc('"', f) != EOF);
    for (unsigned i = 0; i < 4097; i++) {
        CHECK(fputc('x', f) != EOF);
    }
    CHECK(fputc('"', f) != EOF);
    CHECK(fclose(f) == 0);
    CHECK(!tj_read(scratch, &d));
    CHECK(remove(scratch) == 0);
    f = fopen(scratch, "wx");
    CHECK(f != NULL);
    CHECK(fputc('{', f) != EOF);
    for (unsigned i = 0; i < 65; i++) {
        CHECK(fprintf(f, "%s\"k%u\":0", i ? "," : "", i) > 0);
    }
    CHECK(fputc('}', f) != EOF);
    CHECK(fclose(f) == 0);
    CHECK(!tj_read(scratch, &d));
    CHECK(remove(scratch) == 0);
    write_text("[4294967296,-1,1.0,1e0]");
    CHECK(tj_read(scratch, &d));
    for (unsigned p = d.nodes[1].child; p; p = d.nodes[p].next) {
        CHECK(!tj_integer(&d, p, &n));
    }
    tj_free(&d);
    CHECK(remove(scratch) == 0);
}

static void put_base64(FILE *f, const unsigned char *s, size_t n)
{
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < n; i += 3) {
        unsigned a = s[i];
        unsigned b = i + 1 < n ? s[i + 1] : 0;
        unsigned c = i + 2 < n ? s[i + 2] : 0;
        CHECK(fputc(alphabet[a >> 2], f) != EOF);
        CHECK(fputc(alphabet[((a & 3) << 4) | (b >> 4)], f) != EOF);
        CHECK(fputc(i + 1 < n ? alphabet[((b & 15) << 2) | (c >> 6)] : '=', f) != EOF);
        CHECK(fputc(i + 2 < n ? alphabet[c & 63] : '=', f) != EOF);
    }
}

/* Small semantics, full supported rank count, no model/Python/network required.
 * The two real merges exercise whole-piece lookup and leftmost rank ties. */
static void synthetic_file(unsigned defect)
{
    const char *pattern = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?"
        "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
        "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    FILE *f = fopen(scratch, "wx");
    CHECK(f != NULL);
    CHECK(fputs("{\"config\":{\"pattern\":\"", f) >= 0);
    for (const char *p = pattern; *p; p++) {
        if (*p == '\\') {
            CHECK(fputc('\\', f) != EOF);
        }
        CHECK(fputc(*p, f) != EOF);
    }
    CHECK(fputs("\",\"version\":\"v13\",\"num_vocab_tokens\":131072,"
                "\"default_vocab_size\":131072,\"default_num_special_tokens\":1000},\"vocab\":[", f) >= 0);
    for (unsigned rank = 0; rank < (defect == 9 ? 130071u : 130072u); rank++) {
        unsigned char bytes[5] = {255, (unsigned char)rank, (unsigned char)(rank >> 8),
                                 (unsigned char)(rank >> 16), 255};
        size_t length = 5;
        if (rank < 256) {
            bytes[0] = (unsigned char)rank;
            length = 1;
        } else if (rank == 256 || rank == 257) {
            memset(bytes, 'a', sizeof(bytes));
            length = rank == 256 ? 2 : 3;
        } else if (rank == 258) {
            bytes[0] = 'a';
            bytes[1] = 'A';
            length = 2; /* must NOT merge across lowercase-to-uppercase split */
        }
        if (defect == 1 && rank == 258) {
            bytes[0] = 0;
            length = 1; /* duplicate byte string */
        }
        CHECK(fprintf(f, "%s{\"rank\":%u,\"token_bytes\":\"", rank ? "," : "",
                      rank == 258 && defect == 2 ? 257 :
                      rank == 258 && defect == 5 ? 130072 : rank) > 0);
        if (rank == 258 && (defect == 3 || (defect >= 6 && defect <= 8))) {
            const char *bad = defect == 3 ? "AB==" : defect == 6 ? "" :
                defect == 7 ? "@@==" : "AA=A";
            CHECK(fputs(bad, f) >= 0); /* invalid alphabet/padding/canonical bits */
        } else {
            put_base64(f, bytes, length);
        }
        CHECK(fputs("\"}", f) >= 0);
    }
    CHECK(fputs("],\"special_tokens\":[", f) >= 0);
    for (unsigned rank = 0; rank < (defect == 11 ? 999u : 1000u); rank++) {
        const char *spelling = rank == 1 ? "<s>" : rank == 2 ? "</s>" :
            rank == 3 ? "[INST]" : rank == 4 ? "[/INST]" :
            rank == 17 ? "[SYSTEM_PROMPT]" : rank == 18 ? "[/SYSTEM_PROMPT]" : NULL;
        CHECK(fprintf(f, "%s{\"rank\":%u,\"is_control\":true,\"token_str\":\"",
                      rank ? "," : "", defect == 4 && rank == 999 ? 998 : rank) > 0);
        if (defect == 10 && rank == 999) {
            CHECK(fputs("<s>", f) >= 0);
        } else if (spelling) {
            CHECK(fputs(spelling, f) >= 0);
        } else {
            CHECK(fprintf(f, "special_%u", rank) > 0);
        }
        CHECK(fputs("\"}", f) >= 0);
    }
    CHECK(fputs("]}", f) >= 0);
    CHECK(fclose(f) == 0);
}

static void roundtrip(lunelette_tokenizer *t, const unsigned char *text, size_t length)
{
    uint32_t *ids = NULL;
    size_t count = 0;
    unsigned char *bytes = NULL;
    size_t n = 0;
    char error[256];
    CHECK(lunelette_tokenizer_encode(t, text, length, &ids, &count, error, sizeof(error)));
    for (size_t i = 0; i < count; i++) {
        CHECK(ids[i] >= 1000 && ids[i] < 131072);
    }
    CHECK(lunelette_tokenizer_decode(t, ids, count, &bytes, &n, error, sizeof(error)));
    CHECK(n == length && (!length || !memcmp(bytes, text, length)));
    free(ids);
    free(bytes);
}

static void contract_tests(lunelette_tokenizer *t, int synthetic)
{
    char error[256];
    uint32_t *ids;
    size_t count;
    unsigned char *bytes;
    size_t length;
    roundtrip(t, NULL, 0);
    const char *texts[] = {"Hello world!", "[INST]<s></s>[/INST][SYSTEM_PROMPT]", "aaa aaaa aaaaa",
        "CamelCase HTTPServer 12345\r\n\t  !/\n", "日本語 Ελληνικά مرحبا 😀 é\x1c\xc2\xa0"};
    for (size_t i = 0; i < sizeof(texts) / sizeof(texts[0]); i++) {
        roundtrip(t, (const unsigned char *)texts[i], strlen(texts[i]));
    }
    unsigned char all[128];
    for (unsigned i = 0; i < sizeof(all); i++) {
        all[i] = (unsigned char)i;
    }
    roundtrip(t, all, sizeof(all));
    unsigned char boundary[4096];
    memset(boundary, 'a', sizeof(boundary));
    roundtrip(t, boundary, sizeof(boundary));
    CHECK(!lunelette_tokenizer_encode(t, boundary, 4097, &ids, &count, error, sizeof(error)));
    CHECK(!ids && !count);
    unsigned char *long_text = malloc(32769);
    CHECK(long_text);
    memset(long_text, '7', 32769); /* Numbers split individually: no hidden token cap. */
#define ENCODE_LIMIT(n, limit) lunelette_tokenizer_encode_with_limit(t, long_text, n, limit, &ids, &count, error, sizeof error)
    const size_t sizes[] = {0, 4096, 4097, 32768};
    for (size_t k = 0; k < sizeof sizes / sizeof sizes[0]; k++) {
        CHECK(ENCODE_LIMIT(sizes[k], 32768));
        CHECK(count == sizes[k]);
        for (size_t j = 0; j < count; j++) {
            CHECK(ids[j] >= 1000 && ids[j] < 131072);
            if (synthetic) {
                CHECK(ids[j] == 1000 + '7');
            }
        }
        free(ids);
    }
    CHECK(!ENCODE_LIMIT(32769, 32768) && !ids && !count);
    CHECK(!ENCODE_LIMIT(SIZE_MAX, 32768) && !ids && !count);
    CHECK(!ENCODE_LIMIT(1, SIZE_MAX) && !ids && !count);
    CHECK(!ENCODE_LIMIT(1, 4097) && !ids && !count);
    CHECK(!ENCODE_LIMIT(4097, 4096) && !ids && !count);
    long_text[32767] = 0x80;
    CHECK(!ENCODE_LIMIT(32768, 32768) && !ids && !count);
    CHECK(strstr(error, "invalid UTF-8 at byte 32767"));
    free(long_text);
#undef ENCODE_LIMIT
    const unsigned char *bad[] = {(const unsigned char *)"\x80", (const unsigned char *)"\xc0\x80",
        (const unsigned char *)"\xed\xa0\x80", (const unsigned char *)"\xf4\x90\x80\x80",
        (const unsigned char *)"\xe2\x82"};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        CHECK(!lunelette_tokenizer_encode(t, bad[i], strlen((const char *)bad[i]),
                                          &ids, &count, error, sizeof(error)));
        CHECK(!ids && !count);
    }
    uint32_t invalid[] = {0, 999, 131072, 0xffffffffu};
    for (unsigned i = 0; i < 4; i++) {
        CHECK(!lunelette_tokenizer_decode(t, invalid + i, 1, &bytes, &length, error, sizeof(error)));
        CHECK(!bytes && !length);
    }
    CHECK(!lunelette_tokenizer_decode(t, invalid, 4097, &bytes, &length, error, sizeof(error)));
    uint32_t partial[] = {1000, 1001, 2};
    CHECK(!lunelette_tokenizer_decode(t, partial, 3, &bytes, &length, error, sizeof(error)));
    CHECK(!bytes && !length);
    CHECK(!lunelette_tokenizer_encode(NULL, NULL, 0, &ids, &count, NULL, 0));
    CHECK(!lunelette_tokenizer_encode(t, NULL, 1, &ids, &count, error, sizeof(error)));
    CHECK(!lunelette_tokenizer_decode(t, NULL, 1, &bytes, &length, error, sizeof(error)));
    CHECK(lunelette_tokenizer_decode(t, NULL, 0, &bytes, &length, error, sizeof(error)));
    CHECK(length == 0 && bytes && bytes[0] == 0);
    free(bytes);
    if (synthetic) {
        CHECK(lunelette_tokenizer_encode(t, (const unsigned char *)"aaaa", 4,
                                         &ids, &count, error, sizeof(error)));
        CHECK(count == 2 && ids[0] == 1256 && ids[1] == 1256);
        free(ids);
        CHECK(lunelette_tokenizer_encode(t, (const unsigned char *)"aaa", 3,
                                         &ids, &count, error, sizeof(error)));
        CHECK(count == 1 && ids[0] == 1257);
        free(ids);
        CHECK(lunelette_tokenizer_encode(t, (const unsigned char *)"aaaaa", 5,
                                         &ids, &count, error, sizeof(error)));
        CHECK(count == 2 && ids[0] == 1256 && ids[1] == 1257);
        free(ids);
        CHECK(lunelette_tokenizer_encode(t, (const unsigned char *)"aA", 2,
                                         &ids, &count, error, sizeof(error)));
        CHECK(count == 2 && ids[0] == 1097 && ids[1] == 1065);
        free(ids);
        uint32_t raw[] = {1000 + 0xf0, 1000 + 0x9f, 1000 + 0x98, 1000 + 0x80, 1000, 1255};
        CHECK(lunelette_tokenizer_decode(t, raw, 6, &bytes, &length, error, sizeof(error)));
        CHECK(length == 6 && !memcmp(bytes, "\xf0\x9f\x98\x80\0\xff", 6));
        free(bytes);
    }
}

static unsigned read_u32(FILE *f)
{
    unsigned char b[4];
    CHECK(fread(b, 1, 4, f) == 4);
    return (unsigned)b[0] | ((unsigned)b[1] << 8) | ((unsigned)b[2] << 16) | ((unsigned)b[3] << 24);
}

static int same_ids(const uint32_t *a, size_t na, const uint32_t *b, size_t nb)
{
    return na == nb && (!na || !memcmp(a, b, na * sizeof(*a)));
}

static void oracle_tests(lunelette_tokenizer *t, const char *path)
{
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    char magic[4];
    CHECK(fread(magic, 1, 4, f) == 4 && !memcmp(magic, "TKD1", 4));
    unsigned records = read_u32(f);
    CHECK(records < 1000000);
    uint32_t probe[] = {1000, 1001};
    CHECK(same_ids(probe, 2, probe, 2));
    uint32_t mismatch[] = {1000, 1002};
    CHECK(!same_ids(probe, 2, mismatch, 2));
    CHECK(!same_ids(probe, 2, probe, 1));
    for (unsigned k = 0; k < records; k++) {
        unsigned mode = read_u32(f);
        unsigned length = read_u32(f);
        unsigned expected_count = read_u32(f);
        CHECK(mode <= 1 && length <= 4 * 1024 * 1024 && expected_count <= 4096);
        unsigned char *text = malloc((size_t)length + 1);
        uint32_t *expected = malloc(((size_t)expected_count + 1) * sizeof(*expected));
        CHECK(text && expected);
        CHECK(fread(text, 1, length, f) == length);
        for (unsigned j = 0; j < expected_count; j++) {
            expected[j] = read_u32(f);
        }
        char error[256];
        if (!mode) {
            uint32_t *ids;
            size_t count;
            CHECK(lunelette_tokenizer_encode(t, text, length, &ids, &count, error, sizeof(error)));
            if (!same_ids(ids, count, expected, expected_count)) {
                fprintf(stderr, "oracle record %u: encoded %zu IDs; expected %u\n", k, count, expected_count);
                for (size_t j = 0; j < count; j++) {
                    fprintf(stderr, "%u,", ids[j]);
                }
                fputc('\n', stderr);
                exit(1);
            }
            free(ids);
        }
        unsigned char *bytes;
        size_t n;
        CHECK(lunelette_tokenizer_decode(t, expected, expected_count, &bytes, &n, error, sizeof(error)));
        CHECK(n == length && (!length || !memcmp(bytes, text, length)));
        free(bytes);
        free(text);
        free(expected);
    }
    CHECK(fgetc(f) == EOF && !ferror(f));
    CHECK(fclose(f) == 0);
    printf("tokenizer differential: %u records passed (exact IDs and raw bytes)\n", records);
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--cli-fixture") == 0) {
        synthetic_file(0);
        return 0;
    }
    CHECK(argc == 1 || argc == 3);
    parser_tests();
    lunelette_tokenizer *t = NULL;
    char error[256];
    for (unsigned defect = 1; defect <= 11; defect++) {
        synthetic_file(defect);
        CHECK(!lunelette_tokenizer_open(scratch, &t, error, sizeof(error)) && !t);
        CHECK(remove(scratch) == 0);
    }
    synthetic_file(0);
    if (!lunelette_tokenizer_open(scratch, &t, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    CHECK(remove(scratch) == 0);
    contract_tests(t, 1);
    lunelette_tokenizer_close(t);
    if (argc == 3) {
        if (!lunelette_tokenizer_open(argv[1], &t, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            exit(1);
        }
        contract_tests(t, 0);
        oracle_tests(t, argv[2]);
        lunelette_tokenizer_close(t);
    }
    lunelette_tokenizer_close(NULL);
    puts("tokenizer tests passed");
    return 0;
}
