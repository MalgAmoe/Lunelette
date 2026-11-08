#ifndef LUNELETTE_TEST_SUPPORT_TOKENIZER_FIXTURE_H
#define LUNELETTE_TEST_SUPPORT_TOKENIZER_FIXTURE_H

/* Header-local synthetic Tekken writer. CHECK stays active under NDEBUG.
 * Exclusive scratch creation and byte-for-byte fixture construction are preserved.
 * Caller owns scratch cleanup; no parser/encoding contract tests or runner here. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CHECK
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
#endif

static const char *scratch = "test_tokenizer_fixture.tmp";

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


#endif
