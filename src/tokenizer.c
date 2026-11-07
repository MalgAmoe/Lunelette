#include "tokenizer.h"
#include "tokenizer_json.h"
#include "tokenizer_unicode.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TK_ORDINARY 130072u
#define TK_HASH_SIZE 262144u
#define TK_MAX_TOKEN 1024u
#define TK_MAX_BYTES (4u * 1024u * 1024u)
#define TK_MAX_VOCAB_BYTES (16u * 1024u * 1024u)
#define TK_NONE 0xffffffffu

static const char tk_pattern[] =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+"
    "[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

typedef struct {
    unsigned char *bytes;
    size_t length;
} tk_token;

struct lunelette_tokenizer {
    tk_token tokens[TK_ORDINARY];
    unsigned hash[TK_HASH_SIZE]; /* rank + 1; zero is empty */
};

static int fail(char *error, size_t size, const char *format, ...)
{
    if (error && size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return 0;
}

static unsigned hash_bytes(const unsigned char *bytes, size_t length)
{
    unsigned hash = 2166136261u;
    for (size_t i = 0; i < length; i++) {
        hash = (hash ^ bytes[i]) * 16777619u;
    }
    return hash & (TK_HASH_SIZE - 1);
}

static unsigned lookup(const lunelette_tokenizer *t, const unsigned char *bytes, size_t length)
{
    unsigned slot = hash_bytes(bytes, length);
    if (length > TK_MAX_TOKEN) {
        return TK_NONE;
    }
    while (t->hash[slot]) {
        unsigned rank = t->hash[slot] - 1;
        if (t->tokens[rank].length == length &&
            !memcmp(t->tokens[rank].bytes, bytes, length)) {
            return rank;
        }
        slot = (slot + 1) & (TK_HASH_SIZE - 1);
    }
    return TK_NONE;
}

static int b64_value(unsigned c)
{
    if (c >= 'A' && c <= 'Z') {
        return (int)(c - 'A');
    }
    if (c >= 'a' && c <= 'z') {
        return (int)(c - 'a' + 26);
    }
    if (c >= '0' && c <= '9') {
        return (int)(c - '0' + 52);
    }
    return c == '+' ? 62 : c == '/' ? 63 : -1;
}

static int base64(const tj_node *node, unsigned char *out, size_t *length)
{
    size_t n = 0;
    if (node->kind != '"' || !node->length || node->length % 4 ||
        node->length > 4 * ((TK_MAX_TOKEN + 2) / 3)) {
        return 0;
    }
    for (size_t i = 0; i < node->length; i += 4) {
        int a = b64_value(node->text[i]);
        int b = b64_value(node->text[i + 1]);
        int c = b64_value(node->text[i + 2]);
        int d = b64_value(node->text[i + 3]);
        int pad_c = node->text[i + 2] == '=';
        int pad_d = node->text[i + 3] == '=';
        if (a < 0 || b < 0 || (c < 0 && !pad_c) || (d < 0 && !pad_d) ||
            (pad_c && !pad_d) || ((pad_c || pad_d) && i + 4 != node->length) ||
            (pad_c && (b & 15)) || (pad_d && !pad_c && (c & 3))) {
            return 0;
        }
        out[n++] = (unsigned char)((a << 2) | (b >> 4));
        if (!pad_c) {
            out[n++] = (unsigned char)((b << 4) | (c >> 2));
        }
        if (!pad_d) {
            out[n++] = (unsigned char)((c << 6) | d);
        }
    }
    if (n > TK_MAX_TOKEN) {
        return 0;
    }
    *length = n;
    return 1;
}

static int equal_string(const tj_document *d, unsigned node, const char *text)
{
    return node && d->nodes[node].kind == '"' &&
        d->nodes[node].length == strlen(text) &&
        !memcmp(d->nodes[node].text, text, strlen(text));
}

static int equal_integer(const tj_document *d, unsigned object, const char *key, unsigned want)
{
    unsigned value;
    return tj_integer(d, tj_field(d, object, key), &value) && value == want;
}

void lunelette_tokenizer_close(lunelette_tokenizer *t)
{
    if (t) {
        for (unsigned rank = 0; rank < TK_ORDINARY; rank++) {
            free(t->tokens[rank].bytes);
        }
        free(t);
    }
}

int lunelette_tokenizer_open(const char *path, lunelette_tokenizer **out,
                             char *error, size_t error_size)
{
    tj_document doc;
    lunelette_tokenizer *t = NULL;
    unsigned count = 0;
    size_t total = 0;
    unsigned special_nodes[1000] = {0};
    if (!out) {
        return fail(error, error_size, "tokenizer: NULL output pointer");
    }
    *out = NULL;
    if (!path || !tj_read(path, &doc)) {
        return fail(error, error_size, "tokenizer %s: invalid JSON, I/O, or parser limit "
                    "(32 MiB, depth 32, 1500000 nodes)", path ? path : "(null)");
    }
    unsigned config = tj_field(&doc, 1, "config");
    if (!equal_string(&doc, tj_field(&doc, config, "version"), "v13") ||
        !equal_string(&doc, tj_field(&doc, config, "pattern"), tk_pattern) ||
        !equal_integer(&doc, config, "num_vocab_tokens", 131072) ||
        !equal_integer(&doc, config, "default_vocab_size", 131072) ||
        !equal_integer(&doc, config, "default_num_special_tokens", 1000)) {
        fail(error, error_size, "tokenizer %s: unsupported Tekken13 config/pattern", path);
        goto cleanup;
    }
    t = calloc(1, sizeof(*t));
    if (!t) {
        fail(error, error_size, "tokenizer: allocation failed");
        goto cleanup;
    }
    unsigned vocab = tj_field(&doc, 1, "vocab");
    if (!vocab || doc.nodes[vocab].kind != '[') {
        fail(error, error_size, "tokenizer: missing vocab array");
        goto cleanup;
    }
    for (unsigned p = doc.nodes[vocab].child; p; p = doc.nodes[p].next) {
        unsigned rank;
        unsigned char bytes[TK_MAX_TOKEN + 2];
        size_t length;
        unsigned encoded = tj_field(&doc, p, "token_bytes");
        if (!tj_integer(&doc, tj_field(&doc, p, "rank"), &rank) || rank >= TK_ORDINARY ||
            t->tokens[rank].bytes || !encoded || !base64(&doc.nodes[encoded], bytes, &length)) {
            fail(error, error_size, "tokenizer: vocab entry %u invalid rank/base64/duplicate rank", count);
            goto cleanup;
        }
        if (lookup(t, bytes, length) != TK_NONE || total > TK_MAX_VOCAB_BYTES - length) {
            fail(error, error_size, "tokenizer: rank %u duplicate bytes or vocab bytes exceed 16 MiB", rank);
            goto cleanup;
        }
        t->tokens[rank].bytes = malloc(length);
        if (!t->tokens[rank].bytes) {
            fail(error, error_size, "tokenizer: rank %u allocation failed", rank);
            goto cleanup;
        }
        memcpy(t->tokens[rank].bytes, bytes, length);
        t->tokens[rank].length = length;
        total += length;
        unsigned slot = hash_bytes(bytes, length);
        while (t->hash[slot]) {
            slot = (slot + 1) & (TK_HASH_SIZE - 1);
        }
        t->hash[slot] = rank + 1;
        count++;
    }
    if (count != TK_ORDINARY) {
        fail(error, error_size, "tokenizer: expected 130072 ordinary ranks, got %u", count);
        goto cleanup;
    }
    for (unsigned i = 0; i < 256; i++) {
        unsigned char byte = (unsigned char)i;
        if (lookup(t, &byte, 1) == TK_NONE) {
            fail(error, error_size, "tokenizer: missing byte token %u", i);
            goto cleanup;
        }
    }
    unsigned specials = tj_field(&doc, 1, "special_tokens");
    if (!specials || doc.nodes[specials].kind != '[') {
        fail(error, error_size, "tokenizer: missing special_tokens array");
        goto cleanup;
    }
    count = 0;
    for (unsigned p = doc.nodes[specials].child; p; p = doc.nodes[p].next) {
        unsigned rank;
        unsigned str = tj_field(&doc, p, "token_str");
        unsigned control = tj_field(&doc, p, "is_control");
        if (!tj_integer(&doc, tj_field(&doc, p, "rank"), &rank) || rank >= 1000 ||
            special_nodes[rank] || !str || doc.nodes[str].kind != '"' ||
            !doc.nodes[str].length || !control ||
            (doc.nodes[control].kind != 't' && doc.nodes[control].kind != 'f')) {
            fail(error, error_size, "tokenizer: invalid/duplicate special rank or string");
            goto cleanup;
        }
        for (unsigned i = 0; i < 1000; i++) {
            unsigned other = special_nodes[i];
            if (other && doc.nodes[other].length == doc.nodes[str].length &&
                !memcmp(doc.nodes[other].text, doc.nodes[str].text, doc.nodes[str].length)) {
                fail(error, error_size, "tokenizer: duplicate special spelling");
                goto cleanup;
            }
        }
        special_nodes[rank] = str;
        count++;
    }
    if (count != 1000 || !equal_string(&doc, special_nodes[1], "<s>") ||
        !equal_string(&doc, special_nodes[2], "</s>") ||
        !equal_string(&doc, special_nodes[3], "[INST]") ||
        !equal_string(&doc, special_nodes[4], "[/INST]") ||
        !equal_string(&doc, special_nodes[17], "[SYSTEM_PROMPT]") ||
        !equal_string(&doc, special_nodes[18], "[/SYSTEM_PROMPT]")) {
        fail(error, error_size, "tokenizer: expected 1000 specials and supported framing IDs");
        goto cleanup;
    }
    tj_free(&doc);
    *out = t;
    return 1;

cleanup:
    tj_free(&doc);
    lunelette_tokenizer_close(t);
    return 0;
}

static unsigned unicode_mask(unsigned cp)
{
    size_t lo = 0;
    size_t hi = sizeof(tk_unicode) / sizeof(tk_unicode[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < tk_unicode[mid].first) {
            hi = mid;
        } else if (cp > tk_unicode[mid].last) {
            lo = mid + 1;
        } else {
            return tk_unicode[mid].mask;
        }
    }
    return 0;
}

/* Property bits in generated table. A and B overlap for modifier/other letters
 * and combining marks: the first branch must backtrack its greedy A* to B+.
 * Every alternative is tried in regex order, never longest-alternative order. */
#define TK_L 1u
#define TK_A (2u | 4u | 8u | 16u | 32u)
#define TK_B (64u | 8u | 16u | 32u)
#define TK_N 128u
#define TK_S 256u

typedef struct {
    unsigned cp;
    unsigned mask;
    size_t offset;
} tk_character;

static size_t word_branch(const tk_character *c, size_t n, size_t start, int first)
{
    size_t end = start;
    while (end < n && (c[end].mask & TK_A)) {
        end++;
    }
    if (!first) {
        if (end == start) {
            return start;
        }
        while (end < n && (c[end].mask & TK_B)) {
            end++;
        }
        return end;
    }
    for (;;) {
        size_t tail = end;
        while (tail < n && (c[tail].mask & TK_B)) {
            tail++;
        }
        if (tail > end) {
            return tail;
        }
        if (end == start) {
            return start;
        }
        end--;
    }
}

static int newline(unsigned cp)
{
    return cp == '\r' || cp == '\n';
}

static size_t piece_end(const tk_character *c, size_t n, size_t start)
{
    for (int branch = 0; branch < 2; branch++) {
        if (!newline(c[start].cp) && !(c[start].mask & (TK_L | TK_N))) {
            size_t end = word_branch(c, n, start + 1, branch == 0);
            if (end > start + 1) {
                return end;
            }
        }
        size_t end = word_branch(c, n, start, branch == 0);
        if (end > start) {
            return end;
        }
    }
    if (c[start].mask & TK_N) {
        return start + 1;
    }
    /* Optional literal space is greedy, but must backtrack if class+ fails. */
    for (int prefix = 1; prefix >= 0; prefix--) {
        size_t begin = start;
        if (prefix) {
            if (c[start].cp != ' ') {
                continue;
            }
            begin++;
        }
        size_t end = begin;
        while (end < n && !(c[end].mask & (TK_S | TK_L | TK_N))) {
            end++;
        }
        if (end > begin) {
            while (end < n && (newline(c[end].cp) || c[end].cp == '/')) {
                end++;
            }
            return end;
        }
    }
    size_t end = start;
    size_t last_newline = start;
    while (end < n && (c[end].mask & TK_S)) {
        if (newline(c[end].cp)) {
            last_newline = end + 1;
        }
        end++;
    }
    if (last_newline > start) {
        return last_newline; /* greedy whitespace, backtrack to final CR/LF */
    }
    if (end == n || end > start + 1) {
        return end == n ? end : end - 1; /* \s+(?!\S) */
    }
    return end; /* final \s+; all Unicode scalars are covered */
}

/* One heap entry per original byte position, including inactive entries at
 * TK_NONE rank. Links retain byte offsets, so equal ranks sort leftmost.
 * Updating just the two adjacent pairs avoids stale candidates entirely.
 * Scratch is one bounded allocation: length * (sizeof(tk_merge) + sizeof(size_t)). */
typedef struct {
    size_t previous;
    size_t next;
    size_t end;
    size_t position;
    unsigned rank;
} tk_merge;

static int merge_less(const tk_merge *nodes, size_t a, size_t b)
{
    return nodes[a].rank < nodes[b].rank ||
        (nodes[a].rank == nodes[b].rank && a < b);
}

static void merge_fix(tk_merge *nodes, size_t *heap, size_t length, size_t node)
{
    size_t p = nodes[node].position;
    while (p && merge_less(nodes, node, heap[(p - 1) / 2])) {
        heap[p] = heap[(p - 1) / 2];
        nodes[heap[p]].position = p;
        p = (p - 1) / 2;
    }
    while (p < length / 2) {
        size_t child = 2 * p + 1;
        if (child + 1 < length && merge_less(nodes, heap[child + 1], heap[child])) {
            child++;
        }
        if (!merge_less(nodes, heap[child], node)) {
            break;
        }
        heap[p] = heap[child];
        nodes[heap[p]].position = p;
        p = child;
    }
    heap[p] = node;
    nodes[node].position = p;
}

static int bpe(const lunelette_tokenizer *t, const unsigned char *text, size_t length,
               uint32_t *ids, size_t *count)
{
    unsigned rank = lookup(t, text, length);
    if (rank != TK_NONE) {
        ids[(*count)++] = rank + 1000;
        return 1;
    }
    if (!length) {
        return 1;
    }
    if (length > SIZE_MAX / (sizeof(tk_merge) + sizeof(size_t))) {
        return 0;
    }
    tk_merge *nodes = malloc(length * (sizeof(*nodes) + sizeof(size_t)));
    if (!nodes) {
        return 0;
    }
    size_t *heap = (size_t *)(nodes + length);
    for (size_t i = 0; i < length; i++) {
        nodes[i].previous = i ? i - 1 : length;
        nodes[i].next = i + 1;
        nodes[i].end = i + 1;
        nodes[i].position = i;
        nodes[i].rank = TK_NONE;
        heap[i] = i;
    }
    for (size_t i = 0; i + 1 < length; i++) {
        nodes[i].rank = lookup(t, text + i, 2);
        merge_fix(nodes, heap, length, i);
    }
    while (nodes[heap[0]].rank != TK_NONE) {
        size_t left = heap[0];
        size_t right = nodes[left].next;
        size_t next = nodes[right].next;
        size_t previous = nodes[left].previous;
        nodes[left].end = nodes[right].end;
        nodes[left].next = next;
        if (next < length) {
            nodes[next].previous = left;
        }
        nodes[right].rank = TK_NONE;
        merge_fix(nodes, heap, length, right);
        nodes[left].rank = next < length ?
            lookup(t, text + left, nodes[next].end - left) : TK_NONE;
        merge_fix(nodes, heap, length, left);
        if (previous < length) {
            nodes[previous].rank = lookup(t, text + previous, nodes[left].end - previous);
            merge_fix(nodes, heap, length, previous);
        }
    }
    for (size_t i = 0; i < length; i = nodes[i].next) {
        rank = lookup(t, text + i, nodes[i].end - i);
        ids[(*count)++] = rank + 1000;
    }
    free(nodes);
    return 1;
}

int lunelette_tokenizer_encode(const lunelette_tokenizer *t,
                               const unsigned char *text, size_t length,
                               uint32_t **ids, size_t *count, char *error, size_t error_size)
{
    return lunelette_tokenizer_encode_with_limit(t, text, length,
        LUNELETTE_TOKENIZER_MAX_INPUT, ids, count, error, error_size);
}

int lunelette_tokenizer_encode_with_limit(const lunelette_tokenizer *t,
                               const unsigned char *text, size_t length,
                               size_t input_limit, uint32_t **ids, size_t *count, char *error, size_t error_size)
{
    tk_character local_chars[LUNELETTE_TOKENIZER_MAX_INPUT + 1];
    tk_character *chars = local_chars;
    size_t n = 0;
    size_t offset = 0;
    if (ids) {
        *ids = NULL;
    }
    if (count) {
        *count = 0;
    }
    if (!t || !ids || !count || (!text && length) || (input_limit != LUNELETTE_TOKENIZER_MAX_INPUT &&
        input_limit != LUNELETTE_TOKENIZER_LONG_INPUT) || length > input_limit) {
        return fail(error, error_size, input_limit == LUNELETTE_TOKENIZER_MAX_INPUT ?
            "tokenizer encode: invalid arguments or input exceeds 4096 bytes" :
            "tokenizer encode: invalid arguments or input exceeds 32768 bytes");
    }
    if (length > LUNELETTE_TOKENIZER_MAX_INPUT) {
        chars = malloc((length + 1) * sizeof(*chars));
        if (!chars) {
            return fail(error, error_size, "tokenizer encode: allocation failed");
        }
    }
    while (offset < length) {
        size_t width;
        unsigned cp;
        if (!tj_utf8(text + offset, length - offset, &width, &cp)) {
            if (chars != local_chars) {
                free(chars);
            }
            return fail(error, error_size, "tokenizer encode: invalid UTF-8 at byte %zu", offset);
        }
        chars[n].cp = cp;
        chars[n].mask = unicode_mask(cp);
        chars[n++].offset = offset;
        offset += width;
    }
    chars[n].offset = length;
    uint32_t *result = malloc((length ? length : 1) * sizeof(*result));
    if (!result) {
        if (chars != local_chars) {
            free(chars);
        }
        return fail(error, error_size, "tokenizer encode: allocation failed");
    }
    size_t start = 0;
    while (start < n) {
        size_t end = piece_end(chars, n, start);
        if (end <= start || end > n) {
            if (chars != local_chars) {
                free(chars);
            }
            free(result);
            *count = 0;
            return fail(error, error_size, "tokenizer encode: internal splitter coverage error");
        }
        if (!bpe(t, text + chars[start].offset, chars[end].offset - chars[start].offset,
                 result, count)) {
            if (chars != local_chars) {
                free(chars);
            }
            free(result);
            *count = 0;
            return fail(error, error_size, "tokenizer encode: allocation failed");
        }
        start = end;
    }
    if (chars != local_chars) {
        free(chars);
    }
    *ids = result;
    return 1;
}

int lunelette_tokenizer_decode(const lunelette_tokenizer *t, const uint32_t *ids, size_t count,
                               unsigned char **bytes, size_t *length, char *error, size_t error_size)
{
    size_t total = 0;
    if (bytes) {
        *bytes = NULL;
    }
    if (length) {
        *length = 0;
    }
    if (!t || !bytes || !length || (!ids && count) || count > LUNELETTE_TOKENIZER_MAX_INPUT) {
        return fail(error, error_size, "tokenizer decode: invalid arguments or more than 4096 IDs");
    }
    for (size_t i = 0; i < count; i++) {
        if (ids[i] < 1000 || ids[i] >= 131072) {
            return fail(error, error_size, "tokenizer decode: ID %u at %zu is special or out of range", ids[i], i);
        }
        size_t n = t->tokens[ids[i] - 1000].length;
        if (n > TK_MAX_BYTES - total) {
            return fail(error, error_size, "tokenizer decode: output exceeds 4 MiB");
        }
        total += n;
    }
    unsigned char *result = malloc(total + 1);
    if (!result) {
        return fail(error, error_size, "tokenizer decode: allocation failed");
    }
    size_t position = 0;
    for (size_t i = 0; i < count; i++) {
        const tk_token *token = &t->tokens[ids[i] - 1000];
        memcpy(result + position, token->bytes, token->length);
        position += token->length;
    }
    result[total] = 0;
    *bytes = result;
    *length = total;
    return 1;
}
