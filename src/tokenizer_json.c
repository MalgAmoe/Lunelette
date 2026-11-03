#include "tokenizer_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TJ_MAX_FILE (32u * 1024u * 1024u)
#define TJ_MAX_NODES 1500000u

int tj_utf8(const unsigned char *s, size_t n, size_t *width, unsigned *codepoint)
{
    unsigned c;
    size_t k;
    if (!n) {
        return 0;
    }
    if (s[0] < 128) {
        *width = 1;
        *codepoint = s[0];
        return 1;
    }
    if (s[0] >= 0xc2 && s[0] <= 0xdf) {
        k = 2;
        c = s[0] & 31;
    } else if (s[0] >= 0xe0 && s[0] <= 0xef) {
        k = 3;
        c = s[0] & 15;
    } else if (s[0] >= 0xf0 && s[0] <= 0xf4) {
        k = 4;
        c = s[0] & 7;
    } else {
        return 0;
    }
    if (n < k) {
        return 0;
    }
    for (size_t i = 1; i < k; i++) {
        if ((s[i] & 0xc0) != 0x80) {
            return 0;
        }
        c = (c << 6) | (s[i] & 63);
    }
    if ((k == 2 && c < 128) || (k == 3 && c < 2048) ||
        (k == 4 && c < 65536) || c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff)) {
        return 0;
    }
    *width = k;
    *codepoint = c;
    return 1;
}

static void whitespace(tj_document *d)
{
    while (d->position < d->length && strchr(" \t\r\n", d->data[d->position]) &&
           d->data[d->position] != 0) {
        d->position++;
    }
}

static int hex4(tj_document *d, unsigned *value)
{
    unsigned v = 0;
    if (d->length - d->position < 4) {
        return 0;
    }
    for (unsigned i = 0; i < 4; i++) {
        unsigned c = d->data[d->position++];
        if (c >= '0' && c <= '9') {
            c -= '0';
        } else if (c >= 'a' && c <= 'f') {
            c = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            c = c - 'A' + 10;
        } else {
            return 0;
        }
        v = v * 16 + c;
    }
    *value = v;
    return 1;
}

static int string(tj_document *d, unsigned index)
{
    size_t start = ++d->position;
    size_t out = start;
    while (d->position < d->length) {
        if (out - start > 4096) {
            return 0;
        }
        unsigned c = d->data[d->position++];
        if (c == '"') {
            d->nodes[index].text = d->data + start;
            d->nodes[index].length = out - start;
            return 1;
        }
        if (c < 32) {
            return 0;
        }
        if (c == '\\') {
            if (d->position == d->length) {
                return 0;
            }
            c = d->data[d->position++];
            if (c == 'u') {
                if (!hex4(d, &c)) {
                    return 0;
                }
                if (c >= 0xd800 && c <= 0xdbff) {
                    unsigned low;
                    if (d->length - d->position < 6 || d->data[d->position++] != '\\' ||
                        d->data[d->position++] != 'u' || !hex4(d, &low) ||
                        low < 0xdc00 || low > 0xdfff) {
                        return 0;
                    }
                    c = 65536 + ((c - 0xd800) << 10) + low - 0xdc00;
                } else if (c >= 0xdc00 && c <= 0xdfff) {
                    return 0;
                }
                if (c < 128) {
                    d->data[out++] = (unsigned char)c;
                } else {
                    if (c >= 65536) {
                        d->data[out++] = (unsigned char)(0xf0 | (c >> 18));
                        d->data[out++] = (unsigned char)(0x80 | ((c >> 12) & 63));
                    } else if (c >= 2048) {
                        d->data[out++] = (unsigned char)(0xe0 | (c >> 12));
                    }
                    d->data[out++] = (unsigned char)((c < 2048 ? 0xc0 : 0x80) |
                                                    ((c >> 6) & 63));
                    d->data[out++] = (unsigned char)(0x80 | (c & 63));
                }
                continue;
            }
            switch (c) {
                case '"': case '\\': case '/': break;
                case 'b': c = 8; break;
                case 'f': c = 12; break;
                case 'n': c = 10; break;
                case 'r': c = 13; break;
                case 't': c = 9; break;
                default: return 0;
            }
        } else if (c >= 128) {
            size_t width;
            unsigned cp;
            d->position--;
            if (!tj_utf8(d->data + d->position, d->length - d->position, &width, &cp)) {
                return 0;
            }
            memmove(d->data + out, d->data + d->position, width);
            out += width;
            d->position += width;
            continue;
        }
        d->data[out++] = (unsigned char)c;
    }
    return 0;
}

static int digit(unsigned c)
{
    return c >= '0' && c <= '9';
}

static unsigned parse(tj_document *d, unsigned depth)
{
    unsigned index;
    char c;
    if (depth > 32 || d->count == TJ_MAX_NODES) {
        return 0;
    }
    if (d->count >= d->capacity) {
        unsigned capacity = d->capacity ? d->capacity * 2 : 1024;
        tj_node *nodes;
        if (capacity > TJ_MAX_NODES) {
            capacity = TJ_MAX_NODES;
        }
        nodes = realloc(d->nodes, capacity * sizeof(*nodes));
        if (!nodes) {
            return 0;
        }
        d->nodes = nodes;
        d->capacity = capacity;
    }
    index = d->count++;
    memset(&d->nodes[index], 0, sizeof(d->nodes[index]));
    whitespace(d);
    if (d->position == d->length) {
        return 0;
    }
    c = (char)d->data[d->position];
    d->nodes[index].kind = c;
    if (c == '"') {
        return string(d, index) ? index : 0;
    }
    if (c == '{' || c == '[') {
        unsigned tail = 0;
        unsigned key = 0;
        unsigned members = 0;
        char end = c == '{' ? '}' : ']';
        d->position++;
        whitespace(d);
        if (d->position < d->length && d->data[d->position] == end) {
            d->position++;
            return index;
        }
        for (;;) {
            if (++members > (c == '{' ? 64u : 131072u)) {
                return 0;
            }
            unsigned child = parse(d, depth + 1);
            if (!child) {
                return 0;
            }
            if (c == '{') {
                if (d->nodes[child].kind != '"') {
                    return 0;
                }
                for (unsigned p = d->nodes[index].child; p; p = d->nodes[d->nodes[p].next].next) {
                    if (d->nodes[p].length == d->nodes[child].length &&
                        !memcmp(d->nodes[p].text, d->nodes[child].text, d->nodes[p].length)) {
                        return 0;
                    }
                }
                key = child;
                whitespace(d);
                if (d->position == d->length || d->data[d->position++] != ':') {
                    return 0;
                }
                child = parse(d, depth + 1);
                if (!child) {
                    return 0;
                }
                d->nodes[key].next = child;
            }
            if (tail) {
                d->nodes[tail].next = key ? key : child;
            } else {
                d->nodes[index].child = key ? key : child;
            }
            tail = child;
            whitespace(d);
            if (d->position == d->length) {
                return 0;
            }
            if (d->data[d->position] == end) {
                d->position++;
                return index;
            }
            if (d->data[d->position++] != ',') {
                return 0;
            }
        }
    }
    size_t start = d->position;
    if (c == '-' || digit((unsigned char)c)) {
        if (c == '-') {
            d->position++;
        }
        if (d->position == d->length || !digit(d->data[d->position])) {
            return 0;
        }
        if (d->data[d->position++] != '0') {
            while (d->position < d->length && digit(d->data[d->position])) {
                d->position++;
            }
        }
        if (d->position < d->length && d->data[d->position] == '.') {
            d->position++;
            size_t first = d->position;
            while (d->position < d->length && digit(d->data[d->position])) {
                d->position++;
            }
            if (first == d->position) {
                return 0;
            }
        }
        if (d->position < d->length && strchr("eE", d->data[d->position]) &&
            d->data[d->position] != 0) {
            d->position++;
            if (d->position < d->length && (d->data[d->position] == '+' ||
                                           d->data[d->position] == '-')) {
                d->position++;
            }
            size_t first = d->position;
            while (d->position < d->length && digit(d->data[d->position])) {
                d->position++;
            }
            if (first == d->position) {
                return 0;
            }
        }
        d->nodes[index].kind = '0';
    } else {
        const char *literal = c == 't' ? "true" : c == 'f' ? "false" : c == 'n' ? "null" : "";
        size_t n = strlen(literal);
        if (!n || d->length - d->position < n || memcmp(d->data + d->position, literal, n)) {
            return 0;
        }
        d->position += n;
    }
    if (d->position - start > 128) {
        return 0;
    }
    d->nodes[index].text = d->data + start;
    d->nodes[index].length = d->position - start;
    return index;
}

int tj_read(const char *path, tj_document *d)
{
    FILE *file;
    long length;
    memset(d, 0, sizeof(*d));
    file = fopen(path, "rb");
    if (!file) {
        return 0;
    }
    if (fseek(file, 0, SEEK_END) || (length = ftell(file)) < 0 ||
        (unsigned long)length > TJ_MAX_FILE || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return 0;
    }
    d->length = (size_t)length;
    d->data = malloc(d->length + 1);
    if (!d->data) {
        fclose(file);
        return 0;
    }
    size_t got = fread(d->data, 1, d->length, file);
    int failed = ferror(file);
    fclose(file);
    if (got != d->length || failed) {
        tj_free(d);
        return 0;
    }
    d->count = 1; /* zero is the missing/error sentinel */
    if (parse(d, 0) != 1) {
        tj_free(d);
        return 0;
    }
    whitespace(d);
    if (d->position != d->length) {
        tj_free(d);
        return 0;
    }
    return 1;
}

void tj_free(tj_document *d)
{
    free(d->data);
    free(d->nodes);
    memset(d, 0, sizeof(*d));
}

unsigned tj_field(const tj_document *d, unsigned object, const char *key)
{
    if (!object || d->nodes[object].kind != '{') {
        return 0;
    }
    for (unsigned p = d->nodes[object].child; p; p = d->nodes[d->nodes[p].next].next) {
        if (d->nodes[p].length == strlen(key) && !memcmp(d->nodes[p].text, key, strlen(key))) {
            return d->nodes[p].next;
        }
    }
    return 0;
}

int tj_integer(const tj_document *d, unsigned node, unsigned *value)
{
    unsigned v = 0;
    if (!node || d->nodes[node].kind != '0' || !d->nodes[node].length) {
        return 0;
    }
    for (size_t i = 0; i < d->nodes[node].length; i++) {
        unsigned c = d->nodes[node].text[i];
        if (!digit(c) || v > (0xffffffffu - (c - '0')) / 10) {
            return 0;
        }
        v = v * 10 + c - '0';
    }
    *value = v;
    return 1;
}
