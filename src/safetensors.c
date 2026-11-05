#include <math.h>
#include "safetensors.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "requires IEEE binary32 float");
/* Deliberately shallow schema parser: no recursion, implicit conversions, or
 * unbounded scans. Strings are printable ASCII (including JSON escapes). */
typedef struct {
    const unsigned char *p;
    const unsigned char *end;
} json_parser;

static void skip_whitespace(json_parser *j)
{
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\n' || *j->p == '\r' || *j->p == '\t')) {
        j->p++;
    }
}

static int consume_character(json_parser *j, unsigned char c)
{
    skip_whitespace(j);
    if (j->p == j->end || *j->p != c) {
        return 0;
    }
    j->p++;
    return 1;
}

static int parse_string(json_parser *j, char *out, size_t cap)
{
    size_t n = 0;
    if (!consume_character(j, '"')) {
        return 0;
    }
    while (j->p < j->end) {
        unsigned c = *j->p++;
        if (c == '"') {
            out[n] = 0;
            return 1;
        }
        if (c == '\\') {
            if (j->p == j->end) {
                return 0;
            }
            c = *j->p++;
            if (c == 'u') {
                c = 0;
                for (int i = 0; i < 4; i++) {
                    if (j->p == j->end) {
                        return 0;
                    }
                    unsigned h = *j->p++, v;
                    if (h >= '0' && h <= '9') {
                        v = h - '0';
                    }
                    else if (h >= 'a' && h <= 'f') {
                        v = h - 'a' + 10;
                    }
                    else if (h >= 'A' && h <= 'F') {
                        v = h - 'A' + 10;
                    }
                    else {
                        return 0;
                    }
                    c = c * 16 + v;
                }
            } else if (c != '"' && c != '\\' && c != '/') {
                return 0;
            }
        }
        if (c < 32 || c > 126 || n + 1 >= cap) {
            return 0;
        }
        out[n++] = (char)c;
    }
    return 0;
}

static int parse_uint64(json_parser *j, uint64_t *out)
{
    skip_whitespace(j);
    uint64_t v = 0;
    if (j->p == j->end || *j->p < '0' || *j->p > '9') {
        return 0;
    }
    int zero = *j->p == '0';
    do {
        unsigned d = *j->p++ - '0';
        if (v > (UINT64_MAX - d) / 10) {
            return 0;
        }
        v = v * 10 + d;
        if (j->p == j->end || *j->p < '0' || *j->p > '9') {
            break;
        }
        if (zero) {
            return 0;
        }
    } while (1);
    *out = v;
    return 1;
}

static int parse_uint64_array(json_parser *j, uint64_t *a, size_t cap, size_t *n)
{
    *n = 0;
    if (!consume_character(j, '[')) {
        return 0;
    }
    if (consume_character(j, ']')) {
        return 1;
    }
    do {
        if (*n == cap || !parse_uint64(j, &a[(*n)++])) {
            return 0;
        }
        if (consume_character(j, ']')) {
            return 1;
        }
    } while (consume_character(j, ','));
    return 0;
}

static int parse_tensor(json_parser *j, st_tensor *t)
{
    unsigned seen = 0;
    char key[ST_MAX_NAME], dtype[16];
    if (!consume_character(j, '{')) {
        return 0;
    }
    do {
        unsigned bit;
        if (!parse_string(j, key, sizeof key) || !consume_character(j, ':')) {
            return 0;
        }
        if (!strcmp(key, "dtype")) {
            bit = 1;
            if (!parse_string(j, dtype, sizeof dtype)) {
                return 0;
            }
            if (!strcmp(dtype, "F8_E4M3")) {
                t->dtype = ST_F8_E4M3;
            } else if (!strcmp(dtype, "BF16")) {
                t->dtype = ST_BF16;
            } else if (!strcmp(dtype, "F32")) {
                t->dtype = ST_F32;
            } else if (!strcmp(dtype, "I64")) {
                t->dtype = ST_I64;
            } else {
                return 0;
            }
        } else if (!strcmp(key, "shape")) {
            bit = 2;
            if (!parse_uint64_array(j, t->shape, ST_MAX_DIMS, &t->rank)) {
                return 0;
            }
        } else if (!strcmp(key, "data_offsets")) {
            uint64_t a[2];
            size_t n;
            bit = 4;
            if (!parse_uint64_array(j, a, 2, &n) || n != 2) {
                return 0;
            }
            t->begin = a[0];
            t->end = a[1];
        } else {
            return 0;
        }
        if (seen & bit) {
            return 0;
        }
        seen |= bit;
        if (consume_character(j, '}')) {
            return seen == 7;
        }
    } while (consume_character(j, ','));
    return 0;
}

static int parse_metadata(json_parser *j)
{
    char keys[64][ST_MAX_NAME], value[4096];
    size_t n = 0;
    if (!consume_character(j, '{')) {
        return 0;
    }
    if (consume_character(j, '}')) {
        return 1;
    }
    do {
        if (n == 64 || !parse_string(j, keys[n], sizeof keys[n])) {
            return 0;
        }
        for (size_t i = 0; i < n; i++) {
            if (!strcmp(keys[i], keys[n])) {
                return 0;
            }
        }
        n++;
        if (!consume_character(j, ':') || !parse_string(j, value, sizeof value)) {
            return 0;
        }
        if (consume_character(j, '}')) {
            return 1;
        }
    } while (consume_character(j, ','));
    return 0;
}

static uint64_t decode_le64(const unsigned char *p)
{
    uint64_t v = 0;
    for (unsigned i = 0; i < 8; i++) {
        v |= (uint64_t)p[i] << (8 * i);
    }
    return v;
}

void st_close(st_file *s)
{
    if (!s) {
        return;
    }
    if (s->file) {
        fclose(s->file);
    }
    free(s->tensors);
    memset(s, 0, sizeof *s);
}

const st_tensor *st_find(const st_file *s, const char *name)
{
    if (!s || !name) {
        return NULL;
    }
    for (size_t i = 0; i < s->count; i++) {
        if (!strcmp(s->tensors[i].name, name)) {
            return &s->tensors[i];
        }
    }
    return NULL;
}

static int parse_header(st_file *s, const unsigned char *header, size_t length)
{
    json_parser j = {header, header + length};
    int meta_seen = 0;
    /* Safetensors requires the first byte to be '{'. */
    if (*header != '{' || !consume_character(&j, '{')) {
        return -1;
    }
    if (!consume_character(&j, '}')) {
        do {
            char name[ST_MAX_NAME];
            if (!parse_string(&j, name, sizeof name) || !name[0] || !consume_character(&j, ':')) {
                return -1;
            }
            if (!strcmp(name, "__metadata__")) {
                if (meta_seen++ || !parse_metadata(&j)) {
                    return -1;
                }
            } else {
                if (s->count == ST_MAX_TENSORS || st_find(s, name)) {
                    return -1;
                }
                st_tensor *t = &s->tensors[s->count];
                strcpy(t->name, name);
                if (!parse_tensor(&j, t)) {
                    return -1;
                }
                s->count++;
            }
            if (consume_character(&j, '}')) {
                break;
            }
            if (!consume_character(&j, ',')) {
                return -1;
            }
        } while (1);
    }
    skip_whitespace(&j);
    if (j.p != j.end) {
        return -1;
    }
    return 0;
}

static int validate_tensor_layout(const st_file *s, uint64_t data_bytes)
{
    /* Verify byte counts first, then a sorted interval view: gaps, overlaps and
     * trailing unindexed bytes are rejected; zero-length tensors are allowed. */
    for (size_t i = 0; i < s->count; i++) {
        st_tensor *t = &s->tensors[i];
        uint64_t elements = 1;
        int zero = 0;
        for (size_t k = 0; k < t->rank; k++) {
            if (t->shape[k] > INT64_MAX) {
                return -1;
            }
            if (!t->shape[k]) {
                zero = 1;
            }
        }
        if (zero) {
            elements = 0;
        } else {
            for (size_t k = 0; k < t->rank; k++) {
                if (elements > UINT64_MAX / t->shape[k]) {
                    return -1;
                }
                elements *= t->shape[k];
            }
        }
        if (elements > UINT64_MAX / t->dtype || t->end < t->begin ||
            t->end > data_bytes ||
            t->end - t->begin != elements * t->dtype) {
            return -1;
        }
    }
    /* Bounded O(n^2), keeps the public tensor list in header order. */
    uint64_t cursor = 0;
    size_t nonempty = 0;
    for (size_t i = 0; i < s->count; i++) {
        if (s->tensors[i].end > s->tensors[i].begin) {
            nonempty++;
        }
    }
    for (size_t i = 0; i < nonempty; i++) {
        const st_tensor *next = NULL;
        for (size_t k = 0; k < s->count; k++) {
            const st_tensor *t = &s->tensors[k];
            if (t->begin == cursor && t->end > cursor) {
                if (next) {
                    return -1;
                }
                next = t;
            }
        }
        if (!next) {
            return -1;
        }
        cursor = next->end;
    }
    if (cursor != data_bytes) {
        return -1;
    }
    /* Empty tensors must lie on interval boundaries, not inside another tensor. */
    for (size_t i = 0; i < s->count; i++) {
        const st_tensor *empty = &s->tensors[i];
        if (empty->begin != empty->end) {
            continue;
        }
        for (size_t k = 0; k < s->count; k++) {
            const st_tensor *other = &s->tensors[k];
            if (other->begin < empty->begin && empty->begin < other->end) {
                return -1;
            }
        }
    }
    return 0;
}

int st_open(st_file *s, const char *path, char *error, size_t capacity)
{
    unsigned char prefix[8], *header = NULL;
    const char *why = "invalid arguments";
    if (!s) {
        goto report;
    }
    memset(s, 0, sizeof *s);
    if (!path) {
        goto fail;
    }
    why = "cannot open/size file (requires seekable file <= LONG_MAX)";
    s->file = fopen(path, "rb");
    if (!s->file || fseek(s->file, 0, SEEK_END)) {
        goto fail;
    }
    long length = ftell(s->file);
    if (length < 8 || fseek(s->file, 0, SEEK_SET)) {
        goto fail;
    }
    why = "invalid/truncated header length (limit 1 MiB)";
    if (fread(prefix, 1, 8, s->file) != 8) {
        goto fail;
    }
    uint64_t n = decode_le64(prefix);
    if (n < 2 || n > ST_MAX_HEADER || n > (uint64_t)length - 8) {
        goto fail;
    }
    s->base = 8 + n;
    why = "allocation failure";
    header = malloc((size_t)n);
    s->tensors = calloc(ST_MAX_TENSORS, sizeof *s->tensors);
    if (!header || !s->tensors) {
        goto fail;
    }
    why = "truncated header";
    if (fread(header, 1, (size_t)n, s->file) != n) {
        goto fail;
    }
    why = "invalid/unsupported header: require unique ASCII names, F8_E4M3/BF16/F32/I64, rank <=8";
    if (parse_header(s, header, (size_t)n)) {
        goto fail;
    }
    why = "invalid tensor shape/size/offsets or non-contiguous data";
    if (validate_tensor_layout(s, (uint64_t)length - s->base)) {
        goto fail;
    }
    free(header);
    if (error && capacity) {
        error[0] = 0;
    }
    return 0;
fail:
    free(header);
    st_close(s);
report:
    if (error && capacity) {
        snprintf(error, capacity, "%s", why);
    }
    return -1;
}

int st_read(st_file *s, const st_tensor *t, uint64_t offset, void *out, size_t bytes)
{
    if (!s || !s->file || !t || (!out && bytes)) {
        return -1;
    }
    size_t i;
    for (i = 0; i < s->count; i++) {
        if (t == &s->tensors[i]) {
            break;
        }
    }
    if (i == s->count || offset > t->end - t->begin || bytes > t->end - t->begin - offset) {
        return -1;
    }
    uint64_t position = s->base + t->begin + offset;
    if (position > LONG_MAX || fseek(s->file, (long)position, SEEK_SET)) {
        return -1;
    }
    return !bytes || fread(out, 1, bytes, s->file) == bytes ? 0 : -1;
}

float st_bf16_to_f32(uint16_t bits)
{
    uint32_t word = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &word, sizeof value);
    return value;
}

int st_embedding(st_file *s, const st_tensor *t, int64_t token, float *out, size_t count)
{
    if (!t || !out || t->dtype != ST_BF16 || t->rank != 2 || token < 0 ||
        (uint64_t)token >= t->shape[0] || count != t->shape[1] || count > SIZE_MAX / 2) {
        return -1;
    }
    unsigned char *row = malloc(count ? count * 2 : 1);
    if (!row) {
        return -1;
    }
    int rc = st_read(s, t, (uint64_t)token * t->shape[1] * 2, row, count * 2);
    if (!rc) {
        for (size_t i = 0; i < count; i++) {
            out[i] = st_bf16_to_f32((uint16_t)(row[2 * i] | (unsigned)row[2 * i + 1] << 8));
        }
    }
    free(row);
    return rc;
}

int st_load_bf16(st_file *s, const st_tensor *t, float *out, size_t count)
{
    if (!t || !out || t->dtype != ST_BF16 || count > SIZE_MAX / sizeof(float) ||
        t->end < t->begin || t->end - t->begin != count * 2 ||
        st_read(s, t, 0, NULL, 0)) {
        return -1;
    }
    /* Bounded 64 KiB scratch amortizes seeks for large projection tensors.
     * Conversion order is unchanged; no weights are retained between calls. */
    unsigned char bytes[65536];
    for (size_t offset = 0; offset < count;) {
        size_t chunk = count - offset;
        if (chunk > sizeof bytes / 2) {
            chunk = sizeof bytes / 2;
        }
        if (st_read(s, t, (uint64_t)offset * 2, bytes, chunk * 2)) {
            return -1;
        }
        for (size_t i = 0; i < chunk; i++) {
            out[offset + i] = st_bf16_to_f32((uint16_t)(bytes[2 * i] |
                                                      (unsigned)bytes[2 * i + 1] << 8));
        }
        offset += chunk;
    }
    return 0;
}

float st_f8_e4m3_to_f32(unsigned char bits)
{
    unsigned exponent = (bits >> 3) & 15;
    unsigned fraction = bits & 7;
    float value;
    if (exponent == 15 && fraction == 7) {
        return NAN;
    }
    if (exponent == 0) {
        value = ldexpf((float)fraction, -9);
    } else {
        value = ldexpf(1.0f + (float)fraction / 8.0f, (int)exponent - 7);
    }
    return bits & 128 ? -value : value;
}

int st_load_fp8(st_file *s, const st_tensor *t, float scale, float *out, size_t count)
{
    if (!t || !out || t->dtype != ST_F8_E4M3 || !isfinite(scale) || scale <= 0 ||
        count > SIZE_MAX / sizeof(float) || t->end < t->begin ||
        t->end - t->begin != count || st_read(s, t, 0, NULL, 0)) {
        return -1;
    }
    unsigned char bytes[65536];
    for (size_t offset = 0; offset < count;) {
        size_t chunk = count - offset;
        if (chunk > sizeof bytes) {
            chunk = sizeof bytes;
        }
        if (st_read(s, t, offset, bytes, chunk) != 0) {
            return -1;
        }
        for (size_t i = 0; i < chunk; i++) {
            /* Both operands are FP32. Never round the reconstructed weight to BF16. */
            float value = st_f8_e4m3_to_f32(bytes[i]) * scale;
            if (!isfinite(value)) {
                return -1;
            }
            out[offset + i] = value;
        }
        offset += chunk;
    }
    return 0;
}
