/* Private bounded JSON reader, original implementation for the tokenizer. */
#ifndef LUNELETTE_TOKENIZER_JSON_H
#define LUNELETTE_TOKENIZER_JSON_H
#include <stddef.h>

typedef struct {
    unsigned char *text;
    size_t length;
    unsigned child;
    unsigned next;
    char kind;
} tj_node;
typedef struct {
    unsigned char *data;
    size_t length;
    size_t position;
    tj_node *nodes;
    unsigned count;
    unsigned capacity;
} tj_document;
int tj_read(const char *path, tj_document *doc);
/* Copies bounded input; document owns decoded storage. Same strict grammar as files.
 * data must be a valid pointer even when length is zero; doc must be non-NULL. */
int tj_read_memory(const void *data, size_t length, tj_document *doc);
void tj_free(tj_document *doc);
unsigned tj_field(const tj_document *doc, unsigned object, const char *key);
int tj_integer(const tj_document *doc, unsigned node, unsigned *value);
int tj_utf8(const unsigned char *s, size_t n, size_t *width, unsigned *codepoint);
#endif
