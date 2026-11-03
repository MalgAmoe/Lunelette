#ifndef LUNELETTE_TOKENIZER_H
#define LUNELETTE_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define LUNELETTE_TOKENIZER_MAX_INPUT 4096u
#define LUNELETTE_TOKENIZER_VOCAB_SIZE 131072u
#define LUNELETTE_TOKENIZER_SPECIAL_COUNT 1000u

/* Explicit building blocks, not a chat template. No implicit BOS/system prompt. */
enum lunelette_special_token {
    LUNELETTE_TOKEN_BOS = 1,
    LUNELETTE_TOKEN_EOS = 2,
    LUNELETTE_TOKEN_INST = 3,
    LUNELETTE_TOKEN_END_INST = 4,
    LUNELETTE_TOKEN_SYSTEM = 17,
    LUNELETTE_TOKEN_END_SYSTEM = 18
};

typedef struct lunelette_tokenizer lunelette_tokenizer;

/* Tokenizer open/encode/decode return 1 on success, 0 on failure (boolean
 * convention, unlike the engine's 0/-1 numerical and generation APIs). */

/* Load exactly the supported Tekken13 JSON schema/config. No tokenizer.json or
 * Python dependency at runtime. Immutable after load; concurrent reads are safe.
 * On failure *out is NULL; error may be NULL iff error_size is zero. */
int lunelette_tokenizer_open(const char *path, lunelette_tokenizer **out,
                             char *error, size_t error_size);
void lunelette_tokenizer_close(lunelette_tokenizer *tokenizer);

/* Ordinary text only: strict UTF-8, including embedded NUL. Literal special-token
 * spellings are ordinary text, never interpreted as control IDs. No BOS/EOS.
 * At most MAX_INPUT bytes. Caller frees *ids. On error *ids=NULL, *count=0. */
int lunelette_tokenizer_encode(const lunelette_tokenizer *tokenizer,
                               const unsigned char *text, size_t length,
                               uint32_t **ids, size_t *count,
                               char *error, size_t error_size);

/* Lossless raw bytes (NOT necessarily UTF-8), concatenated across token boundaries.
 * Special IDs 0..999 are rejected, never silently skipped/rendered. Caller handles
 * explicit framing/EOS before calling. At most MAX_INPUT IDs, 4 MiB output.
 * Caller frees *bytes; length excludes an extra convenience NUL. On error outputs
 * are NULL/0. No partial output is returned by any operation. */
int lunelette_tokenizer_decode(const lunelette_tokenizer *tokenizer,
                               const uint32_t *ids, size_t count,
                               unsigned char **bytes, size_t *length,
                               char *error, size_t error_size);

#endif
