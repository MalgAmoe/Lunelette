#ifndef LUNELETTE_CHAT_H
#define LUNELETTE_CHAT_H
#include "tokenizer.h"
#include "generation.h"

/* Ordinary text is encoded separately from control IDs (literal special spellings
 * are not controls). Raw means BOS + prompt, and forbids a system message.
 * No implicit system text. Returns 0 on success, -1 on failure; output/count
 * unchanged on failure. Tokenizer and strings are borrowed for the call; caller
 * owns output. Text is UTF-8, NUL-terminated, each field at most
 * TOKENIZER_MAX_INPUT bytes. Error may be NULL iff error_size is zero. */
int lunelette_chat_prompt(const lunelette_tokenizer *tokenizer, const char *prompt,
    const char *system, int raw, int64_t ids[LUNELETTE_CONTEXT_LIMIT], size_t *count,
    char *error, size_t error_size);
#endif
