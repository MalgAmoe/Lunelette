#include "chat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int lunelette_chat_prompt(const lunelette_tokenizer *tokenizer, const char *prompt,
    const char *system, int raw, int64_t ids[LUNELETTE_CONTEXT_LIMIT], size_t *count,
    char *error, size_t error_size)
{
    uint32_t *user_ids = NULL;
    uint32_t *system_ids = NULL;
    size_t user_count = 0, system_count = 0;
    int status = -1;
    if (!tokenizer || !prompt || !ids || !count || (raw && system)) {
        if (error && error_size) {
            snprintf(error, error_size, "Invalid prompt arguments (raw forbids system)");
        }
        return -1;
    }
    if (!lunelette_tokenizer_encode(tokenizer, (const unsigned char *)prompt,
            strlen(prompt), &user_ids, &user_count, error, error_size)) {
        goto cleanup;
    }
    if (system && !lunelette_tokenizer_encode(tokenizer, (const unsigned char *)system,
            strlen(system), &system_ids, &system_count, error, error_size)) {
        goto cleanup;
    }
    size_t total = 1 + user_count + system_count + (raw ? 0 : 2) + (system ? 2 : 0);
    if (total > LUNELETTE_CONTEXT_LIMIT) {
        if (error && error_size) {
            snprintf(error, error_size, "Prompt exceeds 256-token context (%zu tokens)", total);
        }
        goto cleanup;
    }
    size_t n = 0;
    ids[n++] = LUNELETTE_TOKEN_BOS;
    if (system) {
        ids[n++] = LUNELETTE_TOKEN_SYSTEM;
        for (size_t i = 0; i < system_count; i++) {
            ids[n++] = system_ids[i];
        }
        ids[n++] = LUNELETTE_TOKEN_END_SYSTEM;
    }
    if (!raw) {
        ids[n++] = LUNELETTE_TOKEN_INST;
    }
    for (size_t i = 0; i < user_count; i++) {
        ids[n++] = user_ids[i];
    }
    if (!raw) {
        ids[n++] = LUNELETTE_TOKEN_END_INST;
    }
    *count = n;
    status = 0;
cleanup:
    free(user_ids);
    free(system_ids);
    return status;
}
