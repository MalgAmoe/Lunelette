#include "framing.h"
#include <stdlib.h>

int service_frame(const lunelette_tokenizer *tokenizer, const unsigned char *text,
                  size_t length, int initial, unsigned *output, unsigned *capacity)
{
    uint32_t *ids = NULL;
    size_t count = 0;
    char error[512];
    if (!lunelette_tokenizer_encode(tokenizer, text, length, &ids, &count,
                                    error, sizeof(error))) {
        return 0;
    }
    unsigned structural = initial ? 3 : 2;
    if (count + structural > *capacity) {
        free(ids);
        return 0;
    }
    unsigned used = 0;
    if (initial) {
        output[used++] = LUNELETTE_TOKEN_BOS;
    }
    output[used++] = LUNELETTE_TOKEN_INST;
    for (size_t i = 0; i < count; i++) {
        output[used++] = ids[i];
    }
    output[used++] = LUNELETTE_TOKEN_END_INST;
    *capacity = used;
    free(ids);
    return 1;
}
