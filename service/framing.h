#ifndef LUNELETTE_SERVICE_FRAMING_H
#define LUNELETTE_SERVICE_FRAMING_H
#include "tokenizer.h"
/* Native no-system/nonthinking text subset. Output unchanged on failure.
 * capacity is available IDs on entry, exact framed count on success. */
int service_frame(const lunelette_tokenizer *tokenizer, const unsigned char *text,
                  size_t length, int initial, unsigned *output, unsigned *capacity);
/* Explicit 4096/32768-byte admission; same ownership/framing as above. */
int service_frame_with_limit(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, unsigned *output, unsigned *capacity);
#endif
