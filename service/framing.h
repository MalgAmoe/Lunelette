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
/* Exact V13 available-tools JSON (mistral_common json.dumps spacing, no strict field).
 * Borrowed immutable string; used verbatim for fixture generation. */
const char *service_calculator_v13_declaration(void);
/* Opt-in user framing: initial is 1 5 declaration 6 3 text 4, later is 3 text 4.
 * result instead frames 7 text 8 (ignores initial, no EOS); caller validates result schema.
 * input_limit must be 4096 or 32768. Same output/capacity contract as service_frame. */
int service_frame_tools(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, int result,
    unsigned *output, unsigned *capacity);
/* Frozen two-tool V13 declaration; borrowed immutable string.
 * Investigation framing has the same limits and failure contract as tools framing. */
const char *service_investigation_v13_declaration(void);
int service_frame_investigation(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, int result,
    unsigned *output, unsigned *capacity);
/* Single saved synthetic observation lookup; same framing/failure contract. */
const char *service_observation_v13_declaration(void);
int service_frame_observation(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, int result,
    unsigned *output, unsigned *capacity);
#endif
