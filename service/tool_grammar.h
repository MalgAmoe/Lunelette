/* Private stateless schema checks for the three fixed native tool profiles. */
#ifndef LUNELETTE_TOOL_GRAMMAR_H
#define LUNELETTE_TOOL_GRAMMAR_H
#include "tokenizer_json.h"
#include <stdint.h>

#define SERVICE_TOOL_CALCULATOR_FIELDS "|operation||a||b|"
#define SERVICE_TOOL_LOOKUP_FIELDS "|document_id|"
#define SERVICE_TOOL_RESULT_FIELDS "|document_id||passage_id||text|"

typedef struct {
    unsigned operation; /* add = 1, subtract = 2, multiply = 3 */
    int a, b;
} service_tool_calculation;

/* Documents are borrowed, already validated by service_json with the matching
 * field allowlist above (including its scalar, NUL, duplicate and size checks).
 * Result documents additionally require the session's 2048-byte input limit.
 * No document storage is retained or modified. Callers own parsing/freeing,
 * request mutation, synchronization and publication. No session dependency. */
/* On failure output is untouched; arguments admit -0 and magnitude <= 1000000. */
int service_tool_calculator(const tj_document *d, service_tool_calculation *output);
/* Returns document 1/2 for profile 2, document 3 for profile 3, otherwise zero. */
unsigned service_tool_lookup(const tj_document *d, unsigned profile);
/* Checks matching document/passage and nonempty text of at most 1024 bytes. */
int service_tool_lookup_result(const tj_document *d, unsigned document);
/* Borrowed immutable name; document must be in 0..3 (zero names no document). */
const char *service_tool_document(unsigned document);
/* Decimal canonical result, magnitude <= 10^12; rejects -0. text is borrowed
 * for length bytes (need not be NUL-terminated). Failure leaves value untouched. */
int service_tool_calculator_result(const unsigned char *text, size_t length, int64_t *value);
#endif
