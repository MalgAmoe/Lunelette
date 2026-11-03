#ifndef LUNELETTE_H
#define LUNELETTE_H

#define LUNELETTE_VERSION "0.1.0-dev"

/* Normalize count logits into probabilities. Supports in-place operation.
 * Returns 0 on success; -1 for null pointers, zero count, or non-finite input.
 * On invalid input, output is left unchanged.
 */
#include <stddef.h>
int lunelette_softmax(const float *input, float *output, size_t count);

#endif
