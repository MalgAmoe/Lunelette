#ifndef LUNELETTE_BLOCK_H
#define LUNELETTE_BLOCK_H

#include "safetensors.h"

/* Fixed Ministral-3-3B layer 0, CPU FP32, single sequence, no cache/padding.
 * Reads selected BF16 weights from one already-open shard (borrowed, unchanged).
 * IDs[count] must be in [0,131072), count in [1,256]; positions are 0..count-1.
 * Caller owns output[count][3072], disjoint from IDs and shard descriptors.
 * Internally owns/frees workspace and one expanded matrix (at most 108 MiB).
 * No retained state. Same shard cannot be used concurrently (stdio seeks).
 * Returns 0 or -1 with context in error[capacity] when supplied. Output is
 * unchanged on failure. No tokenizer, final model norm, logits or other layers.
 */
int lunelette_first_block(st_file *shard, const int64_t *ids, size_t count,
                          float *output, char *error, size_t capacity);

#endif
