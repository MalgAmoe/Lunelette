#ifndef LUNELETTE_FORWARD_H
#define LUNELETTE_FORWARD_H

#include "safetensors.h"

/* Fixed Ministral-3-3B-Instruct-2512 text layout only: 26 layers, hidden 3072,
 * vocabulary 131072, epsilon 1e-5, tied embeddings. FP32 sequential projection/attention reductions; RMSNorm
 * uses FP32 squares, a double sum, then FP32 mean/scale/output.
 * Two explicitly opened, distinct shards are borrowed (seek positions change).
 * Required tensors may reside in either shard; duplicate required names are rejected.
 * ids[count], count 1..256, IDs 0..131071; positions start at zero.
 * Caller owns hidden[count][3072] (final RMS-normalized) and logits[131072]
 * (last token only). Optional hidden_13[count][3072] is the residual after
 * layer index 12, before layer 13 normalization. All output buffers must be
 * mutually disjoint and disjoint from IDs and shard metadata.
 * Outputs unchanged on any failure. Returns 0 or -1, diagnostic when supplied.
 * Workspace is internally allocated/freed; no retained state. Handles cannot
 * be used concurrently. No config inference, cache, batching, padding, tokenizer,
 * generation, vision, quantization, 8B/14B or accelerated backend support.
 */
int lunelette_forward(st_file *shard1, st_file *shard2, const int64_t *ids, size_t count,
                       float *hidden, float *logits, float *hidden_13,
                       char *error, size_t capacity);

#endif
