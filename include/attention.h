#ifndef LUNELETTE_ATTENTION_H
#define LUNELETTE_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

/* Fixed Ministral-3-3B-Instruct-2512 YaRN, not a generic RoPE configuration.
 * In-place Q[tokens][32][128], K[tokens][8][128], split-half rotation.
 * positions[tokens] are absolute positions, each < 262144; arbitrary order is
 * allowed (causality in the separate attention kernel follows row order).
 * Caller owns all buffers; Q, K and positions must be mutually disjoint.
 * Returns -1 on null, zero/overflow size or out-of-range position, before writes.
 * No allocation/state. Elements are not scanned for finiteness.
 */
int lunelette_ministral3_yarn(float *query, float *key, const uint32_t *positions,
                            size_t tokens);

/* Single-sequence scalar float32 causal prefill, no cache/padding/dropout.
 * Q[tokens][query_heads][head_dim], K/V[tokens][kv_heads][head_dim],
 * output[tokens][query_heads][head_dim], contiguous token-major arrays.
 * Query head h uses KV head h/(query_heads/kv_heads); scale = 1/sqrt(head_dim).
 * Caller owns buffers. Output and scratch[tokens] must be mutually disjoint
 * and disjoint from all inputs. Read-only inputs may overlap. No allocation or
 * retained state; independent buffers permit concurrent calls.
 * Returns -1 before writes on null, zero, non-divisible heads or byte overflow.
 * Returns -1 on non-finite attention scores, possibly after partial output and
 * scratch modification. Inputs should be finite; V is not scanned. Float dot
 * products can overflow. Stable max-subtracted softmax uses float reductions.
 */
int lunelette_causal_gqa(const float *query, const float *key, const float *value,
                         float *output, float *scratch, size_t tokens,
                         size_t query_heads, size_t kv_heads, size_t head_dim);

#endif
