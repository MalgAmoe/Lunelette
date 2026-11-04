#ifndef LUNELETTE_MODEL_FACTS_H
#define LUNELETTE_MODEL_FACTS_H

#include <stddef.h>

/* Fixed Ministral-3-3B-Instruct-2512 text facts, not a configurable backend API.
 * No storage, dtype, residency or execution-state contract is implied. */
#define LUNELETTE_MODEL_HIDDEN 3072
#define LUNELETTE_MODEL_QUERY 4096
#define LUNELETTE_MODEL_KV 1024
#define LUNELETTE_MODEL_INNER 9216
#define LUNELETTE_MODEL_LAYERS 26
#define LUNELETTE_MODEL_VOCAB 131072
#define LUNELETTE_MODEL_QUERY_HEADS 32
#define LUNELETTE_MODEL_KV_HEADS 8
#define LUNELETTE_MODEL_HEAD_DIM 128
#define LUNELETTE_MODEL_MAX_POSITION 262144
#define LUNELETTE_MODEL_ORIGINAL_CONTEXT 16384
#define LUNELETTE_MODEL_NORM_EPS 1e-5f
#define LUNELETTE_MODEL_ROPE_THETA 1e6
#define LUNELETTE_MODEL_ROPE_THETA_F 1000000.0f
#define LUNELETTE_MODEL_YARN_FACTOR 16.0f
#define LUNELETTE_MODEL_YARN_BETA_FAST 32
#define LUNELETTE_MODEL_QUERY_SCALE 0.1f
#define LUNELETTE_MODEL_TIED_EMBEDDINGS 1
#define LUNELETTE_MODEL_BLOCK_WEIGHTS 9

/* Logical checkpoint names/shapes. cols=0 denotes a vector, not a matrix.
 * Ordering matches norm, Q/K/V/O, norm, gate/up/down in the reference block. */
typedef struct {
    const char *suffix;
    size_t rows, cols;
} lunelette_model_weight;
extern const lunelette_model_weight lunelette_model_block_weights[LUNELETTE_MODEL_BLOCK_WEIGHTS];

#endif
