#include "model_facts.h"

const lunelette_model_weight lunelette_model_block_weights[LUNELETTE_MODEL_BLOCK_WEIGHTS] = {
    {"input_layernorm.weight", LUNELETTE_MODEL_HIDDEN, 0},
    {"self_attn.q_proj.weight", LUNELETTE_MODEL_QUERY, LUNELETTE_MODEL_HIDDEN},
    {"self_attn.k_proj.weight", LUNELETTE_MODEL_KV, LUNELETTE_MODEL_HIDDEN},
    {"self_attn.v_proj.weight", LUNELETTE_MODEL_KV, LUNELETTE_MODEL_HIDDEN},
    {"self_attn.o_proj.weight", LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_QUERY},
    {"post_attention_layernorm.weight", LUNELETTE_MODEL_HIDDEN, 0},
    {"mlp.gate_proj.weight", LUNELETTE_MODEL_INNER, LUNELETTE_MODEL_HIDDEN},
    {"mlp.up_proj.weight", LUNELETTE_MODEL_INNER, LUNELETTE_MODEL_HIDDEN},
    {"mlp.down_proj.weight", LUNELETTE_MODEL_HIDDEN, LUNELETTE_MODEL_INNER}
};
