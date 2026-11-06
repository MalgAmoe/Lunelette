#include "metal_weights.h"
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

void lunelette_metal_weights_close(lunelette_metal_weights *weights)
{
    st_close(&weights->file);
    memset(weights, 0, sizeof *weights);
}

static int validate_weight(lunelette_metal_weights *weights, size_t index,
    const char *name, size_t rows, size_t cols, char *error, size_t capacity)
{
    const st_tensor *tensor = st_find(&weights->file, name);
    st_dtype dtype = cols && index != 0 ? ST_F8_E4M3 : ST_BF16;
    if (!tensor || tensor->dtype != dtype || tensor->rank != (cols ? 2u : 1u) ||
        tensor->shape[0] != rows || (cols && tensor->shape[1] != cols)) {
        if (cols) {
            snprintf(error, capacity, "Metal tensor %s: expected %s [%zu,%zu]", name,
                dtype == ST_BF16 ? "BF16" : "F8_E4M3", rows, cols);
        } else {
            snprintf(error, capacity, "Metal tensor %s: expected BF16 [%zu]", name, rows);
        }
        return -1;
    }
    if (dtype == ST_F8_E4M3) {
        const char *suffixes[] = {"weight_scale_inv", "activation_scale"};
        for (size_t i = 0; i < 2; i++) {
            char companion[ST_MAX_NAME];
            snprintf(companion, sizeof companion, "%.*s%s", (int)(strlen(name) - 6),
                name, suffixes[i]);
            const st_tensor *scale = st_find(&weights->file, companion);
            if (!scale || scale->dtype != ST_BF16 || scale->rank != 0 ||
                scale->end < scale->begin || scale->end - scale->begin != 2) {
                snprintf(error, capacity, "Metal scalar %s: expected BF16 rank0", companion);
                return -1;
            }
        }
    }
    size_t elements = rows * (cols ? cols : 1);
    weights->weights[index].scale = 1.0f;
    weights->weights[index].tensor = tensor;
    weights->weights[index].elements = elements;
    weights->bytes += elements * sizeof(float);
    return 0;
}

int lunelette_metal_weights_open(lunelette_metal_weights *weights,
    const char *path, char *error, size_t capacity)
{
    char fallback[512];
    if (!error || !capacity) {
        error = fallback;
        capacity = sizeof fallback;
    }
    memset(weights, 0, sizeof *weights);
    if (!path) {
        snprintf(error, capacity, "Metal requires one explicit FP8 safetensors path");
        return -1;
    }
    if (st_open(&weights->file, path, error, capacity) != 0) {
        goto failure;
    }
    if (lunelette_metal_weights_validate(weights, error, capacity) != 0 ||
        lunelette_metal_weights_read_scales(weights, error, capacity) != 0) {
        goto failure;
    }
    return 0;
failure:
    lunelette_metal_weights_close(weights);
    return -1;
}

/* Separate metadata preflight permits synthetic all-layer validation without
 * model-sized allocations or fabricating sparse checkpoint files. */
int lunelette_metal_weights_validate(lunelette_metal_weights *weights,
    char *error, size_t capacity)
{
    weights->bytes = 0;
    if (validate_weight(weights, 0, "language_model.model.embed_tokens.weight",
            LUNELETTE_MODEL_VOCAB, LUNELETTE_MODEL_HIDDEN, error, capacity) != 0 ||
        validate_weight(weights, 1, "language_model.model.norm.weight",
            LUNELETTE_MODEL_HIDDEN, 0, error, capacity) != 0) {
        return -1;
    }
    for (size_t layer = 0; layer < LUNELETTE_MODEL_LAYERS; layer++) {
        for (size_t i = 0; i < LUNELETTE_MODEL_BLOCK_WEIGHTS; i++) {
            char name[ST_MAX_NAME];
            const lunelette_model_weight *fact = &lunelette_model_block_weights[i];
            snprintf(name, sizeof name, "language_model.model.layers.%zu.%s", layer,
                fact->suffix);
            if (validate_weight(weights, 2 + layer * LUNELETTE_MODEL_BLOCK_WEIGHTS + i,
                    name, fact->rows, fact->cols, error, capacity) != 0) {
                return -1;
            }
        }
    }
    /* Only the known text contract is accepted. Non-text (vision/projector)
     * descriptors remain the generic parser's responsibility, not a vision validator. */
    for (size_t i = 0; i < weights->file.count; i++) {
        const char *name = weights->file.tensors[i].name;
        if (strncmp(name, "language_model.", 15) != 0) {
            continue;
        }
        int known = 0;
        for (size_t j = 0; j < LUNELETTE_METAL_WEIGHT_COUNT && !known; j++) {
            const st_tensor *tensor = weights->weights[j].tensor;
            known = strcmp(name, tensor->name) == 0;
            if (!known && tensor->dtype == ST_F8_E4M3) {
                const char *suffixes[] = {"weight_scale_inv", "activation_scale"};
                for (size_t k = 0; k < 2; k++) {
                    char companion[ST_MAX_NAME];
                    snprintf(companion, sizeof companion, "%.*s%s",
                        (int)(strlen(tensor->name) - 6), tensor->name, suffixes[k]);
                    known |= strcmp(name, companion) == 0;
                }
            }
        }
        if (!known) {
            snprintf(error, capacity, "Unexpected text tensor: %s", name);
            return -1;
        }
    }
    return 0;
}

int lunelette_metal_weights_read_scales(lunelette_metal_weights *weights,
    char *error, size_t capacity)
{
    for (size_t i = 0; i < LUNELETTE_METAL_WEIGHT_COUNT; i++) {
        const st_tensor *tensor = weights->weights[i].tensor;
        if (!tensor || tensor->dtype != ST_F8_E4M3) {
            continue;
        }
        const char *suffixes[] = {"weight_scale_inv", "activation_scale"};
        for (size_t j = 0; j < 2; j++) {
            char name[ST_MAX_NAME];
            snprintf(name, sizeof name, "%.*s%s", (int)(strlen(tensor->name) - 6),
                tensor->name, suffixes[j]);
            const st_tensor *scale = st_find(&weights->file, name);
            float value;
            if (st_load_bf16(&weights->file, scale, &value, 1) != 0 ||
                !isfinite(value) || value <= 0) {
                snprintf(error, capacity, "Metal scalar %s: expected positive finite BF16", name);
                return -1;
            }
            if (j == 0) {
                weights->weights[i].scale = value;
            }
            /* activation_scale is validated, not retained: this is weight
             * dequantization to FP32, not static W8A8 inference. */
        }
    }
    return 0;
}

int lunelette_metal_weights_load(lunelette_metal_weights *weights, size_t index,
    float *out, size_t count)
{
    if (!weights || index >= LUNELETTE_METAL_WEIGHT_COUNT ||
        count != weights->weights[index].elements) {
        return -1;
    }
    const st_tensor *tensor = weights->weights[index].tensor;
    if (tensor && tensor->dtype == ST_F8_E4M3) {
        return st_load_fp8(&weights->file, tensor, weights->weights[index].scale, out, count);
    }
    return st_load_bf16(&weights->file, tensor, out, count);
}

int lunelette_metal_weights_load_raw(lunelette_metal_weights *weights, size_t index,
    unsigned char *out, size_t count, char *error, size_t capacity)
{
    const st_tensor *tensor = NULL;
    const char *reason = "expected exact compact F8 element count and destination";
    if (!weights || !out || !count || index >= LUNELETTE_METAL_WEIGHT_COUNT) {
        goto failure;
    }
    tensor = weights->weights[index].tensor;
    if (!tensor || tensor->dtype != ST_F8_E4M3 ||
        count != weights->weights[index].elements || tensor->end < tensor->begin ||
        tensor->end - tensor->begin != count) {
        goto failure;
    }
    float scale = weights->weights[index].scale;
    reason = "compact scale must be positive FP32-normal widened BF16";
    uint32_t bits;
    memcpy(&bits, &scale, sizeof bits);
    if (scale <= 0 || !isnormal(scale) || (bits & 65535u)) {
        goto failure;
    }
    unsigned char allowed[256];
    for (unsigned code = 0; code < 256; code++) {
        float decoded = st_f8_e4m3_to_f32((unsigned char)code);
        float product = decoded * scale;
        allowed[code] = isfinite(product) && (decoded == 0 || isnormal(product));
    }
    unsigned char bytes[65536];
    for (size_t offset = 0; offset < count;) {
        size_t chunk = count - offset;
        if (chunk > sizeof bytes) {
            chunk = sizeof bytes;
        }
        reason = "compact bounded payload read failed";
        if (st_read(&weights->file, tensor, offset, bytes, chunk)) {
            goto failure;
        }
        for (size_t i = 0; i < chunk; i++) {
            if (!allowed[bytes[i]]) {
                if (error && capacity) {
                    snprintf(error, capacity, "Metal compact tensor %s byte %zu code 0x%02x: "
                        "NaN or non-normal/overflow reconstructed product (scale %.9g)",
                        tensor->name, offset + i, bytes[i], scale);
                }
                return -1;
            }
        }
        memcpy(out + offset, bytes, chunk);
        offset += chunk;
    }
    return 0;
failure:
    if (error && capacity) {
        snprintf(error, capacity, "Metal compact tensor %s: %s",
            tensor ? tensor->name : "(invalid index)", reason);
    }
    return -1;
}

int lunelette_metal_weights_account(const lunelette_metal_weights *weights,
    lunelette_metal_strategy strategy, lunelette_metal_memory *memory)
{
    if (!weights || !memory || (strategy != LUNELETTE_METAL_RESIDENT_FP32 &&
            strategy != LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC)) {
        return -1;
    }
    lunelette_metal_memory result = {0};
    result.strategy = strategy;
    for (size_t i = 0; i < LUNELETTE_METAL_WEIGHT_COUNT; i++) {
        const st_tensor *tensor = weights->weights[i].tensor;
        size_t count = weights->weights[i].elements;
        if (!tensor || !count || count > SIZE_MAX / sizeof(float) ||
            (tensor->dtype != ST_F8_E4M3 && tensor->dtype != ST_BF16)) {
            return -1;
        }
        bool compact = strategy == LUNELETTE_METAL_COMPACT_FP8_ARITHMETIC &&
            tensor->dtype == ST_F8_E4M3;
        uint64_t bytes = count * (compact ? 1 : sizeof(float));
        if (bytes > UINT64_MAX - result.weight_bytes) {
            return -1;
        }
        result.weight_bytes += bytes;
        if (compact && count * sizeof(float) > result.conversion_stage_bytes) {
            result.conversion_stage_bytes = count * sizeof(float);
        }
    }
    result.scratch_bytes = 87819264;
    result.kv_bytes = 54525952;
    uint64_t extras = result.conversion_stage_bytes + result.scratch_bytes + result.kv_bytes;
    if (extras < result.conversion_stage_bytes || result.weight_bytes > UINT64_MAX - extras) {
        return -1;
    }
    result.total_bytes = result.weight_bytes + extras;
    *memory = result;
    return 0;
}
