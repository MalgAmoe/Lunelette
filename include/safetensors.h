#ifndef LUNELETTE_SAFETENSORS_H
#define LUNELETTE_SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define ST_MAX_DIMS 8
#define ST_MAX_NAME 256
#define ST_MAX_TENSORS 4096
#define ST_MAX_HEADER (1024 * 1024)

typedef enum {
    ST_F8_E4M3 = 1,
    ST_BF16 = 2,
    ST_F32 = 4,
    ST_I64 = 8
} st_dtype;

typedef struct {
    char name[ST_MAX_NAME];
    st_dtype dtype;
    size_t rank;
    uint64_t shape[ST_MAX_DIMS];
    uint64_t begin;
    uint64_t end;
} st_tensor;

typedef struct {
    FILE *file;
    uint64_t base;
    size_t count;
    st_tensor *tensors;
} st_file;

/* Returns 0 or -1 with a diagnostic. Open initializes output even on failure.
 * Do not reopen a live handle. Close releases everything. Handles are not
 * thread-safe. Tensor pointers are borrowed until close. */
int st_open(st_file *s, const char *path, char *error, size_t capacity);
void st_close(st_file *s);
const st_tensor *st_find(const st_file *s, const char *name);

/* Bounded raw little-endian byte read; tensor must belong to s. */
int st_read(st_file *s, const st_tensor *t, uint64_t offset, void *out, size_t bytes);

float st_f8_e4m3_to_f32(unsigned char bits);
/* Exact count, bounded scratch. Invalid arguments leave output unchanged;
 * I/O/nonfinite failures may leave a partial destination. Positive finite scale. */
int st_load_fp8(st_file *s, const st_tensor *t, float scale, float *out, size_t count);

float st_bf16_to_f32(uint16_t bits);

/* BF16 rank-2 only. Output count must equal width; invalid tokens never read. */
int st_embedding(st_file *s, const st_tensor *t, int64_t token, float *out, size_t count);

/* Load exactly count BF16 elements into caller-owned out[count], with fixed-size
 * scratch storage and no allocation. Caller validates rank/shape separately;
 * count must equal the whole tensor element count, not merely buffer capacity.
 * Tensor must be borrowed from s, descriptors/file unchanged. Returns -1 on
 * invalid arguments (output unchanged) or I/O failure (output may be partial).
 * Output must not overlap the handle/descriptors. Same thread rules as st_read.
 */
int st_load_bf16(st_file *s, const st_tensor *t, float *out, size_t count);

#endif
