#ifndef LUNELETTE_CPU_WEIGHTS_H
#define LUNELETTE_CPU_WEIGHTS_H
#include "safetensors.h"
/* Private CPU BF16 safetensors checks; callers supply valid open handles.
 * Required names must occur exactly once. Unrelated tensors are ignored. */
const st_tensor *lunelette_require(st_file **files, size_t nfiles, const char *name,
    size_t rows, size_t cols, st_file **owner, char *error, size_t capacity);
/* Exactly two shards; validate the entire supported text layout before I/O. */
int lunelette_cpu_validate_layout(st_file **files, char *error, size_t capacity);
#endif
