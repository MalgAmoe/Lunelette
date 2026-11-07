#ifndef LUNELETTE_RUNTIME_LIMITS_H
#define LUNELETTE_RUNTIME_LIMITS_H
/* Legacy/default policy, NOT the model position limit. Explicit runtime profiles
 * separate total KV positions from the maximum single executed block. */
#define LUNELETTE_CONTEXT_LIMIT 256
#define LUNELETTE_LONG_CONTEXT_LIMIT 4352
#define LUNELETTE_BLOCK_LIMIT 256
/* Generation output budget is independent of total context capacity. */
#define LUNELETTE_GENERATION_LIMIT 256
#endif
