/* Test-only borrowed snapshots after synchronous prefill/decode. No new runtime
 * hooks or execution changes: the final block and norm buffers already survive. */
#include "metal_internal.h"

const float *test_metal_snapshot(lunelette_metal *runtime, int normalized)
{
    LuneletteMetalOwner *owner = (__bridge LuneletteMetalOwner *)runtime->owner;
    return owner->buffers[normalized ? METAL_FINAL : METAL_X].contents;
}
