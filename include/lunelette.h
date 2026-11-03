#ifndef LUNELETTE_H
#define LUNELETTE_H

#define LUNELETTE_VERSION "0.1.0-dev"

/* Normalize count logits into probabilities. Supports in-place operation.
 * Returns 0 on success; -1 for null pointers, zero count, or non-finite input.
 * On invalid input, output is left unchanged.
 */
#include <stddef.h>
int lunelette_softmax(const float *input, float *output, size_t count);

/* Scalar float32 kernels. All buffers are caller-owned, suitably aligned float
 * arrays valid for the stated lengths for the duration of the call. No allocation
 * or retained state; concurrent calls require non-conflicting buffers.
 * Return 0 on completion, -1 for null pointers, zero dimensions, byte-size
 * overflow, or (RMSNorm only) non-positive/non-finite epsilon. Rejection leaves
 * output unchanged. Buffer sizes and forbidden overlap cannot be checked: these
 * are caller preconditions, not rejected inputs.
 * Elements are NOT scanned for finiteness. Arithmetic overflow, NaNs and
 * infinities follow floating-point arithmetic and do not cause a -1 return.
 * Products, sums and nonlinear operations use float (sqrtf/expf), not double.
 * Sequential accumulation can differ from parallel reference reductions; even
 * finite inputs can overflow intermediate products/sums. No bitwise reference
 * equivalence or overflow-resistant RMSNorm is promised.
 */

/* output[i] = input[i] * (1 / sqrt(mean(input^2) + epsilon)) * weight[i].
 * All arrays have count elements. For each input array separately, output must
 * either exactly equal it or be disjoint from it (no partial overlap). Read-only
 * arrays may overlap. Sum of squares is accumulated before any output writes.
 */
int lunelette_rmsnorm(const float *input, const float *weight, float *output,
                     size_t count, float epsilon);

/* output = weight * input, with contiguous row-major weight[rows][cols],
 * input[cols], output[rows], no bias. Each dot product accumulates left to right
 * in float starting at zero. Output must not overlap either read-only array;
 * the read-only arrays may overlap each other.
 */
int lunelette_matvec(const float *weight, const float *input, float *output,
                    size_t rows, size_t cols);

/* output[i] = SiLU(gate[i]) * up[i], all arrays of count elements.
 * Output may exactly equal either/both inputs, but partial output/input overlap
 * is forbidden. Read-only arrays may overlap. Sign-split sigmoid avoids expf
 * overflow for finite gates; the final product can still overflow.
 */
int lunelette_silu_gate(const float *gate, const float *up, float *output, size_t count);

#endif
