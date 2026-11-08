# Frozen native arithmetic v1

`metal-baseline-v1.metal` is a literal full-source snapshot of Lunelette's `src/metal_kernels.metal`, SHA256 `f49e63914633c3ac03ca7f232f11d3b27fd0573a5cd9a9c818073c2510b9f256`. Captured from the dirty working tree based on HEAD `0e36143aba5f8b0a8635e67ed424c8f5891a7a96`; the kernel itself equals that HEAD. It retains the original Transformers-derived math attribution in `NOTICE` and `LICENSES/Apache-2.0.txt`. This is local Lunelette code, not a DS4 kernel. Do not regenerate this fixture from a candidate at build time or runtime. The harness pins its SHA256 before compiling it as an independent library.

The same-input frozen-legacy, non-NaN bit comparison / NaN classification, canary and exact integer-dot test design follows DS4's `tests/test_metal_q8_reduction.m` and `tests/test_metal_q8_gemv_reference.m` at revision `b3f50180ec2c7324892be1f50bc83555ccb8d5e7`. DS4 attribution/license is retained in `DS4-MIT.txt`.

Both libraries use production's Metal 3.2, safe math, precise functions and 32-lane dispatch. The frozen preamble is literal, not generated from current model facts. Only project/norm are dispatched. Passing establishes unchanged native arithmetic on these cases, not mathematical correctness of the model.
