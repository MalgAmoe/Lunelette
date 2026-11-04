# Validation: schemas, commands and historical evidence

All shell commands in this guide run from the repository root, not `docs/`.
External weights and fixtures are optional, caller-supplied, and not bundled or
downloaded automatically. Paths in examples are repository-local placeholders.

## Reading the evidence

The measurements and completed checks below are historical records carried forward
from the pre-cleanup README, not checks rerun by this documentation pass. Current
full-depth acceptance is defined in [the bounded contract](full-depth-validation.md);
strict hidden-state failure counts remain diagnostics, not erased evidence.
Isolated-stage and last-logit tolerances remain unchanged. These external fixtures
were generated previously; their generator and outputs are not bundled here.
Commands consume supplied artifacts, not an end-to-end bundled reproduction.
For current execution see [runtime contracts](metal.md) and
[reader contracts](safetensors.md). CPU API history is [archived](cpu-runtime.md).

## Optional real-weight validation

Default tests generate synthetic files; external weights are never required.
The optional comparator consumes the saved Python `input_ids` (I64 `[1,N]`)
and compares every embedding bit against `hidden_0` (F32 `[1,N,D]`), including
signed zero. It neither tokenizes nor runs Python or the transformer.

The examples below assume you supply the checkpoint under `models/` and the
five reference fixtures under `fixtures/reference/`, both inside this repository
and ignored by Git. These optional resources are not bundled or downloaded
automatically. The fixture files are named `short.safetensors`,
`multilingual.safetensors`, `code.safetensors`, `chat.safetensors`, and
`longer.safetensors`. Default synthetic tests need none of them.

```sh
make integration \
  WEIGHTS=models/Ministral-3-3B-Instruct-2512-BF16/model-00001-of-00002.safetensors \
  FIXTURES=fixtures/reference
# Same five comparisons with ASan/UBSan (after make sanitize):
make BUILD=build/sanitize integration \
  WEIGHTS=models/Ministral-3-3B-Instruct-2512-BF16/model-00001-of-00002.safetensors \
  FIXTURES=fixtures/reference
# Or choose explicit individual fixtures:
./build/compare_embeddings WEIGHT_SHARD FIXTURE.safetensors [MORE_FIXTURES...]
```

The local checkpoint's `model.safetensors.index.json` maps
`language_model.model.embed_tokens.weight` to shard **00001** of two; its shape
is `[131072,3072]`, BF16 (805,306,368 bytes). Shard 00002 is not needed for
embeddings. Five local fixtures match exactly: short (6 tokens), multilingual
(24), code (22), chat (24), longer (150), each with width 3072. This validates
embedding lookup only, not checkpoint-wide compatibility or full model inference.

Synthetic tests cover supported dtypes, scalar/empty shapes, BF16 special bits,
invalid tokens/read bounds, duplicate/missing fields and names, malformed JSON,
overflow, rank limits, invalid offsets, overlap, and every truncation position
of a small valid file. Tests also exercise metadata/escape handling, parser
capacity boundaries, reordered intervals, and empty tensors at valid and interior
boundaries. `make test` generates tiny weights and matching/mismatching fixtures:
the comparator must accept the exact match and reject a one-bit F32 difference.
These tests remain active with `-DNDEBUG`. `make sanitize` runs them under ASan
and UBSan.

A post-open truncation/short-read test is intentionally omitted: file mutation
violates the unchanged-file contract, and stdio may retain previously buffered
payload bytes. Portable C11 has no reliable truncation/buffer-invalidation hook
for that scenario. Truncation before open is tested at every byte position;
read failures are checked by the reader and diagnosed by the comparator.

> The remaining sections are historical CPU numerical provenance from
> `4d4f8f8`. Their consumers and commands are retired, not current build targets.
> Frozen fixtures, thresholds and recorded failures remain evidence; use
> `compare_metal` as documented in [the runtime guide](metal.md).

## Optional layer-0 pre-attention comparison

```text
make build/compare_projections
./build/compare_projections FIXTURE.safetensors WEIGHT_SHARD [MORE_SHARDS...]
```

All paths are explicit caller-supplied resources; no downloads or Python are
needed by C. Supply 1–16 shards containing exactly one copy of each selected
weight. The comparator does not parse the shard index. It supports this 3B layout
only: `language_model.model.embed_tokens.weight` BF16 `[131072,3072]`,
`language_model.model.layers.0.input_layernorm.weight` BF16 `[3072]`, and
`language_model.model.layers.0.self_attn.{q,k,v}_proj.weight` BF16
`[4096,3072]`, `[1024,3072]`, `[1024,3072]`. Other shapes/dtypes fail.
Only embedding rows, the learned norm vector and one projection matrix at a time
are read/expanded. Peak matrix storage is 48 MiB, not the entire checkpoint.

The fixture must contain I64 `input_ids` `[1,N]` (1–256 tokens), and F32
`input_norm`, `q`, `k`, `v` with shapes `[1,N,3072]`, `[1,N,4096]`,
`[1,N,1024]`, `[1,N,1024]`. Capture projections **before head reshape and
RoPE**, with no bias. Reference generation must independently expand selected
BF16 tensors to float32, look up the supplied IDs, compute
`weight * (embedding * rsqrt(mean(embedding**2) + 1e-5))`, then PyTorch
`F.linear(normalized, W)` (`W` is row-major `[out,in]`). Record model revision,
IDs/tokenizer settings, library versions, dtype/device, shapes and capture points
alongside fixtures. The official [Transformers v5.0.0 implementation](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/ministral3/modeling_ministral3.py)
confirms norm ordering and projection placement; PyTorch
[linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html)
defines the matrix orientation.

C uses sequential float32 reductions; PyTorch CPU reductions may use a different
order. Acceptance is elementwise `abs(actual-reference) <= 2e-4 +
1e-4*abs(reference)`; nonfinite values fail. These explicit allowances cover
3072-term float32 accumulation differences and cancellation near zero, not BF16
runtime arithmetic. Every stage reports maximum absolute error and maximum
relative error (`abs_error / max(abs(reference),1e-12)`). Near-zero relative errors
can be large even when absolute errors are tiny; neither metric alone determines
acceptance. Tolerances are fixed, not automatically relaxed.

Independent CPU float32 PyTorch 2.15.0 / Transformers 5.0.0 / safetensors 0.8.0
fixtures for revision `b6d637bef2393152b3da2b2fde72eecdee30557e` passed for the
6-token short and 24-token multilingual inputs. Maximum absolute errors across
both (optimized normal build): norm `2.289e-5`, Q `2.098e-5`, K `1.526e-5`, V `2.146e-6`.
Maximum relative errors respectively: `2.752e-6`, `0.01151`, `0.000904`,
`0.03411`. Sanitizer comparisons also pass; different optimization/reduction
behavior gives a largest absolute error of `2.289e-5` and relative error of
`0.04241`. NDEBUG comparisons match the normal build. A deliberately corrupted
real Q fixture is rejected by the complete CLI pipeline. External
fixture-generation scripts/results are not bundled.

`st_load_bf16` (see [reader API](safetensors.md)) is a reusable bounded tensor-to-F32 helper: the caller owns the
float buffer, validates the desired shape, and passes the exact whole-tensor
count (not a capacity). It uses 64 KiB (65,536 bytes) of scratch space, validates dtype,
count and handle membership, and may partially modify output on I/O failure.
It allocates nothing and retains no state. See the header contract.
Synthetic tests cover chunk boundaries, empty/scalar tensors, invalid counts,
dtype, foreign descriptors and special BF16 values. The comparator's tiny
hand-calculated weighted norm/non-square projection self-test checks acceptance,
deliberate mismatch and nonfinite rejection, but does not test its whole file
pipeline synthetically. Real fixtures exercise that pipeline.

The projection comparator is a targeted diagnostic tool, not an execution framework: it recomputes
embedding/norm per projection to bound storage. The projection comparator alone does not validate attention, RoPE, residuals or MLP;
see the complete first-block comparison below. That comparator does not validate cache or generation. Batching,
quantization remain unimplemented; this diagnostic does not exercise the optional Metal runtime.

## Isolated attention comparison

Optional isolated reference comparison (fixtures supplied separately):

```text
make build/compare_attention
build/compare_attention fixtures/attention/short.safetensors
build/compare_attention fixtures/attention/multilingual.safetensors
build/compare_attention fixtures/attention/boundary.safetensors
```

Fixture schema: I64 `positions[T]`; F32 `q`, `rotated_q`, `attention`
`[1,T,4096]`; F32 `k`, `v`, `rotated_k` `[1,T,1024]`, with `1<=T<=256`.
`rotated_q` includes query scaling. `attention` is official eager causal
attention before o_proj. Export from official `Ministral3RotaryEmbedding`,
`apply_rotary_pos_emb`, query scaling and `eager_attention_forward`, using
CPU FP32 actual layer-0 QKV captured before rotary. Keep model revision,
source revision, environment, input IDs/tokenizer settings, fixture hashes and
capture points alongside fixtures. A synthetic small-position-list fixture
can test context boundaries without allocating a 16K-token sequence.

Comparison reports max absolute and relative errors, rejecting nonfinite data
and any element outside `atol=2e-4, rtol=1e-4` (scalar versus parallel FP32
reductions; absolute allowance covers cancellation). Near-zero relative errors
alone can be large. The comparator tests known matches, mismatches and NaNs.
This isolates the new kernels from projection errors; it is not an end-to-end
C model or full long-context validation.


## First-block comparison

```text
make build/compare_block
./build/compare_block WEIGHT_SHARD FIXTURE.safetensors [MORE_FIXTURES...]
# Five named fixtures, explicit caller-supplied paths:
make block-integration WEIGHTS=/path/to/shard.safetensors FIXTURES=/path/to/fixtures
# Also supported after the corresponding test builds:
make BUILD=build/sanitize block-integration WEIGHTS=/path/to/shard.safetensors FIXTURES=/path/to/fixtures
make BUILD=build/ndebug block-integration WEIGHTS=/path/to/shard.safetensors FIXTURES=/path/to/fixtures
```

Fixture schema: I64 `input_ids[1,N]`, F32 `hidden_1[1,N,3072]`, 1<=N<=256.
The independent existing fixtures came from full official Transformers eager
CPU FP32 forward with `use_cache=False`, `output_hidden_states=True`, model
revision `b6d637bef2393152b3da2b2fde72eecdee30557e`, Python 3.12.11,
PyTorch 2.15.0, Transformers 5.0.0, safetensors 0.8.0, four CPU threads.
Raw prompts use tokenizer `encode(add_special_tokens=True)`; chat and longer
use the checkpoint chat template with a generation prompt and system message
“Be concise.” Exact token IDs are stored in each fixture. No C-derived outputs
were used to generate the reference.

Historical first-block sequential-norm baseline: all five passed normal,
ASan/UBSan and NDEBUG builds (Apple clang 17):

| Fixture | Tokens | Normal/NDEBUG max absolute | Normal/NDEBUG max relative |
|---|---:|---:|---:|
| short | 6 | 1.216e-5 | 0.05129 |
| multilingual | 24 | 1.216e-5 | 0.33297 |
| code | 22 | 1.216e-5 | 0.50538 |
| chat | 24 | 1.216e-5 | 0.09705 |
| longer | 150 | 1.216e-5 | 38.30 |

Sanitizer maximum absolute error is 1.193e-5 across all five; maximum relative
is 39.60 (near-zero reference). Acceptance is elementwise `abs_error <= 2e-4 +
1e-4*abs(reference)`, unchanged from existing projection/attention comparisons,
chosen for sequential versus parallel FP32 reductions. Large relative errors
near zero do not determine acceptance alone. No tolerance relaxation or extra
intermediate captures were needed.

Default `make test` tests the comparator's exact match, deliberate mismatch and
nonfinite rejection, plus API null/count/token validation, missing weights,
wrong embedding dtype/shape and unchanged output on rejection. Existing kernel
tests cover the component math. There is intentionally no miniature configurable
block or synthetic full fixed-3B weight payload: the default suite does **not**
prove full-block numerical composition; the optional five independent real-weight
comparisons provide that evidence. No full model, cache or long-context
correctness claim follows from this first-block result.

## Full-forward comparison

```text
make forward-integration \
  WEIGHTS=models/Ministral-3-3B-Instruct-2512-BF16/model-00001-of-00002.safetensors \
  WEIGHTS2=models/Ministral-3-3B-Instruct-2512-BF16/model-00002-of-00002.safetensors \
  FIXTURES=fixtures/reference
# Individual cases; same executable under build/sanitize after make sanitize:
./build/compare_forward SHARD1 SHARD2 FIXTURE.safetensors [MORE_FIXTURES...]
```

The existing five independent fixtures supply I64 `input_ids[1,N]`, F32
`hidden_13[1,N,3072]`, `hidden_26[1,N,3072]`, `last_logits[1,131072]`.
Provenance/tokenization is the same as the first-block fixtures above:
checkpoint revision `b6d637bef2393152b3da2b2fde72eecdee30557e`, official full
uncached eager CPU FP32 forward, Python 3.12.11 / torch 2.15.0 / Transformers
5.0.0 / safetensors 0.8.0, four reference threads. Inspected upstream source:
[Transformers v5.0.0, commit 08810b1](https://github.com/huggingface/transformers/blob/08810b1e278938278c50153ee1edfd7a20a759da/src/transformers/models/ministral3/modeling_ministral3.py).
No reference outputs were regenerated from C. The comparator reports maximum
absolute/relative errors, RMSE, failing-element count, worst tolerance ratio and
top-1 IDs, exact ties and margins. Logits retain the **unchanged** elementwise
`abs_error <= 2e-4 + 1e-4*abs(reference)` allowance. Only full-depth hidden_13/26
use the [provisional per-token RMS/cosine/absolute contract](full-depth-validation.md),
calibrated using PyTorch-only variation and frozen before held-out outputs.
Their strict failure counts remain diagnostics. Nonfinite data always rejects;
isolated-layer comparisons are unchanged.

### Reduction drift investigation

The initial implementation retained all sequential float reductions from commit
`5e1062e`. All five layer-13 checkpoints and last logits passed, with top-1
agreement, but final hidden states failed for chat and longer. For chat exactly
one of 73,728 hidden elements failed: flattened index 19662, C `0.0709941611`
versus reference `0.0712088495`, 1.03654 times the existing allowance. Initial
maximum errors were 0.00262451 at layer 13, 0.00121737 at final hidden (longer),
and 0.000188351 at last logits (longer). Relative errors near zero are not a
reliable accuracy summary.

Controlled external experiments changed only reductions, with identical weights,
architecture, fixtures and tolerance. On chat, widening only matrix-vector sums
still failed (final-hidden maximum 0.00253677); widening both norm and matvec sums
also failed (0.00210571). Widening **only RMSNorm's positive sum of squares**
passed (0.000152588), with last-logit error 0.0000495911 instead of 0.0000956059.
Therefore the targeted norm change addresses accumulated norm-rounding drift;
there is no evidence here for an architecture, sharding or head-layout error.
Remaining differences include sequential FP32 dot products, reference parallel
reductions and nonlinear rounding; the experiments do not assign every residual
error to one kernel or establish that the parallel FP32 reference is exact.
No tolerance was loosened. First-block real comparisons still pass, now with
normal-build maximum absolute error 1.1921e-6 across all five. The historical
first-block error table above describes the original sequential-norm baseline.

Default tests cover every required tensor's dtype/rank/dimensions, missing tensors,
duplicate required tensors within/across shards, invalid IDs/counts/nulls/handles,
upper token/count boundaries reaching I/O, zero-weight layer execution at indices
0 and 25, in-place layer execution, late matrix/logit read failure with unchanged
outputs, and the widened norm's rounding and in-place behavior. Private synthetic
handles use a sparse 54 MiB all-zero backing store shared by descriptors; they do
not construct a configurable miniature model or replace real-weight evidence.
Comparator self-tests accept known matches and reject deliberate mismatches/NaNs.
Existing embedding, first-block, reader and kernel tests are retained.

Limits: one unpadded sequence of 1–256 supplied IDs starting at position zero;
this uncached forward API itself provides no cache, generation or tokenization.
This CPU API has no GPU execution; the optional Metal runtime is separate.
No batching, vision, quantization, 8B/14B or arbitrary configurations.
Finite checkpoint values are assumed; there is no
exhaustive nonfinite-weight scan. Optional real fixtures exercise up to 150 tokens,
not a real 256-token or long-context forward. These are bounded correctness tests,
not throughput benchmarks or claims about cached execution.

### Historical strict full-forward results and remaining gap

The measurements below predate the separate bounded contract. The later recorded normal
reruns passed all five original and three held-out fixtures under that contract;
the strict longer hidden diagnostic still fails at exactly 1,220 elements.
See [current criteria, provenance and limitations](full-depth-validation.md).

Apple clang 17, portable `-O2` build, base Apple M5. CPU seconds measure forward
only, excluding fixture loading/comparison; the complete five-case command took
329.80 seconds wall time (other validation jobs also ran during that interval).

| Fixture | Tokens | CPU seconds | max abs hidden_13 | max abs hidden_26 | max abs logits | Top-1 |
|---|---:|---:|---:|---:|---:|---|
| short | 6 | 9.62 | 2.4414e-4 | 1.5259e-4 | 3.0518e-5 | 6993 = 6993 |
| multilingual | 24 | 35.24 | 2.4414e-4 | 1.5259e-4 | 3.2902e-5 | 11461 = 11461 |
| code | 22 | 32.59 | 2.4414e-4 | 1.5259e-4 | 3.5763e-5 | 1032 = 1032 |
| chat | 24 | 35.33 | 2.4414e-4 | 1.5259e-4 | 4.9591e-5 | 1052 = 1052 |
| longer | 150 | 216.36 | 2.4414e-4 | **1.0195e-3** | 6.5088e-5 | 12359 = 12359 |

All layer-13 and last-logit comparisons pass; final hidden passes **four of five**.
**Known validation gap:** longer final hidden fails the unchanged tolerance at
1,220 / 460,800 elements, RMSE 4.4723e-5. Worst allowance ratio is 3.82315 at
flattened index 380137: actual -0.0549602099, reference -0.055746153. Its maximum
relative error is 4.28806 near zero; last-logit maximum relative error is 0.142956.
The historical strict five-fixture comparator correctly exited nonzero. Do not interpret top-1
agreement, passing logits, or the norm experiment as proof that this residual
hidden-state discrepancy is fully diagnosed. Norm summation was one demonstrated
contributor, not an exclusive attribution of the remaining longer-case error.
Subsequent identical-input replay of all 26 layers passed strict elementwise
checks, while a PyTorch row-wise versus batched control itself failed strict
full-depth hidden checks. This supports reduction-order accumulation sensitivity,
not a proof that every remaining discrepancy is harmless. Strict failures remain
visible alongside the new separately calibrated full-depth acceptance rule.

Follow-up diagnosis (scratch-only variants, unchanged production kernels): widening
attention accumulation still fails longer hidden_26 (679 elements, RMSE 4.0812e-5,
max absolute error 1.0147e-3). Widening projection accumulation instead worsens it
(9,643 elements, RMSE 1.2909e-4). Neither is a demonstrated fix.
On identical captured final-layer Q/K/V inputs with shared rotated inputs, C and
official Torch eager attention differ by at most 1.7286e-6; explicit Torch calls
using the C scale and triangular mask reproduce official eager output exactly.
Wide attention is closer to an FP64 oracle but farther from Torch eager locally.
A model-weight projection probe also shows Torch GEMV versus batched GEMM differing
by 4.2916e-6 on identical vector values. Thus reference reduction/backend shape
sensitivity is measurable; it does not prove the entire remaining full-forward
error is harmless accumulation drift. Full forward remains experimental. The
separate bounded acceptance contract does not erase this strict discrepancy.

Measured peak resident memory for the normal five-case command was 153,747,456
bytes (146.6 MiB). ASan/UBSan short and chat both pass all checkpoints and top-1:
short 18.34 s wall / 396,476,416 bytes RSS; chat 61.06 s / 409,370,624 bytes RSS.
Sanitizer max absolute errors (hidden_13, hidden_26, logits) were respectively
(1.2207e-4, 4.1962e-5, 3.1948e-5) and (1.2207e-4, 1.1826e-4, 5.1975e-5).
Optimization/instrumentation changes floating-point contraction and rounding;
these are not expected to be bitwise identical builds.

Historical verification completed: normal, ASan/UBSan and NDEBUG synthetic suites;
all-five embedding exact comparisons; all-five first-block comparisons in normal,
ASan/UBSan and NDEBUG; full-forward normal all five (the longer hidden failure
above); full-forward ASan/UBSan short and chat; `git diff --check`.
Full-forward NDEBUG real comparisons and sanitizer multilingual/code/longer were
not run. These historical forward checks imply no cache, generation, accelerated or
arbitrary-configuration validation. No commits, downloads or dependency installation were made.

## Fixed-ID generation comparison

Synthetic tests cover EOS, zero/exact budgets, total context 256, rejection of
257-token prompts, negative-score ties, nonfinite logits, and failure atomicity.
Optional real validation accepts independently exported safetensors per step,
each with `input_ids` I64 `[1,N]` and `last_logits` F32 `[1,131072]`:

```text
make build/compare_generation
./build/compare_generation SHARD1 SHARD2 STEP0.safetensors STEP1.safetensors
# A direct forward-only comparison also permits the full 256-token boundary:
./build/compare_generation --forward-only SHARD1 SHARD2 CONTEXT256.safetensors
```

Each step checks the complete prefix, every logit with the existing strict
`atol=2e-4, rtol=1e-4` criterion, and exact greedy token agreement. This tolerance
allows scalar C versus PyTorch FP32 reduction ordering, not arbitrary text
agreement. The comparator itself is tested with a match, a deliberate mismatch,
and NaN. This fixed-ID baseline is uncached CPU-only, not evidence for cache, tokenizer,
longer contexts, or other model sizes.

## Cache comparison

Optional per-step verification (fixtures contain `input_ids` and `last_logits`):

```text
make build/compare_cache
build/compare_cache SHARD1 SHARD2 STEP0.safetensors STEP1.safetensors STEP2.safetensors
```

This compares every cached logit against both same-prefix uncached C and supplied
reference logits at `2e-4 + 1e-4*abs(reference)`, checks greedy tokens and the cached
generation adapter/reset. Default tests need no models or Python and compare the
shared offset-attention seam exactly through the 256-token boundary. Full-prefix
execution and its existing comparison tolerances are unchanged.
