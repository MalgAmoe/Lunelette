# Lunelette

A learning-first Ministral 3 inference engine in C, with a path toward an
optimized, portable runtime.

**Status:** complete first transformer block validated against real-weight Python
fixtures, from embedding lookup through attention and MLP residuals. This is
not full model inference: no tokenizer, remaining layers or generation loop.

## Initial target

- Ministral 3 **3B**, text-only; eventually the **8B and 14B** variants.
- Development machine: Apple M5, 32 GB unified memory.
- Readable CPU correctness baseline first; profile before adding Apple-specific
  acceleration. Keep platform-specific code separate when it becomes necessary.
- Python is reference/testing tooling only, not a runtime dependency.

## Build and check

Requires a C11 compiler and Make; on macOS, Apple's command-line developer tools
provide both. No external runtime libraries beyond the C/math libraries.

```sh
make
./build/lunelette --help
make test
make sanitize
```

Independently tested scalar CPU kernels include stable softmax, learned-weight
RMSNorm with epsilon, row-major matrix-vector multiply (`y = W*x`, explicit rows
and columns, no bias), and SiLU gated product (`SiLU(gate) * up`). These are
primitives used by the first-block path described below, not full model inference.

The new kernels use float32 products/accumulation and `sqrtf`/`expf`; softmax
retains its original double-precision sum. RMSNorm sums squares before writing
output; it supports exact in-place input or weight output. Gated product supports
exact in-place output on either input. Partial output/input overlap is forbidden;
matrix-vector output must be disjoint from its inputs. See `include/lunelette.h`
for buffer lengths, ownership, overlap and error contracts.

Null pointers, zero dimensions, byte-size overflow, and non-positive/non-finite
RMSNorm epsilon are rejected before writing output. Array elements are not
scanned for finiteness; arithmetic overflow and non-finite results are not errors.
Sign-split SiLU avoids exponential overflow for finite gates, but does not prevent
final-product overflow. Sequential float reductions need not match PyTorch's
reduction order bit-for-bit; RMSNorm is not an overflow-resistant norm.

Always-active synthetic tests cover hand-calculated values, weighted/zero-vector
RMSNorm, epsilon, non-square matrices in both directions, finite SiLU extremes,
in-place cases and invalid arguments. Arithmetic checks use combined absolute
(1e-6) and relative (2e-6) tolerances for these small examples, with error reports
on failure. Real-weight layer-0 norm/projection comparisons are described below; the small
unit tests alone are not evidence of model correctness.

## Layout

- `include/`: public C declarations
- `src/`: CLI and portable numerical code
- `tests/`: small deterministic C tests
- `tools/reference/`: reference-validation plan; Python harness not yet implemented

## Safetensors / embedding API

`include/safetensors.h` exposes `st_open`, `st_find`, `st_read`, `st_close`,
`st_bf16_to_f32`, and `st_embedding`. Open validates the entire header and file
layout without loading weight payloads. Find returns a borrowed tensor pointer;
read takes a byte offset relative to that tensor and returns raw little-endian
bytes. Embedding accepts a BF16 rank-2 tensor, a nonnegative in-range token,
and an output buffer of exactly the row width. It reads only that row and
converts BF16 bits to IEEE binary32 via `memcpy`, preserving signed zero,
subnormals, infinities and NaN payload bits (no arithmetic conversion).
Functions return 0 on success, -1 on failure; open provides a diagnostic.
Initialize/open a fresh handle, then close it, including on error. Handles are
not thread-safe. Do not modify public tensor descriptors or reopen live handles.

Deliberate limits (unsupported inputs fail, not silently reinterpret):

- BF16, F32, I64 only; at most 8 dimensions, 4,096 tensors, 1 MiB header.
- Nonempty unique tensor names, up to 255 printable ASCII bytes. JSON escapes
  for quote, slash, backslash and printable ASCII `\uXXXX` are decoded.
  Non-ASCII/control strings are unsupported, including metadata strings.
- Optional string-to-string metadata: at most 64 unique keys, 255-byte keys,
  4,095-byte values. Unknown tensor fields and duplicate fields are rejected.
- Unsigned decimal integer syntax only for shapes/offsets; dimensions <= INT64_MAX.
  Scalars and zero-sized tensors work. Shape products and byte sizes are checked
  for overflow. Offsets must exactly cover the payload, without gaps, overlaps,
  or unindexed trailing bytes; empty tensors must be on interval boundaries.
- Seekable files no larger than `LONG_MAX`; on 32-bit-long systems large model
  shards cannot be opened. Requires IEEE binary32 floats (compile-time checked).
- File must remain unchanged while open. No sharded-index parsing or tokenizer.

Standard C `fseek`/`fread` avoids POSIX mmap and whole-model RAM residency.
The implementation holds a bounded header and descriptor table plus one BF16
row during lookup; the caller owns the float output. Bounded quadratic metadata
validation favors clarity over large-header performance. No new dependencies.

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

## Optional layer-0 pre-attention comparison

```sh
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

`st_load_bf16` is a reusable bounded tensor-to-F32 helper: the caller owns the
float buffer, validates the desired shape, and passes the exact whole-tensor
count (not a capacity). It uses 4096 bytes of scratch space, validates dtype,
count and handle membership, and may partially modify output on I/O failure.
It allocates nothing and retains no state. See the header contract.
Synthetic tests cover chunk boundaries, empty/scalar tensors, invalid counts,
dtype, foreign descriptors and special BF16 values. The comparator's tiny
hand-calculated weighted norm/non-square projection self-test checks acceptance,
deliberate mismatch and nonfinite rejection, but does not test its whole file
pipeline synthetically. Real fixtures exercise that pipeline.

This is a targeted diagnostic tool, not an execution framework: it recomputes
embedding/norm per projection to bound storage. That comparator alone does not validate attention, RoPE, residuals or MLP;
see the complete first-block comparison below. Cache, generation, batching,
quantization and GPU behavior remain unimplemented.

## Next milestones

Record immutable checkpoint/reference revisions and license details, then add
and compare transformer operations incrementally. Check cached and uncached
decoding separately; matching generated text alone is not a correctness test.

## DwarfStar as a reference

Lunelette is inspired by antirez's [DwarfStar](https://github.com/antirez/ds4):
a focused native engine rather than a universal model framework. Study and reuse
suitable pieces deliberately; do not inherit unrelated model or infrastructure
scope. DwarfStar’s separation of tensor descriptors, data-base offsets, and
lazy shard file access informed the design; no local upstream checkout is required.
The broader database, hash maps and threading are unnecessary here. The parser
and reader are original; no DwarfStar code was reused or modified. The YaRN
formula adaptation from Transformers is attributed in NOTICE; adapted code
must retain its required attribution and license notices.

The project's own distribution license has not yet been selected.

## Rotary embeddings and attention (scalar baseline)

`include/attention.h` exposes two independent, caller-buffer APIs:

- `lunelette_ministral3_yarn`: in-place Q `[T,32,128]`, K `[T,8,128]`,
  split-half rotation, absolute positions in `[0,262144)`. Fixed to
  Ministral-3-3B-Instruct-2512 checkpoint revision
  `b6d637bef2393152b3da2b2fde72eecdee30557e`: theta=1e6, factor=16,
  original context=16384, beta_fast/slow=32/1, default truncated correction
  range. mscale=mscale_all_dim=1 makes the cos/sin amplitude ratio **1**.
  Only Q receives `1 + 0.1*log(1 + floor(position/16384))` afterward.
  This intentionally is not a generic model configuration API.
- `lunelette_causal_gqa`: token-major FP32 Q/K/V, row-order causal prefill,
  query head `h` maps to KV head `h/4` for 32/8 heads. Generic positive head
  dimensions and divisible head counts are supported for synthetic tests.
  The output is `[T,32,128]` (flattenable to `[T,4096]`), **before o_proj**.
  Caller supplies `T` float scratch elements. No weights, cache or allocation.

Contracts, overflow checks and failure behavior are documented in the header.
The scalar implementation uses float dot products and max-subtracted float
softmax. It is quadratic in sequence length; it is a correctness baseline,
not an optimized long-context implementation. No padding, batching, dropout,
KV cache is provided by these attention APIs; output projection and feedforward
are composed separately in the first-block API below.
See NOTICE for upstream formula attribution and license.

`make test` includes basis rotations, golden official FP32 YaRN samples at
positions 0/1/16383/16384/16385, causal masking, GQA mapping, nonuniform
probabilities, stable large-score softmax, invalid arguments and size overflow.
The small golden constants are self-contained; no Python/model is required.

Optional isolated reference comparison (fixtures supplied separately):

```sh
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


## Complete first transformer block

`include/block.h` / `src/block.c` expose `lunelette_first_block`: actual embedding
rows -> input RMSNorm -> QKV -> YaRN -> causal GQA -> output projection ->
residual -> post-attention RMSNorm -> SiLU(gate projection) * up projection ->
down projection -> residual. Returns token-major FP32 `[N,3072]`, corresponding
to official `hidden_states[1]` (before layer 1, not final model normalization).
The [official decoder and MLP forward](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/ministral3/modeling_ministral3.py)
confirm the two pre-norm residual branches and bias-free gated MLP.

This deliberately fixes the checkpoint's 3B architecture: hidden 3072,
intermediate 9216, 32 query / 8 KV heads, head dimension 128, RMS epsilon 1e-5,
vocabulary 131072. It does not parse model configuration or the shard index.
The inspected checkpoint index places embeddings and **all layer-0 weights in
shard 00001**. Every required tensor's BF16 dtype, rank and exact shape are
checked before use. One row-major matrix is expanded at a time (maximum
108 MiB); embedding rows only are read. Token-major activation storage is
bounded to 256 tokens (about 37 MiB maximum). No entire model allocation,
retained weight cache, framework, dependencies or Python runtime is needed.
Weights are reread on each call: clarity and bounded memory, not speed.

The borrowed shard remains open and unchanged; the function owns and frees its
workspace. Caller owns IDs and output; output is unchanged on failure. See the
header for lifetimes and concurrency restrictions. Finite checkpoint values
are assumed, consistent with the underlying scalar kernels; arbitrary corrupt
nonfinite weights are not exhaustively scanned. Positions are consecutive from
zero, one sequence, no padding/cache. This stops at the first block only.

```sh
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

All five pass normal, ASan/UBSan and NDEBUG builds (Apple clang 17):

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
