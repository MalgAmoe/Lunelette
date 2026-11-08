# Engine reference

The engine is one resident Metal/MPS implementation with CPU host work, not an operator-dispatch framework. This guide describes the current C runtime; the [service contract](service.md) describes framing, multi-turn state, pending tokens and HTTP controls. The single-turn CLI remains limited to 256 total tokens; that is **not** the capacity limit of every runtime or service profile. See [running](../guides/running.md) for startup and flags.

Requires macOS 15+, Apple Silicon supporting Apple7 GPU family or newer, Command Line Tools/macOS SDK and sufficient unified memory. Development validation used Apple M5/32 GiB; neither other devices nor smaller-memory configurations inherit that validation. Unsupported hardware or insufficient admission headroom fails, without silent fallback. Portable host tests remain available on other platforms.

All commands run from the repository root. `make` builds the CLI and `make server` builds the native service; neither runs inference. See [testing](../development/testing.md). Python is not a native runtime dependency. No weights or dependencies are downloaded.

## Fixed architecture and operation sequence

[model_facts.h](../../include/model_facts.h) and [model_facts.c](../../src/model_facts.c) define a fixed checkpoint, not a configuration loader: 26 layers, hidden width 3072, inner width 9216, vocabulary 131072, 32 query heads and 8 KV heads of width 128 (query width 4096, KV width 1024), tied embeddings and model position bound 262144. The position bound is not allocated runtime capacity. [runtime_limits.h](../../include/runtime_limits.h) separately defines default capacity 256, opt-in capacity 4352, maximum block 256 and generation budget 256; a 128-token block is available only in the long profile. Tokenizer byte limits are separate again.

Input IDs gather rows from the expanded embedding. Each layer applies pre-RMSNorm, Q/K/V projections, split-half rotary transformation of Q/K at absolute cached positions, and appends rotated K and V to that layer's cache. Causal grouped-query attention maps four query heads to each KV head, normalizes scores, combines values and projects O back to hidden width. The first residual is followed by a second RMSNorm, gate/up projections, SiLU(gate) × up, down projection and the second residual. After 26 layers, final RMSNorm and the tied custom vocabulary projection produce logits for the last newly processed token, not a returned logit matrix for every input position. The encoder uses scratch intermediates and explicit copies; compact storage does not fuse or change this graph.

### Logical weight shapes

Matrices are output rows × input columns, as described by the checkpoint’s fixed logical names. Each of the 26 blocks has the following nine weights; projection companions do not count as executable matrices.

| Block suffix | Shape |
|---|---|
| `input_layernorm.weight` | 3072 |
| `self_attn.q_proj.weight` | 4096 × 3072 |
| `self_attn.k_proj.weight`, `self_attn.v_proj.weight` | 1024 × 3072 each |
| `self_attn.o_proj.weight` | 3072 × 4096 |
| `post_attention_layernorm.weight` | 3072 |
| `mlp.gate_proj.weight`, `mlp.up_proj.weight` | 9216 × 3072 each |
| `mlp.down_proj.weight` | 3072 × 9216 |

The shared embedding/head is 131072 × 3072 and final norm is 3072. These shape facts do not license row permutation or arbitrary compatible-looking models.

## Model-file and numerical contract

Only the official single-file HF-layout mixed FP8/BF16 Ministral 3 3B checkpoint is supported. No shard index, second shard, BF16-model fallback or arbitrary model configuration. There are 236 required text tensors: 182 F8_E4M3 projections, 53 BF16 norms and one BF16 embedding shared with the vocabulary head. Shapes are fixed. Each projection also requires rank-0 positive finite BF16 `weight_scale_inv` and `activation_scale`; all 364 companions are read and checked. Other namespaces are parsed safely, not treated as a supported vision architecture. Unknown `language_model` tensors are rejected.

The reference asset is [Ministral-3-3B-Instruct-2512-FP8](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-FP8/tree/b35d4dfe56c142746f54dbd64f579faab2744308), revision `b35d4dfe56c142746f54dbd64f579faab2744308`. Its single `model.safetensors` is 4,672,099,184 bytes, SHA256 `728f1826cd0e38191ca7b1379e81f78cf0555c6ffd95882aabd2404632346f86`; the paired Tekken file is SHA256 `600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843`. These are reference identities, not bundled files or fresh verification of a caller's assets. The loader checks format/domain and the service reports actual hashes; startup does not attest independent numerical correctness.

Reconstruction is FP8 decoded to FP32 multiplied by the BF16 weight scale widened to FP32, **without BF16 rounding of the product**. Activation scales are unused after validation. Embedding/head, norms, activations and KV are FP32 in execution. This is not W8A8 or native FP8 matrix arithmetic. Vision, batching and 8B/14B models are unsupported. The reader contract is specified below.

Custom kernels use safe/precise FP32 Metal math with contraction disabled; projection uses compensated per-lane accumulation and SIMDgroup reduction. RMSNorm epsilon is 1e-5. YaRN uses theta 1e6, factor 16, original context 16384 and beta 32/1, split-half rotation and integer-block query scaling. Attribution is in [NOTICE](../../NOTICE). The checkpoint's larger position limit does not authorize runtime profiles beyond the explicit capacities below.

MPS uses Float32 storage, tight row strides, transposed right operand, alpha 1 and beta 0. Its internal accumulation precision is unspecified; Float32 storage is not proof of full-FP32 accumulation. Independent numerical acceptance remains **FAIL/open**. Exact native agreement, operator checks and generated text do not replace [independent validation](../development/validation.md).

## Projection storage and execution

| Storage option | Multi-token block projections | Single-token block projections |
|---|---|---|
| `resident-fp32` (explicit supported option) | Resident FP32 weights → MPS or custom | Precise custom FP32 |
| `compact-fp8-arithmetic` (C/CLI default) | Packed weights → reusable FP32 stage → MPS or custom | Arithmetic FP8 reconstruction during projection |
| `compact-fp8-lookup` | Same conversion/stage path as arithmetic | Unscaled constant lookup then FP32 scaling |

Prefill policy in `lunelette_metal_options` is independent of storage; startup selectors belong to [running](../guides/running.md). Multi-token MPS is the default in C/CLI options. Single-token prefill/decode and the tied head use custom kernels. Lookup changes only single-token block decoding, not the head.

Compact admission is stricter than the general loader: scales must be positive FP32-normal widened BF16, FP8 codes finite, and nonzero reconstructed products normal and finite. Unsupported values fail, never fall back to resident storage. One stable conversion stage is reused; no full host FP32 model or duplicate head is retained. The lookup reconstruction is specified below; configuration guidance belongs to [running](../guides/running.md).

### Single-token arithmetic and lookup

Arithmetic reconstructs each packed E4M3FN byte, widens the BF16 scale to FP32, multiplies in FP32 and then performs activation multiplication, compensated (Kahan) per-lane accumulation and SIMDgroup summation. Lookup substitutes only the unscaled byte reconstruction with a 256-entry constant table; scale multiplication and reduction order are unchanged. The tied vocabulary head never uses this table.

For exponent 0 the unsigned value is `fraction × 2^-9`; otherwise it is `(8 + fraction) × 2^(exponent - 10)`, with sign applied. All 254 finite codes are exact, including negative zero. Exact hexadecimal FP32 literals occupy 1 KiB of shader constant data, not a per-weight table or explicitly allocated MTLBuffer. The two unreachable NaN entries do not relax loader admission. Compiler/pipeline storage is not an exact driver-memory measurement.

Both compact strategies use the same arithmetic conversion stage for multi-token work. Lookup is a supported choice, not a different precision mode or automatic speedup. Its same-schedule full-vector contract is finite bitwise agreement with arithmetic; changed partitions and independent correctness have separate [validation](../development/validation.md) contracts. Source-table auditing includes every literal and signed zero. No failure authorizes tolerance widening.

### Why one conversion stage is sufficient

The compact representation changes projection storage, not the transformer execution graph. Each context keeps packed projection bytes and host scale metadata. For a multi-token projection, [metal_encode.m](../../src/metal_encode.m) encodes conversion of exactly that matrix into the stage, then its MPS/custom consumer, before encoding the next projection's conversion. These operations share one ordered command stream. There is no CPU stage rewrite, second queue, double buffering or overlapping use of the stage by independent calls. The serialized-context API and synchronous call completion are part of this lifetime argument; introducing concurrency would require revisiting it.

The stage is sized for the largest projection: 9,216 × 3,072 FP32 elements, or 113,246,208 bytes. A smaller conversion writes only its matrix extent; the retained tail is neither zeroed nor part of that projection's input. [metal_mps.m](../../src/metal_mps.m) binds each weight view to the same stable stage with the appropriate dimensions, while input/output views retain their existing scratch buffers. Views are descriptors over storage, not another expanded model. A new token-count shape is prepared completely before replacing the old cache; reset may reuse the current cache. The vocabulary head remains on its shared expanded FP32 embedding buffer and does not borrow the compact stage.

The bounded loader in [metal_weights.c](../../src/metal_weights.c) checks actual packed payloads using a 64 KiB read buffer and a 256-entry admission table. Both signed zeros are accepted; both FP8 NaN codes are rejected. Every nonzero reconstructed weight must be normal and finite, and the scale itself must be positive FP32-normal widened BF16 even if some larger code could otherwise produce a normal product. This controls conversion behavior; it does **not** prove that arbitrary activations or intermediate arithmetic cannot underflow/overflow. Activation scales retain their separate positive/finite validation and are not used to quantize execution. A failed read can partially write its destination, but no failed load is published as a usable context.

[Compact projection tests](../../tests/test_metal_compact.m) exercise production dispatch with expanded synthetic FP32 controls, A/B/A stage reuse, captured matrix extents and retained tails, guards, shape changes and cache ownership. They protect this ordering/lifetime contract; they are not proof of independent full-model agreement. Supported full FP32 residency and these regression controls remain maintained.

## Capacity and memory

`lunelette_metal_limits` separates total KV capacity C from maximum call size B:

| C / B | Scratch bytes | KV bytes |
|---|---:|---:|
|256 /256 (default) |87819264 |54525952 |
|4352 /256 |222036992 |926941184 |
|4352 /128 |111280640 |926941184 |

KV layout is 26 layers ×2 buffers ×C rows ×1024 FP32 values. Limits are immutable; there is no resize or implicit multi-block ingestion in the runtime. The service or another caller partitions longer input into blocks of at most B. Last blocks may be shorter. B is newly processed tokens, not accumulated history.

Resident weight bytes are 13716025344. Either compact strategy uses 4637454336 weight bytes plus 113246208 stage bytes. Add the selected scratch/KV above for named GPU totals. The memory query reports actual owned buffer lengths, excluding host checkpoints, MPS/driver-internal allocations and allocator overhead. These are not physical peaks.

Before buffer allocation, shape/length arithmetic and device `maxBufferLength` are checked. The unchanged guard charges weights, stage, scratch/KV growth and host allowances, with a 2 GiB reserve and 24 GiB ceiling. Its default allowance is 310 MiB; long profiles additionally charge positive scratch/KV growth over the default plus 8 MiB host allowance. Required availability must be strictly greater than the estimate. Initially insufficient availability may be polled at up to 50 ms intervals under one 3-second monotonic deadline; clock/query errors fail closed, late sufficient readings are rejected. Destroy does not wait. This is a heuristic, not a memory-pressure monitor or allocation guarantee; do not reduce the guard or retry a refused validation run automatically. Avoid another full resident model.

Each prefill/append/decode encodes one command buffer and waits once before publishing logits. Scratch/KV/weights are reused, with no per-operator CPU reads or warm-path weight uploads. MPS retains only the current shape: seven kernels and 182 matrix-view triples. Shape changes replace the cache transactionally before encoding; views retain buffers but not the owner. Framework-internal behavior is opaque. A failed shape preparation requires reset, not a silent custom fallback.

### Allocation accounting versus admission

The shared [allocation planner](../../src/metal_plan.h) drives both requested buffer lengths and preflight accounting. Most scratch is proportional to B, attention scores use `4 × 32 × C × B` bytes, and KV uses `26 × 2 × C × 1,024 × 4` bytes. The logit vector stays 131,072 FP32 values. Reducing B therefore shrinks scratch but not retained KV or the conversion stage; raising C does not enlarge a single allowed block automatically. These are allocation relationships, not predictions of throughput or numerical equivalence across block partitions.

The guard does not simply add the named total to another full scratch allowance. Its charged base is weights plus stage plus positive scratch/KV growth over the original 142,345,216-byte combined baseline, plus 8 MiB for a long profile. The existing 310 MiB allowance and 2 GiB reserve are then added. The long-profile increment covers bounded additional host lifetimes rather than only GPU payloads; future host growth must be charged, not hidden by weakening reserves.

Availability is a heuristic `(free_count + inactive_count) × page_size`, with checked arithmetic. Mach already includes speculative pages in `free_count`; they must not be added again. This estimate, named buffer lengths and sampled physical footprint are different quantities. The [memory-guard tests](../../tests/test_metal_memory_guard.c) check conversion bounds and policy thresholds/deadlines without allocating a model; they do not guarantee successful allocation under real memory pressure.

## API and ownership

[include/metal.h](../../include/metal.h) is the precise C API; no Apple types cross it. Calls on one context, including handle operations/reset/destroy, must be serialized. The C wrapper retains one ARC owner; destroy releases buffers, queue, pipelines and cache. Model files close after loading, including on failure.

| Operation | Contract |
|---|---|
| `create`, `create_with_options` | Default C256/B256; NULL options select compact arithmetic/MPS |
| `create_configured` | Explicit supported limits above; NULL limits select 256/256 |
| `get_limits`, `get_memory` | Nonmutating queries; invalid arguments leave output unchanged |
| `prefill` | Empty valid sequence; 1..B valid IDs |
| `append` | Nonempty valid prefix; 1..B IDs; new total≤C |
| `decode` | One ID appended to nonempty valid prefix; new total≤C |
| `reset` | Empty valid state; invalidates checkpoint and masks stale KV tails without clearing buffers or reloading weights |
| `destroy` | Release context; NULL permitted; destroy all handles first |

Successful model calls return borrowed 131072-element FP32 logits, valid until the next **mutating** context call or destroy. Memory/limits queries and checkpoint create/destroy do not invalidate that borrow. Calls returning logits set the output pointer NULL on failure. Invalid arguments reject before GPU/cache mutation; execution/allocation/nonfinite failures invalidate the sequence and checkpoint. No partial prefix is published as valid. All logits must be finite before length and endpoint publication. Model operations, reset and queries return 0 on success and -1 on failure; constructors return a context or NULL. Diagnostics use caller-provided error/capacity storage. IDs must lie in 0..131071; the runtime does not interpret chat framing.

The compact arithmetic default applies to `create()`, NULL options and `LUNELETTE_METAL_OPTIONS_DEFAULT`. Compatibility enum identities matter: `{0}` and partial initializers omitting strategy still select resident FP32; omitted prefill still means MPS. Zero initialization is not the default-options macro. Explicit resident/arithmetic/lookup are honored; invalid enums fail before checkpoint I/O. No admission fallback or retry is introduced.

### Host generation and selection

[generation.h](../../include/generation.h) defines the shared `lunelette_select_greedy`: scan every borrowed vocabulary score, reject any NaN/Inf and use strict greater-than so ties (including signed zero) select the lowest ID. Selection allocates nothing, retains no pointers, leaves the token output unchanged on failure and requires disjoint input/output/error storage with immutable live logits throughout the call.

`lunelette_generate_with_logits` supplies the full prefix to its callback each step, with default capacity 256; `lunelette_generate_with_logits_capacity` explicitly accepts 256 or 4352. Config budget is 0..256 (HTTP instead requires 1..256), default initializer is budget 32/EOS 2. Prompt IDs are validated even for zero budget; a prompt ending EOS is not itself a stop signal. Generated EOS is included in the caller-owned fixed result. Stop precedence is generated EOS, then token budget, then context. Results are wholly unchanged on failure, although callback side effects cannot be rolled back. Long-prefix scratch and logits are internally owned/freed on every path; callbacks only borrow IDs/logits during the call and must not retain them or mutate config/prompt/result. This full-prefix callback contract is not the native service’s incremental materialization API.

### Checkpoint handles

A handle owns a full finite logit copy and a prefix boundary, **not a KV copy**. Only one handle is active per context. Successful create replaces the active handle; the caller still owns and must destroy older handles. Use a separate output variable during replacement so the old pointer is not lost.

Create requires a valid materialized nonempty endpoint and an available checkpoint epoch; epoch exhaustion rejects without replacing the prior handle. Allocation/argument failure preserves the prior checkpoint; nonfinite endpoint detection invalidates state. Repeated restore rewinds to the retained prefix and copies saved logits, without GPU work. Append/decode preserve the active checkpoint; reset or execution failure invalidates it. Stale/wrong-context restore preserves state and returns NULL logits. Handles must be destroyed before their context; forged or destroyed pointers are invalid C usage.

Each handle owns 524288 host payload bytes, excluded from the GPU memory query; replacement temporarily requires two payloads plus metadata. The runtime does not own/free caller handles. Service pending-token materialization and asynchronous publication are separate [service semantics](service.md#checkpointrestore-and-recovery).

### Why restoring a boundary can reuse KV

Successful append/decode writes KV only after the current prefix. The saved prefix therefore remains intact while later rows are appended. Restore shortens the visible length and replaces endpoint logits; stale tail rows are masked by that length, and a new branch overwrites them as it grows. This is why the handle needs a prefix boundary and copied logits rather than a KV snapshot. It is not a portable serialized checkpoint or permission to transfer state between contexts.

The runtime checks active-handle identity, context identity and checkpoint epoch before restoration. Replacement makes older handles stale; reset and failed execution invalidate the active handle. Execution temporarily withdraws endpoint/checkpoint validity before scratch/model work and republishes it only after command completion and full finite-logit checking. Argument rejection before mutation differs from execution failure after mutation. Keep those publication boundaries when changing selection or service ownership. The implementation is in [metal_runtime.m](../../src/metal_runtime.m); [runtime tests](../../tests/test_metal.m) cover checkpoint/state and injected failure seams.

## Implementation and verification

[metal_weights.c](../../src/metal_weights.c) validates/loads the model; [metal_yarn.c](../../src/metal_yarn.c) prepares positions; [metal_encode.m](../../src/metal_encode.m) encodes transformer operations; [metal_mps.m](../../src/metal_mps.m) owns shape views; [metal_runtime.m](../../src/metal_runtime.m) owns lifetime/state; [metal_kernels.metal](../../src/metal_kernels.metal) contains FP32 shaders. The C embedder stores shader text in the binary: no runtime shader file, absolute source path, `xcrun metal`, Python or repository working directory is needed.

## Verification

[Testing](../development/testing.md) separates host, GPU, protocol and supplied-resource suites. Synthetic tests include shapes/tails, scalar reconstruction, allocation/encoder failure seams, finite checks, prefix guards and checkpoint state. They do not establish full-model correctness or actual GPU-device-fault recovery. ASan/UBSan instrument host code, not shader execution. Independent fixture commands/schemas are centralized in [validation](../development/validation.md); frozen native controls are described in [native regression](../development/validation.md).

## Safetensors / embedding API

[include/safetensors.h](../../include/safetensors.h) exposes `st_open`, `st_find`, `st_read`, `st_close`, `st_bf16_to_f32`, and `st_embedding`. The file starts with an 8-byte little-endian header length, then that many JSON header bytes; tensor offsets are relative to the following payload base. Open validates the entire header and file layout without loading weight payloads. Find returns a borrowed tensor pointer; read takes a byte offset relative to that tensor and returns raw little-endian bytes. Embedding accepts a BF16 rank-2 tensor, a nonnegative in-range token, and an output buffer of exactly the row width. It reads only that row and converts BF16 bits to IEEE binary32 via `memcpy`, preserving signed zero, subnormals, infinities and NaN payload bits (no arithmetic conversion). Open/read/embedding and bulk loaders return 0 on success, -1 on failure; open provides a diagnostic. Find returns NULL when absent, numeric conversion helpers return floats and close returns no value. Initialize/open a fresh handle, then close it, including on error. Handles are not thread-safe. Do not modify public tensor descriptors or reopen live handles.

Deliberate limits (unsupported inputs fail, not silently reinterpret):

- F8_E4M3 (one byte), BF16, F32, I64 only; at most 8 dimensions, 4,096 tensors, 1 MiB header.

- Nonempty unique tensor names, up to 255 printable ASCII bytes. JSON escapes for quote, slash, backslash and printable ASCII `\uXXXX` are decoded. Non-ASCII/control strings are unsupported, including metadata strings.

- Optional string-to-string metadata: at most 64 unique keys, 255-byte keys, 4,095-byte values. Unknown tensor fields and duplicate fields are rejected.

- Unsigned decimal integer syntax only for shapes/offsets; dimensions <= INT64_MAX. Scalars and zero-sized tensors work. Shape products and byte sizes are checked for overflow. Offsets must exactly cover the payload, without gaps, overlaps, or unindexed trailing bytes; empty tensors must be on interval boundaries.

- Seekable files no larger than `LONG_MAX`; on 32-bit-long systems large model shards cannot be opened. Requires IEEE binary32 floats (compile-time checked).

- File must remain unchanged while open. No sharded-index parsing or tokenizer.

Standard C `fseek`/`fread` avoids POSIX mmap and whole-model RAM residency. The implementation holds a bounded header and descriptor table plus one BF16 row during lookup; the caller owns the float output. Bounded quadratic metadata validation favors clarity over large-header performance. No new dependencies.

`st_load_bf16` is a reusable bounded tensor-to-F32 helper: the caller owns the float buffer, validates the desired shape, and passes the exact whole-tensor count (not a capacity). It uses 64 KiB (65,536 bytes) of scratch space, validates dtype, count and handle membership, and may partially modify output on I/O failure. It allocates nothing and retains no state. See the header contract. See [validation](../development/validation.md) for supplied-resource comparisons.

## FP8 numeric reads

`st_f8_e4m3_to_f32` implements the finite E4M3 variant: signed zero, subnormals in units of 2^-9, finite extrema +/-448 and NaN bytes 0x7f/0xff. Only the exact safetensors dtype spelling `F8_E4M3` is accepted. `st_load_fp8` fills exactly the caller's FP32 destination using 64 KiB scratch, rejecting NaNs and nonfinite reconstructed products. The positive finite FP32 scale is multiplied after decoding, without intermediate BF16 rounding. Invalid arguments leave output unchanged; I/O/nonfinite failures may leave partial data. The checkpoint-specific module owns companion names, scalar policy and shapes.
