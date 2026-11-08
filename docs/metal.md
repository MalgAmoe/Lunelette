# Current Metal/MPS runtime contract

The engine is one resident Metal/MPS implementation with CPU host work, not an operator-dispatch framework. This guide describes the current C runtime; the [service contract](service.md) describes framing, multi-turn state, pending tokens and HTTP controls. The [single-turn CLI](../README.md#single-turn-cli) remains limited to 256 total tokens; that is **not** the capacity limit of every runtime or service profile.

Requires macOS 15+, Apple Silicon supporting Apple7 GPU family or newer, Command Line Tools/macOS SDK and sufficient unified memory. Development validation used Apple M5/32 GiB; neither other devices nor smaller-memory configurations inherit that validation. Unsupported hardware or insufficient admission headroom fails, without silent fallback. Portable host tests remain available on other platforms.

All commands run from the repository root. `make` builds the CLI and `make server` builds the native service; neither runs inference. See [testing](testing.md). Python is not a native runtime dependency. No weights or dependencies are downloaded.

## Model-file and numerical contract

Only the official single-file HF-layout mixed FP8/BF16 Ministral 3 3B checkpoint is supported. No shard index, second shard, BF16-model fallback or arbitrary model configuration. There are 236 required text tensors: 182 F8_E4M3 projections, 53 BF16 norms and one BF16 embedding shared with the vocabulary head. Shapes are fixed. Each projection also requires rank-0 positive finite BF16 `weight_scale_inv` and `activation_scale`; all 364 companions are read and checked. Other namespaces are parsed safely, not treated as a supported vision architecture. Unknown `language_model` tensors are rejected.

Reconstruction is FP8 decoded to FP32 multiplied by the BF16 weight scale widened to FP32, **without BF16 rounding of the product**. Activation scales are unused after validation. Embedding/head, norms, activations and KV are FP32 in execution. This is not W8A8 or native FP8 matrix arithmetic. Vision, batching and 8B/14B models are unsupported. See [model facts](model-facts.md) and [safetensors](safetensors.md).

Custom kernels use safe/precise FP32 Metal math with contraction disabled; projection uses compensated per-lane accumulation and SIMDgroup reduction. RMSNorm epsilon is 1e-5. YaRN uses theta 1e6, factor 16, original context 16384 and beta 32/1, split-half rotation and integer-block query scaling. Attribution is in [NOTICE](../NOTICE). The checkpoint's larger position limit does not authorize runtime profiles beyond the explicit capacities below.

MPS uses Float32 storage, tight row strides, transposed right operand, alpha 1 and beta 0. Its internal accumulation precision is unspecified; Float32 storage is not proof of full-FP32 accumulation. Independent numerical acceptance remains **FAIL/open**. Exact native agreement, operator checks and generated text do not replace [independent validation](validation.md).

## Strategy selection

| Storage option | Multi-token block projections | Single-token block projections |
|---|---|---|
| `resident-fp32` (explicit supported option) | Resident FP32 weights → MPS or custom | Precise custom FP32 |
| `compact-fp8-arithmetic` (C/CLI default) | Packed weights → reusable FP32 stage → MPS or custom | Arithmetic FP8 reconstruction during projection |
| `compact-fp8-lookup` | Same conversion/stage path as arithmetic | Unscaled constant lookup then FP32 scaling |

Select CLI storage with `--metal-strategy`; the server uses required `--strategy`. Prefill policy is independent: `--metal-prefill mps|custom` for CLI, required `--prefill mps|custom` for server, or `lunelette_metal_options` for C. Multi-token MPS is the default in C/CLI options. Single-token prefill/decode and the tied head use custom kernels. Lookup changes only single-token block decoding, not the head.

Compact admission is stricter than the general loader: scales must be positive FP32-normal widened BF16, FP8 codes finite, and nonzero reconstructed products normal and finite. Unsupported values fail, never fall back to resident storage. One stable conversion stage is reused; no full host FP32 model or duplicate head is retained. See [lookup details](fp8-lookup.md) and the bounded [configuration recommendation](useful-workload-panel.md).

### Why one conversion stage is sufficient

The compact representation changes projection storage, not the transformer execution graph. Each context keeps packed projection bytes and host scale metadata. For a multi-token projection, [metal_encode.m](../src/metal_encode.m) encodes conversion of exactly that matrix into the stage, then its MPS/custom consumer, before encoding the next projection's conversion. These operations share one ordered command stream. There is no CPU stage rewrite, second queue, double buffering or overlapping use of the stage by independent calls. The serialized-context API and synchronous call completion are part of this lifetime argument; introducing concurrency would require revisiting it.

The stage is sized for the largest projection: 9,216 × 3,072 FP32 elements, or 113,246,208 bytes. A smaller conversion writes only its matrix extent; the retained tail is neither zeroed nor part of that projection's input. [metal_mps.m](../src/metal_mps.m) binds each weight view to the same stable stage with the appropriate dimensions, while input/output views retain their existing scratch buffers. Views are descriptors over storage, not another expanded model. A new token-count shape is prepared completely before replacing the old cache; reset may reuse the current cache. The vocabulary head remains on its shared expanded FP32 embedding buffer and does not borrow the compact stage.

The bounded loader in [metal_weights.c](../src/metal_weights.c) checks actual packed payloads using a 64 KiB read buffer and a 256-entry admission table. Both signed zeros are accepted; both FP8 NaN codes are rejected. Every nonzero reconstructed weight must be normal and finite, and the scale itself must be positive FP32-normal widened BF16 even if some larger code could otherwise produce a normal product. This controls conversion behavior; it does **not** prove that arbitrary activations or intermediate arithmetic cannot underflow/overflow. Activation scales retain their separate positive/finite validation and are not used to quantize execution. A failed read can partially write its destination, but no failed load is published as a usable context.

[Compact projection tests](../tests/test_metal_compact.m) exercise production dispatch with expanded synthetic FP32 controls, A/B/A stage reuse, captured matrix extents and retained tails, guards, shape changes and cache ownership. They protect this ordering/lifetime contract; they are not proof of independent full-model agreement. Original rationale and bounded evidence remain in [M4](real-workload-m4.md); single-token lookup details remain in [the lookup guide](fp8-lookup.md). Neither those controls nor supported full FP32 residency are retirement candidates.

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

The shared [allocation planner](../src/metal_plan.h) drives both requested buffer lengths and preflight accounting. Most scratch is proportional to B, attention scores use `4 × 32 × C × B` bytes, and KV uses `26 × 2 × C × 1,024 × 4` bytes. The logit vector stays 131,072 FP32 values. Reducing B therefore shrinks scratch but not retained KV or the conversion stage; raising C does not enlarge a single allowed block automatically. These are allocation relationships, not predictions of throughput or numerical equivalence across block partitions.

The guard does not simply add the named total to another full scratch allowance. Its charged base is weights plus stage plus positive scratch/KV growth over the original 142,345,216-byte combined baseline, plus 8 MiB for a long profile. The existing 310 MiB allowance and 2 GiB reserve are then added. The long-profile increment covers bounded additional host lifetimes rather than only GPU payloads; future host growth must be charged, not hidden by weakening reserves. The original census and its assumptions remain in [M6](real-workload-m6.md).

Availability is a heuristic `(free_count + inactive_count) × page_size`, with checked arithmetic. Mach already includes speculative pages in `free_count`; they must not be added again. This estimate, named buffer lengths and sampled physical footprint are different quantities. The [memory-guard tests](../tests/test_metal_memory_guard.c) check conversion bounds and policy thresholds/deadlines without allocating a model; they do not guarantee successful allocation under real memory pressure.

## API and ownership

[include/metal.h](../include/metal.h) is the precise C API; no Apple types cross it. Calls on one context, including handle operations/reset/destroy, must be serialized. The C wrapper retains one ARC owner; destroy releases buffers, queue, pipelines and cache. Model files close after loading, including on failure.

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

Successful model calls return borrowed 131072-element FP32 logits, valid until the next **mutating** context call or destroy. Memory/limits queries and checkpoint create/destroy do not invalidate that borrow. Calls returning logits set the output pointer NULL on failure. Invalid arguments reject before GPU/cache mutation; execution/allocation/nonfinite failures invalidate the sequence and checkpoint. No partial prefix is published as valid. All logits must be finite before length and endpoint publication.

The storage-only default migration applies to `create()`, NULL options and `LUNELETTE_METAL_OPTIONS_DEFAULT`. Enum identities are unchanged: `{0}` and partial initializers omitting strategy still select resident FP32; omitted prefill still means MPS. Zero initialization is not the default-options macro. Explicit resident/arithmetic/lookup are honored; invalid enums fail before checkpoint I/O. No admission fallback or retry is introduced.

### Checkpoint handles

A handle owns a full finite logit copy and a prefix boundary, **not a KV copy**. Only one handle is active per context. Successful create replaces the active handle; the caller still owns and must destroy older handles. Use a separate output variable during replacement so the old pointer is not lost.

Create requires a valid materialized nonempty endpoint. Allocation/argument failure preserves the prior checkpoint; nonfinite endpoint detection invalidates state. Repeated restore rewinds to the retained prefix and copies saved logits, without GPU work. Append/decode preserve the active checkpoint; reset or execution failure invalidates it. Stale/wrong-context restore preserves state and returns NULL logits. Handles must be destroyed before their context; forged or destroyed pointers are invalid C usage.

Each handle owns 524288 host payload bytes, excluded from the GPU memory query; replacement temporarily requires two payloads plus metadata. The runtime does not own/free caller handles. Service pending-token materialization and asynchronous publication are separate [service semantics](service.md#checkpointrestore-and-recovery).

### Why restoring a boundary can reuse KV

Successful append/decode writes KV only after the current prefix. The saved prefix therefore remains intact while later rows are appended. Restore shortens the visible length and replaces endpoint logits; stale tail rows are masked by that length, and a new branch overwrites them as it grows. This is why the handle needs a prefix boundary and copied logits rather than a KV snapshot. It is not a portable serialized checkpoint or permission to transfer state between contexts.

The runtime checks active-handle identity, context identity and checkpoint epoch before restoration. Replacement makes older handles stale; reset and failed execution invalidate the active handle. Execution temporarily withdraws endpoint/checkpoint validity before scratch/model work and republishes it only after command completion and full finite-logit checking. Argument rejection before mutation differs from execution failure after mutation. Keep those publication boundaries when changing selection or service ownership. The implementation is in [metal_runtime.m](../src/metal_runtime.m); [runtime tests](../tests/test_metal.m) cover checkpoint/state and injected failure seams. Historical design/evidence remains in [M5](real-workload-m5.md).

## Implementation and verification

`metal_weights.c` validates/loads the model; `metal_yarn.c` prepares positions; `metal_encode.m` encodes transformer operations; `metal_mps.m` owns shape views; `metal_runtime.m` owns lifetime/state; `metal_kernels.metal` contains FP32 shaders. The C embedder stores shader text in the binary: no runtime shader file, absolute source path, `xcrun metal`, Python or repository working directory is needed.

## Verification

[Testing](testing.md) separates host, GPU, protocol and supplied-resource suites. Synthetic tests include shapes/tails, scalar reconstruction, allocation/encoder failure seams, finite checks, prefix guards and checkpoint state. They do not establish full-model correctness or actual GPU-device-fault recovery. ASan/UBSan instrument host code, not shader execution. Independent fixture commands/schemas are centralized in [validation](validation.md#full-model-reference-comparison); frozen native controls are described in [native regression](native-regression.md).
