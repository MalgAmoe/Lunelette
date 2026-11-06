# Production text runtime

The default application uses one resident Metal/MPS engine with required CPU
host work. It is a concrete runtime, not an operator-dispatch framework. It requires
macOS 15+, Apple Silicon (Apple7 GPU family or newer), the Command Line Tools
with a macOS SDK, and enough unified memory. Development validation uses a
32 GiB machine; smaller-memory configurations are not promised to fit.

All shell commands here run from the repository root, not `docs/`.

```sh
make BUILD=build/metal test
build/metal/lunelette \
  --weights models/Ministral-3-3B-Instruct-2512-FP8/model.safetensors \
  --tokenizer models/tekken.json --prompt 'What is 2 + 2?' \
  --max-new-tokens 32 --dump-output-ids
```

Supply the official HF-layout Ministral-3-3B-Instruct-2512-FP8 single checkpoint
yourself; no files are downloaded and no index file is read. Supply Tekken separately. Use distinct build directories for different compiler flags.
Other platforms can run `make core-test`, but cannot build the application.
A build with no supported device
or insufficient headroom returns an error, never falls back silently.

Tokenization, optional system message, `--raw` (BOS plus literal user text),
greedy tie-breaking, EOS=2, output decoding and ID dumps retain shared host policy.
The removed `--backend` and `--uncached` selectors fail explicitly. Total returned
prompt plus generated IDs is at most 256. The output budget is
`min(max_new_tokens, 256 - prompt_count)`; a full prompt or zero budget performs
no forward. A generated EOS is included in IDs and omitted from decoded text.
There is no batching, static W8A8 execution, vision, multi-turn reuse or larger-context
support. The larger checkpoint position limit is not a supported context claim.

## Strategy selection

The storage description below describes the resident FP32 default. Select
`--metal-strategy compact-fp8-arithmetic` (or the corresponding C options enum) for
the optional compact strategy. It keeps packed FP8 block projections and one stable
FP32 conversion stage; embedding/head, norms, activations and KV stay FP32. Prefill
policy `mps|custom` is independent of storage. No fallback occurs on admission or
execution failure. See [M4](real-workload-m4.md) for the admitted payload domain,
compact buffer/guard accounting, validation scope and measured prefill/decode tradeoff.

## Storage and numerical contract (resident default)

* Unique text weights are resident FP32, reconstructed directly from bounded mixed FP8/BF16
  reader chunks into final shared Metal buffers. No full host FP32 copy or
  duplicated vocabulary head is retained. The head shares the embedding buffer.
* All 236 required text tensors (embedding, final norm, nine in each of 26
  blocks) have exact logical shapes: 182 F8_E4M3 projections, 53 BF16 norms and
  a BF16 embedding. Each projection requires rank-0 positive finite BF16
  weight_scale_inv and activation_scale. The former multiplies decoded FP8 in
  FP32 with no BF16 rounding; the latter is validated but unused (not W8A8).
  Unknown language_model tensors are rejected. Other namespaces are parsed
  safely but never uploaded or validated as a vision architecture.
* Unique weights occupy 13,716,025,344 bytes (12.774 GiB). KV is
  `[26,2,256,1024]` FP32, exactly 54,525,952 bytes (52 MiB).
* Scratch is reused across layers and steps. Each prefill or decode uses one
  command buffer, with one completion wait before publishing last-token logits.
  There are no per-operator waits/readbacks, warm-path weight uploads or new
  weight/scratch/KV buffers. Multi-token block projections default to MPS GEMM.
* **Default change:** `create()` and NULL/default `create_with_options()` options
  now select MPS for the seven block projections when count > 1. Select
  `--metal-prefill custom` or `LUNELETTE_METAL_PREFILL_CUSTOM`
  through `lunelette_metal_options` for the precise custom path. The CLI rejects unknown/duplicate prefill modes. Decode,
  single-token prefill and the vocabulary head always retain custom arithmetic.
* MPS descriptors use Float32 storage, explicit tight rowBytes and
  `transposeLeft=NO, transposeRight=YES`, alpha=1, beta=0. Inputs are token-major;
  existing output-major weights are viewed directly, without copying/repacking.
  Apple does not specify full-FP32 internal accumulation in the inspected API;
  agreement is empirical, not an arithmetic guarantee. The strict acceptance
  bound remains `2e-4 + 1e-4*abs(reference)`, with exact greedy IDs. Synthetic
  evidence has been verified on Apple M5/macOS; other devices need validation.
* Only the current prefill shape is cached: seven kernels and 182 projection
  triples of matrix views. Repeating a shape reuses them; changing count replaces
  them transactionally, including after reset. Views retain buffers but not the
  owner, so no ownership cycle exists. Count remains bounded to 256.
  All descriptors/kernels/views are prepared before encoding any command work.
  Allocation failure invalidates the sequence without publishing a prefix; reset
  is required. There is no silent numerical or allocation fallback: select custom
  explicitly after an actionable MPS error. Framework-internal allocations and
  execution behavior remain opaque.
* Custom kernels use FP32 safe/precise Metal math, contraction disabled, compensated per-lane
  projection accumulation and a SIMDgroup reduction. RMSNorm epsilon is 1e-5.
  YaRN is independently implemented from the pinned checkpoint formula, with
  theta 1e6, factor 16, original context 16384 and beta 32/1. Split-half rotation
  and integer-block query scaling match the reference. See `NOTICE` for source
  and license attribution. Kernels deliberately specialize the fixed model.
* Before allocation a conservative bound includes scratch/KV and a 2 GiB runtime
  reserve. Estimated peak must be <=24 GiB and below currently reclaimable
  headroom. These are guardrails, not an allocator guarantee. Device recommended
  working-set size is performance guidance, not hard capacity. Avoid keeping
  another full model (for example PyTorch) resident simultaneously. Physical VM
  reclamation can lag Metal object release; immediate destroy/recreate may be
  handled by a bounded availability wait before data allocation: an initially sufficient
  estimate passes immediately; otherwise the same estimate is polled at up to 50 ms
  intervals against one 3-second monotonic budget. The observed accounting lag was
  up to about one second; three seconds provides margin, not a reclamation guarantee.
  Each sleep is capped to the remaining budget, including after interruption;
  clock/query errors fail closed, and late sufficient results are rejected.
  This heuristic does not check memory pressure, change the reserve, or guarantee
  allocation success; allocation can still fail cleanly. Destroy never waits.
  A final guard refusal ends a validation run; do not lower the guard.

## API and ownership

`include/metal.h` contains only C types: opaque create/create_with_options/destroy/reset/prefill/decode.
A context owns an ARC Objective-C owner through an explicitly retained C wrapper.
Destroy transfers that reference back to ARC, releasing weights, scratch, KV,
queue and pipelines. The checkpoint file is closed after uploads, including on failure.
Calls on one context must not be concurrent. No Python is involved at runtime.

Prefill requires an empty valid sequence; decode requires a nonempty sequence
with room for one token. Returned logits borrow 131072 floats until **any next
context call**. Failure returns a null logits pointer. Invalid arguments reject
before GPU/cache mutation. Once execution begins, command allocation, encoder,
command completion or nonfinite-logit failure invalidates the sequence; reset is
required. Reset makes stale KV tails inaccessible without clearing GPU storage.
Length is published only after completion and a finite check of every logit.

The tiny C build tool embeds the shader as a data header. No Python, `xxd`,
`xcrun metal`, absolute source path, runtime shader file or repository working
directory is required. The resulting binary can be moved elsewhere.

Implementation modules: `metal_weights.c` stages reader/model-facts validation;
`metal_yarn.c` computes positions; `metal_encode.m` encodes transformer work;
`metal_mps.m` prepares the bounded MPS view/kernel cache;
`metal_runtime.m` owns resources and sequence publication;
`metal_kernels.metal` holds the fixed FP32 kernels. Apple headers remain private.

## Verification

`make test` remains offline/model-free/Python-free, including synthetic complete
metadata preflight. `make BUILD=build/metal test` runs
small deterministic GPU MPS short/long-K awkward shapes against independent double sums,
current-shape cache replacement and allocation-failure checks, custom projection tails, RMSNorm, element, attention and YaRN
rotation comparisons, state rejection/reset checks, every encoder's nil path,
and simulated command/nonfinite failures. These do **not** establish full-model
correctness or actual device-fault recovery. Test checks remain active under
`-DNDEBUG`. No hardware fault is deliberately induced.

Optional real-weight validation accepts explicit paths only:

```sh
make BUILD=build/metal build/metal/compare_metal
build/metal/compare_metal --self-test
build/metal/compare_metal models/Ministral-3-3B-Instruct-2512-FP8/model.safetensors \
  fixtures/prompt.ids fixtures/generation.safetensors
# Repeat with a trailing --metal-prefill=custom to validate the custom path.
```

`prompt.ids` is whitespace-separated decimal IDs (1..256 tokens).
The independent reference safetensors file must contain `generated` I64 `[N]`
and `logits_0` through `logits_N-1` F32 `[1,131072]`, plus `block_25_STEP`
and `final_norm_STEP` F32 `[1,rows,3072]` (rows=prompt length for prefill,
otherwise 1). These are test-only snapshots of existing buffers, with
`N <= 256 - prompt_count`. Run multiple pairs in one invocation to exercise
reset/reuse across distinct prompts. The comparator runs two cycles, checks every
logit finite with `abs(error) <= 2e-4 + 1e-4*abs(reference)`, and requires exact
lowest-index greedy IDs. It also exercises invalid appends, the final KV slot
for a 255-token prompt, and same-process single-file destroy/recreate.
A final memory-guard refusal is final, without inference retries. Those extra state checks are not extra independent
numerical comparisons.
Fixture/oracle provenance must be verified separately before interpreting passes.
Generate independent fixtures in a separate process, never alongside GPU residency.

Real-weight validation and performance results are not part of the default
suite. Synthetic tests alone are not a claim of full-model or CPU/GPU equivalence.

The FP8-derived strict numerical baseline is pending: fitting longer-context
reference mismatches must be reported, not repaired by loosening thresholds.
