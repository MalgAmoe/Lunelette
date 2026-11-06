# M4: optional compact FP8 arithmetic execution

Status: **A/B/C/D complete in bounded scope; compact remains optional.** Resident
FP32 remains the default and provisional engineering control, not an independently
validated numerical oracle or quality-approved baseline. Operator/integration and
native full-model/state checks passed; task outputs were unchanged. Compact lowers
memory use and decode latency, at a cost to multi-token prefill. No longer context,
precision change or default promotion is implied.

Independent-reference checking remains part of the agreed workflow. Known strict
failures remain failures; reproduced unchanged discrepancies do not automatically
reopen investigation or block authorized engineering. See `AGENTS.md` and the
[M3 disposition](real-workload-m3.md) for the separate comparison contracts.

## Selection and execution

The C options append `strategy` after `prefill`. NULL options, the default macro,
and an aggregate initializing only the first field select resident FP32 (enum zero).
Unknown enum values fail before opening the checkpoint; there is no fallback.

CLI selection is `--metal-strategy resident-fp32|compact-fp8-arithmetic`.
The native service requires explicit `--strategy` with the same pair, alongside its
existing explicit model, tokenizer, context256, chunk256, prefill and port settings.
The Python client accepts `--native-strategy resident-fp32|compact-fp8-arithmetic`
(default resident) alongside `--native-model` and `--native-tokenizer`. It verifies
the returned strategy, rejects a mismatch with child cleanup, and records strategy
identity even when the scenario fails. Programmatic `Service` callers opt into
this check with `expected_strategy`; the CLI always supplies it. It does not infer numerical acceptance from
startup readiness or published validation metadata.

Unknown or duplicate server options fail before loading. `--metal-prefill mps|custom`
(CLI), or `--prefill mps|custom` (service), is an independent choice.

| Path | Resident FP32 | Compact FP8 arithmetic |
|---|---|---|
| One-token block projection | precise FP32 custom | arithmetic FP8 decode, FP32 scale, same precise custom reduction |
| Multi-token block, MPS policy | existing MPS | GPU conversion into shared FP32 stage, then existing MPS |
| Multi-token block, custom policy | precise FP32 custom | GPU conversion into stage, then existing precise FP32 custom |
| Embedding, 53 norms | FP32 | FP32, unchanged |
| Tied head | precise FP32 custom | precise FP32 custom, unchanged |

This is FP8 **storage**, not native FP8 arithmetic or static W8A8. Activations,
residuals and KV remain FP32; MPS internal accumulation precision is unspecified.
The compact scalar kernel preserves decode, scale, activation multiplication,
Kahan lane accumulation and SIMD sum order, with contraction disabled.

One context owns 182 packed projection buffers, host scale metadata, and one stable
FP32 conversion stage. All 26 layers' MPS weight views bind that stage with their
own dimensions. Conversion and consumption are ordered in the same command stream;
no CPU stage rewrites, extra queues, pools or double buffering are used. Shape
caches publish only after complete preparation. Reset preserves the current shape
cache while clearing logical sequence state. Existing invalidation and synchronous
completion rules remain in force.

## Compact loading domain

Raw reads use a 64 KiB byte buffer plus a 256-entry admission table, validating the
actual payload before a context can be published. It never creates a full CPU-expanded projection copy.
Only exact-length F8_E4M3 projection destinations are accepted. The weight scale
must be a positive, finite, FP32-normal widened BF16 value. Both NaN encodings
(`7f`, `ff`) are rejected. Each nonzero decoded value times scale must be finite
and FP32-normal; both signed zeros are permitted, including an all-zero payload.
A subnormal scale is rejected even if a large code would produce a normal product.
Errors identify the tensor and, for invalid payloads, byte position/code and scale.
A failed destination may be partially written but is never published.

Original rank-zero BF16 activation scales remain validated positive/finite and
unused. The public `st_load_fp8` acceptance rules and resident loader are unchanged.
This normal-conversion contract is **not** a universal activation-arithmetic proof:
arbitrary activations may still underflow or overflow intermediate products.

## Named allocation accounting

| Allocation, bytes | Resident FP32 | Compact |
|---|---:|---:|
| Weight buffers | 13,716,025,344 | 4,637,454,336 |
| Conversion stage | 0 | 113,246,208 |
| Scratch | 87,819,264 | 87,819,264 |
| KV | 54,525,952 | 54,525,952 |
| Total | 13,858,370,560 | 4,893,045,760 |

The stage fits 9216 × 3072 FP32 elements. Allocation-size preflight checks counts
and sums before admission/allocation. The guard source/policy is unchanged:
310 MiB overhead, 2 GiB reserve, 24 GiB ceiling, strict greater-than availability,
and a three-second settlement budget. Its base is 13,716,025,344 resident bytes,
or **4,750,700,544** compact bytes (weights plus the entire stage). The stage is
not hidden in overhead.

`lunelette_metal_get_memory` sums owned buffer lengths without mutating the context.
Service identity uses that query, separates conversion-stage bytes, and reports
shape/prefill/head kernel policy. For compact, `compact_model_validation` is
`external_evidence_only` (resident: `not_applicable`), replacing the stale B-era
`NOT_RUN` label. `numerical_comparison_performed_at_startup` is false and
`published_validation_document` identifies `docs/real-workload-m4.md`. These fields
point to published bounded evidence; they do not certify the supplied model or run
a comparator at startup. Loader/domain admission checks still run normally.
The legacy `numerical_validation` diagnostic label refers
to published strict reference mismatches, not a new comparison of the loaded file.
Known strict failures remain separately documented. Info formatting
checks its fixed capacity. Host metadata, command/MPS objects, load transients and
allocator/driver costs are excluded; driver memory remains null. Named totals are
not physical peaks or performance measurements.

## Model-free verification

Run from the repository root, without models, network access or Python ML libraries:

```sh
make all test
make service-test
make sanitize
make BUILD=build/ndebug CFLAGS='-std=c11 -O2 -DNDEBUG -Wall -Wextra -Wpedantic' test
```

Portable tests cover accounting, raw loading, signed zero/all 254 finite codes,
invalid scales/products, NaNs, malformed counts/dtypes/indices and later-chunk
failures. CLI tests cover explicit selection, zero-work flow and preload rejection.
The native adapter link-time stub test additionally accepts an explicitly supplied
compatible tokenizer, never opens a model, and checks strategy identity, both
prefill policies, accounting, JSON capacity and cleanup on query failure.

The unchanged frozen project/norm test retains its 192 cases. The new maintained
`test_metal_compact` uses production pipelines, projection dispatch and MPS cache
preparation with synthetic Q/down/K matrices across three layers. It checks 32
finite bitwise projection comparisons for token counts 1/2/6/2 and both prefill
policies, A/B/A staging in one command, stage snapshots/retained tails, output and
stage guards, reset/shape reuse, cache release, actual buffer accounting, conversion
shape tails/all finite codes/signed zero, and deliberate comparator mismatches.
Only three compact/control matrices are allocated, not full-model dummy weights.
GPU unavailability returns 77 and is reported as SKIP, not PASS.

These checks passed on the development Apple GPU, including ASAN/UBSAN and NDEBUG
builds. Those model-free tests alone do not establish full-model parity, task
quality or performance; the separately executed bounded campaigns follow.

## Full-model and session evidence

Validation used the official Ministral-3-3B-Instruct-2512-FP8 checkpoint at revision
`b35d4dfe56c142746f54dbd64f579faab2744308`, SHA256
`728f1826cd0e38191ca7b1379e81f78cf0555c6ffd95882aabd2404632346f86`, and tokenizer
SHA256 `600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843`.
Execution sources were the reviewed M4 implementation on top of `c79c292`, not a
clean build of that commit. Detailed raw campaigns remain external development
artifacts; this guide does not imply they are bundled maintained test targets.

- Compact and fresh resident each passed 64/64 finite bitwise native full-vocabulary
  comparisons per prefill mode: prefixes 6/24/150/248, prefill plus seven decodes,
  two resets. Each vector has 131072 scores. References were not recaptured.
- Compact passed 14 service/direct full-vector pairs and four pause/reset comparisons
  per mode; all 28 retained session vectors per mode matched M3 resident controls.
- Independent frozen CPU PyTorch/Transformers checks in C retained the known strict
  discrepancies: reported metrics and failing sidecars matched the established
  control. This was not an independent numerical pass. The reference reconstructs
  the same checkpoint into FP32, not native static W8A8 execution.
- D replayed seven frozen examples/eight turns twice per strategy/mode: all 64
  responses matched M3 and each other in IDs/text/stops; official-tokenizer framing,
  ledger and decoded-text checks passed. Three of seven examples passed each time.
  Arithmetic budget/format failures, fenced JSON and the recall-format failure
  remain failures. Operational completion is not task success.

These observations are bounded; they do not prove every-prompt equivalence.
Full-model experiments ran sequentially, without a concurrent Python model.

## Measured tradeoff and disposition

Apple M5, 32 GiB, macOS 26.2; battery-powered measurement, with no controlled
thermal/OS isolation. Three fresh processes per strategy/mode each ran one excluded
warmup sweep and three measured rotated-prefix sweeps. Strategy order was R,C /
C,R / R,C. Nine case observations per row are clustered in three processes.
Inputs and seven subsequent token IDs were fixed, not chosen to favor either path.

Primary MPS medians (milliseconds):

| Prefix | Resident prefill | Compact prefill | Resident decode/token | Compact decode/token | Resident total | Compact total |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 115.19 | 222.58 | 115.74 | 58.20 | 925.39 | 629.98 |
| 24 | 158.13 | 264.79 | 116.32 | 58.74 | 972.32 | 676.62 |
| 150 | 363.94 | 481.46 | 119.77 | 62.11 | 1201.51 | 916.03 |
| 248 | 525.07 | 640.06 | 122.58 | 64.70 | 1381.99 | 1092.92 |

Total is one prefill plus seven decode calls; decode/token is the median of case
seven-step means. Total medians are not sums of separately aggregated medians.
Synchronous API timing includes conversion, MPS preparation, GPU wait and runtime
finite checks; extra digest scans, reset, network, tokenization and sampling are
excluded. These are not end-to-end successful-task latencies.

Compact total latency was 21–32% lower. Process-paired total speedup ranges were
1.454–1.518x, 1.423–1.457x, 1.301–1.317x and 1.260–1.270x respectively. Decode was
roughly 1.9–2.0x faster, but prefill took 107–118ms longer. Short replies and
first-token-sensitive use can therefore favor resident. Long custom prefill remained
much slower than MPS; its results were evaluated separately, not pooled with MPS.

Warm-file-cache constructor medians were 5.332s resident and 1.275s compact;
observed ranges were 4.716–5.433s and 1.266–1.828s. Preflight hashing had read the
model: these are not cold-disk or complete service-startup measurements.

Median process `phys_footprint` after warmup was 12.990 GiB resident versus
4.637 GiB compact. Median largest observed point samples were 12.992 and 4.639 GiB.
These are process samples, **not measured peaks**, driver accounting or DRAM traffic.
They are distinct from the named buffer table above. Smaller-memory hardware was
not tested. No energy-efficiency claim follows from a battery-powered latency run.

**Disposition:** retain the explicit compact option for memory savings and longer
responses; keep resident FP32 as default. Task quality did not improve. Faster
failed/truncated answers are not faster successful tasks. Further optimization,
context expansion or default changes need separate scope and validation.
