# M6: longer context and chunked service ingestion

> **Historical milestone record.** Current interface details supersede this page: see the [native service contract](service.md) and [Metal runtime](metal.md). The historical results and limitations below are preserved, not current defaults.

M6 closes a **bounded engineering milestone**, not independent numerical or task-quality acceptance. Opt-in context 4352 and chunks 128/256 passed the retained real-model runtime, KV, session, cancellation and HTTP checks described below. Defaults remain context 256, resident storage and legacy reuse; generation and history remain 256, activations/KV FP32.

## Explicit runtime profiles

`lunelette_metal_create_configured` takes the existing options plus a separate `lunelette_metal_limits`. Existing options ABI and constructors are unchanged; NULL limits selects 256/256. Supported context/block pairs are:

| Total KV capacity C | Maximum single-call block B |
|---:|---:|
|256|256|
|4352|256|
|4352|128|

The limits query is nonmutating. Limits cannot be changed through the public API on an existing context. Prefill/append execute one block (1..B IDs), enforcing total<=C; callers must orchestrate logical ingestion. A short final block is allowed. Checkpoint ownership/invalidation contracts remain as in [M5](real-workload-m5.md); the endpoint may reach C, but no further append/decode is allowed at capacity.

## Allocation and admission

Shared planning drives preflight accounting and actual scratch/KV allocation. Named requested bytes, excluding weights and compact's 113246208-byte conversion stage:

| Profile | Scratch | KV |
|---|---:|---:|
|256/256|87819264|54525952|
|4352/256|222036992|926941184|
|4352/128|111280640|926941184|

Default resident/compact totals remain 13858370560/4893045760 bytes. At 4352/256 they are 14865003520/5899678720 bytes, approximately 13.844/5.495 GiB. Named buffer lengths are not physical peaks or driver memory.

The guard preserves the old default threshold, 310 MiB allowance, 2 GiB reserve, 24 GiB ceiling, strict available-memory comparison and 3-second wait budget. Long profiles add positive scratch/KV growth over the old 142345216-byte combined baseline plus an additional 8 MiB host allowance. At 4352/256 the requirements are >17203589120 bytes resident or >8238264320 compact (~16.022/7.672 GiB). The stage 2 conservative incremental host census is 7902913 bytes including 1 MiB allocator/rounding slack, below 8 MiB; future growth must be charged, not hidden by reducing reserves. This is named requested payload/lifetime accounting, not RSS. Every weight, stage, scratch and KV buffer length is checked against device.maxBufferLength before model buffer allocation.

These calculations alone do not establish sufficient live headroom. Retained model runs constructed these profiles successfully. The guard may refuse; no fallback or bypass is added.

## Verification scope

Normal, ASAN+UBSAN and NDEBUG model-free suites passed, including existing 192 frozen GPU cases and 32 compact-projection cases. New checks cover:

- Default/long allocation geometry, overflow/invalid profiles and guard thresholds.
- Descriptor-only failure injection at every scratch/KV allocation, ownership cleanup, zero-allocation max-buffer rejection and actual-length accounting through the query.
- B/B+1 prefill/append bounds; nonzero-prefix absolute positions and checkpoint restore/rebranch at 4350→4352; full capacity and 4353 rejection.
- Real GPU blits into all 52 KV buffers, retaining prefix and guard rows, while model computation is mocked. Prefix state is seeded, not obtained by running a long model.
- Isolated real score/attention shaders at 4352 stride, causal masking and output guards.

ASAN/UBSAN instruments host code, not GPU shaders. Same-function YaRN upload checks establish absolute-position plumbing, not a new independent 4351 reference. These model-free checks alone do not establish full-model correctness. See the retained model evidence below; real-model memory peaks remain unmeasured. Prior M5 evidence remains bound to its historical sources and short schedules.

## Service integration

Native server options `--context 4352 --prefill-chunk 256` (or 128) select the configured constructor. Omitted options default to 256/256. The Python workload client uses `--native-context 4352 --native-chunk 256`, checks `/info` against the requested profile, and otherwise retains 256 defaults. Generation budget, retained per-request ID/timing arrays and request-history count remain 256.

Long-profile admission is 262144 HTTP body bytes and 32768 decoded UTF-8 bytes; default admission stays 8192/4096. The service parser alone is limited to 64 nodes (including its sentinel) and depth 4. Legacy memory/file parser defaults are unchanged. Decoded strings alias the owned parser body copy. Connection metadata/output and all accepted-connection inputs are heap owned; shutdown and disconnect free them. Input is framed/tokenized once; materialization consumes contiguous ID chunks.

The worker drives READY/MORE/ERROR steps, each executing at most one prefill, append or decode. Rebuild/reuse is chosen once; rebuild resets once. The worker publishes committed progress and checks cancellation after every block, including the last, before sampling or snapshotting. Pending EOS/non-EOS and save/restore retain M5 invalidation and needs-reset semantics. No runtime call runs on a socket thread. CPU framing still runs on the socket owner under the session mutex; it is not asynchronous and does not promise zero health/cancel latency.

Metrics version 3/work version 2 count disjoint actual API `prefill_*`, `append_*`, `decode_*` attempted/committed tokens plus calls. Append includes rebuilt chunks, not just reuse. `recompute_*` counts previously committed tokens repeated after reset (including a strict-off saved recipe); `pending_*` counts emitted tokens reconciled once. Both are subsets: never add either to the three API categories. Ingestion timing aggregates include every block/head. Timing steps count complete materialization windows, not runtime API calls: a window containing prefill and append chunks is charged wholly to prefill timing; append-only windows to suffix timing, otherwise scalar windows to decode timing. Use work counters for actual API counts; these timers cannot give individual chunk/API latency. Legacy `newly_processed` describes successfully materialized initial prompt coverage, not partial or repeated work; use work receipts on failed/cancelled jobs. No unbounded timing array is added. Historical captures retain their original schema and meaning.

### Envelope and host bounds

Full ledger 4352 uses 30465 bytes (`[131071,...]`); snapshot envelope is below 512, so 30977 bytes suffice, below the transport's 64000-byte route reply and 65536 output. Every ID append is checked; message admission reserves the complete ledger before mutating it. Retained request output contains only 256 generated IDs. A conservatively escaped 4096-byte retained text is 24576 bytes; IDs need at most 2817 bytes even for unsigned maxima; metrics 7200 and envelope 2048 yield 36641, below 64000. Terminal events have no text delta: metric 7200 + bounded envelope<800 stays<8192. Text-delta events have no metrics: at most 1027 decoded bytes ×6 escape expansion plus 800 gives 6962, below 8192. Executable maximal-counter/timing tests exercise 6668-byte metrics and 7152-byte terminal records, full 4352-ID serialization and worst escaping. Nonfinite/astronomical diagnostic durations are not valid timer samples; serialization failure remains explicit, never truncates silently.

All 16 connection structs (including retained outputs) total 1051136 bytes; long inputs total 4259856. Session payload 2111888 includes 1945600 bytes of history (256 ×7600), 131072 bytes of events and both 4352-ID ledgers. Native owner 39072 includes both prefix mirrors, receipts and 4096-byte info. Long tokenization can simultaneously hold 524304 character-heap +131072 result +1572864 BPE bytes, plus 65552 default-character stack bytes. Parser adds 262145 body +2048 node bytes, not a second decoded-string allocation. Generation heap prefix 34816 and logits 524288 coexist; snapshot replacement uses two 524320-byte handles, but does not coincide with generation/tokenization scratch. The persistent tokenizer has 3129728 struct bytes plus vocabulary bytes (<=16 MiB); it is unchanged by this profile. Runtime Objective-C/framework/allocator ownership remains covered by the unchanged baseline allowance/reserve, not claimed as an enumerated physical peak.

The conservative incremental census charges all input growth 4063232, the entire 2293792-byte long tokenizer payload/stack, entire 262145 parser copy, 69632 for prefix/generation scratch, 65536 for both pairs of ledger growth, 100000 extra stack, and 1048576 allocator slack:7902913 bytes. It deliberately overcounts sequential lifetimes; the transport test checks it against the planner's unchanged 8 MiB H. No increase or weakening of the 310 MiB baseline, 2 GiB reserve or 24 GiB ceiling.

### Model-free checks

`make test` includes long parser bounds, full-ledger/envelope serialization and deterministic cancellation barriers after mid/last chunks for generation and save, including failed attempted work. Optional macOS CPU-only adapter/localhost tests:

```sh
make service-test
make service-long-test TOKENIZER=/path/to/official/tekken.json
make service-checkpoint-test TOKENIZER=/path/to/official/tekken.json
```

These use the actual native adapter and generation/tokenizer APIs with explicit link-time runtime stubs, never weights. Both blocks and all reuse policies cover B/B+1/4096/4352, 4353 rejection, pending EOS/non-EOS, restore/rebranch and failure. Long HTTP checks measure complete preprocessing and concurrent health response separately from isolated BPE, across six bounded byte patterns including escaping and multibyte text. These are diagnostic measurements, not a hard real-time bound.

## Retained model evidence and limitations

| Check | Retained outcome | Scope |
|---|---|---|
| Seven reduced MPS cells | All seven pass, 313.85 seconds total | Short compact/resident ON, compact OFF, B128 cancellation; long resident/compact B256 and compact B128 |
| Numerical/state contracts | Full finite 131072-value vectors; 52 KV hashes per endpoint; fresh replay and repeated restore/rebranch | Same schedules bitwise; changed partitions use `abs(error) <= 2e-4 + 1e-4 * abs(reference)` |
| Real HTTP slice | Pass; 83 requests, 35 model calls and 2 restores, 4621 input tokens | Integration/state/finite evidence; natural-EOS-ready and incomplete-UTF8 branches not exercised |
| Custom resident B256 continuation | 305 operations and 42 comparisons pass | Explicit bounded continuation; not all custom configurations |
| Useful-work panel, matched ON/OFF | Each policy 4/5 response checks, 2/3 distinct tasks | Identical answers and logical trajectories; replay is not an independent task |

The arithmetic task returned **33 instead of 43** under both policies. The other answers were LARCH and memory values 17/9/9. Do not attribute the arithmetic failure to model versus engine without independent evidence. Recorded ON/OFF session times were 30.39/47.07 seconds and processed-token totals 6411/10661; the 4250 avoided tokens are diagnostic work accounting, **not a successful-whole-panel speedup claim**.

The original 32-job exhaustive campaign was not completed: 11 jobs were retained, and remaining 21 were suspended in favor of risk-driven coverage. Historical compiler-file-limit failure, timeout/cleanup failure, memory admission refusal and HTTP-client failure remain failures, not erased by successful continuations. Known short independent-reference discrepancies remain unchanged failures; **independent long-context correctness is unverified**. Native replay/parity is not an independent oracle. No peak-memory, energy, general performance or broad quality acceptance follows. Sampled physical footprints are not peaks.

Evidence was collected against a frozen 104-file implementation inventory. New builds must have new source/product receipts; do not rewrite historical evidence. The runtime `/info` field describes external evidence, not certification of the current binary or of every supported configuration.

## Regression tiers

- **Fast:** `make test` (model-free C/Metal checks, no Python/network/models), `make sanitize`, NDEBUG; optional `make service-test` and tokenizer-backed `make service-long-test`. These are not model-quality checks.
- **Focused model-backed:** [current-source seven-cell tooling](long-context-testing.md). Build into a fresh directory, then explicitly supply model/tokenizer and a fresh result root. Historical ~5 m 14 s is guidance, not a deadline guarantee.
- **Extended/change-triggered:** custom backend changes, altered allocation geometry, precision/reduction order, checkpoint semantics or tokenizer/admission changes require their relevant additional configurations and independent checks. This is not a routinely launched Cartesian campaign.

A [current-source real-HTTP repeat](long-context-http-testing.md) is also available. The promoted seven-cell run passed. The HTTP child completed; its initial campaign stopped on an offline environment-census mismatch. Corrected offline replay passed on the retained data; the stopped receipt remains unchanged. See the HTTP document for the precise correction and subsequent fresh full-launch pass at commit `4d06d326acccfd8201fe38c7d45c1f3b783346a5`. The earlier stopped receipt remains historical evidence; the corrected-launch gap is closed. No defaults or memory guards are promoted or weakened. The [current coverage report](real-workload-status.md) separates later tool/client additions from these source-bound M6 observations.
