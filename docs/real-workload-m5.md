# M5: experimental prefix checkpoints

Runtime/service integration, bounded full-model numerical/state checking and real-model
HTTP workload evaluation are complete. Prefix reuse remains **experimental and
explicit**, not the default. Independent numerical acceptance remains unresolved;
unchanged task outputs do not imply quality approval. See results and limits below.

## Startup policies

Add `--prefix-reuse on` or `--prefix-reuse off` to the native server invocation in
[the M2 guide](real-workload-m2.md). This selection is independent of storage
(`resident-fp32` or `compact-fp8-arithmetic`) and prefill (`mps` or `custom`).
Context256, FP32 activations/KV and memory guards are unchanged.

| Selection | Generation boundary | Checkpoint restore |
| --- | --- | --- |
| omitted (`legacy`) | Existing full new-turn prefill / cached partial continuation | Unsupported (501) |
| `on` | Validate exact retained prefix and execute only outstanding known IDs | Restore retained KV endpoint, saved logits and ledger |
| `off` | Full authoritative-input prefill at **every** generation request, including partial/no-suffix continuation | Logical ledger only; clear cache, committed=0 |

Within each generation request, later generated IDs still use scalar cached decode.
Weights remain loaded in all policies. The existing scenario CLI remains legacy;
use direct HTTP or the Python `Service` helper with explicit `native_args` for M5.
`/info` reports `prefix_reuse`, checkpoint capability and
`prefix_reuse_validation: "external_evidence_only"` for explicit policies.
The scope is published source-bound evidence, not runtime numerical attestation;
no numerical comparison runs at startup. Archived measured binaries retain their
historical pre-validation metadata rather than being silently relabelled.

## Asynchronous controls

After creating a session and supplying a user message:

```text
POST /sessions/1/checkpoint  {"id":"save1"}
GET  /requests/save1
POST /sessions/1/restore     {"id":"restore1","checkpoint_id":"1"}
GET  /requests/restore1
```

Use the checkpoint ID returned by the **completed** save, not an assumed value.
Both POSTs return non-streaming **202** with `id`, `operation`, `status:"running"`
and the admitted session snapshot. Poll the request until `completed`, `error` or
`cancelled`. Request results include `operation`, `checkpoint_id` (decimal string;
`"0"` means no applicable/published ID), `cache_restored`, counters and metrics.
On restore, the checkpoint ID also identifies the attempted target if it fails.

One replaceable checkpoint per session; replaced, reset/deleted, foreign or
invalidated IDs are rejected. IDs never wrap. Controls and generation share the
same bounded 256-request history and request-ID uniqueness, including across reset
and session deletion. Full history requires service restart. Mutation is rejected
while a job runs. Controls require a ready/partial session; save additionally
requires a nonempty ledger and empty incomplete-UTF-8 carry.

Save materializes all outstanding known IDs, including pending EOS exactly once,
without sampling a new token. A successful save establishes logical=committed,
pending=0 and preserves conversation phase. Restore discards later branch state
and UTF-8 carry, but does not rewind request history. Restoring ready-after-EOS
still requires a new user message before generation. A partial checkpoint can
continue directly from its saved endpoint logits with `on`.

Accepted controls survive closing the admission/poll connection. Generation-stream
disconnect cancels only its owning request. Explicit cancellation uses
`POST /requests/{id}/cancel {}`. Cancellation before publication wins; after runtime
mutation, failure/cancellation invalidates the checkpoint and enters `needs_reset`.
There is no rollback claim. A late cancel after completion does not undo publication.
A snapshot allocation failure preserves the previous checkpoint; if materialization
already succeeded, counters reflect its committed work, not the old pending state.

## Work and memory reporting

Metrics are explicitly versioned: `metrics_version:2`, `work_version:1`.

- `reused_tokens`: retained prefix at the first materialization (or control operation).
- Generation `newly_processed_tokens`: committed initial request input, not all
  subsequent generated-token work. For controls it counts that control's commits.
- `work.full_*`, `suffix_*`, `decode_*`: attempted/committed token counts for actual
  runtime API calls. Attempted means submitted to the API, not proof that every GPU
  kernel executed. Sum these three categories for model-token work.
- `pending_*` is a **subset**, not an additional token category.
- `prefill_calls`, `append_calls`, `decode_calls` count actual API attempts.
  `logit_restores` counts actual runtime logit-copy restores, even if cancellation
  later prevents controller publication; `cache_restored` denotes published success.
- `prefill_us`, `suffix_us`, `decode_total_us` classify generation materialization
  time by the actual API. Sampling restored logits is not a decode step.
- Controls report `materialize_ms`, `snapshot_ms`, `restore_ms` and `total_ms`.
  Pending CPU framing is charged once to the next work request, including a restore
  that discards that branch; it is never replayed into a later generation's metrics.

Snapshot payload is 524288 host bytes per handle; replacement temporarily holds two
payloads (1048576 bytes), plus metadata/allocator overhead. Old handles are promptly
released. No full KV copy occurs. The existing named allocation query is GPU-only;
process footprint point samples include more than named allocations and are not peaks.
Off-policy save pays materialization/snapshot costs too, but its saved logits never
bypass the next generation boundary's full recomputation.

## Verification limits

Model-free tests cover worker ownership, pending EOS, phase/UTF-8 boundaries,
replacement/reset/delete, bounded IDs, failure/cancel publication, request-owned
disconnect, real HTTP on/off control lifecycles, and adapter runtime-call boundaries
(100 retained + 12 suffix versus 112 recomputed). Runtime tests separately execute
sentinel KV blits with model compute mocked. These are not full-model parity tests.

Optional macOS native-stub HTTP tests require only an existing tokenizer, not weights:

```sh
make service-checkpoint-test TOKENIZER=/path/to/tekken.json
```

Further validation must keep same-schedule bitwise checks, changed-partition comparisons,
independent frozen-reference checks and task quality separate. A partition tolerance
exceedance remains a failure and receives a bounded assessment—not silent widening
or an automatic open-ended tracing study. Resident and legacy defaults stay unchanged.

## Bounded full-model results

The measured source was the 95-file pre-reporting snapshot with archive SHA256
`1393c61ff1c8d8e323b257b12e2772423f8699c71817d5489e4c0824c93309c2`.
Both storage strategies and both prefill modes were checked. Later changes to this
guide and metadata are reporting-only, not retroactively attributed to the captures.

- 256 frozen-native full-logit checks and 214 new exact-contract pairs were bitwise.
- 24 changed-partition pairs passed `abs(error) <= 2e-4 + 1e-4*abs(control)`;
  maximum absolute error 1.8596649169921875e-5, maximum normalized error 0.0728564
  (failure above 1), no sampled argmax changes.
- Real-model retained K/V prefix hashes, repeated restore/branch, zero suffix,
  capacity256 and rejection cases passed. Intercepted native API arguments confirm
  prefix 100 + suffix 12 uses append 12 on versus full 112 off, with matched endpoints.
- Independent frozen-reference checks **still fail**: 10/64 captured cases in each
  of four panels. Full normalized metrics and failed-sidecar sets/hashes match M3;
  matching failure counts alone was not used. These cases include intermediates,
  not necessarily failing logits. Native control remains provisional.

Initial post-EOS controller checking was explicitly teacher-forced. The subsequent
HTTP memory workload reached naturally selected EOS, saved/restored ready, and
processed a new-user suffix. This closes that bounded phase-coverage gap, not a
universal conversational-correctness claim.

## HTTP tasks and session timings

Eight task processes (both modes/strategies/policies, two repeats) produced 128
responses: 112 EOS and 16 budget stops, no missing turns. All IDs/text/stops match
M3/M4 and each other. Strict score stays 3/7 examples per run (48/112 example
executions, 64/128 turns). Known arithmetic/JSON-format/memory failures remain.
Official-tokenizer framing/decode and independent regrading were checked offline.

A separate 24-process timing campaign used three counterbalanced off/on pairs per
mode/strategy, one warmup sweep and three measured sweeps of three schedules per
load: 216 measured complete sessions plus 72 warmups. Weights remain loaded; all
864 generated streams match across configurations, including warmups. Timings include
reset/framing, first prefill/generation, save/pending reconciliation/snapshot, branch,
restore, replay, HTTP/metadata reads and control polling. Load is separate.

- short: 14 prompt IDs, 4 generated; checkpoint L18, then 4-token branch/replay.
- long: 20 prompt IDs, 8 generated; checkpoint L28, then 8-token branch/replay.
  Both deliberately stop at budget; neither is successful task-throughput evidence.
- memory: 22 prompt IDs, READY+EOS in 3 IDs; checkpoint L25. New user adds 17 IDs:
  on appends 17 versus off recomputes 42. Branch/replay each produces 25 IDs through
  EOS, retaining the failing fenced Python comment rather than exact requested code.

Median complete-session seconds; nine observations per cell clustered in three loads.
Range is the three paired ratios of process medians, not a confidence interval:

| Prefill | Storage | Schedule | Off median s | On median s | Paired process speedup range |
|---|---|---|---:|---:|---:|
| mps | resident | short | 1.708 | 1.354 | 1.22–1.30× |
| mps | resident | long | 3.191 | 2.880 | 1.11–1.17× |
| mps | resident | memory | 6.559 | 6.606 | 0.99–1.01× |
| mps | compact | short | 1.445 | 0.935 | 1.55–1.60× |
| mps | compact | long | 2.210 | 1.700 | 1.30–1.30× |
| mps | compact | memory | 3.946 | 3.945 | 1.00–1.01× |
| custom | resident | short | 6.152 | 2.577 | 2.34–2.41× |
| custom | resident | long | 10.224 | 4.671 | 2.16–2.24× |
| custom | resident | memory | 16.500 | 11.599 | 1.39–1.44× |
| custom | compact | short | 5.698 | 2.108 | 2.69–2.77× |
| custom | compact | long | 8.788 | 3.384 | 2.59–2.62× |
| custom | compact | memory | 13.324 | 8.689 | 1.53–1.54× |

MPS memory-turn benefit is essentially flat here; do not infer a general new-turn
speedup. Custom improves versus its own full-recompute control but remains slower
in absolute terms. These gains are **not versus legacy**: legacy already caches
partial continuation and has no checkpoint restore. Faster truncated/failed outputs
are not faster successful tasks. No significance or broad quality claim.

Per-generation physical-footprint points (1728 across benchmark processes) are not
peaks: median largest sampled/process about 13.043/4.689GiB resident/compact for MPS,
13.035/4.682GiB custom. On/off differences are negligible at that precision. Named
allocations remain 13,858,370,560/4,893,045,760 bytes respectively, distinct from process
footprint, driver memory and host checkpoint payload. No reuse memory saving is claimed.

Load-to-ready medians by mode/policy range 6.342–6.616s resident, 2.991–3.295s compact;
prelaunch hashes warm file cache, so these are not cold loads or pure constructor
timings. Mac17,2/32GiB/10 CPUs, macOS26.2 build25C56, clang17.0.0/SDK26.2; AC power,
charging 34%→85%. No energy/thermal, peak-memory or longer-context claim. Defaults,
precision, memory guards and tolerances are unchanged.
