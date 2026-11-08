# M2 native short-session service (operational milestone complete)

> **Historical milestone record.** Current interface details supersede this page:
> see the [native service contract](service.md) and [Metal runtime](metal.md).
> The historical results and limitations below are preserved, not current defaults.

The Darwin binary uses the existing resident FP32 runtime, not scripted IDs.
After initially refused admission, real-weight MPS short-session checks passed
under the unchanged memory guard: resident continuation, EOS multi-turn, streamed
versus buffered text, reset recovery, cancellation/disconnect and a fresh CLI
comparison. Measurement reporting has also passed independently run real-session
checks in MPS and custom-prefill modes. M2 is operationally complete, **not an
accepted numerical or performance baseline**: strict independent reference
discrepancies remain open.

## Explicit launch

```sh
make build/lunelette-server
./build/lunelette-server --model /path/to/official-FP8/model.safetensors \
  --tokenizer /path/to/tekken.json --strategy resident-fp32 \
  --context 256 --prefill-chunk 256 --prefill mps --port 8081

python3 tools/real_workload.py --server build/lunelette-server \
  --native-model /path/to/official-FP8/model.safetensors \
  --native-tokenizer /path/to/tekken.json --native-prefill mps \
  --scenario tools/scenarios/native-short-v1.json --output /path/to/new-results
```

All native configuration is required; `mps|custom` and context/chunk 256 remain
fixed choices. M2 used `resident-fp32`; [M4](real-workload-m4.md) also supports
`--strategy compact-fp8-arithmetic` in the server. The convenience Python client
shown above defaults to resident FP32; add
`--native-strategy compact-fp8-arithmetic` to select compact explicitly. The client
requires the native model/tokenizer pair and rejects a missing or different strategy
in the server's `/info` response. It records the verified strategy in `result.json`,
including scenario failures, and retains raw `/info` in `client.ndjson`. No fake
switch, default weights or fallback.
The separate `build/test_service_server` remains model-free. The client has a
180-second startup deadline, explicit native generation deadline 1..120 seconds
(default 120), and four-second control-request sockets. Native shutdown has a
120-second safe-boundary grace period; the fake service retains three seconds.
Native stderr is drained while running, not left to fill a pipe. Unterminated
startup diagnostics are retained. Native scenarios admit up to 4096 message bytes
(subject also to the HTTP body and framed-token limits); fake scenarios retain
their 256-byte limit. A scenario/launch-mode mismatch fails before model startup.
No automatic retry.
Free-greedy scenarios always report `task_success: null`; protocol completion is
not task correctness. The short scenario exercises two paused sessions, partial
continuation and reset/re-prompt; it has passed a bounded real-model MPS run, not task-quality evaluation.

## Framing and state

Protocol `native-no-system-nonthinking-text-v1`: the native mistral_common 1.13.1
text-only user/assistant format, **not** equivalence to the HF bundled template
(which injects a default system prompt). Initial user is `[1,3,text_ids...,4]`;
a subsequent user after generated EOS=2 appends `[3,text_ids...,4]`. No hidden
system prompt or implicit thinking tokens. System/tool roles return 501.
Literal `[INST]` and `</s>` in user text are ordinary encoded text.

Empty-session generation and a second user before generation return 409. Budget
stop is partial: continue or reset only. EOS is ready for the next user, not
permission to generate an empty assistant turn. Generated IDs are authoritative;
assistant text is never re-encoded. Session snapshots include the full bounded
`token_ids`; request snapshots include that request's emitted IDs (including EOS).
Unsupported generated structural controls, invalid UTF-8, embedded NUL, and text
or piece overflow fail with `needs_reset`, never silent skipping/truncation.

Each model ID yields at most one complete-prefix text event; incomplete UTF-8
(up to three trailing bytes) persists across partial continuations. Native pieces
are bounded to 1024 bytes. Event cap is 8192, retained request text 4096 bytes,
HTTP response buffer 65536 bytes, request history 256 identities. Exactly-full
ring cancellation remains authoritative; dropped deltas never imply completion.

## Runtime ownership and recomputation

Main loads/hashes the model and tokenizer before announcing readiness. Successful
session initialization transfers runtime ownership to one worker. Every subsequent
runtime prefill/decode/reset/destroy is serialized there, outside the session
mutex. HTTP status and cancel can run during GPU execution. Shutdown joins the
worker before releasing the session. Startup failures release untransferred state.

Every materialization checks the exact runtime prefix mirror, including any
emitted pending ID. Partial continuation decodes that ID once. New user turns
reset and prefill the **entire authoritative ledger** with weights resident:
the legacy policy, not retained-prefix reuse or M5's strict off control. Logits are
invalidated before reset. This remains the omitted-policy default; see the
[M5 guide](real-workload-m5.md) for explicit experimental on/off checkpoint controls.
Tools, chunked ingestion and capacity changes remain unsupported.

## Identity and measurement limits

`/info` reports actual file SHA256 (64-KiB incremental read), build/dirty identity,
load duration including hashing/tokenizer startup, successful constructor count,
precision and readiness separately from
`diagnostic_known_reference_mismatches`. Named resident weight bytes are the
existing allocation contract, not physical memory or a peak. `/info` also names
scratch (87,819,264), KV (54,525,952) and total buffers (13,858,370,560 bytes).
Driver observations remain explicitly null. MPS accumulation remains unspecified.

`/info.process_memory_sample` samples process `task_info(TASK_VM_INFO)`
`phys_footprint` after resident load, with method, monotonic sample time and bytes.
Request metrics sample the same process value at worker selection and safe stop.
These are instantaneous whole-process observations, not engine attribution, a
peak, available memory, or a comparison benchmark. OS failure/unsupported hosts
serialize null; test generators never report synthetic process memory.

Terminal events and GET request snapshots share one bounded metrics formatter;
started/text_delta events have null metrics to keep large escaped pieces bounded.
Terminal IDs, session ID, before/after state and ledger counts, validity and reset
recovery are retained. GET preserves terminal state even after session reset.

Measurement definitions (controller CLOCK_MONOTONIC, never per-layer):
- `total_ms` is **worker-selected to safe stop**, excluding admission/queue wait,
  transport delivery and final memory sampling. It includes the before sample.
  Framing is separately measured at message admission, not generation.
- Each materialize callback is timed until host return, excluding `next` sampling,
  piece conversion, delivery and lock reacquisition. Durations are truncated once
  to integer microseconds. Initial full-ledger materialize is `prefill_us` (one
  step); continuation's first pending-ID materialize is a decode step.
- `decode_step_duration_us` records **executed callback attempts**, including
  failed/cancelled work whose sampled ID is discarded, not emitted tokens.
  `decode_total_us` is their exact integer sum; ms aggregates divide integer totals
  by 1000. `prefill_steps`/`decode_steps` are distinct from `generated_tokens`
  (which includes emitted EOS). No throughput or GPU kernel-time claim.
- `first_token_ms` observes first nonnegative ID immediately after `next` returns,
  before controller-lock reacquisition/delivery, even if cancellation discards it.
  Null means no ID became available. Client first-text arrival is different.
- Cancellation stores the **first** receipt monotonic time/source (explicit,
  disconnect, overflow, shutdown). Repeats never rewrite it. `cancel_latency_ms`
  spans receipt to worker safe stop after callbacks return. It remains null until
  a cancelled request terminates (and for requests never cancelled). Historical
  cancellation flags survive subsequent requests. Other live fields are provisional;
  final values are authoritative at terminal.
- Client `cancel_ack_seconds` and `cancel_completion_seconds` measure from cancel
  send to acknowledgement and terminal arrival, respectively, not GPU stop.

CLOCK_MONOTONIC is required; failure or an epoch beyond signed-64-bit nanoseconds
aborts rather than emitting fabricated/nonfinite times. This bounds decimal timer
widths independently of client deadlines; no server request-latency bound is claimed.
JSON appends check every variable-length addition. Up to 256 uint64 samples use
5376 bytes including separators; terminal metadata remains below the 8192 event
cap. GET additionally allows 24576 escaped text bytes plus <=2816 ID-array bytes
and metrics/metadata, fitting the 64000-byte JSON reply and 65536-byte transport buffer. Capacity failure is
an explicit error/cancellation, never a silently truncated completion.

## Verification

```sh
make service-test
make build/test_service_framing build/test_service_native
./build/test_service_framing /path/to/tekken.json /path/to/native-fixtures.json
./build/test_service_native /path/to/tekken.json
```

The last binary uses link-time runtime stubs and opens **no model**. It tests actual
adapter callback order, load count, prefix identity, EOS/new-turn recompute,
continuation, failures/nonfinite policy and reset. It is not native numerical
validation. The framing test compares independent native tokenizer CPU fixtures;
fixtures are supplied explicitly, never generated by the engine. Portable
`make test` includes synthetic ID/piece/state/capacity tests without models/network.

Bounded real-weight checks passed in **both MPS and custom-prefill modes** for
repeated residency, fresh CLI IDs/text, streamed/buffered text, 1+3 versus 4
continuation, actual EOS multi-turn, cancel/disconnect/reset recovery, terminal/GET
metric equality, sample sums/counts/units, process observations and client cancel
timings. Native regression separately passed 64 full-vocabulary vectors per mode
across two reset cycles. Normal, ASAN/UBSAN, NDEBUG and TSAN service checks passed.
The main agent personally reviewed the complete changes and new files, fixed
reporting bugs with regression coverage, and independently ran these checks.

This closes the bounded M2 operational milestone, not full-spec acceptance or
independent numerical correctness. Driver memory remains explicitly unavailable,
process samples are not peaks, and no performance/task-quality claim is made.
At M2 closure, M3 and later milestones had not started. For subsequent bounded
investigation, task results and the provisional-control decision, see the
[M3 disposition](real-workload-m3.md). M2 operational closure did not depend on
resolving every independent-reference discrepancy.
