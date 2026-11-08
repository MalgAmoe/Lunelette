# Native service contract

This reference describes [native.c](../../service/native.c), [session.c](../../service/session.c), [session.h](../../service/session.h), [http.c](../../service/http.c), [framing.c](../../service/framing.c) and the stateless bounded [tool grammar](../../service/tool_grammar.h). See [Metal runtime](engine.md) for runtime, model-file and memory policy; service framing and history are distinct from single-prompt CLI behavior.

## Process and execution ownership

The native server binds only 127.0.0.1, validates caller-supplied model/tokenizer/configuration, hashes files and loads once before readiness. There is one session, one worker and one generation, with no hidden inference queue. Startup flags, accepted profiles and omission behavior are owned by [running](../guides/running.md). Python is client tooling, not a native service dependency. Startup failures release untransferred runtime state; after initialization the worker owns all runtime calls, including reset/destroy, and shutdown joins it before session release.

`/info` reports actual strategy, prefill, capacities, protocol, capabilities, model/tokenizer SHA256, build identity/dirty status, load count/duration, precision policy, named allocation bytes and a startup process-footprint sample. Validation labels and documentation paths in that metadata are published-evidence pointers, **not runtime attestation**; startup performs no numerical comparison. Independent numerical validation remains **FAIL/open**. Native parity, successful HTTP exchanges and generated text do not make an independently validated baseline.

## Authoritative history and framing

There is one live session and one worker owning runtime execution. Native ordinary chat is `native-no-system-nonthinking-text-v1`, checked against the pinned `mistral_common==1.13.1` text-only format, not the HF bundled template that injects a default system prompt: first user input is `1 3 <ordinary text IDs> 4`, later input `3 <ordinary text IDs> 4`. EOS is ID 2, included in generated IDs but omitted from text. There is no system or assistant-message injection API. Special-token spellings in content are ordinary text, not caller-supplied structural IDs.

Session `token_ids` are the authoritative complete logical ledger; retained request `token_ids` are that request's generated suffix. Never decode/re-encode assistant text to reconstruct history. Logical tokens include accepted frames and published outputs; committed tokens are actually materialized KV. Newly emitted tokens, including EOS, remain pending until consumed exactly once by later materialization. Accepted messages may add further unmaterialized logical input. Check counts and state rather than assuming every logical token already has KV.

* **legacy:** reuses an unchanged materialized prefix and the immediate single pending non-EOS decode path; other extensions rebuild (not general new-turn reuse).

* **on:** retains matching committed prefixes and materializes only missing input, including pending EOS plus the next message frame.

* **off:** resets/recomputes at each generation boundary, while preserving normal within-response decode. Restore recovers logical state, not a committed cache.

Budget completion enters `partial`; another generate continues without a new user message. Natural EOS enters `ready`, requiring user input before another generation. User messages require ready state and either an empty ledger or preceding EOS. No silent history truncation or automatic budget extension occurs.

### Materialization and publication boundaries

A selected token is not yet a model input: logits from the committed prefix select the next ID, and consuming that ID in a subsequent runtime call creates its KV. For example, after publishing one output at a materialized prefix of length L, the logical ledger has L + 1 IDs while committed KV still has L and one output is pending. EOS follows the same rule even though it is omitted from text. With reuse ON, the next user frame can be materialized together with pending EOS; neither may be dropped or consumed twice. Accepted input can extend the unmaterialized suffix beyond that single pending output, so `logical - committed` is not generally a pending-token count.

[native_materialize](../../service/native.c) verifies the committed prefix and pending ID, chooses rebuild/reuse once per ingestion window, and executes at most one block per step. Rebuild resets once, then subsequent blocks append. A MORE result requests another step, not a second reset. [session.c](../../service/session.c) publishes committed progress under its mutex and checks cancellation after every step, including the final one, before token selection or checkpoint publication. A completed block can therefore count as executed work without producing a published output.

Runtime callbacks remain on the worker and outside the session mutex; state/receipt publication is synchronized separately. This lets HTTP inspection/cancel intent proceed while a GPU call waits, but does not preempt that call. CPU framing still runs on the socket owner under the mutex and is not promised asynchronous latency. Keep grammar extraction separate from these synchronization and publication rules. Timing windows can contain multiple prefill/append calls; use the work-v2 call/token counters rather than interpreting timing-step counts as individual GPU API calls.

## HTTP API

HTTP/1.1 requires exactly one nonempty Host and one Content-Length, **including zero-length GETs**. Transfer coding, pipelining and embedded NUL are rejected. Connections close after one response. Mutation bodies are strict flat JSON objects; unknown/duplicate keys and invalid schemas fail rather than being ignored.

| Method and path | Body / result |
|---|---|
| GET `/health`, `/info` | Empty body; readiness / launch metadata |
| POST `/sessions` | `{}`; create the sole session (reject if one is active) |
| GET `/sessions/N` | Empty body; current state, counts and complete native ledger |
| POST `/sessions/N/messages` | `{"role":"user","content":"…"}`; frame/append input, no model execution |
| POST `/sessions/N/generate` | `{"id":"g1","budget":32}`; admitted HTTP 200 NDJSON stream |
| GET `/requests/ID` | Empty body; retained status, text, IDs, state, metrics and recovery metadata |
| POST `/requests/ID/cancel` | `{}`; explicit cancel intent, not a completion receipt |
| POST `/sessions/N/checkpoint` | `{"id":"save1"}`; asynchronous HTTP 202 control |
| POST `/sessions/N/restore` | `{"id":"restore1","checkpoint_id":"1"}`; asynchronous HTTP 202 control |
| POST `/sessions/N/reset` | `{}`; clear session/runtime state and checkpoint |
| DELETE `/sessions/N` | `{}`; reset and close session |

Generation events are `started`, `text_delta`, then `completed`, `cancelled` or `error`. Stream text is provisional until terminal/retained-state checks; drain it promptly. HTTP inspect and explicit cancellation remain responsive during GPU execution; cancellation takes effect at worker safe boundaries. Other mutations reject busy state; a new generation also waits for the previous stream's transport delivery. Old closed sessions return 410; missing identities 404, schema errors 400, state/identity conflicts 409, capacity/framing errors 422, unsupported operations 501 and exhausted process history 503.

### State and operation matrix

Inspection and request cancellation remain available in every phase; they do not confer permission to mutate a busy runtime. Session IDs are server-issued increasing positive integers, distinct from caller-supplied request IDs and public decimal checkpoint IDs. Closed/older sessions cannot be revived. Client NDJSON command IDs and client wall deadlines are separate [interactive](../guides/interactive.md) contracts.

| Session phase | Admitted mutations when no job/undelivered generation stream blocks them |
|---|---|
| `closed` | Create a new session only |
| `ready`, empty | User message, reset, close; generation requires input and save requires a boundary |
| `ready`, accepted user input | Generate; user message is rejected until assistant turn completes; save/restore subject to capability/boundary checks |
| `ready`, ending EOS | User message, save/restore; generate requires a new user message |
| `partial` | Generate continuation, save/restore, reset, close; no new user message |
| `awaiting_tool` | Matching validated tool result, reset or close; no ordinary user input/generation |
| `generating` / `control` | Inspect/cancel; competing mutations rejected |
| `needs_reset` | Reset or close after the active job finishes; no resumability claim |

Unsupported roles return 501; supported roles at the wrong phase reject rather than silently append. Tool-result admission is the additional schema `{"role":"tool","tool_call_id":"…","content":"…"}` and requires the outstanding call ID. Details and fixed content validators belong to [tools](tools.md). Ordinary messages must not contain `tool_call_id`.

### Response fields and transport identity

Session snapshots report `session_id`, `state`, `logical_tokens`, `committed_tokens`, `pending_tokens`, `remaining` and the complete native `token_ids`; inspect capacity through `/info` and counts, not visible text length. Retained request records contain `id`, `status`, `text`, `generated_tokens`, `cancel_requested`, `stop_reason`, `state`, logical/committed/pending counts, `operation`, `checkpoint_id`, `cache_restored`, `session_id`, `before_state`, `before_logical_tokens`, `before_committed_tokens`, `before_pending_tokens`, `session_valid`, `recovery`, `metrics` and request output `token_ids`. Terminal snapshots remain historical even after session reset; active snapshots are provisional synchronized copies.

NDJSON events contain `event`, `id`, `session_id`, `text`, `stop_reason`, `generated_tokens`, current and before counts/state, `session_valid`, `recovery` and `metrics`. Started/delta events have null metrics; terminal events carry the same formatter as retained inspection. Concatenate text deltas, not terminal `text` (terminal records do not repeat the full response). Completed status alone is not task success. Tool/replay-specific metadata is conditional on those protocols.

Structured route errors contain `error`, `request_id`, `session_id`, `session_valid`, `state` and `recovery`. Recovery may direct `correct_request`, `restart_service`, `inspect_existing_request`, `reset`, `wait_for_request_completion` or `new_session`; terminal request recovery is `reset` for needs-reset, otherwise `none`. Duplicate IDs must be inspected, never treated as automatic retry permission.

Native decoding emits at most one complete-prefix text event per selected model ID. Incomplete UTF-8 (up to three trailing bytes) carries across partial continuations. Ordinary generated structural controls other than EOS, invalid UTF-8, embedded NUL or piece/text overflow fail explicitly with needs-reset. Native pieces are bounded to 1024 bytes. Tool-call bytes are buffered/validated under their own protocol and never authorize execution from a partial delta.

Controls survive closing their admission/poll connection. Disconnect cancellation is tied to the owning generation stream and request identity; a stale socket must not cancel a later job. Exactly-full event-ring overflow still records cancellation intent even if no more event fits. Poll retained state after transport failure; no dropped delta or truncated socket response implies completion.

### Checkpoint/restore and recovery

Controls require reuse on/off and no tools; legacy and **all tool profiles return 501**. Save requires a nonempty ready/partial token boundary without incomplete UTF-8. It first materializes all logical input, including pending output, then saves tokens, phase and logits while retaining a KV prefix boundary. It does not copy the KV buffers. One saved checkpoint is retained; a successful replacement invalidates the previous opaque decimal ID. Reset/close invalidate it.

HTTP 202 is admission only: poll `/requests/ID` to completion and inspect session state/ledger. The admission response contains `id`, `operation`, `status:"running"` and the session snapshot. `checkpoint_id` is a decimal string; `"0"` means no applicable/published ID, and a failed restore record can identify its attempted target. IDs never wrap; exhaustion requires service restart. Restore does not rewind process request history, and restoring ready-after-EOS still requires new user input. ON restore recovers saved IDs/phase, committed KV and logits (`cache_restored:true`, one logit restore). OFF restore recovers IDs/phase with committed=0 and pending=0; the next generation rebuilds. Restore itself performs no prefill/append/decode in either policy.

Failures are not blanket rollback: snapshot rejection after successful materialization can preserve the prior checkpoint while leaving the prefix newly committed. Pre-mutation rejection can preserve a usable phase. Runtime-invalid or mutated cancelled controls enter `needs_reset` and invalidate the checkpoint. Read terminal `session_valid`/`recovery` and state; do not retry blindly.

Generation cancellation, disconnect, event overflow or runtime/protocol errors enter `needs_reset`, invalidating checkpoints. Cancellation is observed between completed blocks, not GPU preemption; work may have executed with zero published tokens. Reset after the job ends is required before reuse. Reset/close do not clear process-lifetime request-ID history or replay reservations.

### Checkpoint records and ownership

The runtime handle retains endpoint logits and a KV boundary; the native adapter separately keeps the corresponding prefix IDs; the session keeps its saved logical ledger, phase and public checkpoint ID. These are complementary records, not interchangeable snapshots. Save first reconciles pending/logical input, then creates the runtime replacement, and only publishes the session checkpoint after success and cancellation checks. Allocation rejection cannot roll back already completed materialization. The old handle is destroyed only after a replacement has been created successfully.

Reuse OFF intentionally treats saved IDs as a replay recipe: restore clears runtime commitment, and the next generation recomputes. Reuse ON can retain the prefix because append-only execution has not overwritten it. Neither policy restores arbitrary stale runtime state after an execution failure. See [runtime checkpoint rationale](engine.md#why-restoring-a-boundary-can-reuse-kv), [native adapter tests](../../tests/test_service_native.c) and [checkpoint HTTP tests](../../tests/test_service_checkpoint_http.py).

## Bounds and measurement

* Context includes all structural, input and generated IDs: C=256 or 4352. Generation budget is 1..256 and must fit the remaining logical capacity at admission; no optimistic early-EOS overbooking.

* User content: 4096 UTF-8 bytes at C256, 32768 at C4352; tool results remain bounded separately. HTTP header limit 4096 bytes; body limit 8192 /262144 respectively.

* At most 16 socket clients; one generation stream. Ordinary connection/read/write deadlines are 2 seconds; native stream inactivity is 120 seconds (not a task wall budget). Event ring 16 entries, each 8192 bytes; retained generated text 4096 bytes.

* At most 256 admitted generation/control request IDs per process, no eviction; IDs are 1..64 ASCII letters/digits/underscore/hyphen, unique across reset/close. Generation stores at most 256 output IDs. Larger context does not enlarge these generation/history limits. The runtime's 24 GiB guard remains in force.

Terminal and retained-request metrics use `metrics_version:3`, `work_version:2`:

| Group | Actual JSON fields |
|---|---|
| Timing | `framing_ms`, `prefill_ms`, `decode_ms`, `first_token_ms`, `total_ms`, `prefill_us`, `prefill_steps`, `suffix_us`, `suffix_steps`, `decode_total_us`, `decode_steps`, `decode_step_duration_us` |
| Cancellation | `cancel_latency_ms`, `cancel_source`, `cancel_received_monotonic_ms` |
| Counts | `prompt_tokens`, `reused_tokens`, `newly_processed_tokens` |
| Memory / scope | `test_only`, `process_phys_footprint_before_bytes`, `process_phys_footprint_after_bytes`, `driver_memory_bytes` |
| Controls only | `materialize_ms`, `snapshot_ms`, `restore_ms` |
| Nested `work` | `prefill_attempted`, `prefill_committed`, `append_attempted`, `append_committed`, `decode_attempted`, `decode_committed`, `pending_attempted`, `pending_committed`, `recompute_attempted`, `recompute_committed`, `prefill_calls`, `append_calls`, `decode_calls`, `logit_restores` |

Work receipts count actual API calls/attempts and commits, including failed work. Prefill/append/decode are disjoint model-token categories; append includes rebuild chunks. Pending and recompute are subsets, **not extra tokens to add**. Recompute means repeating previously committed input after reset. Generated-token counts are not executed-model-token counts; old timing labels are not a substitute for work v2 accounting. Unavailable footprint/first-token/cancel-latency values are null; driver memory is null. Named buffers and sampled process footprint are neither driver allocation accounting nor measured peak memory.

### Metric dictionary and timing boundaries

All service intervals use controller `CLOCK_MONOTONIC`, not per-layer GPU timestamps. Clock failure or nanosecond-epoch overflow aborts rather than fabricating valid measurements. Integer-microsecond callback durations are truncated once; millisecond aggregates divide integer sums by 1000. JSON serialization is bounded and fails explicitly rather than emitting truncated/nonfinite metrics.

| Fields | Definition |
|---|---|
| `framing_ms` | CPU framing/tokenization at message admission; accumulated pending framing is charged once to the next work request, including restore discarding that branch |
| `total_ms` | Worker-selected to terminal state processing, including before-footprint sampling; excludes queue/admission wait and transport delivery. Generation samples this before final footprint sampling; controls sample it after final footprint sampling |
| `prefill_us`, `prefill_steps`, `prefill_ms` | Initial/rebuild materialization windows containing prefill, including their append chunks and heads; steps count windows, not API calls |
| `suffix_us`, `suffix_steps` | Append-only materialization windows; whole-window timing, not individual append latency |
| `decode_total_us`, `decode_steps`, `decode_step_duration_us`, `decode_ms` | Scalar materialization attempt durations, including failed/cancelled work whose selected ID is discarded; array sums exactly to total; emitted EOS counts as output but not necessarily executed decode |
| `first_token_ms` | Worker start to first nonnegative ID returned by selection, before mutex reacquisition/delivery, even if later discarded; null if none. Not client first-text arrival |
| `cancel_received_monotonic_ms`, `cancel_source` | First receipt time/source (`explicit`, `disconnect`, `overflow`, `shutdown`); repeats do not overwrite it; absent receipt has numeric time 0 and empty source |
| `cancel_latency_ms` | Receipt to worker safe stop after callbacks return; null until cancelled work is terminal, and for uncancelled work |
| `prompt_tokens` | Logical input count at work admission, not visible words or output IDs |
| `reused_tokens` | Retained prefix at first materialization/control |
| `newly_processed_tokens` | Successfully materialized initial request input (controls: their commits), not all later generated work; failed/partial work requires the work receipts |
| `materialize_ms`, `snapshot_ms`, `restore_ms` | Control-only timings. Materialize/restore start at control worker selection, so include initial footprint sampling and setup before their callback; snapshot measures only the save callback. These are not isolated GPU timings |
| `process_phys_footprint_before_bytes`, `process_phys_footprint_after_bytes` | Instantaneous whole-process TASK_VM_INFO samples at worker selection/safe stop; null on unsupported/query failure; neither peaks nor engine-only memory |
| `driver_memory_bytes` | Always null: no driver accounting claim |
| `test_only` | Distinguishes injected generator measurements from native execution |

Each `work` attempted/committed pair counts tokens submitted to/successfully committed by the named runtime API; attempted is not proof every GPU kernel ran. `prefill_calls`, `append_calls`, `decode_calls` count actual attempts. `pending_*` counts emitted IDs reconciled exactly once; `recompute_*` counts previously committed IDs processed again after reset, including strict-off restored recipes. `logit_restores` counts actual runtime logit copies even if cancellation prevents controller publication; `cache_restored` means published restore success. Never infer these from output length or timing steps.

`/info.process_memory_sample` additionally labels `method`, units, phase `after_resident_load`, monotonic sample time and bytes. Model/tokenizer hashing uses incremental 64 KiB reads; loading duration includes hashing/tokenizer startup. Samples, named engine buffers and admission headroom are three different quantities. Neither process memory nor elapsed time establishes DRAM traffic or energy. Client send-to-first-text, cancellation acknowledgement/completion and whole-task durations are separate client measurements.

### Envelope and allocation bounds

The service JSON profile is limited to 64 nodes including sentinel and depth 4. Decoded strings alias its owned body copy. Connections, accepted inputs and output storage are heap-owned and released on disconnect/shutdown. Full 4352-ID ledger JSON needs at most 30465 bytes; its envelope fits the 64000-byte route reply and 65536-byte transport output. Retained output remains 256 IDs and 4096 text bytes even in long context. Terminal records exclude text deltas; delta records exclude metrics, preserving the 8192-byte event bound under worst escaping. Every append checks capacity; overflow is error/cancellation, never silent completion.

Long tokenization, connection body growth, both ledgers, parser storage and checkpoint replacement are bounded host lifetimes charged by the runtime's additional 8 MiB long-profile allowance. That allowance is admission policy, not a measured peak or permission for arbitrary future host growth. The exact GPU geometry and guard formula are owned by [engine memory](engine.md#capacity-and-memory).

## Specialized current guides

* [Native tools](tools.md): fixed calculator, frozen stock investigation, and saved synthetic observation profiles. Initial tools frame is `1 5 <declaration IDs> 6 3 <user IDs> 4`; calls are `9 <name IDs> 32 <argument IDs> 2`; results append `7 <result IDs> 8`. Completed calls enter `awaiting_tool` with `stop_reason:tool_call`; submit `role:tool`, matching `tool_call_id` and validated content before generation resumes. The server does not execute tools; malformed/truncated/cancelled calls authorize no local work. No arbitrary registration, shell, network or live sensor.

* [Interactive adapter](../guides/interactive.md): persistent bounded NDJSON client, its separate command schema, reservations, evidence and shutdown rules.

* [Scenarios](../guides/workloads.md): versioned recipes, asynchronous control polling, repetition/identity/deadline bounds and scoring.

* [Controlled replay](../development/replay.md): optional diagnostic only; paired `--evaluation-mode controlled-replay --replay-logits FILE`, exclusive capture, at most 16 process-lifetime forced selections, no tools or task-quality claims. Ordinary greedy requests do not accept replay tokens.

* [Runtime](engine.md): execution/storage, checkpoint ownership and named memory. Published evidence is not certification of a new binary.

## Python client ownership

`tools/workload_client.py` owns `Service`, `AdmissionError` and duplicate-key rejection (`strict_object`). Both workload entrypoints import it directly; importing any of these modules does not parse arguments or launch a child. The shared client preserves bounded HTTP/event validation, deadlines, cancellation, retained-state reconciliation, evidence files and owned-child cleanup. Scenario policy, scoring and CLI parsing remain in their callers; tool validation ownership is unchanged.

`real_workload` re-exports these three names for archived scripts. Maintained transport tests should import from `workload_client` and patch transport dependencies there (for example `workload_client.subprocess.Popen`); tests replacing a CLI launch should patch the entrypoint’s local `Service` binding. This extraction does not introduce a public SDK or change CLI defaults.
