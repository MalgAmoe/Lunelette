# Current native service contract

This is the current service contract, not a milestone result or numerical acceptance. It describes [native.c](../service/native.c), [session.c](../service/session.c), [session.h](../service/session.h), [http.c](../service/http.c) and [framing.c](../service/framing.c). See [Metal runtime](metal.md) for runtime, model-file and memory policy; service framing and history are distinct from single-prompt CLI behavior.

## Startup and profiles

A direct native-server invocation **requires all five** options:

```sh
build/lunelette-server --model models/model.safetensors \
  --tokenizer models/tekken.json --strategy resident-fp32 --prefill mps --port 0
```

Paths must identify caller-supplied official mixed FP8/BF16 single-file weights and Tekken tokenizer. Port is decimal 0..65535; 0 selects an ephemeral loopback port, reported by the stderr `listening` record. The server binds only 127.0.0.1. Unknown/duplicate options or missing values fail startup; there is no fallback.

| Selector | Accepted values / omission behavior |
|---|---|
| Required `--strategy` | `resident-fp32`, `compact-fp8-arithmetic`, `compact-fp8-lookup` |
| Required `--prefill` | `mps`, `custom`, independent of storage |
| Optional `--context` | `256` (default), `4352` |
| Optional `--prefill-chunk` | `256` (default), `128` only with context 4352 |
| Optional `--prefix-reuse` | `on`, `off`; **omit for legacy** (literal `legacy` is not a server value) |
| Optional `--tools` | `calculator-v13`, `investigation-v13`, `observation-v13`; omitted means ordinary chat |

Thus supported C/B pairs are 256/256, 4352/128 and 4352/256. B bounds one materialization call, not total history. Runtime construction defaults and workload client defaults are not missing-option defaults for this executable: `tools/real_workload.py` and the interactive adapter supply resident-fp32/MPS, context 256/chunk 256 and omit the reuse flag for legacy. Their `--native-*` selectors differ from direct server flags. No default promotion is implied by optional compact, lookup, larger-context or reuse evidence.

`/info` reports actual strategy, prefill, capacities, protocol, capabilities, model/tokenizer SHA256, build identity/dirty status, load count/duration, precision policy, named allocation bytes and a startup process-footprint sample. Validation labels and historical document paths in that metadata are published-evidence pointers, **not runtime attestation**; startup performs no numerical comparison. Independent numerical validation remains **FAIL/open**. Native parity, successful HTTP exchanges and generated text do not make an independently validated baseline.

## Authoritative history and framing

There is one live session and one worker owning runtime execution. Native ordinary chat is `native-no-system-nonthinking-text-v1`: first user input is `1 3 <ordinary text IDs> 4`, later input `3 <ordinary text IDs> 4`. EOS is ID 2, included in generated IDs but omitted from text. There is no system or assistant-message injection API. Special-token spellings in content are ordinary text, not caller-supplied structural IDs.

Session `token_ids` are the authoritative complete logical ledger; retained request `token_ids` are that request's generated suffix. Never decode/re-encode assistant text to reconstruct history. Logical tokens include accepted frames and published outputs; committed tokens are actually materialized KV. Newly emitted tokens, including EOS, remain pending until consumed exactly once by later materialization. Accepted messages may add further unmaterialized logical input. Check counts and state rather than assuming every logical token already has KV.

* **legacy:** reuses an unchanged materialized prefix and the immediate single pending non-EOS decode path; other extensions rebuild (not general new-turn reuse).
* **on:** retains matching committed prefixes and materializes only missing input, including pending EOS plus the next message frame.
* **off:** resets/recomputes at each generation boundary, while preserving normal within-response decode. Restore recovers logical state, not a committed cache.

Budget completion enters `partial`; another generate continues without a new user message. Natural EOS enters `ready`, requiring user input before another generation. User messages require ready state and either an empty ledger or preceding EOS. No silent history truncation or automatic budget extension occurs.

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

### Checkpoint/restore and recovery

Controls require reuse on/off and no tools; legacy and **all tool profiles return 501**. Save requires a nonempty ready/partial token boundary without incomplete UTF-8. It first materializes all logical input, including pending output, then saves tokens, phase and logits while retaining a KV prefix boundary. It does not copy the KV buffers. One saved checkpoint is retained; a successful replacement invalidates the previous opaque decimal ID. Reset/close invalidate it.

HTTP 202 is admission only: poll `/requests/ID` to completion and inspect session state/ledger. ON restore recovers saved IDs/phase, committed KV and logits (`cache_restored:true`, one logit restore). OFF restore recovers IDs/phase with committed=0 and pending=0; the next generation rebuilds. Restore itself performs no prefill/append/decode in either policy.

Failures are not blanket rollback: snapshot rejection after successful materialization can preserve the prior checkpoint while leaving the prefix newly committed. Pre-mutation rejection can preserve a usable phase. Runtime-invalid or mutated cancelled controls enter `needs_reset` and invalidate the checkpoint. Read terminal `session_valid`/`recovery` and state; do not retry blindly.

Generation cancellation, disconnect, event overflow or runtime/protocol errors enter `needs_reset`, invalidating checkpoints. Cancellation is observed between completed blocks, not GPU preemption; work may have executed with zero published tokens. Reset after the job ends is required before reuse. Reset/close do not clear process-lifetime request-ID history or replay reservations.

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

## Specialized current guides

* [Native tools](native-tool-loop.md): fixed calculator, frozen stock investigation, and saved synthetic observation profiles. Initial tools frame is `1 5 <declaration IDs> 6 3 <user IDs> 4`; calls are `9 <name IDs> 32 <argument IDs> 2`; results append `7 <result IDs> 8`. Completed calls enter `awaiting_tool` with `stop_reason:tool_call`; submit `role:tool`, matching `tool_call_id` and validated content before generation resumes. The server does not execute tools; malformed/truncated/cancelled calls authorize no local work. No arbitrary registration, shell, network or live sensor.
* [Interactive adapter](interactive-workload.md): persistent bounded NDJSON client, its separate command schema, reservations, evidence and shutdown rules.
* [Scenarios](workload-scenarios.md): versioned recipes, asynchronous control polling, repetition/identity/deadline bounds and scoring.
* [Controlled replay](controlled-replay.md): optional diagnostic only; paired `--evaluation-mode controlled-replay --replay-logits FILE`, exclusive capture, at most 16 process-lifetime forced selections, no tools or task-quality claims. Ordinary greedy requests do not accept replay tokens.
* [Runtime](metal.md) and [lookup strategy](fp8-lookup.md): execution/storage and strategy-specific evidence. Milestone pages preserve historical observations, not the current interface contract.
