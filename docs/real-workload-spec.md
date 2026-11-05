# Builder spec: Lunelette real-workload test system

User-provided specification. Markdown formatting normalized; requirements retained.
This is the target specification, not a statement of implemented capabilities.

## 1. Objective

Build a persistent local inference service and workload client so we can evaluate complete tasks—not just individual kernels.

The system must let us test:

- Requests separated by realistic pauses.
- Multiple turns without reloading weights.
- Shared-prefix reuse versus recomputation.
- Tool-result continuations.
- Document ingestion with different prefill chunk sizes.
- Different execution strategies under identical workloads.
- Numerical correctness, task success, latency and memory together.

This is a testable local service, not a general-purpose inference platform.

## 2. Scope and architecture

```text
Workload client
  ├── prepares requests and schedules pauses
  ├── executes or replays tools
  ├── records client-visible timings
  └── evaluates task success
             │
       Local HTTP API
             │
Persistent Lunelette process
  ├── loads one model once
  ├── owns session/token/KV state
  ├── generates and streams responses
  └── reports execution measurements
```

### Initial constraints

- Keep the existing C host/core and Metal/MPS runtime.
- One model, one active session and one generation at a time initially.
- Bind to loopback only.
- Reject competing generation requests explicitly; no hidden queue initially.
- Control requests, especially cancellation, must remain responsive during generation.
- All engine calls remain serialized. HTTP handling must not call reset/destroy concurrently with GPU execution.
- Python may be used for workload tooling, but must not become a requirement of the native engine/service.
- Do not introduce a generic backend framework or language rewrite.
- Ask before adding dependencies or downloading models.

## 3. Persistent service

### Required lifecycle

1. Start with explicit model, tokenizer, execution strategy and context configuration.
2. Validate inputs and memory admission.
3. Load weights once.
4. Announce readiness.
5. Accept repeated requests without reloading.
6. Shut down cleanly, releasing resources.

Readiness means operationally ready, not “numerically validated.” Report numerical-validation status separately.

### Required service information

Expose:

- Build/commit identifier.
- Model and tokenizer identity.
- Execution strategy and precision/storage policy.
- Configured context capacity.
- Prefill chunk size.
- Supported conversation/tool protocol.
- Numerical-validation status and scope.
- Loading duration.
- Memory-accounting fields, with their measurement definitions.

No silent fallback to a different model, strategy or precision policy.

## 4. Minimal API

Exact route names may change, but these operations and semantics are required.

| Operation | Suggested endpoint | Purpose |
|---|---|---|
| Health/readiness | GET /health | Process and model readiness |
| Configuration/capabilities | GET /info | Reproducible execution identity |
| Create session | POST /sessions | Start an empty logical conversation |
| Inspect session | GET /sessions/{id} | State, token counts and remaining capacity |
| Append input | POST /sessions/{id}/messages | Add a supported user or tool-result message |
| Generate/resume | POST /sessions/{id}/generate | Produce the next response or continue a partial one |
| Inspect request | GET /requests/{id} | Running/completed/failed status and result metadata |
| Cancel request | POST /requests/{id}/cancel | Request cancellation |
| Reset session | POST /sessions/{id}/reset | Clear conversation/KV state without unloading weights |
| Close session | DELETE /sessions/{id} | Release session state, retain model |
| Save/restore prefix | Session checkpoint operations | Controlled reuse experiments |

### Request requirements

- Every generation has a client-supplied unique request ID.
- Duplicate IDs must not accidentally execute twice.
- Bound request size, message size, generation budget and retained result history.
- Reject malformed JSON, unsupported roles/options and capacity overflow clearly.
- Return structured errors with:
  - Error code.
  - Request/session ID where applicable.
  - Whether session state remains valid.
  - Required recovery action.
- Do not imply OpenAI API compatibility unless it is deliberately implemented and tested.

## 5. Conversation and session correctness

This is a core requirement, not merely HTTP plumbing.

### Required session states

At minimum distinguish:

- Ready for input/generation.
- Generating.
- Partial assistant response awaiting continuation.
- Needs reset after failure.
- Closed.

Document which operations are allowed in each state.

### Conversation framing

- Implement the supported multi-turn format from a pinned authoritative tokenizer/template reference.
- Do not invent assistant or tool-role delimiters.
- Treat literal special-token spellings in ordinary input as text, not injected control tokens.
- Preserve generated token IDs as authoritative. Do not assume decoding and re-encoding assistant text reproduces the same IDs.
- Keep displayed text, structural tokens and the execution token ledger consistent.

### Pending-token handling

The last emitted token may not yet be represented in KV.

The implementation must explicitly distinguish:

- Tokens logically present in the conversation.
- Tokens already committed to KV.
- Any emitted-but-uncommitted token.

On continuation, commit or otherwise reconcile that token exactly once. Never omit or duplicate it.

### Partial responses

Reaching a token budget is not necessarily completing an assistant turn.

- Report the stop reason.
- Allow explicit continuation.
- Do not silently manufacture a completed turn or tool call.
- Define how a caller abandons a partial response; resetting is an acceptable initial behavior.

## 6. Prefix reuse and checkpoints

We need to compare reuse against recomputation with identical logical inputs.

### Required behavior

- Save a checkpoint at an explicit, materialized token boundary.
- Restore that prefix and append a different suffix.
- Report the number of tokens actually reused and newly processed.
- Invalidate checkpoints on reset or incompatible state changes.
- Checkpoints are session-local initially; disk persistence is unnecessary.

A checkpoint must preserve enough state to generate correctly, including appropriate last-token logits—or a documented reconstruction step. KV length alone is insufficient.

A logical KV truncation is acceptable where earlier causal KV entries remain valid. Copying the entire KV cache is not a requirement.

### Reuse-off control

The workload client must be able to request recomputation of the same authoritative token sequence while keeping model weights resident.

Do not compare reuse against a control that unnecessarily reloads the model unless model residency is the specific variable being tested.

## 7. Streaming, cancellation and failure behavior

### Streaming

Provide structured events, for example:

- started
- text_delta
- completed
- cancelled
- error

Requirements:

- Stream only valid UTF-8 text.
- Buffer incomplete byte sequences until valid.
- Final assembled text must match buffered decoding.
- Include final token counts, stop reason, session validity and metrics.
- Distinguish internal first-token availability from first actual text received by the client.
- Bound buffering for slow clients.

Debug logging must not corrupt the response stream.

### Cancellation and disconnects

- Cancellation must be acknowledged while generation is running.
- It may take effect at the next safe boundary; immediate interruption of a submitted GPU command is not required.
- Report cancellation latency and terminal state.
- Initially, marking an interrupted session needs_reset is acceptable and safer than claiming resumability.
- A disconnect must not leave an uncontrolled generation running indefinitely.
- Failed or cancelled requests must never be silently retried.

## 8. Tool-loop support

Tools execute in the client, not inside the inference engine.

Required loop:

```text
Submit task
→ generate assistant response
→ identify a complete tool request
→ execute or replay the tool
→ append its result
→ continue the same session
→ stop at a final answer or an explicit limit
```

### Requirements

- Use a documented, versioned tool-request schema.
- Distinguish a tool request from a final answer.
- Validate complete tool-call output before executing anything.
- Malformed or truncated JSON must not trigger a tool.
- Start with deterministic local tools:
  - Calculator.
  - Lookup in a frozen document collection.
  - Recorded sensor/log observations.
- Bound tool iterations and total generated tokens.

If the checkpoint’s native tool framing is not supported, use an explicitly labelled application-level protocol. Do not advertise native function calling.

The client must distinguish a model protocol failure from a service/runtime failure.

## 9. Context capacity and ingestion

### First milestone

Support repeated sessions correctly within the maintained 256-token total limit.

Expose capacity usage and reject overflow. Do not silently truncate history.

### Long-context milestone

A concrete first target is:

- Up to 4,096 framed input tokens.
- Up to 256 generated tokens.
- A validated 4,352-token total capacity, if memory admission permits.

Counts include framing and structural tokens, not just user-visible text.

Requirements:

- Audit generation limits, token arrays, KV allocation and attention/scratch allocation together.
- Implement validated chunked prefill.
- Make chunk size explicit and recorded.
- Test chunk boundaries and cached continuation.
- Preserve memory reserves and ceilings.
- If a configuration cannot fit, report it as unavailable—not as a benchmark failure or a reason to weaken guards.

Whole-document prefill is an optional comparison where it fits safely; it must not be mandatory for running document workloads.

## 10. Execution strategies to compare

Select strategy at startup initially. Do not switch silently during a session.

| Strategy | Required role |
|---|---|
| Current resident-FP32 execution | Control |
| Compact FP8 projections; conversion+MPS for multi-token work; arithmetic-decoded scalar execution for single tokens | First compact candidate |
| Same strategy with lookup-decoded scalar execution | Second compact candidate |
| Shape-specific small-token tiles | Follow-up candidate |
| Paired gate/up fusion for selected small shapes | Separate follow-up experiment |

Requirements:

- Preserve declared FP32 arithmetic, norms and residuals initially.
- Specify embedding/head storage and execution explicitly.
- Include conversion in measured request execution.
- Validate reusable scratch across different matrices and successive layers, not just repeated multiplication by one matrix.
- Report the actual selected kernel policy for each relevant shape.
- Test fusion against a compact control, so storage changes do not masquerade as fusion benefits.

Do not include the tested half-reduction policies or the current naïve MPP kernel in the initial workload comparison. They remain separate research candidates.

## 11. Measurement contract

### Server-side measurements

Record, per request where applicable:

- Tokenization/framing time.
- Prefill/append duration.
- Time to first generated token.
- Decode duration and per-token samples.
- Total request duration.
- Prompt, reused, newly processed and generated token counts.
- Stop reason.
- Session state before and after.
- Numerical/nonfinite failures.
- Memory observations.

### Client-side measurements

Record:

- Send-to-first actual text.
- Send-to-completion.
- Complete session/task duration.
- Inter-request pauses.
- Tool duration separately.
- Cancellation response and completion times.

### Measurement rules

- Separate model loading from already-resident requests.
- Include first ingestion in complete-session measurements.
- Preserve raw samples, failures and timeouts.
- Record warmup/conditioning explicitly.
- Do not mix sustained and intermittent results.
- Do not label sampled memory as an exact peak.
- Distinguish named engine buffers, process memory and driver observations; do not double-count unified memory.
- Avoid intrusive per-kernel instrumentation in headline timings. Provide a separate diagnostic mode where necessary.

## 12. Workload client and test cases

Provide a reusable client with versioned scenario files—not one bespoke script per benchmark.

Each scenario defines:

- Initial instructions and input.
- Session operations.
- Request schedule and pauses.
- Generation limits.
- Tool definitions/results.
- Reuse mode.
- Expected outcomes.
- Repetitions and timeouts.

### Initial scenarios

1. Intermittent log/sensor summaries: short requests in bursts and after pauses.
2. Repeated structured tasks: same foundation, changing small inputs; reuse on/off.
3. Document retrieval/questions: increasing document size and varied prefill chunks.
4. Tool-assisted investigation: several generations separated by deterministic tool results.
5. Growing conversation: increasing history and different response lengths.

### Two distinct evaluation modes

- Controlled replay: fixed token trajectories for matched computational work and numerical comparison.
- Free-running greedy tasks: genuine generated responses and task-success evaluation.

Fixed trajectories must never be presented as evidence that the model successfully completed a task.

## 13. Acceptance and correctness gates

### A. Model-free service tests

Must cover:

- Request parsing and bounds.
- Session-state transitions.
- Busy handling.
- Duplicate requests.
- Streaming framing and split UTF-8 sequences.
- Cancellation and disconnects.
- Slow-client limits.
- Invalid operations and recovery.
- Context accounting, including pending tokens.

Use an injectable deterministic generator to test these without loading a model.

### B. Current-checkpoint numerical baseline

The maintained documentation reports unresolved strict reference mismatches.

Before declaring the system a validated comparison baseline:

- Resolve or explicitly disposition those mismatches.
- Use independent fixtures from the exact current checkpoint.
- Preserve the existing finite comparison criterion:
  `abs(error) <= 2e-4 + 1e-4 * abs(reference)`.
- Keep raw mismatches.
- Do not silently widen tolerances.

A service can be operational while this work is pending, but its results must remain labelled diagnostic.

### C. Session and strategy gates

Require:

- Fresh versus resident execution comparisons.
- Reset/recompute versus retained-prefix comparisons.
- Buffered versus streamed token/text equality.
- Continuation without token omission or duplication.
- Chunked ingestion comparisons.
- Candidate whole-model logits and intermediate checks.
- Near-capacity and overflow tests.
- Error recovery without stale or partially valid KV publication.

### D. Workload acceptance

Each workload must have a task-specific criterion, such as:

- Correct calculation.
- Correct lookup with supporting passage.
- Required facts present in a summary.
- Correct structured fields.
- Valid tool protocol and bounded completion.

Performance and task success must be reported together.

## 14. Delivery order

1. Persistent service and client, with model-free protocol/state tests.
2. Resident short-session execution, streaming and cancellation.
3. Numerical baseline and session correctness gates.
4. Compact strategies in complete-model experimental builds.
5. Prefix checkpoint/reuse comparisons.
6. Validated longer-context and chunked-ingestion support.
7. Reproducible real-workload campaign and report.

Reuse the existing resident-streaming, prefix-reuse and 4K experimental work where appropriate, but do not treat historical acceptance as automatic validation of the new strategies.

### Definition of done

A client can launch the service once, run a multi-request or tool-assisted task through it, collect valid streamed results and complete measurements, reset/reuse state deliberately, and repeat the same workload against another explicitly identified strategy—with correctness and task-success checks.

Not required initially: concurrent inference, batching, remote deployment, authentication infrastructure, a UI, durable on-disk sessions, a vector database, or an embedded agent framework.
