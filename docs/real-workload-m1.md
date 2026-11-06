# M1: diagnostic protocol foundation (prototype v1)

**Not a real-model service.** This implements only the model-free foundation of
`real-workload-spec.md`. The production resident Metal adapter is separate; see `real-workload-m2.md` for its operational scope and numerical limitations.
No model is loaded, no inference is run, no official tokenizer/template is claimed,
and fixed replay is not task-success or numerical/performance evidence.

## Build and run

POSIX sockets, poll, pthreads and C11; no Foundation/Metal or Python runtime link:

```sh
make build/test_service_server
./build/test_service_server --port 8081
make service-protocol-test  # native, also included in core-test/test
make service-test           # optional Python 3 standard-library live tests
python3 tools/real_workload.py --server build/test_service_server \
  --scenario tools/scenarios/partial-v1.json --output build/partial-run-1
python3 tools/real_workload.py --server build/test_service_server \
  --scenario tools/scenarios/cancel-reset-v1.json --output build/cancel-run-1
```

Output directories must be new. No automatic retries. The client launches only its
own child, validates `/health` and `/info`, terminates/joins it and kills only that
child after a three-second shutdown timeout (recorded as failure). Startup and
network deadlines are bounded. Raw arrival bytes/times, operations, responses,
failures and child stderr are retained. First actual text time is the raw chunk
arrival timestamp at which a complete nonempty text_delta can be decoded; it is
not internal first-token time. Intermittent pauses remain separate log operations.
`protocol_success` is distinct from `task_success: null`. Free-greedy mode is
rejected, not evaluated by a fake generator. Generate/continue require `completed`
by default; cancel requires `cancelled`. A generation operation may explicitly set
`expect.event` to `completed`, `cancelled`, or `error` for an expected-terminal
failure scenario. Other event names are rejected at scenario validation. Unexpected
terminals or unsuccessful child shutdown make the runner exit nonzero and retain
`protocol_success: false` plus raw diagnostics; no retry occurs. The stream must
begin with one `started` event, use the admitted request ID throughout, and end
with exactly one terminal event. Unknown, reordered and post-terminal events fail.
Events are bounded to 8192 bytes and budget+2 records. A 15-second diagnostic
request budget is checked at payload reads, alongside the four-second socket
read timeout; these are test-generator limits, not real-model latency limits.
Startup failures are retained in the client log, including executable-launch errors.

## Transport and routes

Binds **127.0.0.1 only**, explicit `--port`; port 0 is a test-only ephemeral
selection announced as one stderr JSON line. No wildcard/bind-address option.
HTTP/1.1 only, Host and exactly one Content-Length required, including `0` on GET.
No transfer coding, pipelining or keepalive. Headers <=4096 bytes, body <=8192,
16 simultaneous sockets, two-second incomplete-request/idle-output deadline.
Responses close the connection; admitted generations use
`Content-Type: application/x-ndjson`, terminated by connection close.

| Method/route | Body | Result |
|---|---|---|
| GET /health | empty | test operational true, model_loaded/runtime_ready false |
| GET /info | empty | diagnostic generator identity, build commit/dirty, unsupported capabilities |
| POST /sessions | `{}` | one active session, monotonic integer ID |
| GET /sessions/N | empty | state, logical/committed/pending counts, remaining |
| POST /sessions/N/messages | `{"role":"user","content":"hi"}` | append synthetic input bytes |
| POST /sessions/N/generate | `{"id":"unique-id","budget":3}` | NDJSON events |
| GET /requests/unique-id | empty | status, text, live active counters/state or retained terminal snapshot |
| POST /requests/unique-id/cancel | `{}` | immediate acknowledgement, safe-boundary cancellation |
| POST /sessions/N/reset | `{}` | reset fixture and ledger, retain request identities |
| DELETE /sessions/N | `{}` | close/reset fixture, retain identities |

Every object is strict: duplicate/unknown fields, nested values, invalid types,
invalid UTF-8, invalid surrogate escapes and decoded NUL are rejected. Generation
budget is an integer 1..256, no larger than remaining logical capacity. Request IDs
are 1..64 ASCII letters/digits/underscore/hyphen. Exactly 256 admitted IDs can be
retained for the process lifetime; no eviction even on reset/close. Exhaustion
returns 503/restart-required. Duplicate IDs return 409, and GET /requests/ID
returns the original request; they never execute twice, even with identical body.
Rejected admissions do **not** reserve IDs. One generation, no hidden queue.
Competing generation/mutations return 409. Previous sessions return 410, unknown
future sessions 404. See the plan's state/operation table.

`tool_result` is explicitly 501, unknown roles 400. Checkpoint/recompute/tool and
other planned operations are unsupported (501 for unimplemented routes); no
placeholder successful operation. Partial input cannot be abandoned by appending
messages; reset or close first. `needs_reset` cannot generate or append.

## Synthetic ledger, not conversation framing

`conversation_protocol=test-ledger-v1`, `tokenizer=raw-synthetic-bytes-v1`.
Input UTF-8 bytes become byte-valued synthetic token IDs with no added delimiters.
Output fixture raw IDs are `[65,226,130,172,240,159,152,128,90,256]`:
`A€😀Z`, then structural EOS=256. EOS remains in the authoritative ledger.
The fixture repeats for another generation after EOS; reset restarts the row.
Special-token spellings are simply bytes. There is no official model framing.

Capacity 256 counts every logical byte token and EOS, including the pending output
token. Before producing each output token, the worker materializes the entire
current prefix, checked against a separate fixture execution ledger. Each emitted
token is logically present but uncommitted until the next tick/continuation.
`pending_tokens` describes only emitted output (0/1), not unprocessed input;
`logical_tokens - committed_tokens` can also include newly appended input bytes.
Budget stop means `partial`, not a completed assistant turn. EOS means `ready`.
The UTF-8 decoder retains incomplete bytes across partial continuations, never
flushes replacement characters at a budget boundary, and emits only complete
valid scalars. Invalid sequences/NUL or incomplete EOS produce safe JSON error
and `needs_reset`. C tests use an injected invalid-output fixture as well.

## Ownership, bounds and cancellation

Two implementation modules: `service/http.c` owns sockets and a single poll loop;
`service/session.c` owns strict schema validation, state, and one serialized
worker. The narrow injected generator has next/materialize/reset/destroy callbacks;
all run on the worker, never the HTTP thread. No generic backend framework.
No generator callback runs while holding the shared mutex. Reset/close reject
active execution; idle reset synchronizes with the worker. Shutdown flags cancel,
joins the worker, then worker reset/destroy callbacks release state. The signal
handler only sets sig_atomic_t; poll is bounded to 50ms. SIGPIPE is ignored and
send errors handled as disconnects.

A 16-event ring never blocks the worker. Overflow cancels, discards queued deltas
if necessary to reserve terminal delivery, and marks needs_reset. This also applies
when the last delta exactly fills the ring: dropping text must never result in a
`completed` event. Do not treat
text before a cancelled/error event as complete. Per-event storage is 8192 bytes;
retained results are capped at 4096 text bytes per admitted ID (the fixture's
256-token context is tighter). Transport has one bounded output buffer per socket.
Disconnect cancels active work; idle-output deadlines bound stalled sockets.
The test-only startup knob `--tick-ms 0..1000` delays a safe-boundary tick, never
adds HTTP control fields or allows an indefinite engine block. At extreme delays,
the idle streaming deadline can deliberately cancel a no-text interval.

Terminal `completed`, `cancelled`, `error` events carry stop reason, generated and
ledger counts, session state/validity and metric fields. Timing/model/memory
measurements not implemented in the fake are null; loading duration 0 means no
model load. Client wall-clock observations describe test transport only. Sampled
memory, GPU memory, logits, numerical correctness, model TTFT and scientific task
success are not claimed. Cancellation latency is observable in client raw send,
ack and terminal arrival records; the native metric remains null.

## Verification boundaries

Native always-active CHECK tests cover strict JSON/UTF-8, each Euro/emoji byte
boundary, pending continuation, exact execution ledger, invalid-output recovery,
ring overflow, lifecycle and all 256 identity slots. Optional live tests cover
HTTP framing/bounds/deadlines, competing generation, duplicate reuse, invalid
state, capacity, streaming equality, responsive cancel during a one-second tick,
disconnect, active shutdown and both versioned scenarios. Existing model-free
core/Metal synthetic tests do not establish real-model correctness.

Still absent: resident adapter, authoritative pinned framing, numerical gates,
checkpoints/recompute, tools, ingestion chunking, compact strategies, long context,
real workloads and task-success evaluation. No full-spec completion claim.
