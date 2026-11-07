# Bounded interactive workload adapter

`tools/interactive_workload.py` is a persistent diagnostic NDJSON client of the
existing service, not a new inference engine or a quality/numerical acceptance
check. It starts and owns exactly one child, retains one session across turns,
and never retries a command. No shell, arbitrary tool execution, downloads or
Python model are involved.

```sh
python3 tools/interactive_workload.py --server build/lunelette-server \
  --native-model models/model.safetensors --native-tokenizer models/tekken.json \
  --output evidence/interactive-001 --seconds 300 --total-budget 1024
```

Use your actual service executable and checkpoint/tokenizer paths. Native defaults
remain resident-fp32, mps prefill, context256/chunk256, legacy prefix reuse. Optional
`--native-strategy`, `--native-prefill`, `--native-prefix-reuse` select existing
profiles. Checkpoint/restore require an on/off prefix profile and advertised
checkpoint support. `--native-context 4352` enables long history (32768 message bytes);
`--native-chunk 128|256` selects the existing chunk profile (128 requires 4352).
The adapter checks /info context, chunk, prefix, prefill and tools profile before ready.
Replay, scoring and tuning interfaces are not exposed.

For model-free protocol testing only:

```sh
python3 tools/interactive_workload.py --server build/test_service_server \
  --test-only --output evidence/protocol-001
python3 tests/test_interactive_workload.py ./build/test_service_server
python3 -O tests/test_interactive_workload.py ./build/test_service_server
```

## Wire protocol

Each stdin line is a UTF-8 JSON object with exactly `id`, `op`, and the fields
listed below. Duplicate keys, unknown fields and invalid schemas terminate the
adapter. IDs are unique for the entire process (including rejected commands),
1–48 ASCII letters/digits/hyphens/underscores. No request IDs are reused after reset.
Command count includes rejected commands. Duplicate IDs receive an error, not a
retry or another operation.

| op | Additional fields | Behavior |
|---|---|---|
| create | none | Create the sole session; a second create is rejected |
| message | content | Append user text, at most 4096 UTF-8 bytes (32768 with context4352) |
| generate / continue | budget | Same service generation endpoint; budget integer 1–256 |
| inspect | none | Current authoritative session metadata, including while generating |
| reset | none | Reset session and forget local checkpoint receipt |
| checkpoint / restore | none | Existing validated workload_scenarios controls; one saved receipt |
| cancel | none | Cancel active generation; never implicitly starts generation |
| quit | none | Acknowledge then shut down owned service |
| tools | profile, budget, max_calls, max_generated, wall_seconds, answer | Fixed bounded native tool loop below |

Example input (send successive turns after reading their final events):

```json
{"id":"s1","op":"create"}
{"id":"m1","op":"message","content":"Hello"}
{"id":"g1","op":"generate","budget":32}
{"id":"i1","op":"inspect"}
{"id":"q1","op":"quit"}
```

Stdout contains only JSON lines: an initial `ready` with service info, live `text`
events carrying the originating command `id`, and a compact `final` with `ok` and
`stats` or `result`. Tool text also includes its underlying `request_id`. Generation
stats include terminal event, token count, stop reason, state, latency and native
metrics when supplied. Text events are delivered after Service validates event
identity, ordering and text, but before final retained-result/ledger checks; they
are provisional evidence, not proof that the whole request passed. Callback
exceptions abort the stream and adapter. A disconnect/EOF can leave no final event.
Non-schema admission errors such as busy or budget exhaustion emit `ok:false`;
transport/protocol/control failures terminate with a stderr diagnostic.

Generation runs in one worker; inspect/cancel remain available on stdin. All other
mutations are rejected until generation completion is published. There is no internal command
queue; callers must still pace writes (the OS stdin pipe is naturally buffered).
Cancellation is an explicit HTTP request, not a local synthesized terminal. A
cancel before native admission can return a recoverable admission error. Cancel intent
is nevertheless latched when the command is accepted. Every subsequent tool request
and generation checks it under a short state lock; no lock spans GPU generation.
If cancellation races with publication/admission of a new request, its validated
`started` callback rechecks intent and sends one explicit cancel to that newly
admitted ID (unless already acknowledged). This is not a generation retry. Between
tool requests, cancel acknowledges `cancel_pending` without inventing a service
terminal; with no live worker it rejects `no active generation`. A rejected early
cancel can therefore still prevent subsequent tool work. Already-started HTTP/tool
work may finish. No next-turn messages are injected automatically.

Quit acknowledges the quit command, not generation success. Quit/EOF latch active
cancel and allow up to three seconds for the worker's real terminal/error before
terminating the child; errors are retained and yield nonzero exit, not a fabricated
clean terminal. Signals and fatal transport failures interrupt the owned child
immediately. Unsupported checkpoint capability or restore without a saved receipt
is a recoverable ID-bearing `final` with `ok:false`; corrupt control responses remain
fatal. Adapter fatal errors are recorded in `client.ndjson` when the log permits,
independently of child cleanup.

## Restricted native tools

Launch with `--tools calculator`, `investigation`, or `observation` (native only),
then use `op:tools` with that same `profile`. These call the existing
`native_workload_tools.run_loop` and its capability, ledger, argument and corpus
checks. Only bounded integer calculator operations and lookups in hash-checked
bundled fixture corpora are possible. No paths, commands, network requests, dynamic
code, arbitrary document queries, or user-defined tools are accepted.

The existing handler limits apply: budget ≤128, calls ≤2 (observation ≤1), total
loop generation ≤384 (observation ≤256), bounded wall time, and fixed scoring
answers: calculator `43`, investigation `"19 [LARCH-STOCK:P1]"`, observation
`"pump-A 84 ALERT"`. Supply all six tool fields; e.g. calculator budget 64,
max_calls 2, max_generated 128, wall_seconds 30, answer 43. These are diagnostic
fixtures, not a general assistant tool interface. The input message must have the
handler's required native framing. A cancellation flag prevents subsequent tool
messages/generations even if cancellation arrives between loop requests. Internal
IDs `<command-id>-<call-index>` are reserved persistently; avoid collisions with
user command IDs. Tool terminal stats retain the bounded handler report.

## Bounds, evidence and shutdown

* Lifetime includes startup and idle: default 300 seconds, configurable 1–1800.
  Startup also retains Service's native 180-second / test 3-second readiness bound.
* At most 256 input commands; 262144 bytes per line, including newline. Oversize
  lines, incomplete EOF lines, invalid UTF-8/JSON and schema failures are fatal.
* Total generation budget defaults to 1024, configurable 1–4096. Budgets (tool
  `max_generated` included) are reserved before starting and never refunded, even
  on cancellation/failure. There is no hidden continuation or extra generation.
* Stdout must be a pipe, socket or terminal; regular-file redirection is refused
  because nonblocking mode cannot bound regular-file writes (pipe through a consumer
  if a transcript is needed). Stdout is capped at 4 MiB and each complete event has at most 250 ms to write,
  further limited by lifetime. Broken/slow stdout is fatal and does not gate owned
  child cleanup. Partial final lines are possible on output failure.
* The output directory must not exist. Service creates it exclusively; no append,
  overwrite, fallback path or recovery from an old evidence root. `config.json`
  retains launch settings (≤64 KiB); `client.ndjson` retains concurrent request,
  arrival, command and failure evidence (≤8 MiB); `server.stderr` retains merged
  child stdout/stderr (≤8 MiB). Evidence is never automatically deleted. Limit
  exhaustion is fatal; logs can be incomplete and shutdown records can be absent
  when the log cap is reached. Protect evidence: messages/config paths are not
  redacted. There are no token-vector captures or additional log files.
* EOF, quit, SIGINT, SIGTERM, deadline and transport failure terminate the owned
  child, then kill it if necessary. Cleanup has bounded grace beyond the lifetime
  (child wait up to 3 seconds then kill wait up to 3 seconds, reader joins). No
  unrelated processes/resources are touched. SIGKILL cannot provide cleanup.

Model-free tests cover schema and byte/count/budget bounds, persistent IDs,
competing mutations, live inspect/cancel, streamed text, EOF during generation,
idle deadline, signal handling, broken/slow output refusal and exclusive evidence
roots under normal and optimized Python. Native tool execution, actual model
latency/quality and GPU cleanup require separate explicitly authorized validation.
