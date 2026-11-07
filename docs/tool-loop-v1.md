# Client tool loop v1 (diagnostic only)

`tools/real_workload.py` accepts the reusable `tool_loop` operation in a version-1
scenario. Tools execute only in the Python client. **native_tools: false**: this
is an application JSON protocol, not native function calling. No engine changes,
new role, or assistant decode/re-encode are involved. The engine retains its
assistant tokens; results append through the existing user-message framing in
the same session.

## Protocol

The entire assistant response must be one JSON object, optionally surrounded by
JSON whitespace. No prose or markdown fences. Exactly these fields are accepted:

```json
{"version":1,"type":"tool","id":"sum1","name":"calculator","arguments":{"operation":"add","a":19,"b":24}}
```

```json
{"version":1,"type":"final","answer":43}
```

Calculator operations are `add`, `subtract`, and `multiply`. Inputs must be exact
JSON integers in [-1000000, 1000000]; results and final answers are bounded to
+/-1000000000000. Booleans, floats, nonfinite constants, duplicate or unknown
fields, trailing content, oversized integers, strings as numbers, and unknown
operations/tools are rejected. Responses are limited to 2048 UTF-8 bytes; call
IDs match `[A-Za-z0-9_-]{1,32}` and cannot repeat within a loop. There is no eval,
arbitrary expression language, path access, shell, network, lookup, or sensor tool.

Execution requires a `completed` terminal event, `stop_reason: eos`, and
`state: ready`, matching request/session identities, `session_valid: true`, and
`recovery: none` in an active integer session. Native responses require retained
request and session metadata: completed generate identity, matching counts/state,
valid token IDs with exactly one final generated EOS, and coherent ready ledger
counters, remaining capacity, and generated suffix. Contradictions are rejected
before executing a tool or accepting a final. A budget stop
never executes a tool, even if its JSON happens to be complete and valid.
Cancellation, runtime errors, and invalid accounting also execute nothing.

Results are deterministic, compact, sorted JSON with this label, appended as
`role: user` using `/sessions/{id}/messages`:

```text
APPLICATION_TOOL_RESULT_V1 {"id":"sum1","name":"calculator","result":43,"version":1}
```

## Scenario operation and limits

See `tools/scenarios/calculator-v1.json` (19 + 24 = 43). Historical quality cases
are unchanged. Submit the initial task using ordinary `create`/`message`
operations, then `tool_loop`; follow with `close` if successful.

Required operation fields: `op`, `id` (ASCII identifier, 1–48 characters),
`budget` (1–256 tokens per request), `max_calls` (1–16 calculator calls),
`max_generated` (1–4096 tokens across responses, including EOS), `wall_seconds`
(1–120 integer seconds), and `answer` (expected bounded integer).
Request IDs are deterministically `{id}-{turn}`, reserved alongside ordinary
generation IDs; collisions fail rather than replaying. A final-response request
is permitted after the last allowed tool call. Each request's budget is clamped
to the remaining total token allowance.

Use the existing CLI with `--scenario tools/scenarios/calculator-v1.json` and
its normal `--server`, `--output`, `--native-model`, `--native-tokenizer` options.
Use `--native-context 4352 --native-chunk 256` for multi-turn headroom; context256
can exhaust its ledger before the final response. This free-greedy example needs
an explicit native launch; it is not a new launcher.
The fixed diagnostic server does not produce calculator JSON. Tests instead use
scripted model-free service responses.

An absolute monotonic deadline covers generation, local tools, and result
append, with checks at each boundary and socket timeouts capped by the remaining
wall time using the existing transport timeout handling. It is not a hard
process-preemption guarantee. Timeouts do not imply server cancellation or a
valid reusable KV state. Generation may already have mutated the session on
failure; it is not left untouched. Failure stops the scenario; inspect/reset or close
explicitly before reusing the session. No automatic repair or retry is attempted.

`client.ndjson` retains responses, tool observations/time, and loop reports.
`result.json` separates `failure_kind: protocol`, `runtime`, or `limit` from a
valid but wrong final (`protocol_success: true`, `task_success: false`). Protocol,
runtime, and limit failures leave task score null and yield CLI failure. Wrong
finals retain normal diagnostic CLI success; inspect the score. Tool timing and
scores are diagnostic only, never numerical correctness or performance evidence.
A direct final is allowed; tool usage is observed, not required by the score.
Context exhaustion is a runtime failure, not silently truncated input. Successful model tool compliance, native tool support and successful-task latency
remain unestablished. See the bounded observation below.

Model-free checks (also run with `python3 -O`):

```sh
python3 tests/test_workload_tools.py
python3 tests/test_service_http.py ./build/test_service_server
```

## First bounded real-model observation

One unchanged calculator-v1 run used compact FP8/MPS, context4352/chunk256 and
the scenario limits above. It generated37 tokens ending at natural EOS, but
returned fenced JSON with an invented `type: "tool_result"` and `result: 43`.
That is neither a valid tool request nor the required final-answer schema.

The client rejected the response: protocol failure, task score null, zero tools
executed, no result append or continuation. Server shutdown was clean. No prompt
repair or inference retry occurred. The presence of43 is not successful tool use.
This establishes fail-closed behavior on this real response, not a successful
real-model tool loop; positive continuation coverage remains model-free.
