# Native calculator-v13 service and workload client

The interface passed model-free checks and **one real native calculator run**
(see the bounded result below). This is not general tool-reliability evidence. The previous
custom JSON calculator experiment failed; its implementation (`tool_loop` in
`workload_tools.py`), scenario, and historical evidence remain separate and intact.
They are not native-tool evidence.

`tools/scenarios/calculator-v13.json` uses `native_tool_loop`. The runner launches
the supplied native server with `--tools calculator-v13` and requires `/info`
`tools: true` and `conversation_protocol: "native-calculator-v13"`. Supply the
usual explicit `--server`, `--native-model`, `--native-tokenizer`, `--scenario`,
and new `--output` directory arguments to `tools/real_workload.py`. This is an
opt-in model run, not part of the model-free tests.

The initial user prompt asks for calculator use for 19 + 24 and an answer only.
The client inspects the accepted input ledger before generation. It requires a
completed `tool_call` terminal, `awaiting_tool`, matching retained request identity,
status, session validity/recovery, counts, metadata and authoritative token suffix.
The generated call must be `9 ordinary-name 32 ordinary-arguments 2`, with nonempty
ordinary spans entirely within 1000..131071. Calls emit no text deltas and empty
text. No assistant text is decoded, re-encoded, or replayed by this client.

Only exact calculator metadata/argument dictionaries are allowed. Operations are
add/subtract/multiply; operands are integers within ±1,000,000 (not bool or float).
Validation finishes before local execution. The result is a canonical integer
string within ±1,000,000,000,000, posted as `role: "tool"`, with `tool_call_id`
equal to the generation ID. It is never fabricated as a user message. The reply
must preserve the prior ledger/cache and append `7 ordinary-result 8`, returning
to ready. The client cannot independently tokenize content: checking that the
ordinary span encodes the exact string belongs to separate official-tokenizer
fixture tests. Fake API consistency is not tokenizer or C correctness evidence.

Final generation requires natural EOS and ready state, matching retained metadata
and the complete preserved ledger plus generated suffix. Scoring is fixed:
`text.strip() == "43"`, with at least one executed tool call. A direct final may
be protocol-valid, but is not tool-loop/task success; `tool_used` and
`tool_loop_success` expose that distinction. Wrong final answers remain separate
from protocol failure. Runtime, protocol, and limit failures have distinct
`failure_kind` values.

Limits are at most two calls, 128 generated tokens per response, 384 total
(including native protocol tokens), and 120 seconds. Lower scenario limits are
allowed. One absolute deadline covers generation, inspection, local work, and
result submission, respecting any earlier enclosing deadline. There are no
retries, repairs, extra turns beyond those limits, or deadline extensions.
Malformed/truncated calls execute zero local operations. Bounded work reported
with matching request/session identity is retained even when validation fails.

Model-free checks:

```sh
python3 -m unittest discover -s tests -p 'test_*workload_tools.py'
python3 -O -m unittest discover -s tests -p 'test_*workload_tools.py'
```

## Service contract and framing

Native service startup accepts `--tools calculator-v13`. Without it, ordinary-chat
framing and capabilities remain unchanged. This is a single allowlisted calculator,
not arbitrary tool registration, parallel calls, or a general tool API. In tool
mode checkpoint and restore operations return 501; context and memory defaults
and the runtime memory guard are unchanged.

The actual structural IDs are:

- First user: `1 5 <tool-list JSON tokens> 6 3 <user text tokens> 4`.
- Assistant call: `9 <calculator name tokens> 32 <argument JSON tokens> 2`.
- Tool result: `7 <result text tokens> 8` (no user framing or extra EOS).

V13 does not encode call IDs. The API associates the call with its generation
request ID. A completed call emits `stop_reason: "tool_call"`, enters
`awaiting_tool`, and exposes `tool_call: {id, name, arguments}` on the terminal
and retained request. Submit the result to the same session's `/messages`:

```json
{"role":"tool","tool_call_id":"GENERATION_ID","content":"43"}
```

Only one call can be outstanding. Generation and user messages are blocked until
its matching result arrives; stale/duplicate results are rejected. The server
validates framing/schema but trusts the client's bounded calculator result—it does
not execute the calculator itself. Reset/close clears outstanding state. Partial,
malformed or cancelled calls require reset and cannot authorize execution. The
pending assistant EOS stays in the authoritative ledger and is materialized once
alongside the result; generation resumes after token 8 without assistant replay.

`tests/fixtures/calculator-v13.json` freezes exact official `mistral_common 1.13.1`
vectors, tokenizer/source hashes, the implemented declaration, escaping and literal
special-token spellings. These spellings in user content remain ordinary tokens.
The optional check compares C and official encoders against frozen vectors and
includes a deliberate-mismatch check. Actual native-adapter CPU stubs cover pending
EOS and result continuation under legacy/on/off policies and 128/256 chunks.
A scripted socket test connects the actual service and client with prescribed
logits, including one tool call, result submission and final answer. It is **not
model inference, numerical parity, or task-quality evidence**.

On macOS, with an existing tokenizer and the pinned official Python environment:

```sh
make service-tools-tokenizer-test TOKENIZER=/path/to/tekken.json \
  SERVICE_TOOLS_PYTHON=/path/to/python
```

Core call-parser/state tests are also part of `make test`, requiring no model,
Python, or network. Optional tokenizer tests load no model weights.

## Bounded real-model result

One unchanged calculator scenario passed with compact FP8 arithmetic/MPS,
context4352/chunk256, legacy reuse and native V13 framing. The actual model called
`calculator` with `operation: add, a: 19, b: 24`; the client executed it, returned
`43`, and the model resumed the same session to answer `43` with natural EOS.
There were 172 initial tokens, 25 generated call tokens, four result tokens and
three generated final tokens. Official-tokenizer comparison matched the actual
initial/call/result transcript exactly; retained ledger and pending counts matched.

The single launch used the existing two-call/128-token-response/384-token-total/
120-second-loop limits, with no retry or prompt repair. Client and server exited
zero. The reviewed binary SHA256 was
`d98e15bda1b6b44ac8d1200a1d67b1506c8e046e8f526426a8b31978db990f94`;
model SHA256 was
`728f1826cd0e38191ca7b1379e81f78cf0555c6ffd95882aabd2404632346f86`.

This establishes one real calculator call/result/GPU-continuation success, not
independent numerical correctness, general tool reliability, peak memory or a
performance comparison. Historical custom-protocol failure remains a failure.

## Frozen lookup investigation

`tools/scenarios/investigation-v13.json` selects `native_investigation_loop` and
launches with `--tools investigation-v13`. It requires `/info` `tools: true` and
`conversation_protocol: "native-investigation-v13"`. Mixed native profiles are
rejected. Calculator-v13's API, default fixture and real-run evidence above are
unchanged. One bounded real investigation completed both tools correctly but
failed the exact final-output criterion, as recorded below. Use context4352 for
this scenario: its 260-token initial framed input exceeds the default256 limit.

The same bounded native loop validates authoritative call tokens, retained request
metadata and ledgers before execution, and submits `role: tool` with the generation
call ID. Lookup accepts exactly `{"document_id":"larch-stock"}` or the cedar
identifier. There is no path argument, arbitrary user-file access, remote tool or
network lookup. The only collection is `tools/fixtures/stock-documents-v1.json`,
version 1, pinned in the client to SHA256
`670215ba689b16ce7fbc3444c67a728682d039b7ad0fd830d917bd876b432258`.
It is bounded-read and every record validated once per loop before generation;
drift fails closed. Corpus identity is recorded before task execution.

Results use `json.dumps(..., ensure_ascii=False, sort_keys=True,
separators=(",", ":"))`. Each record has document ID, matching uppercase passage
ID ending `:P1`, and 1..1024 UTF-8 bytes of text; serialized content is at most
2048 bytes. Tool results append `7 <ordinary JSON tokens> 8`. The service checks
result document/passage identity and bounds; the client supplies frozen contents.
Calculator results remain canonical integer strings.

The prompt requests larch-stock, subtraction of dispatched from opening stock,
and `<remaining> [<passage_id>]`, without exposing the numbers. Task success
requires the exact final `19 [LARCH-STOCK:P1]` and actual lookup of larch followed
by actual calculator subtraction of 31 and 12, in that order. A valid cedar lookup,
wrong arithmetic, wrong answer, reordered calls, or direct final is a quality
failure, not a protocol error. No retry or repair is attempted. Limits remain two
calls, 128 output tokens per generation, 384 total, 120 seconds. Reports retain
classified failures and generated-work counts, plus local operation durations per
call and total (`tool_call_seconds`, `tool_seconds`), excluding HTTP submission.
Model-free tests cover the two-call/three-generation path and rejection/quality,
deadline, cancellation and corpus-drift cases; these do not establish service,
tokenizer, GPU or model correctness.

### Bounded real investigation result

One unchanged run used compact FP8 arithmetic/MPS, context4352/chunk256 and legacy
reuse. The model called `lookup(larch-stock)`, received the pinned passage, then
called `calculator(subtract,31,12)` and received `19`. It returned
`19 <LARCH-STOCK:P1>` rather than the required `19 [LARCH-STOCK:P1]`.

Therefore **protocol and tool-sequence success are true; task success is false**.
The arithmetic and passage identity are correct, but the frozen exact-format
criterion is not relaxed. No retry or prompt repair was performed. There were
260 initial tokens, 14/26/13 generated tokens across three responses, and 44/4
result tokens. Independent official-tokenizer replay matched the actual transcript
and retained ledger exactly. Client/server exited zero and the owned process group
was gone. This is one tool-assisted integration observation, not general quality,
independent numerical correctness or comparative performance evidence.

## Saved-observation lookup

`tools/scenarios/observation-v13.json` uses `native_observation_loop` and starts
`--tools observation-v13` (`native-observation-v13` in `/info`). This reuses the
existing `lookup` call, result schema, native V13 framing and session machinery;
it is not a separate sensor subsystem. This profile exposes only
`lookup({"document_id":"pump-readings"})`. Calculator and stock IDs are rejected
in this profile; existing calculator/investigation declarations remain unchanged.

The client reads the bundled `tools/fixtures/observations-v1.json`, version1,
SHA256 `136b7bcb4e89246fa7756854ce7b67264f6f32356faed379999f8098a74d53a0`.
It returns passage `PUMP-READINGS:P1` containing explicitly synthetic recorded
pump-A84 C / pump-B72 C readings. No live sensor, arbitrary path, network request
or user-selectable data file is involved. The prompt does not contain the readings;
the model must request them. The same strict result identity/byte/UTF-8 bounds,
pending-EOS handling, reset/cancellation and checkpoint restrictions apply.

The fixed task requires one actual pump-readings lookup followed by exact text
`pump-A 84 ALERT`. A direct final without lookup or wrong final text is a task
failure even if the protocol is valid. Limits: one executed lookup,128 generated
tokens per response,256 total,120-second loop. No retry or prompt repair.

Model-free verification includes frozen official-tokenizer vectors, old-profile
compatibility, cross-profile refusals, malformed/stale/bounded results, reset,
cancellation and scripted actual service/client continuation. Optional macOS check:

```sh
make service-observation-tokenizer-test TOKENIZER=/path/to/tekken.json \
  SERVICE_TOOLS_PYTHON=/path/to/python
```

These scripted checks do not establish real-model task quality. One subsequent
unchanged real run used compact FP8 arithmetic/MPS, context4352/chunk256 and legacy
reuse. The model requested `lookup(pump-readings)`, received the saved record, and
answered `Pump-A 84 ALERT`. The expected exact string was `pump-A 84 ALERT`.

**Protocol and tool-loop success passed; strict task success failed solely on the
case-sensitive device label.** The temperature and ALERT decision were correct;
the scorer was not relaxed. No retry, prompt repair or extra generation occurred.
There were127 initial tokens,14 generated call tokens,51 result tokens and9 final
tokens. Official-tokenizer replay matched all actual transcript IDs and the retained
ledger exactly. Client/server exited zero and the process group was gone. This is
one saved-observation tool integration observation, not live sensor access, broad
reliability, independent numerical correctness or comparative performance evidence.
