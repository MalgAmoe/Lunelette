# Fixed native tools

Tools execute in the client, never inside the inference engine. These are diagnostic fixtures, not arbitrary execution or general tool-reliability evidence. The retired application-JSON `tool_loop` operation is rejected before startup. See [service](service.md) for protocol/state ownership and [workloads](../guides/workloads.md) for scenario execution.

`tools/scenarios/calculator-v13.json` uses `native_tool_loop`. The runner launches the supplied native server with `--tools calculator-v13` and requires `/info` `tools: true` and `conversation_protocol: "native-calculator-v13"`. Supply the usual explicit `--server`, `--native-model`, `--native-tokenizer`, `--scenario`, and new `--output` directory arguments to `tools/real_workload.py`. This is an opt-in model run, not part of the model-free tests.

The initial user prompt asks for calculator use for 19 + 24 and an answer only. The client inspects the accepted input ledger before generation. It requires a completed `tool_call` terminal, `awaiting_tool`, matching retained request identity, status, session validity/recovery, counts, metadata and authoritative token suffix. The generated call must be `9 ordinary-name 32 ordinary-arguments 2`, with nonempty ordinary spans entirely within 1000..131071. Calls emit no text deltas and empty text. No assistant text is decoded, re-encoded, or replayed by this client.

Only exact calculator metadata/argument dictionaries are allowed. Operations are add/subtract/multiply; operands are integers within ±1,000,000 (not bool or float). Validation finishes before local execution. The result is a canonical integer string within ±1,000,000,000,000, posted as `role: "tool"`, with `tool_call_id` equal to the generation ID. It is never fabricated as a user message. The reply must preserve the prior ledger/cache and append `7 ordinary-result 8`, returning to ready. The client cannot independently tokenize content: checking that the ordinary span encodes the exact string belongs to separate official-tokenizer fixture tests. Fake API consistency is not tokenizer or C correctness evidence.

Final generation requires natural EOS and ready state, matching retained metadata and the complete preserved ledger plus generated suffix. Scoring is fixed: `text.strip() == "43"`, with at least one executed tool call. A direct final may be protocol-valid, but is not tool-loop/task success; `tool_used` and `tool_loop_success` expose that distinction. Wrong final answers remain separate from protocol failure. Runtime, protocol, and limit failures have distinct `failure_kind` values.

Limits are at most two calls, 128 generated tokens per response, 384 total (including native protocol tokens), and 120 seconds. Lower scenario limits are allowed. One absolute deadline covers generation, inspection, local work, and result submission, respecting any earlier enclosing deadline. There are no retries, repairs, extra turns beyond those limits, or deadline extensions. Malformed/truncated calls execute zero local operations. Bounded work reported with matching request/session identity is retained even when validation fails.

Model-free checks:

```sh
python3 -m unittest discover -s tests -p 'test_*workload_tools.py'
python3 -O -m unittest discover -s tests -p 'test_*workload_tools.py'
```

## Service contract and framing

Native service startup accepts `--tools calculator-v13`. Without it, ordinary-chat framing and capabilities remain unchanged. This is a single allowlisted calculator, not arbitrary tool registration, parallel calls, or a general tool API. In tool mode checkpoint and restore operations return 501; context and memory defaults and the runtime memory guard are unchanged.

The actual structural IDs are:

- First user: `1 5 <tool-list JSON tokens> 6 3 <user text tokens> 4`.
- Assistant call: `9 <calculator name tokens> 32 <argument JSON tokens> 2`.
- Tool result: `7 <result text tokens> 8` (no user framing or extra EOS).

V13 does not encode call IDs. The API associates the call with its generation request ID. A completed call emits `stop_reason: "tool_call"`, enters `awaiting_tool`, and exposes `tool_call: {id, name, arguments}` on the terminal and retained request. Submit the result to the same session's `/messages`:

```json
{"role":"tool","tool_call_id":"GENERATION_ID","content":"43"}
```

Only one call can be outstanding. Generation and user messages are blocked until its matching result arrives; stale/duplicate results are rejected. The server validates framing/schema but trusts the client's bounded calculator result—it does not execute the calculator itself. Reset/close clears outstanding state. Partial, malformed or cancelled calls require reset and cannot authorize execution. The pending assistant EOS stays in the authoritative ledger and is materialized once alongside the result; generation resumes after token 8 without assistant replay.

`tests/fixtures/calculator-v13.json` freezes exact official `mistral_common 1.13.1` vectors, tokenizer/source hashes, the implemented declaration, escaping and literal special-token spellings. These spellings in user content remain ordinary tokens. The optional check compares C and official encoders against frozen vectors and includes a deliberate-mismatch check. Actual native-adapter CPU stubs cover pending EOS and result continuation under legacy/on/off policies and 128/256 chunks. A scripted socket test connects the actual service and client with prescribed logits, including one tool call, result submission and final answer. It is **not model inference, numerical parity, or task-quality evidence**.

On macOS, with an existing tokenizer and the pinned official Python environment:

```sh
make service-tools-tokenizer-test TOKENIZER=models/tekken.json \
  SERVICE_TOOLS_PYTHON=python3
```

Core call-parser/state tests require no model, Python, or network; the enclosing `make test` also executes GPU tests. Optional tokenizer tests load no model weights. `SERVICE_TOOLS_PYTHON=python3` assumes that interpreter already supplies `mistral_common==1.13.1`; otherwise select the existing supplied environment explicitly. Do not install packages implicitly.

## Frozen lookup investigation

`tools/scenarios/investigation-v13.json` selects `native_investigation_loop` and launches with `--tools investigation-v13`. It requires `/info` `tools: true` and `conversation_protocol: "native-investigation-v13"`. Mixed native profiles are rejected. The calculator profile remains separate and unchanged. Use context 4352 for this scenario: its 260-token initial framed input exceeds the default 256 limit.

The same bounded native loop validates authoritative call tokens, retained request metadata and ledgers before execution, and submits `role: tool` with the generation call ID. Lookup accepts exactly `{"document_id":"larch-stock"}` or the cedar identifier. There is no path argument, arbitrary user-file access, remote tool or network lookup. The only collection is `tools/fixtures/stock-documents-v1.json`, version 1, pinned in the client to SHA256 `670215ba689b16ce7fbc3444c67a728682d039b7ad0fd830d917bd876b432258`. It is bounded-read and every record validated once per loop before generation; drift fails closed. Corpus identity is recorded before task execution.

Results use `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))`. Each record has document ID, matching uppercase passage ID ending `:P1`, and 1..1024 UTF-8 bytes of text; serialized content is at most 2048 bytes. Tool results append `7 <ordinary JSON tokens> 8`. The service checks result document/passage identity and bounds; the client supplies frozen contents. Calculator results remain canonical integer strings.

The prompt requests larch-stock, subtraction of dispatched from opening stock, and `<remaining> [<passage_id>]`, without exposing the numbers. Task success requires the exact final `19 [LARCH-STOCK:P1]` and actual lookup of larch followed by actual calculator subtraction of 31 and 12, in that order. A valid cedar lookup, wrong arithmetic, wrong answer, reordered calls, or direct final is a quality failure, not a protocol error. No retry or repair is attempted. Limits remain two calls, 128 output tokens per generation, 384 total, 120 seconds. Reports retain classified failures and generated-work counts, plus local operation durations per call and total (`tool_call_seconds`, `tool_seconds`), excluding HTTP submission. Model-free tests cover the two-call/three-generation path and rejection/quality, deadline, cancellation and corpus-drift cases; these do not establish service, tokenizer, GPU or model correctness.

## Saved-observation lookup

`tools/scenarios/observation-v13.json` uses `native_observation_loop` and starts `--tools observation-v13` (`native-observation-v13` in `/info`). This reuses the existing `lookup` call, result schema, native V13 framing and session machinery; it is not a separate sensor subsystem. This profile exposes only `lookup({"document_id":"pump-readings"})`. Calculator and stock IDs are rejected in this profile; existing calculator/investigation declarations remain unchanged.

The client reads the bundled `tools/fixtures/observations-v1.json`, version 1, SHA256 `136b7bcb4e89246fa7756854ce7b67264f6f32356faed379999f8098a74d53a0`. It returns passage `PUMP-READINGS:P1` containing explicitly synthetic recorded pump-A 84 C / pump-B 72 C readings. No live sensor, arbitrary path, network request or user-selectable data file is involved. The prompt does not contain the readings; the model must request them. The same strict result identity/byte/UTF-8 bounds, pending-EOS handling, reset/cancellation and checkpoint restrictions apply.

The fixed task requires one actual pump-readings lookup followed by exact text `pump-A 84 ALERT`. A direct final without lookup or wrong final text is a task failure even if the protocol is valid. Limits: one executed lookup, 128 generated tokens per response, 256 total, 120-second loop. No retry or prompt repair.

Model-free verification includes frozen official-tokenizer vectors, old-profile compatibility, cross-profile refusals, malformed/stale/bounded results, reset, cancellation and scripted actual service/client continuation. Optional macOS check:

```sh
make service-observation-tokenizer-test TOKENIZER=models/tekken.json \
  SERVICE_TOOLS_PYTHON=python3
```


## Fixed input profiles and corpus contents

The shipped prompts are part of the diagnostic task; changing them changes the task, not just its presentation. All three scenarios create a session, append the following message, run the named loop with the fixed answer, then close. Lower budgets are permitted but cannot authorize retries or extra calls.

### Calculator input

```json
[
  {
    "op": "message",
    "content": "Use the calculator to compute 19 + 24, then report the answer only."
  },
  {
    "op": "native_tool_loop",
    "id": "calculator-v13",
    "budget": 128,
    "max_calls": 2,
    "max_generated": 384,
    "wall_seconds": 120,
    "answer": 43
  }
]
```

### Investigation input

```json
[
  {
    "op": "message",
    "content": "Look up larch-stock, use calculator to subtract dispatched from opening stock, and answer '<remaining> [<passage_id>]' only."
  },
  {
    "op": "native_investigation_loop",
    "id": "investigation-v13",
    "budget": 128,
    "max_calls": 2,
    "max_generated": 384,
    "wall_seconds": 120,
    "answer": "19 [LARCH-STOCK:P1]"
  }
]
```

### Observation input

```json
[
  {
    "op": "message",
    "content": "Use lookup to read pump-readings. Report the hottest device, its integer temperature, and ALERT if temperature is at least 80 else OK. Answer exactly <device> <temperature> <status>."
  },
  {
    "op": "native_observation_loop",
    "id": "observation-v13",
    "budget": 128,
    "max_calls": 1,
    "max_generated": 256,
    "wall_seconds": 120,
    "answer": "pump-A 84 ALERT"
  }
]
```

### Frozen records

These exact record texts are returned, not user-supplied files or live observations:

`tools/fixtures/stock-documents-v1.json`:

```json
{
  "version": 1,
  "documents": [
    {
      "document_id": "larch-stock",
      "passage_id": "LARCH-STOCK:P1",
      "text": "Opening stock: 31 units. Dispatched: 12 units."
    },
    {
      "document_id": "cedar-stock",
      "passage_id": "CEDAR-STOCK:P1",
      "text": "Opening stock: 80 units. Dispatched: 5 units."
    }
  ]
}
```

`tools/fixtures/observations-v1.json`:

```json
{
  "version": 1,
  "documents": [
    {
      "document_id": "pump-readings",
      "passage_id": "PUMP-READINGS:P1",
      "text": "Synthetic recorded observation (not live): pump-A temperature 84 C; pump-B temperature 72 C."
    }
  ]
}
```

Corpus reads are capped at 8192 bytes (an extra byte detects overflow); exact hashes, version 1, exact collection keys/counts and every record are checked before generation. Lookup results serialize with `ensure_ascii=False`, sorted keys and compact separators. Argument objects admit no additional fields; booleans and floats are not integer operands. Loop operation fields are exactly `op`, `id`, `budget`, `max_calls`, `max_generated`, `wall_seconds`, `answer`; IDs are 1–48 ASCII letters/digits/underscore/hyphen. All bounds are positive integers; wall time is 1–120 seconds.

The service validates result schema, identity and bounds, not the truth of arbitrary client-supplied contents. Frozen client corpus checks and independent official-tokenizer fixture checks are distinct safeguards. Incomplete/malformed/cancelled calls authorize no local execution. Natural EOS/ready final validation precedes scoring; protocol completion, tool sequence and exact task success remain separate.
