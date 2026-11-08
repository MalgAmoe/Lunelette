# Workload scenarios

`tools/real_workload.py` retains version-1 scenarios and adds version 2 for explicit prefix policy, bounded repetitions and native checkpoint/restore recipes. These are client controls over the existing service API; they do not change the engine, KV format, memory guard or default context/strategy/reuse policy.

## Version 2

See `tools/scenarios/checkpoint-repeat-v2.json`. Its top-level fields are exactly:

```json
{
  "version": 2,
  "mode": "free_greedy",
  "prefix_reuse": "on",
  "repetitions": 2,
  "wall_seconds": 120,
  "operations": [
    {"op":"create"},
    {"op":"message","content":"Reply exactly READY."},
    {"op":"checkpoint","id":"save","checkpoint":"base"},
    {"op":"generate","id":"first","budget":16},
    {"op":"score","id":"first","text":"READY"},
    {"op":"restore","id":"restore","checkpoint":"base"},
    {"op":"generate","id":"second","budget":16},
    {"op":"score","id":"second","text":"READY"},
    {"op":"close"}
  ]
}
```

The process/model is loaded **once**. Each repetition starts a fresh session and closes it; there are no implicit warmups, retries, repairs or extra generations. The sample has four planned generations, at most 64 generated tokens total. To compare OFF, make an explicit copy changing only `prefix_reuse` to `off` and use a separate output directory. Run model processes sequentially.

```sh
python3 tools/real_workload.py \
  --server build/lunelette-server \
  --scenario tools/scenarios/checkpoint-repeat-v2.json \
  --output results/checkpoint-on-01 \
  --native-model models/Ministral-3-3B-Instruct-2512-FP8/model.safetensors \
  --native-tokenizer models/tekken.json \
  --native-strategy compact-fp8-arithmetic --native-prefill mps
```

Paths are caller-supplied; no resources are downloaded. The output directory must not exist. Version 1 still defaults to legacy reuse; optional `--native-prefix-reuse legacy|on|off` selects the existing native policy. For version 2 this flag must agree with the scenario; conflicts and use with a model-free launch are rejected before startup. `/info` must match the selection.

### Bounds and identity

- `repetitions`: integer 1..16; `wall_seconds`: integer 1..120, shared by all operations/repetitions after service startup. Earlier enclosing deadlines win.
- Operations: 2..256 per repetition, create first/close last, no interior create/close. Reset and the other existing operations remain available.
- Template IDs and checkpoint aliases: 1..40 ASCII letters/digits/underscore/hyphen. IDs are expanded to `r1-save`, `r2-save`, etc.; score references and tool-loop IDs receive the same prefix. Aliases are local to a repetition.
- At most 256 possible admitted requests across all repetitions, including checkpoint/restore and every potential tool-loop response. Tool-generated ID collisions are rejected before launch. The service history is not evicted.
- Existing message, per-response 256-token, pause 0..5 s and tool-loop bounds remain. There is no silent context truncation or automatic output extension.

Control polling and ordinary response-body reads recheck deadlines; response bodies are bounded to 65536 bytes. Connection inactivity is a runtime failure, not application-budget exhaustion. Startup and cleanup have separate existing bounds; the scenario deadline is not a hard OS-level supervisor for the whole process. For real-weight experiments use an external wall/output/free-space supervisor as well; neither sampled memory nor a polling disk cap is a peak/quota guarantee.

### Checkpoints

Only `on` and `off`, without any tool-loop operations, support these recipes. Native tool-mode checkpoint/restore remains unsupported. A checkpoint operation publishes one alias **only after** the asynchronous request completes successfully. The client uses the returned opaque decimal checkpoint ID, not a guessed counter. A newer save replaces the previous alias; reset/close invalidate it. Unknown or stale alias references are rejected before execution.

Admission 202 is not completion. The client polls the same request ID, verifies identity, before-state, terminal outcome and authoritative session token ledger. Checkpoint materialization preserves logical IDs and phase, commits the full prefix and clears pending tokens. Restore preserves saved IDs and phase; ON restores the committed cache/logits, while OFF restores logical state with zero committed tokens. Restore itself must not run prefill/append/decode: work counters are checked, including the one ON logit restore. No client decode/re-encode or assistant replay substitutes for the saved ledger.

Failures stop the scenario; no automatic rollback or retry is assumed. Raw request/response records preserve runtime recovery metadata. The owning service is shut down on exit, including interruption.

## Results and evidence

Each repetition retains start/end, elapsed time, controls, task checks and tool reports. A task-quality failure does not stop later **already planned** repetitions. A protocol/runtime/limit failure does; completed repetitions are counted honestly and partial reports retained. Overall task success is null if the scenario is incomplete; completed per-task scores remain visible. With no scores it is null. Timings remain diagnostic (`performance_evidence:false`), not automatic benchmark or numerical acceptance. `score` retains exact EOS/ready output criteria.

See [testing](../development/testing.md) for model-free and optional-resource coverage. Scripted CPU-stub responses are protocol evidence, not real-model answers.

```sh
python3 -m unittest discover -s tests -p 'test_*workload*.py'
python3 -O -m unittest discover -s tests -p 'test_*workload*.py'
# Optional macOS socket integration; caller supplies the tokenizer:
make workload-scenarios-test TOKENIZER=models/tekken.json
```

Version 2 also supports explicitly labelled [native controlled replay](../development/replay.md), with fixed ID trajectories, bounded full-vector captures and a separate comparator. That mode prohibits task scoring and reserves at most 16 selections per process. `controlled_protocol_only` remains synthetic protocol testing, not native replay. These recipes do not establish independent numerical correctness, general quality/reliability, increasing-history quality sweeps or energy/peak-memory acceptance. Known independent-reference discrepancies remain unresolved.

## Version 1 and shared operation schemas

Version 1 has exactly `version:1`, `mode`, `operations`. Modes are `free_greedy` (explicit native model/tokenizer launch) and `controlled_protocol_only` (model-free server only). Version 2 supports `free_greedy` and `controlled_replay`, always native. A mismatch is rejected before model startup. Duplicate JSON keys and unknown fields are errors. Version 1 has at most 256 operations; unlike version 2 it permits multiple create/close sessions and does not require create first/close last at schema admission. Runtime state still constrains operations.

| Operation | Required fields beyond `op` | Contract |
|---|---|---|
| create, reset, close | none | Existing session lifecycle; reset/close invalidate saved checkpoint |
| message | content | User text, no assistant replay |
| generate, continue | id, budget | Same generation endpoint; default expected terminal `completed` |
| cancel | id, budget | Start generation then explicitly cancel; default terminal `cancelled` |
| pause | seconds | Number 0–5; explicit schedule, not hidden warmup |
| score | id, text | Immediate matching generate/continue, free-greedy only |
| native_tool_loop | id, budget, max_calls, max_generated, wall_seconds, answer | Fixed calculator profile |
| native_investigation_loop | same | Fixed stock lookup/calculator profile |
| native_observation_loop | same | Fixed saved-observation profile |
| checkpoint, restore (v2) | id, checkpoint | Async control with repetition-local alias |
| replay (v2) | id, tokens | Controlled-replay mode only |

All operations except score may have `expect`, subject to stricter exact-field control/replay/tool schemas. Its permitted keys are `text`, `state`, `logical_tokens`, `committed_tokens`, `pending_tokens`, `event`; these assert protocol values, not task quality. `expect.event` is only for generate/continue/cancel and must be `completed`, `cancelled` or `error`. Unexpected terminals stop the scenario; an expected error does not make an invalid session reusable. Reset is still required according to recovery metadata.

Budgets are integers 1–256, excluding booleans. Native content is at most 4096 UTF-8 bytes (32768 at context 4352); synthetic content is at most 256 bytes. Version-1 request IDs are nonempty strings at most 64 characters at schema validation; the service additionally enforces ASCII letters/digits/underscore/hyphen and process-lifetime uniqueness. Tool IDs have the tighter 1–48 ASCII contract, and version-2 templates the 1–40 contract. Duplicate IDs never authorize replay. See [tools](../reference/tools.md) for exact fixed fields, prompts, corpus hashes, safety limits and tool-specific scoring. The retired `tool_loop` application-JSON route is rejected before launch; mixing native profiles is prohibited.

## Exact-answer scoring

```json
{"op":"score","id":"previous-generation-id","text":"exact expected answer"}
```

Score is client-only and allowed immediately after the matching `generate` or `continue` in free-greedy mode. Expected text is a string of at most 4096 UTF-8 bytes; `expect` is forbidden on score. A pass requires `event == "completed"`, `stop_reason == "eos"`, `state == "ready"` and actual text exactly equal to expected text. There is no stripping, case folding or normalization. A valid budget stop is a quality failure, not a completed answer. Calculator tool scoring deliberately has a different whitespace rule; see [tools](../reference/tools.md).

Each task check retains request ID, expected/actual text, `completed_turn` and `success`. Overall task success combines all explicit scores and tool-task scores using logical all; no scores means null. Quality failure does not stop already planned operations/repetitions. Protocol/runtime/limit failure does stop; version-2 overall task success is null when incomplete even if earlier scores exist, with completed and partial checks retained. A quality-only failure is not a runner nonzero exit: the process exit tracks protocol/shutdown failure. Inspect `task_success`, not just exit status. Scores create no HTTP request, extra generation, retry, repair or budget extension.

## Shipped scenarios and fixed schedules

| File under `tools/scenarios/` | Purpose / planned work |
|---|---|
| partial-v1.json | Synthetic byte-ledger partial response and UTF-8 continuation, budgets 3 + 7, pause 0.05 s |
| cancel-reset-v1.json | Synthetic warm budget 1, cancellation budget 100, reset, budget 10 |
| native-short-v1.json | Native 1 + 3 continuation versus reset/re-prompt budget 4; pause 1 s, close, pause 1 s, new session budget 4; no task score |
| recorded-sensors-v1.json | Three independent synthetic recorded-reading prompts with two resets, budgets 32 each (96 total), one 1 s pause |
| calculator-v13.json | Calculator input and fixed loop, at most two calls / 384 generated IDs |
| investigation-v13.json | Stock lookup then subtraction, at most two calls / 384 IDs; use context 4352 because its framed prompt exceeds 256 |
| observation-v13.json | Saved pump-reading lookup, one call / 256 IDs |
| checkpoint-repeat-v2.json | Two repetitions, two 16-token generations each, one load, four generations / 64 IDs maximum |
| replay-v2.json | Fixed trajectories and controls, not task evidence; see replay guide |

The recorded-reading inputs are explicitly synthetic, not live sensor data: (pump-A 84, pump-B 72), (pump-A 70, pump-B 75), (pump-A 81, pump-B 89). Each asks for the hottest device, integer temperature and ALERT at temperature >=80, otherwise OK, with nothing else. Expected outputs are exactly `pump-A 84 ALERT`, `pump-B 75 OK`, `pump-B 89 ALERT`. Resets prevent reliance on prior answers or partial output. This is not an observation-tool call or representative arrival-rate benchmark. Exact full prompts remain in the versioned scenario file.

```sh
python3 tools/real_workload.py --server build/lunelette-server \
  --native-model models/Ministral-3-3B-Instruct-2512-FP8/model.safetensors \
  --native-tokenizer models/tekken.json \
  --scenario tools/scenarios/recorded-sensors-v1.json \
  --output results/readings-001

# Explicit model-free protocol launch; no native arguments:
python3 tools/real_workload.py --server build/test_service_server \
  --scenario tools/scenarios/partial-v1.json --output results/partial-001
```

The protocol executable is built with `make build/test_service_server`; it is not an inference engine. Its `test-ledger-v1` / `raw-synthetic-bytes-v1` input IDs are UTF-8 bytes without model delimiters. Output IDs `[65,226,130,172,240,159,152,128,90,256]` encode `A€😀Z` then synthetic EOS 256, not native EOS 2. Context is 256 byte tokens; pending output and incomplete UTF-8 span continuations. It cannot establish official framing/tokenization, model quality, numerical correctness or performance.

## Launch, timing and retained results

Required runner flags are `--server`, `--scenario`, `--output` (new directory). Native launch requires both `--native-model` and `--native-tokenizer`; optional selectors are `--native-strategy resident-fp32|compact-fp8-arithmetic|compact-fp8-lookup` (default arithmetic), `--native-prefill mps|custom` (default mps), `--native-context 256|4352` (default 256), `--native-chunk 128|256` (default 256, 128 requires 4352), `--native-prefix-reuse legacy|on|off`. Version 1 omits reuse for legacy; version 2's scenario policy governs and conflicting flags fail. `--request-seconds` is 1–120, default 120 for native requests. No model/tokenizer means the separate synthetic launch, not fallback after native failure.

One child is launched and owned; weights remain resident across planned sessions. Native startup is bounded to 180 seconds, test startup to 3 seconds; native safe-boundary shutdown allows 120 seconds, synthetic shutdown 3 seconds, followed by owned-child termination handling. No unrelated process is killed. Startup, runtime and shutdown failures remain evidence; unsuccessful shutdown makes protocol success false. Raw arrivals/times, operations, responses, failure records and merged child output are retained in `client.ndjson` and `server.stderr`; `result.json` contains the report. Evidence roots are never overwritten or automatically deleted. Protect message/path data: it is not redacted.

Streams must begin with one `started`, retain the admitted request ID, and end with one terminal event. Unknown/reordered/post-terminal events fail. Each event is limited to 8192 bytes and each stream to budget + 2 events. The synthetic request deadline is 15 seconds with at most four seconds of socket inactivity; native requests use their explicit request deadline. Ordinary control reads retain a four-second inactivity bound under enclosing deadlines. These are client limits, not GPU latency guarantees.

Client first actual text time is send-to-raw-chunk-arrival when a complete nonempty text delta can be decoded, not native first-token computation time. Client completion and cancel acknowledgement/terminal arrival are transport observations; tool local-operation time excludes HTTP submission. Load time, framing, generation and explicit pauses must remain distinguishable. See [service metrics](../reference/service.md) for the canonical server dictionary. Preserve raw samples, failures, planned work, conditioning and pauses; include initial ingestion in whole-session comparisons and do not call sampled memory a peak. `performance_evidence:false` is intentional, not benchmark certification.

Native controlled replay permits only create/message/reset/pause/checkpoint/restore/close/replay, prohibits scoring/tools/free-greedy generation and reserves at most 16 selections across the process. Its capture formats, trajectory constraints, verification and comparator contracts belong to [replay](../development/replay.md); forced IDs are never evidence of successful task completion. Free-greedy task results, native parity and protocol delivery do not close the independent numerical FAIL/open gate or establish broad quality, energy or long-context acceptance.
