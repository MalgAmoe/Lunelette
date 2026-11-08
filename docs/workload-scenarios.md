# Reusable workload scenarios

`tools/real_workload.py` retains version-1 scenarios and adds version 2 for explicit prefix policy, bounded repetitions and native checkpoint/restore recipes. These are client controls over the existing service API; they do not change the engine, KV format, memory guard or default context/strategy/reuse policy.

## Version2

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
  --native-tokenizer models/Ministral-3-3B-Instruct-2512-BF16/tekken.json \
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

Model-free tests exercise schema/refusal, identity collisions, limits, asynchronous controls, malformed ledgers/work, failure classification, trickling response bodies, repetition aggregation and interruption. The optional socket test uses the actual native adapter/controller/client/CLI with CPU runtime stubs, ON/OFF and two sessions per process. Its two-token responses deliberately exercise partial-state restore and fail the READY quality score; they are **not real-model answers**. The maintained sample's 16-token budget is not changed by this test.

```sh
python3 -m unittest discover -s tests -p 'test_*workload*.py'
python3 -O -m unittest discover -s tests -p 'test_*workload*.py'
# Optional macOS socket integration; caller supplies the tokenizer:
make workload-scenarios-test TOKENIZER=/path/to/official/tekken.json
```

Version 2 also supports explicitly labelled [native controlled replay](controlled-replay.md), with fixed ID trajectories, bounded full-vector captures and a separate comparator. That mode prohibits task scoring and reserves at most 16 selections per process. `controlled_protocol_only` remains synthetic protocol testing, not native replay. These recipes do not establish independent numerical correctness, general quality/reliability, increasing-history quality sweeps or energy/peak-memory acceptance. Known independent-reference discrepancies remain unresolved.
