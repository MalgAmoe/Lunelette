# Build and run

Run every command below from the engine repository root. Native execution requires macOS 15+, supported Apple Silicon, Make and Command Line Tools/macOS SDK; it links Foundation, Metal and MetalPerformanceShaders. Python 3 is optional client tooling, not an engine runtime dependency. Nothing is downloaded automatically.

Supply the official HF-layout `Ministral-3-3B-Instruct-2512-FP8/model.safetensors` and its Tekken 13 `tekken.json`. Examples use ignored local `models/` placeholders; HF tokenizer JSON is not a substitute. Only the mixed FP8/BF16 single-file checkpoint is supported, not shards or BF16 fallback. Independent numerical acceptance remains **FAIL/open**; see [limitations](../limitations.md).

## Persistent service

```sh
make server
python3 tools/interactive_workload.py \
  --server build/lunelette-server \
  --native-model models/Ministral-3-3B-Instruct-2512-FP8/model.safetensors \
  --native-tokenizer models/tekken.json \
  --native-strategy compact-fp8-arithmetic --native-prefill mps \
  --native-context 4352 --native-chunk 256 --native-prefix-reuse on \
  --output evidence/session-001 --seconds 300 --total-budget 1024
```

The output directory must be new. After `ready`, send these lines, waiting for each command's `final` before the next mutation:

```json
{"id":"s1","op":"create"}
{"id":"m1","op":"message","content":"What is 17 + 25? Answer with the number only."}
{"id":"g1","op":"generate","budget":32}
{"id":"q1","op":"quit"}
```

This is an invocation recipe, not an answer guarantee. See [interactive](interactive.md) for command schemas, bounds and owned-child cleanup; [workloads](workloads.md) for scripted scenarios.

## Configuration choices and omission

| Interface | Storage / prefill omission | Capacity / chunk omission | Prefix omission |
|---|---|---|---|
| CLI | compact arithmetic / MPS | fixed total context 256 | single turn |
| Python native clients | compact arithmetic / MPS | 256 / 256 | legacy |
| C constructors, NULL options or default macro | compact arithmetic / MPS | 256 / 256 | service-owned policy |
| Zero-initialized C options | resident FP32 / MPS (enum-zero compatibility) | NULL limits: 256 / 256 | service-owned policy |
| Direct server | strategy and prefill required explicitly | 256 / 256 | legacy |
| Numerical/native comparators | resident FP32 compatibility default | interface-specific | not client defaults |

Storage choices are `resident-fp32`, `compact-fp8-arithmetic`, `compact-fp8-lookup`. Compact arithmetic is a memory-conscious starting point; arithmetic versus lookup is not a universal speed ranking. Resident expands projections to FP32; compact choices keep projections packed and use a shared FP32 conversion stage for multi-token work. These choices do not imply changed numerical acceptance. See [engine](../reference/engine.md) for math, compact admission and memory ownership.

`mps` and `custom` are independent prefill selectors. MPS is the default for multi-token projections; decode/head use custom arithmetic. FP32 storage does not specify MPS internal accumulation precision. Unsupported compact inputs, allocation failures or the 24 GiB estimated peak guard cause refusal, not fallback. Admission refusal means that configuration is unavailable for the run, not a failed model answer or a throughput measurement. Named allocations and sampled footprint are not exact process/driver peaks; do not add unified-memory categories together.

Supported service/runtime context/chunk pairs are 256/256, 4352/256 and 4352/128. Context includes frames, input and generated IDs; chunk bounds one materialization call, not history. Capacity is fixed at construction; no silent truncation or resizing. Context 4352/chunk 256 with reuse ON is a useful explicit document/history profile, not a default promotion, speed guarantee or long-context correctness claim. The CLI remains 256 even though the runtime supports long context. Whole-document single-call prefill is not required for document workloads; longer input is partitioned into supported blocks, never silently admitted beyond the call-size bound.

## Direct server flags

```sh
build/lunelette-server \
  --model models/Ministral-3-3B-Instruct-2512-FP8/model.safetensors \
  --tokenizer models/tekken.json \
  --strategy compact-fp8-arithmetic --prefill mps --port 8081 \
  --context 4352 --prefill-chunk 256 --prefix-reuse on
```

Required flags are `--model`, `--tokenizer`, `--strategy`, `--prefill`, `--port`. Port is decimal 0–65535; 0 selects an ephemeral port announced in stderr. The service binds only 127.0.0.1. Unknown/duplicate options and missing values fail startup.

Optional selectors are `--context 256|4352`, `--prefill-chunk 256|128`, `--prefix-reuse on|off`, and `--tools calculator-v13|investigation-v13|observation-v13`. Omit prefix reuse for legacy: literal `legacy` is not a direct-server value. Python clients accept `--native-prefix-reuse legacy|on|off` and implement legacy by omission. Tools default off. Checkpoints require explicit on/off and no tools. See [service](../reference/service.md) for protocol, state, recovery and the canonical metrics dictionary; `/info` evidence labels are not startup numerical attestation.

## Single-turn CLI

```sh
make
./build/lunelette --help
./build/lunelette \
  --weights models/Ministral-3-3B-Instruct-2512-FP8/model.safetensors \
  --tokenizer models/tekken.json \
  --metal-strategy compact-fp8-arithmetic --metal-prefill mps \
  --system 'Be concise.' --prompt 'What is 17 + 25?' \
  --max-new-tokens 32 --dump-output-ids
```

`make` builds the CLI by default; `make server` builds the service without inference. CLI flags are `--weights`, `--tokenizer`, `--prompt`, optional `--system`, `--raw`, `--max-new-tokens`, `--metal-prefill`, `--metal-strategy`, `--dump-prompt-ids`, `--dump-output-ids`; `--help` and `--version` are standalone modes. Unknown/duplicate arguments fail. Retired `--backend` and `--uncached` fail explicitly.

Tokenizer and prompt are required; weights are required except with `--dump-prompt-ids`, which prints IDs without opening weights. Text fields are strict UTF-8, each at most 4096 bytes. Chat framing is BOS, optional system block, then `[INST]user[/INST]`, with no implicit system. `--raw` means BOS plus ordinary text and forbids `--system`.

`--max-new-tokens` is decimal digits only, 0–256, default 32; execution never exceeds total context 256. Greedy ties choose the lowest ID. Output is buffered, EOS omitted, other generated special IDs fail; a budget cutoff can leave incomplete UTF-8. `--dump-output-ids` writes generated IDs including EOS to stderr. No streaming or multi-turn state is exposed by this CLI.

## Execution boundaries

Builds are not inference. Model-free tests can still execute the GPU; consult [testing](../development/testing.md) before choosing targets. Optional assets must be supplied explicitly. Use fresh build directories for different compiler flags, and new output roots for client evidence. `make clean` removes all of `build/`, regardless of `BUILD`; do not use it on retained evidence.
