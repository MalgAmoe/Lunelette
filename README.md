# Lunelette

A learning-first Ministral 3 3B **text-only** inference engine in C, using one Metal/MPS runtime on Apple Silicon. It provides a persistent local diagnostic service, streamed multi-turn sessions, bounded tool profiles, checkpoint controls, and a single-turn CLI.

**Experimental; independent numerical acceptance remains FAIL/open.** The agreed operational diagnostic scope is delivered, not a claim of production readiness or general model quality. See the [current status](docs/real-workload-status.md) and [acceptance checklist](docs/spec-closeout-checklist.md).

## Start here: persistent service

Run commands from the repository root. Requires macOS 15+, supported Apple Silicon, Make and Command Line Tools/macOS SDK; development validation used an Apple M5 with 32 GiB unified memory. The native application links Foundation, Metal and MetalPerformanceShaders. Python 3 is needed for the optional interactive client, not for the native engine/service. Nothing is downloaded automatically.

Supply the official HF-layout `Ministral-3-3B-Instruct-2512-FP8/model.safetensors` and its Tekken 13 `tekken.json`. The paths below are Git-ignored local placeholders, not bundled resources. HF tokenizer JSON is not a substitute for the Tekken file.

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

The output directory must be new. After the client reports `ready`, send one JSON command per line, waiting for each command's `final` response before the next mutation:

```json
{"id":"s1","op":"create"}
{"id":"m1","op":"message","content":"What is 17 + 25? Answer with the number only."}
{"id":"g1","op":"generate","budget":32}
{"id":"q1","op":"quit"}
```

This example is an invocation recipe, not a guaranteed answer. The adapter owns one service process and preserves evidence. Inspect/cancel can run during generation; other mutations must wait. See the [interactive guide](docs/interactive-workload.md) for complete-text output, errors, cleanup and checkpoint/tool limitations.

### Recommended profile versus defaults

For the measured document/history workload, use **compact arithmetic + MPS + context 4352/chunk 256 + reuse ON**, as above. This is a memory-conscious recommendation from a bounded [workload panel](docs/useful-workload-panel.md), not a universal speed or correctness claim.

Client defaults are **compact arithmetic, MPS, context 256/chunk 256, legacy reuse**. C default constructors likewise select compact arithmetic/MPS and 256/256. This intentional storage-only compatibility change replaces resident FP32 on omission; use `--metal-strategy resident-fp32` for the CLI or `--native-strategy resident-fp32` for Python clients to retain previous storage behavior. A direct native-server launch requires explicit strategy and prefill flags; see the [service contract](docs/service.md). Lookup decoding, custom prefill and resident FP32 remain supported alternatives; lookup was not a consistent whole-workload winner. No failed optimization candidate has been integrated. The [status report](docs/real-workload-status.md) preserves the research outcomes separately from operational delivery.

## Interfaces and limits

| Interface | Scope |
|---|---|
| [`lunelette-server`](docs/service.md) | Local persistent service; multi-turn streaming; context 256 or opt-in 4352; chunks 256 or 128 (128 requires 4352) |
| `interactive_workload.py` | Optional bounded NDJSON client; session controls and fixed native tool profiles |
| `lunelette` | Single system/user turn; 256-token total context; buffered output |
| Scenario/replay tooling | Reproducible diagnostic workloads; forced replay is never task-quality evidence |

Checkpoints are plain-chat only and require an explicit on/off reuse policy. Cancellation may leave `needs_reset`; do not treat partial streamed text as a successful completed answer. Native tools are fixed diagnostic profiles, not arbitrary shell execution. Vision, batching, 8B/14B models, arbitrary checkpoint layouts and static W8A8 execution are unsupported.

## Single-turn CLI

`make` still builds the CLI; this default has not changed.

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

CLI framing is BOS, optional system block, then `[INST]user[/INST]`, without an implicit system message. `--raw` uses BOS plus ordinary text and forbids `--system`. Each text field is strict UTF-8, at most 4096 bytes. Budget is 0..256 (default 32); prompt plus generated context must fit 256. Greedy ties choose the lowest ID. `--dump-prompt-ids` needs no weights. Output is buffered, not streamed; EOS is omitted, other generated special IDs fail, and a cutoff may leave incomplete UTF-8. Retired `--backend`/`--uncached` selectors fail explicitly. Full contracts are in [the runtime guide](docs/metal.md) and public headers.

## Precision and evidence

The engine reconstructs FP8 projections as decoded FP32 values multiplied by BF16 scales widened to FP32, without BF16 product rounding. Compact storage keeps 182 projections packed; multi-token calls use a shared FP32 conversion stage. Embedding, tied head, 53 norms, activations and KV remain FP32 during execution. Activation scales are validated but unused: this is not W8A8.

Multi-token projections use MPS by default; `custom` is explicit, with no silent fallback. Single-token projections and the head use custom arithmetic. MPS FP32 storage does not establish its unspecified internal accumulation precision.

Keep operator checks, native regression, independent references and task outcomes separate. The fixed independent tolerance is `2e-4 + 1e-4*abs(reference)`; strict mismatches remain unresolved. Matching text or native controls does not close that gate. Historical CPU/BF16 findings are provenance, not current FP8 acceptance. See [validation](docs/validation.md), [native regression](docs/native-regression.md) and the [numerical review](docs/numerical-closeout-review.md).

## Build and test

For the optional standard local workflow: `make BUILD=build/dev-check dev-check`. It uses the GPU, standard-library Python and loopback sockets, but no model weights. The original `make test` remains Python-free.

| Command | What it does |
|---|---|
| `make` / `make server` | Build CLI / native service; no inference |
| `make host-test` | Portable C host/protocol tests; no Metal execution, Python or weights |
| `make gpu-test` | Synthetic Metal/operator/native regression tests; uses the GPU, no weights |
| `make test` | Existing host + GPU + CLI suite; offline, model-free, Python-free |
| `make python-test` | Python client/scenario/comparator unit tests; no model; optional integration cases may skip |
| `make interactive-test` | Adapter tests with the deterministic protocol server; loopback sockets, no model/GPU |
| `make service-test` | Protocol/transport tests with deterministic server; Python and loopback sockets |
| `make shader-facts-test dependency-test` | Shader constant and build-dependency checks; no inference |
| `make build-path-test` | Optional macOS Python/compiler/Make checks of shader build paths and publication; no GPU/model execution |
| `make entrypoint-test` | Dry-run checks of build/test execution boundaries; no compilation or GPU execution |

**Model-free does not mean GPU-free.** Real-weight tests remain explicit and optional. Supply your own weights/tokenizer/fixtures; see the [testing guide](docs/testing.md). Use fresh build directories when changing flags:

```sh
make BUILD=build/check-ndebug CFLAGS='-std=c11 -O2 -DNDEBUG -Wall -Wextra -Wpedantic' test
make BUILD=build/check-asan CFLAGS='-std=c11 -O1 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer' test
```

`make clean` removes **all of `build/`**, regardless of `BUILD`. Do not run it on retained evidence builds; inventory and preserve referenced binaries first. Runtime evidence directories are caller-owned and should not be committed.

## Project map

- [Documentation index](docs/README.md): current guides, validation and historical milestones.
- `include/`, `src/`: public contracts, host/CLI and Metal/MPS implementation.
- `service/`: persistent native service and session/transport logic.
- `tests/`: regression tests and frozen synthetic controls.
- `tools/`: optional clients, scenarios and validation helpers.

## Attribution and distribution

Inspired by antirez's [DwarfStar](https://github.com/antirez/ds4): its tensor descriptors, data-base offsets and lazy shard access informed the design. The parser and reader are original; no DwarfStar code was reused or modified, and no upstream checkout is required. Transformers YaRN adaptations and Unicode-table notices are retained in [NOTICE](NOTICE) and the referenced license files. Adapted code must retain the required attribution and license notices.

The project's own distribution license has not yet been selected.
