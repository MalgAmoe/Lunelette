# Lunelette

A learning-first Ministral 3 3B **text-only** inference engine in C, with one resident Metal/MPS runtime on Apple Silicon. It provides a persistent local service, streamed multi-turn sessions, fixed tool profiles, checkpoints and a single-turn CLI.

**Experimental: independent numerical acceptance remains FAIL/open.** Operational diagnostic use, native regression agreement and task quality are different claims. Read [supported scope and limitations](docs/limitations.md).

## Quickstart

Requires macOS 15+, supported Apple Silicon, Make and Command Line Tools/macOS SDK. Development validation used M5/32 GiB. Python 3 is needed for this optional client, not the native engine. Supply the official mixed FP8/BF16 single-file checkpoint and Tekken 13 `tekken.json`; the ignored paths below are placeholders, not bundled resources. Nothing is downloaded automatically.

Run from the repository root:

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

Use a new output directory. After `ready`, send one JSON line at a time, waiting for each command's `final` response:

```json
{"id":"s1","op":"create"}
{"id":"m1","op":"message","content":"What is 17 + 25? Answer with the number only."}
{"id":"g1","op":"generate","budget":32}
{"id":"q1","op":"quit"}
```

This is a usage example, not an answer guarantee. It explicitly selects the larger context and reuse; client defaults are compact arithmetic/MPS, 256/256 and legacy reuse. Full FP32 and compact lookup remain supported. See [running](docs/guides/running.md) for all configuration choices, direct HTTP startup and the single-turn CLI (`make`). The CLI stays limited to 256 total tokens.

## Learn and develop

- [Documentation](docs/README.md): task-oriented guide to the engine.
- [Interactive commands](docs/guides/interactive.md) and [repeatable workloads/scoring](docs/guides/workloads.md).
- [Engine design](docs/reference/engine.md), [service contract](docs/reference/service.md), [fixed tools](docs/reference/tools.md) and [tokenizer](docs/reference/tokenizer.md).
- [Testing](docs/development/testing.md) and [validation contracts](docs/development/validation.md).

`make host-test` runs host/protocol tests without GPU, weights or Python. **`make test` uses the GPU**, although it is model-free and Python-free. Optional `make dev-check` also uses Python and loopback sockets. Choose the appropriate tier from the testing guide; use fresh build directories when changing flags. `make clean` removes all of `build/`, including retained evidence builds.

Source is organized as `include/` and `src/` for C contracts/runtime, `service/` for HTTP/session logic, `tests/` for maintained regression coverage and `tools/` for optional clients/validation. [Project standards](AGENTS.md) govern changes. Weights, environments and generated run evidence do not belong in Git.

## Attribution and distribution

Inspired by antirez's [DwarfStar](https://github.com/antirez/ds4): tensor descriptors, data-base offsets and lazy shard access informed the design. The parser and reader are original; no DwarfStar code was reused or modified, and no upstream checkout is required. Transformers YaRN adaptations and Unicode-table notices are retained in [NOTICE](NOTICE) and its referenced license files. Adapted code must retain the required attribution and licenses.

The project's own distribution license has not yet been selected.
