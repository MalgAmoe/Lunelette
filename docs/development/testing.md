# Build and test entrypoints

Run from the repository root. No target downloads weights, fixtures or dependencies. `make` still builds the single-turn CLI; `make server` builds the native service. Both require the supported macOS/Apple Silicon toolchain. Build-only commands do not run inference.

## Standard development workflow

On supported macOS/Apple Silicon with Python 3, run:

```sh
make BUILD=build/dev-check dev-check
```

This optional workflow builds the native server, then runs the existing C/CLI/GPU suite, Python units, deterministic adapter/server HTTP checks, shader constants, build dependencies and entrypoint schedules. It needs no weights, tokenizer, third-party Python package or network download, but **does use GPU and loopback sockets**. Suite invocations are ordered; it does not launch model inference. Optional resource-dependent unit cases may still skip; tokenizer integrations below are deliberately excluded. The original `make test` stays Python-free. For host-only work use `host-test`, not this aggregate.

## Choose the execution boundary explicitly

| Target | Requirements / execution |
|---|---|
| `host-test` (`core-test`) | C compiler, Git/POSIX shell fixture tools, host/protocol code; no Metal execution, weights, supplied tokenizer or Python |
| `gpu-test` (`metal-test`) | macOS/Metal; synthetic GPU and frozen native arithmetic checks, no weights |
| `test` | Host + GPU + CLI parser tests; Python-free and model-free, **not GPU-free** |
| `python-test` | Python 3 standard library; workload/adapter, replay comparator, FP8 lookup-table and Unicode-generator admission unit checks |
| `interactive-test` | Builds deterministic test server; Python adapter checks including loopback sockets |
| `service-test` | C protocol/transport and Python HTTP checks with deterministic test server |
| `shader-facts-test` | Python/compiler checks of embedded model constants, no Metal execution |
| `build-path-test` | macOS, Python 3, C compiler and Make; host shader-embedding/path/publication checks and mocked builder delegation; no Metal/model execution |
| `dependency-test` | macOS build/dry-run dependency checks; compiles but does not run Metal tests |
| `build-identity-test` | Requires Git and POSIX shell tools; disposable Git fixture and recording compiler; checks both service recipes without GPU, models or Python; included in `core-test` |
| `entrypoint-test` | Inspect Make schedules to guard host/GPU boundaries and convenience aliases; dry-run only |
| `integration` / `compare_fp8` | Supplied weights and independent fixtures/hashes; host-only comparison, no GPU or Python; `compare_fp8` requires macOS CommonCrypto |
| `compare_metal` / model `compare_native` / long-context model runs | macOS/Metal plus explicit weights and supplied fixtures; long-context runners additionally need Python and tokenizer, HTTP runner needs the pinned tokenizer package |

`build-path-test` is separately invoked, not included in `python-test` or `dev-check`; run `make build-path-test` and `make build-path-test PYTHON='python3 -O'`. It checks relative/absolute shader-helper builds, exact generated-header equality, dependency scheduling and failed-helper publication preservation. This does not promise absolute-path support for every other Make recipe or paths containing spaces. The same target also runs `tests/test_build_timeout.py`: both builders are exercised with successful/nonzero shader helpers, timeout and cancellation, with and without a prior header. Delayed helpers must leave no live process-group writers or later file changes; unrelated process groups survive. Fast-exit notification races and signal-mask restoration on notification cleanup failure are covered. Run both normal and optimized Python; these are lifecycle tests with mocked compiler queries, not GPU dispatch or model inference.

Both long-context builders use these Make targets while retaining their fresh-output, environment, receipt and resource boundaries.

The Unicode-generator admission tests in `python-test` inject a stub `regex` module and stop before scanning or writing generated tables. They cover missing/wrong/matching versions and output preservation, including with `PYTHON='python3 -O'`; no real regex package or Unicode-table regeneration is required.

Python unit discovery can skip tests requiring an explicitly supplied service or tokenizer. A green unit suite is not proof that all HTTP integration cases ran. `interactive-test` covers the adapter's optional deterministic-server tests; other native-stub/tokenizer integrations remain separate targets below. `PYTHON` can select the interpreter for the Python/interactive entrypoints, for example `make python-test PYTHON='python3 -O'`. Existing specialized targets retain their own interpreter settings.

Use separate build directories to avoid stale objects when flags change:

```sh
make BUILD=build/check-host host-test
make BUILD=build/check-native server test
make BUILD=build/check-ndebug CFLAGS='-std=c11 -O2 -DNDEBUG -Wall -Wextra -Wpedantic' test
make BUILD=build/check-asan CFLAGS='-std=c11 -O1 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer' test
make python-test
make BUILD=build/check-protocol interactive-test service-test
```

To constrain sanitizers to host-only execution, use the same sanitizer flags with `host-test`, not `test`. Existing `make sanitize` uses `build/sanitize` and runs the full `test` target, including GPU tests. Do not try to suppress a target's recipe while leaving its GPU prerequisites active.

Service builds recheck `SERVICE_COMMIT` and `SERVICE_DIRTY` each invocation using a content-stable `$(BUILD)/service-identity` input. Both service binaries rebuild when either effective value changes, not when it stays unchanged. Command-line overrides remain supported (for example `make server SERVICE_COMMIT=release SERVICE_DIRTY=false`). Without Git metadata the existing fallback is `unknown` / `false`.

## Deterministic server versus native stubs

The deterministic HTTP server advertises `conversation_protocol=test-ledger-v1` and `tokenizer=raw-synthetic-bytes-v1`: input UTF-8 bytes are synthetic byte-valued IDs without official delimiters. Its output fixture is `A€😀Z` followed by synthetic EOS 256, with capacity 256 counting byte tokens and pending output. It exercises split UTF-8, authoritative ledgers, pending continuation, cancellation and recovery; it is not official tokenization or inference. The current formatter marks these measurements `test_only:true`; callback/worker durations describe synthetic host work, not model inference. Native process-footprint samples are unavailable/null for this generator and driver memory is null. Protocol-only scenarios have no task-success claim.

Build the synthetic server with `make build/test_service_server`, then run `build/test_service_server --port 8081 --tick-ms 50`. The test-only tick delay accepts 0..1000 ms to exercise safe-boundary cancellation; it is not a native-model latency setting or an HTTP control. Port 0 announces an owned ephemeral listening port on stderr.

Native adapter tests instead combine official framing (when a tokenizer is supplied) with CPU runtime stubs and mutable synthetic logits/checkpoints. Neither these stubs nor deterministic-server success establishes engine numerical correctness. Report which integration cases actually ran rather than treating discovery skips as passes.

## Offline long-context checker tests

The HTTP checker suite is model-free, but not dependency-free: provider, replay and environment-route tests require an existing interpreter with `mistral_common==1.13.1`. Choose that environment explicitly rather than assuming bare `python3` or an installed `uv` command selects it. Run from the repository root, preserving the virtualenv launcher path (do not resolve its symlink to the base interpreter):

```sh
REFERENCE_PYTHON=/path/to/existing/.venv/bin/python
"$REFERENCE_PYTHON" -B -c 'import sys; from importlib.metadata import version; print(sys.executable); v = version("mistral_common"); print(v); sys.exit(0 if v == "1.13.1" else 1)'
# Proceed only if the version check succeeds.
"$REFERENCE_PYTHON" -B -m unittest discover -s tests/long_context/http -p 'test_*.py'
"$REFERENCE_PYTHON" -O -B -m unittest discover -s tests/long_context/http -p 'test_*.py'
```

This suite exercises mock schedules, synthetic replay, subprocess environment checks and pinned original-driver equivalence; it does not launch the model, dispatch GPU work or require the downloaded tokenizer file. Missing packages are environment failures, not skipped passes. Do not install or synchronize an environment automatically. Other long-context suites have different prerequisites; this command is not blanket discovery of all long-context tests.

## Optional supplied-data checks

These require macOS native-stub builds and an explicitly supplied official tokenizer, not model weights. The following four use standard-library Python 3 (their existing recipes invoke `python3` directly):

```sh
make workload-replay-test TOKENIZER=models/tekken.json
make workload-scenarios-test TOKENIZER=models/tekken.json
make service-long-test TOKENIZER=models/tekken.json
make service-checkpoint-test TOKENIZER=models/tekken.json
```

The standalone ordinary-chat framing comparator and native-adapter stub can also be invoked explicitly:

```sh
make build/test_service_framing build/test_service_native
build/test_service_framing models/tekken.json fixtures/native-framing.json
build/test_service_native models/tekken.json
```

The ordinary framing fixture is externally supplied, not regenerated by C: a JSON object with `cases`, exactly seven cases, each with `messages` (`role`/`content`, supported user/assistant histories) and the expected full `tokens` array, independently generated with pinned native `mistral_common` rather than the HF default-system template. Each case must fit the 256-ID comparator buffer. The native-adapter executable uses link-time CPU runtime stubs; it does not open model weights. Tool-profile frozen framing fixtures and optional official-encoder checks are separate below.

The tool/lookup/observation parity targets additionally require an **existing Python environment with `mistral_common==1.13.1` and its dependencies**. Their official encoder comparisons verify that pinned version against frozen fixture provenance. The tokenizer alone is insufficient. Select that interpreter using `SERVICE_TOOLS_PYTHON`, not `PYTHON`; nothing is installed automatically:

```sh
make service-tools-tokenizer-test TOKENIZER=models/tekken.json \
  SERVICE_TOOLS_PYTHON=/path/to/existing/python
make service-lookup-tokenizer-test TOKENIZER=models/tekken.json \
  SERVICE_TOOLS_PYTHON=/path/to/existing/python
make service-observation-tokenizer-test TOKENIZER=models/tekken.json \
  SERVICE_TOOLS_PYTHON=/path/to/existing/python
```

These tests use official framing plus native CPU stubs/socket fixtures, not model inference. See [native tools](../reference/tools.md) for fixture provenance.

See [validation](validation.md), [controlled replay](replay.md) and [long-context diagnostics](long-context.md) for supplied-artifact comparisons and real-model commands. Native agreement, independent-reference acceptance, protocol correctness, task quality and speed remain separate gates. Do not regenerate fixtures or relax tolerances to erase failures.

## Cleaning local artifacts

`make clean` currently deletes the entire literal `build/` directory, even when `BUILD` selects another path. It is unsuitable for an evidence-bearing checkout. Before deleting any old build, identify external launch receipts, pinned binaries, frozen source snapshots and unique logs that reference it. Preserve those artifacts and their hashes without rewriting historical receipts. An absence of a filename search hit alone does not establish that a build is disposable.

Python caches are generated, but avoid broad ignored-file deletion: ignored model, fixture, environment and evidence files may be irreplaceable or costly to recreate. Use a reviewed, explicit removal list rather than `git clean -fdx`. Convenience test aliases do not authorize artifact deletion.

Comparator strategy/parser and LUN1/LUN2 metadata checks run in portable `native-self-test`, including NDEBUG and sanitizer builds without models, Python or a GPU. See [strategy compatibility](validation.md#strategy-identity-and-compatibility). Current model checks should pass an explicit final `--strategy=resident-fp32` (or an intentionally chosen compact selector); legacy omission stays fixed resident. These model-free tests do not establish full-model compact parity.

Storage-default compatibility is tested model-free by `make metal-options-test` (the constructor option-resolution seam), `test_metal_zero_work` (actual CLI with stub boundaries), and `python3 -B tests/test_storage_workload_defaults.py` (actual Python entrypoints with stub Service launches). Run Python also with `-O`. These host tests establish selection only, not real compact model acceptance. `core-test` is host-only and Python-free; full `make test` additionally runs `metal-test`, requiring macOS/Apple Silicon Metal GPU execution, including synthetic kernel tests. No GPU run is needed for the storage-selection matrix. Comparator omission stays fixed resident, and reference metadata is unchanged.

## Shared C fixtures and intentional inclusion seams

`tests/support/` owns only reusable synthetic fixtures, deterministic callbacks and CPU runtime stubs; scenario assertions and `main` remain in each `test_*.c`. No test borrows another whole test program or renames its runner. Fixture checks use always-active `CHECK`, including file creation/I/O and callback invariants, so NDEBUG does not erase setup or validation.

These are deliberately header-local support modules, not a generic test framework: `tokenizer_fixture.h` writes the same exclusive-create synthetic Tekken file; `session_fixture.h` shares cancellation callbacks/state; `lookup_fixture.h` shares tool callbacks and bounded event/route helpers. `native_runtime.h` is a single-translation-unit implementation header providing link-time Metal API stubs, mutable logits/counters and the synthetic checkpoint owner. It must be included once per test executable, never linked with the production Metal runtime. Header inclusion preserves the tools-native `create_configured` symbol-renaming/capture seam and direct private runtime inspection without introducing a new runtime API or changing operation order. None of these stubs establishes model correctness.

Production-source inclusions remain intentional: tokenizer allocation/BPE tests inject allocation failures or inspect private merge state; generation allocation tests inject failures; zero-work CLI tests replace runtime boundaries; session lookup/observation/tools/metrics tests inspect bounded grammar, serialization or private state; native replay/long tests inspect adapter ownership/state. HTTP transport and scripted checkpoint/tools/lookup/observation servers include the transport to replace entrypoint/callback boundaries. Long-context observers retain native/session/HTTP inclusion for source-bound observation; `short.c` retains its controller-probe inclusion for live observation/replay/model-admission helpers, not the retired runner or long scenarios. Its stub entrypoint also executes admission, journal-identity and comparator self-tests. Do not compile those included production units a second time into the same executable.

Frozen Metal shaders, synthetic expanded-FP32 controls and `tests/long_context/http/original/driver.py` remain separately maintained controls for their respective suites; none is an independent full-model numerical oracle. The long-context `engine-sources.json` census includes the support headers so a changed fixture invalidates current-source receipts; historical receipts are not rewritten. Header dependencies are explicit in Make; host tests remain Python-free and acquire no Metal/GPU prerequisites.

## Shared Python tool validation

`tools/tool_validation.py` owns the shared native tool-loop exceptions, bounded integer checks and operation-field validation. Native profiles, framing checks and task scoring stay in `native_workload_tools.py`. The application-JSON executor is unsupported; `tool_loop` is explicitly rejected before service effects rather than reinterpreted as native. `tests/test_workload_validation.py` checks shared bounds, exception identity and native import independence; `tests/test_workload_retirement.py` checks retired operation rejection. Both run in `make python-test`, including optimized Python. These checks do not establish model correctness.

## Bounded C tool schemas

`service/tool_grammar.c` owns stateless calculator integer/schema checks and fixed lookup/observation document and passage matching. `make tool-grammar-test` exercises it without a session or runtime; it is also part of `service-protocol-test` and the host suite. The private API takes borrowed, already bounded/allowlisted parsed documents and plain outputs; it does not call service/session APIs. `session.c` retains generic JSON parsing, the 2048-byte lookup-result input cap, request mutation, serialization/metadata and all locking, worker/state/cancellation/checkpoint publication. Calculator failure leaves existing arguments untouched; lookup parse failure leaves the document untouched, while a parsed but invalid schema/profile clears it to zero. The private session tools/lookup tests pin these distinct failure behaviors in addition to the deterministic state/cancellation flows. Link `service/tool_grammar.c` once alongside every compiled or source-included session implementation; maintained Make and manual long-context builds include it, and the current-source census pins both module files and the direct test. Frozen sources and the original HTTP driver remain unchanged.
