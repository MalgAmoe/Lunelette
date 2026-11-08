# Long-context diagnostics

These optional tools remain supported at their current `tools/` and `tests/long_context/` paths. They are fixed diagnostic schedules, not a general context, strategy or workload framework. Independent long-context numerical correctness remains unverified; native parity, state/work checks and synthetic checker tests do not close that gap. See [validation](validation.md) for acceptance policy and [testing](testing.md) for execution dependencies.

## Runtime suite

This optional tooling is separate from `make test` and requires Python 3 standard library; no model or network is needed to build. macOS with the existing Metal toolchain is required. No production defaults, guards, or checkpoint arithmetic are changed.

```
python3 tools/long_context.py --self-test
python3 tools/long_context.py --build /absolute/fresh/build
python3 tools/long_context.py --build /absolute/fresh/build \
  --weights /absolute/model.safetensors --tokenizer /absolute/tekken.json \
  --out /absolute/fresh/results
```

The final command loads weights: review all delivered changes before launching. No real-model execution is implied by a successful model-free test. Build and result directories must be fresh and exclusively owned; no resume, downloads, or dependency installation. Build receipts hash the current `tests/long_context/engine-sources.json` manifest plus the promoted tooling collected by the builder, and record compiler/commands/products. The current manifest and receipt define the source inventory; do not assume a fixed historical file count. Objects are rebuilt from current sources using Make's BUILD override, never copied from archived builds. Source drift invalidates an existing build. Paths and all ancestors are checked before creation without resolving away symlinks (on macOS, use `/private/tmp`, not the `/tmp` symlink). This assumes exclusive ownership; it is not protection against concurrent hostile filesystem replacement.

Build subprocesses use an allowlisted environment, an explicit xcrun-selected compiler and SDK, and explicit Make flags matching production Metal contraction settings (`-ffp-contract=off`). Inherited Make/compiler/toolchain flags are not used. `build.log` persists commands, compiler path/hash/version, SDK selection, environment and compiler output, including failed builds. Individual build commands have 30- or 180-second deadlines. Full Xcode is optional: an unavailable `xcodebuild -version` is recorded; Command Line Tools can build these probes.

Seven serial cells: short compact ON, resident ON, compact OFF, cancellation; long resident B256, compact B256, compact B128. Structural schedules, complete finite 131072-float logits, 52 KV-buffer hashes, restore and independent replay remain required. Resident/compact comparisons are bitwise; OFF partition and B128 use the fixed changed-schedule rule `abs(error) <= 2e-4 + 1e-4 * abs(reference)`; this is not permission to relax same-schedule failures. Timing normalization only removes explicit documented timing fields and cross-lane interleaving, not state events. Native parity is not an independent numerical oracle or task-quality approval.

The reviewed POSIX supervisor owns one child process group at a time and cleans that group on failure. Limits: 1200 seconds/cell, 14400 total, 256 MiB artifacts, 1 GiB free-space headroom, declared bounded stdout/stderr and sidecar writers. Real children have no RLIMIT_FSIZE (it can break Metal compiler internal files); this is trusted-writer accounting, not a hostile-process sandbox or filesystem quota. Preserve failed artifacts; do not refresh contracts or tolerances to erase failures.

## Runtime fixtures and coverage boundaries

`tests/long_context/contracts.json` holds normalized structural contracts, not model-generated numerical goldens. The suite uses the short observer/controller probe, long runtime probe and the existing POSIX supervisor; it does not implement another inference engine. The short observer retains journal, endpoint, replay and model-admission helpers, and its stub entrypoint executes admission, journal-identity and comparator self-tests. Source inclusion is deliberate and source-bound; do not link included production units twice.

The runtime `--self-test` builds fresh probes, generates a synthetic tokenizer, prepares forced branch IDs with valid UTF-8, runs short-stub checks and long/supervisor/checker unit discovery. It does not load weights or dispatch model inference. The stub tokenizer IDs differ from the official tokenizer, so these runs do not establish agreement with frozen official-token schedules. Checker tests copy normalized contracts and provide synthetic finite vectors and input-state hashes; rejection tests do not become independently observed state/numerical evidence. Stub hashes are explicitly not native KV evidence.

The seven-cell comparison is intentionally narrow: every probe uses capacity C=4352 and MPS prefill. Short normal cells use block B=256 for compact arithmetic ON, resident ON and compact arithmetic OFF; cancellation uses compact arithmetic ON at B=128. Long cells use resident B=256, compact arithmetic B=256 and compact arithmetic B=128; lookup/custom prefill and arbitrary reuse/profile combinations are not exercised. The B=256 runtime schedule reaches prefix 4096 and capacity 4352, while B=128 is a reduced smoke schedule with prefix 257 and maximum tested length 513, not a full-capacity B=128 validation. The long schedule tests lengths B-1/B/B+1, irregular tails, save/append/restore branches and reset plus identical-chunk replay, with separate input-token budgets of 20,000 for B=256 and 2,048 for B=128. Here replay is an independently reset native trajectory, not an independent framework oracle. Complete-vector, 52-buffer hash, restore/replay and normalized trace checks are complementary, not independent reference acceptance. Timing normalization must not remove state events. A successful self-test/build says nothing about a later real-model result; pin each run's own sources/products/inputs and retain failures.

Both builders delegate shader embedding to Make and use `tests/long_context/build_command.py` for owned build process groups. Timeout/cancellation waits for no live group writers before returning, so a killed Make cannot leave its helper modifying output later. Existing headers are preserved on failed publication. [Build-path tests](testing.md#choose-the-execution-boundary-explicitly) cover relative/absolute helper paths, publication, delayed-writer teardown, fast exits and signal-mask cleanup without GPU/model execution.

## HTTP suite

The optional entrypoint is `tools/long_context_http.py`. It uses the fixed driver/provider schedule, not a second inference implementation or supervisor. Only `D-256-mps-compact-fp8-arithmetic` is executable: capacity 4352, chunk 256, prefix reuse on. There are no prerequisite jobs, retries/resume, implicit model locations, downloads or installations. It is schedule/state/work diagnostic coverage, not numerical or task-quality approval.

### Explicit commands

Use an already-installed official tokenizer environment. `--python` is mandatory and must provide `mistral_common==1.13.1`; missing/wrong versions fail before building or launching. The immutable tokenizer SHA256 remains `600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843`. No C-tokenizer self-consistency is substituted for official framing/decode. Each run records Python and installed-distribution versions in `reference-environment.json`; offline replay rejects version drift. The executable and virtualenv configuration are pinned, **installed package contents are not**. Use an exclusively owned, unchanged environment: in-place package modification without a version change is a remaining reproducibility gap.

```sh
python3 tools/long_context_http.py --python /absolute/reference/bin/python --self-test

# Optional tokenizer-only independent framing check, also in normal and -O:
python3 tools/long_context_http.py --python /absolute/reference/bin/python \
  --self-test --tokenizer /absolute/checkpoint/tekken.json

# Fresh build outside the source tree; never opens a model:
python3 tools/long_context_http.py --python /absolute/reference/bin/python \
  --build /absolute/fresh-http-build
```

An explicitly invoked model run must supply **all** inputs:

```sh
python3 tools/long_context_http.py --python /absolute/reference/bin/python \
  --build /absolute/http-build --out /absolute/fresh-http-results \
  --weights /absolute/checkpoint/model.safetensors \
  --tokenizer /absolute/checkpoint/tekken.json --port 18341
```

Build/output/input paths reject symlinks and parent traversal; build/output must be outside the source tree, fresh and non-overlapping. On systems where `/tmp` is a symlink, use its canonical directory instead. The explicit virtualenv interpreter is the sole symlink exception: the supervisor pins its resolved executable and virtualenv configuration; child/replay reject target drift. There is no implicit model location or port selection.

### Fixture and writer contracts

`tests/long_context/http/fixtures/` contains the four byte-exact texts, their original official framed IDs and the 4,594-token capacity-negative fixture. `original-sha256.json` identifies the original inputs/helpers. The original framing provenance (Python 3.12.11, mistral_common 1.13.1, tokenizer hash and ID hashes) remains in `fixtures/texts.json` and `negative.json`. The original driver is retained solely as a hashed, injected test oracle under `original/`; no historical workspace is needed at runtime.

- `driver.py` owns the fixed schedule, validators, ledger, stream terminals, conditional checkpoint branches, request bounds, metrics schema 3/work 2 and work envelopes. Launch requires immutable fixture checks and current-source/product receipts; journal census measures the output directory.
- `SocketTransport.request` declares Content-Length 0 for non-POST requests absent an explicit declaration. Explicit malformed POST lengths remain testable. Response cap is 262,144 bytes; every connection closes on success and exception.
- `provider.py` uses official Tekkenizer framing/byte decoding, streaming UTF-8 tails and EOS policy, and owns observation reads, memory requests and separate restore accounting. Version detection is explicit. Fixtures are immutable; `check_tokenizer.py` verifies them against an explicitly supplied tokenizer without importing model libraries.
- `child.py` owns server startup/capture/cleanup. Evidence/tokenizer exists before server memory admission; readiness requires this child's own listening record. Its server inherits the supervisor-owned process group.
- `campaign_http.py` composes exactly one job with `supervisor/{manifest,supervisor}.py`. The owner/process limit and total campaign deadline are each 1200 seconds. No retries/resume.
- `replay.py` is the offline callback: it replays the full ordered HTTP journal through the same validators and official decoder, with ordered native-call reconciliation, memory evidence membership, exact journal consumption and failure-vector veto. It does not make network requests or open weights.

The 37-model-call / 4,635-attempted-token block 256 upper bounds remain upper bounds, not exact required totals. Restores are distinct from model calls. Coverage records distinguish partial replay, natural EOS/bare-generation rejection, and incomplete-UTF8 checkpoint rejection; missing conditional coverage is never renamed a pass for a branch that did not occur. The eight evidence memory requests remain outside the driver's HTTP request counter.

Original helper/fixture hashes are retained in `tests/long_context/http/original-sha256.json`; required original inputs and helpers are supplied in this suite. These hashes establish fixture identity, not numerical acceptance of the current binary.

### Observation and build

The observation-only translation unit includes **current** native.c/http.c and forwards the same four native operations. `native_service.length`, `pending`, `pending_position`, `recompute_length`, `checkpoint` and `runtime` still have the required meaning in current `native_materialize`/`native_restore`. The wrapper captures cursor/pending/recompute before the native caller updates those fields; restore receipts remain separate. Request attribution uses the active session history entry during its worker's operation. Idle memory reads occur under the session lock only when neither job nor reset_job is active. No routing/computation body is replaced.

Each forwarded call scans all 131,072 logits. A failed/nonfinite operation aborts affected validation; only failure writes an exclusive full FP32 vector. The observer preserves the original named memory fields and physical-footprint method; allocator guard details remain explicitly unavailable, not inferred. Argument checks occur before observation creation or model/tokenizer access. Only the fixed HTTP profile is admitted, and every argument must be explicit.

`build.py` uses an allowlisted compiler environment, explicit C/Metal flags, contraction disabled, current source pins, compiler digest/version, SDK path/version, developer directory, command list and product/dependency-file digests. It refuses source/product/inventory drift. Build subprocesses have timeouts, core dumps disabled and a 2 MiB per-file output limit. The fixed recipe has fewer than 32 flat products, with a 64 MiB census stop and 256 KiB receipt bound; it is not a general build sandbox. Missing, duplicate, unknown, empty and invalid arguments are checked against the real linked binary with nonexistent input names, requiring exit 2, the refusal marker, and no observation file.

Direct run writers are exclusive: journal 2 MiB, observation 4 MiB, server stdout and stderr 512 KiB each, optional failure vector 512 KiB, validation 16 KiB. The existing supervisor reserves/censuses these and its bounded pipe outputs under a 32 MiB campaign cap. Child captures enforce their caps before writes, and journal/capture census has a 64 MiB hard stop. No model files are copied into artifacts.

### Model-free verification

The explicit self-test runs the owned suite in normal Python and `python -O`; it uses loopback sockets/subprocesses and the required tokenizer package, but no weights, GPU dispatch or downloaded tokenizer unless the tokenizer-only option is supplied:

- Original/promoted deterministic transcripts agree for partial, EOS and UTF-8 branches at block 128 and block 256. Requests, bodies/declarations, journal, generated IDs/text, final ledgers, histories, streams, native work, coverage and totals are compared without timing/state normalization. Corrupt comparator inputs are rejected. This is schedule/checker evidence, not C correctness.
- Real loopback sockets demonstrate the original bodyless GET's advertised two bytes/time-out and promoted zero-byte completion on info/session/request reads. Server reads have deadlines/caps; malformed POST declarations, response caps, malformed-response propagation and peer-observed client cleanup are checked.
- Tiny subprocesses cover listening-record ownership, occupied ports, early and unexpected exits, capture overflow and cleanup. Other regressions cover state drift, work receipt corruption, nonfinite/missing native evidence, journal caps, stale products/sources/inventory and symlink/fresh-directory refusal.
- Offline replay is exercised separately for each conditional branch and rejects corrupted calls, finiteness, state, ordering and coverage.
- Optional explicit tokenizer checking verifies all original frozen framing, capacity-negative IDs/hash and official café byte decode, in both modes.

Model-free verification includes subprocess equivalence of child and offline reference environments and refusal of malformed observing-server arguments before model access. It does not establish C engine correctness, independent numerical acceptance or task quality. Conditional natural EOS-ready and incomplete-UTF8 coverage must be reported as reached/not reached for each actual run; synthetic branch tests are not evidence those branches ran with a real model.
