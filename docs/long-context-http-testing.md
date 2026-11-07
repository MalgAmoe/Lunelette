# Bounded long-context HTTP repeat

The optional entrypoint is `tools/long_context_http.py`. It promotes the original
D schedule/provider, not a second inference implementation or supervisor. Only
D-256/mps/compact-fp8-arithmetic is executable: context4352, prefix-reuse on,
no prerequisites, retries, default model paths, downloads or installations.

**Acceptance boundary:** model-free checks, a fresh observing-server build and
one real-weight HTTP child completed. The first campaign remained **stopped**:
its offline validator used a direct virtualenv launcher while the child used the
pinned resolved interpreter with virtualenv paths added. Their package censuses
differed by the base environment's `pip`, not tokenizer dependency versions.
The shared bootstrap correction preserves the exact original child environment;
main's corrected offline replay of the retained real artifacts passed, without
rerunning inference or rewriting the stopped receipt. Subprocess regressions cover
both launch routes. A fresh full campaign with that correction has not been run.
This is schedule/state/work smoke evidence, not numerical or task-quality approval.

## Explicit commands

Use an already-installed official tokenizer environment. `--python` is mandatory
and must provide `mistral_common==1.13.1`; missing/wrong versions fail before
building or launching. The immutable tokenizer SHA256 remains
`600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843`.
No C-tokenizer self-consistency is substituted for official framing/decode.
Each run records Python and installed-distribution versions in
`reference-environment.json`; offline replay rejects version drift. The executable
and virtualenv configuration are pinned, **installed package contents are not**.
Use an exclusively owned, unchanged environment: in-place package modification
without a version change is a remaining reproducibility gap.

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

Build/output/input paths reject symlinks and parent traversal; build/output must
be outside the source tree, fresh and non-overlapping. On systems where `/tmp`
is a symlink, use its canonical directory instead. The explicit virtualenv
interpreter is the sole symlink exception: the supervisor pins its resolved
executable and virtualenv configuration; child/replay reject target drift.
There is no implicit model location or port selection.

## Preserved contracts and intentional changes

`tests/long_context/http/fixtures/` contains the four byte-exact texts, their
original official framed IDs and the 4,594-token capacity-negative fixture.
`original-sha256.json` identifies the original inputs/helpers. The original
framing provenance (Python 3.12.11, mistral_common 1.13.1, tokenizer hash and ID
hashes) remains in `fixtures/texts.json` and `negative.json`.
The original driver is retained solely as a hashed, injected test oracle under
`original/`; no historical workspace is needed at runtime.

- `driver.py`: schedule, validators, ledger, stream terminals, conditional
  checkpoint branches, request bounds, work schema3/work2 and envelopes preserved.
  Historical engine-pin checks become immutable fixture checks plus mandatory
  current-source/product receipts at launch. Journal census now measures its
  output directory rather than the historical source directory.
- `SocketTransport.request`: the reviewed adapter's only wire change is retained:
  absent an explicit declaration, non-POST requests declare Content-Length 0.
  Explicit malformed POST lengths are untouched. Response cap is 262,144 bytes;
  every connection closes on success and exception.
- `provider.py`: official Tekkenizer framing/byte decoding, streaming UTF-8 tails,
  EOS policy, observation reads, memory requests and separate restore accounting
  preserved. Version detection is explicit. The historical fixture-regeneration
  entrypoint is removed; fixtures are immutable. `check_tokenizer.py` checks them
  against an explicitly supplied tokenizer without importing model libraries.
- `child.py`: only the original child startup/capture/cleanup logic is adapted.
  Evidence/tokenizer exists before server memory admission; readiness requires
  this child's own listening record. Its server inherits the supervisor-owned
  process group. No historical outer launcher or owned.py is imported/copied.
- `campaign_http.py`: composes exactly one job with the existing, unmodified
  `supervisor/{manifest,supervisor}.py`. The owner/process limit is 1200 seconds,
  and the total campaign deadline is also 1200 seconds. No retries/resume.
- `replay.py`: offline callback replays the full ordered HTTP journal through the
  same validators and official decoder, with ordered native-call reconciliation,
  memory evidence membership, exact journal consumption and failure-vector veto.
  It does not make network requests or open weights.

The 37-model-call / 4,635-attempted-token block256 upper bounds remain upper
bounds, not exact required totals. Restores are distinct from model calls.
Coverage records distinguish partial replay, natural EOS/bare-generation
rejection, and incomplete-UTF8 checkpoint rejection; missing conditional coverage
is never renamed a pass for a branch that did not occur. The eight evidence
memory requests remain outside the driver's HTTP request counter.

Original helper/fixture hashes are retained in `original-sha256.json`; the
function mapping above describes intentional adaptations. Promotion review diffs
are external engineering evidence, not a runtime dependency.

## Current-source observation audit and build

The observation-only translation unit includes **current** native.c/http.c and
forwards the same four native operations. `native_service.length`, `pending`,
`pending_position`, `recompute_length`, `checkpoint` and `runtime` still have the
required meaning in current `native_materialize`/`native_restore`. The wrapper
captures cursor/pending/recompute before the native caller updates those fields;
restore receipts remain separate. Request attribution uses the active session
history entry during its worker's operation. Idle memory reads occur under the
session lock only when neither job nor reset_job is active. No routing/computation
body is replaced.

Each forwarded call scans all 131,072 logits. A failed/nonfinite operation aborts
affected validation; only failure writes an exclusive full FP32 vector. The
observer preserves the original named memory fields and physical-footprint
method; allocator guard details remain explicitly unavailable, not inferred.
Argument checks occur before observation creation or model/tokenizer access.
Only the promoted profile is admitted, and every argument must be explicit.

`build.py` uses an allowlisted compiler environment, explicit C/Metal flags,
contraction disabled, current source pins, compiler digest/version, SDK path/
version, developer directory, command list and product/dependency-file digests.
It refuses source/product/inventory drift. Build subprocesses have timeouts,
core dumps disabled and a 2MiB per-file output limit. The fixed recipe has fewer
than 32 flat products, with a 64MiB census stop and 256KiB receipt bound; it is not
a general build sandbox. Missing, duplicate, unknown, empty and invalid
arguments are checked against the real linked binary with nonexistent input
names, requiring exit 2, the refusal marker, and no observation file.

Direct run writers are exclusive: journal 2MiB, observation 4MiB, server stdout
and stderr 512KiB each, optional failure vector 512KiB, validation 16KiB. The
existing supervisor reserves/censuses these and its bounded pipe outputs under a
32MiB campaign cap. Child captures enforce their caps before writes, and journal/
capture census has a 64MiB hard stop. No model files are copied into artifacts.

## Model-free verification

The explicit self-test runs the owned suite in normal Python and `python -O`:

- Original/promoted deterministic transcripts agree for partial, EOS and UTF-8
  branches at block128 and block256. Requests, bodies/declarations, journal,
  generated IDs/text, final ledgers, histories, streams, native work, coverage and
  totals are compared without timing/state normalization. Corrupt comparator
  inputs are rejected. This is schedule/checker evidence, not C correctness.
- Real loopback sockets demonstrate the original bodyless GET's advertised two
  bytes/time-out and promoted zero-byte completion on info/session/request reads.
  Server reads have deadlines/caps; malformed POST declarations, response caps,
  malformed-response propagation and peer-observed client cleanup are checked.
- Tiny subprocesses cover listening-record ownership, occupied ports, early and
  unexpected exits, capture overflow and cleanup. Other regressions cover state
  drift, work receipt corruption, nonfinite/missing native evidence, journal caps,
  stale products/sources/inventory and symlink/fresh-directory refusal.
- Offline replay is exercised separately for each conditional branch and rejects
  corrupted calls, finiteness, state, ordering and coverage.
- Optional explicit tokenizer checking verifies all original frozen framing,
  capacity-negative IDs/hash and official café byte decode, in both modes.

The suite has17 test methods (with branch/corruption subtests), including real
subprocess equivalence of child and offline reference environments. No dependencies
were installed. The observing server refused malformed arguments before model
access. The real child and corrected offline replay are described above; neither
establishes task quality or independent numerical correctness.
