# Lunelette project standards

## Purpose and scope

Lunelette is a learning-first Ministral 3 inference engine in C. Understandable
correctness comes before optimization. Initial target: 3B, text-only, macOS
resident Metal/MPS runtime on a base Apple M5 with 32 GB unified memory.
Standalone CPU inference is retired; portability applies to host/core tests only.

Verify architecture details against the exact checkpoint configuration and trusted
implementation. Do not assume all model sizes share every architectural detail.
Describe implemented capabilities accurately: embedding lookup is not full inference.

## C style

- Use C11 and the standard library for shared host code. Isolate and explain
  platform-specific code when a concrete need arises.
- Use four-space indentation, no tabs in C; retain recipe tabs in Makefiles.
- Put function opening braces on a new line; control-flow braces on the same line.
  Use `} else {` for alternatives and braces around control-flow bodies.
- Prefer one statement per line. Do not compress allocation, checks, cleanup, and
  returns onto one line. Readability matters more than line count.
- Use descriptive `snake_case` names; uppercase macros/constants. Short loop
  indices are fine. Avoid cryptic names for parser state, sizes, or offsets.
- Keep functions focused. Separate parsing, validation, and execution when that
  makes their contracts easier to understand; avoid tiny wrappers without purpose.
- Keep headers small and document public contracts. Use existing module naming
  conventions (`lunelette_*`, `st_*`) rather than introducing inconsistent prefixes.
- Comment the reason, numerical assumption, layout, or invariant—not obvious syntax.
  Explain unfamiliar model math and link relevant reference details when useful.
- Prefer roughly 100-column lines; clarity takes priority over a rigid limit.
- Apply style to new/touched code. Do not perform unrelated mass reformatting.

## Safety and API contracts

- Validate external file lengths, shapes, dtypes, offsets, token IDs, and supported
  limits before use. Check arithmetic overflow before multiplication/addition.
- Treat byte order and tensor layout explicitly. Use `memcpy` for float bit
  conversion; avoid aliasing violations, unaligned casts, and host-endian assumptions.
- Make ownership, lifetimes, allocation/free responsibility, and borrowed pointers
  explicit. Document initialization, close behavior, and thread-safety constraints.
- Specify failure behavior, including whether output may be partially modified.
- Return actionable errors: identify the file/tensor and failed expectation where
  possible. Unsupported input must fail clearly rather than be silently accepted.
- Keep cleanup paths simple and complete. `goto cleanup` is acceptable in C.
- Use explicit bounded-format limitations instead of pretending to support every
  safetensors/model variant. Document them and test their boundaries.

## Architecture and performance

- Keep model execution separate from weight-format parsing and platform kernels.
- No speculative backend frameworks, generic tensor systems, or dependencies.
- Python is development/reference tooling, never a runtime requirement for C.
- Keep the existing resident Metal/MPS engine; measure before optimizing.
  Use frozen native regression controls and isolated operator checks for changes,
  not a second C inference engine. CPU tokenization, parsing, metadata, generation policy and
  YaRN preparation remain required. Do not introduce backend dispatch wrappers.
- Avoid loading entire model files when a bounded read suffices. Do not introduce
  complex caching or allocators before measurements justify them.
- Use upstream projects as references, not templates to copy wholesale.
  Preserve licenses and attribution whenever adapting upstream code.
- Keep this repository self-contained. Do not reference developer-specific paths
  or sibling checkouts in code, instructions, or documented commands. Workspace
  arrangements belong in workspace instructions, outside this repository.

## Checkpoint contract

- Support only the official HF-layout mixed FP8/BF16 single `model.safetensors`.
  No second-shard API, BF16-checkpoint fallback, or row permutation.
- Reconstruct projection weights as FP8 decoded to FP32 times BF16 scale widened
  to FP32, with no BF16 product rounding. Validate scalar positive finite weight
  and activation scales; activation scales are unused in FP32 execution (not W8A8).
- Keep 182 FP8 projections, 53 BF16 norms, BF16 embedding/tied head and context256.
  Resident weights remain 13,716,025,344 bytes; preserve the 24 GiB peak guard.
- Fresh fixtures must come from this checkpoint and true cached decode. Archived
  BF16 fixtures are historical, not an oracle for the current runtime.

## Tests and reference evidence

- Use always-active `CHECK`-style assertions in tests. Never put required setup,
  I/O, or tested calls inside standard `assert`: NDEBUG removes them.
- Add deterministic synthetic tests for normal cases, malformed input, bounds,
  overflow, and relevant failure paths. Test both acceptance and rejection.
- Test the comparator itself with a known match and deliberate mismatch.
- `make test` must work without downloaded models, Python, or network access.
- Keep real-weight comparisons optional and provide explicit paths/commands.
- Exact equality is appropriate for embedding lookup/BF16 expansion. Arithmetic
  comparisons require justified tolerances and reported absolute/relative errors.
- Record fixture provenance: model revision, prompt/token IDs, tokenizer settings,
  library versions, dtype/device, tensor shapes, layout, and capture points.
- Generated text alone does not establish correctness. Use immutable native
  full-vector regression controls, isolated operator tests and prefill/decode/reset
  checks; assess task quality and matched-work performance separately.
- Retain agreed independent PyTorch-reference checks using frozen fixtures; this
  does not imply regenerating references or launching a Python model. Preserve
  finite-value checks and `abs(error) <= 2e-4 + 1e-4 * abs(reference)`; do not
  relabel strict failures or refresh goldens/tolerances to erase discrepancies.
- Keep independent checking, candidate/control regression and task behavior separate.
  The native control is provisional, not an independent oracle or quality approval.
  Match inputs, trajectories and execution conditions; check full finite logits,
  relevant intermediates and continuation/reset/session state. Bitwise contracts
  require preserved operations and rounding/reduction order, not just FP32 labels.
  Explicitly revise the contract before interpreting changed precision/order.
- Known independent mismatches, established unchanged from comparable artifacts and
  metrics rather than failure counts, do not automatically reopen investigation or
  block authorized engineering. Distinguish nonzero exits caused by infrastructure,
  missing coverage or new mismatches; never blanket-ignore comparator failure.
- Withhold acceptance for candidate/control contract regressions and debug within
  authorized scope. Stop affected validation for nonfinite values, memory errors or
  invalid session/KV state. Flag new/materially worse independent discrepancies.
- A check does not authorize expanded cross-framework tracing. Before proposing it,
  state what is new, the suspected defect, discriminating experiment, stopping
  condition and decision impact; ask approval before expanding the study. This
  supersedes the earlier blanket restriction on routine independent checks.
- Preserve task prompts, scoring and control failures. Matching selected tokens is
  not the only numerical test; faster failed/truncated tasks are not faster successes.
- Do not claim untested GPU, quantization, long-context, batching, or vision support.

## Verification and completion

Run from this repository, proportionate to the change:

```sh
make test
make sanitize
make BUILD=build/ndebug CFLAGS='-std=c11 -O2 -DNDEBUG -Wall -Wextra -Wpedantic' test
git diff --check
```

For weight-reader/embedding changes, when optional resources have been supplied
under the repository-local ignored `models/` and `fixtures/` directories
(these are not bundled or downloaded automatically):

```sh
make integration \
  WEIGHTS=models/Ministral-3-3B-Instruct-2512-FP8/model.safetensors \
  FIXTURES=fixtures/fp8
```

Use separate build directories when changing compiler flags so stale binaries do
not hide failures. Report checks actually run and any skipped checks/gaps.
Inspect diffs, including untracked files, before committing. Preserve unrelated
changes. Keep commits focused; do not commit weights, virtual environments, build
outputs, or exploratory research. Ask before installing dependencies or downloading
large files. Never push/publish without explicit permission.
