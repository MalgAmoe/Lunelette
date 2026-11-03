# Lunelette project standards

## Purpose and scope

Lunelette is a learning-first Ministral 3 inference engine in C. Understandable
correctness comes before optimization. Initial target: 3B, text-only, portable
CPU baseline on a base Apple M5 with 32 GB unified memory. Future targets include
8B/14B, Mac acceleration, and portable edge deployment.

Verify architecture details against the exact checkpoint configuration and trusted
implementation. Do not assume all model sizes share every architectural detail.
Describe implemented capabilities accurately: embedding lookup is not full inference.

## C style

- Use C11 and the standard library for the portable baseline. Isolate and explain
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
- Start with transparent numerical kernels; measure before optimizing. Preserve
  a simple correctness baseline when adding accelerated implementations.
- Avoid loading entire model files when a bounded read suffices. Do not introduce
  complex caching or allocators before measurements justify them.
- Use upstream projects as references, not templates to copy wholesale.
  Preserve licenses and attribution whenever adapting upstream code.
- Keep this repository self-contained. Do not reference developer-specific paths
  or sibling checkouts in code, instructions, or documented commands. Workspace
  arrangements belong in workspace instructions, outside this repository.

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
- Generated text alone does not establish correctness. Compare intermediate
  activations/logits and cached versus uncached execution as those features land.
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
  WEIGHTS=models/Ministral-3-3B-Instruct-2512-BF16/model-00001-of-00002.safetensors \
  FIXTURES=fixtures/reference
```

Use separate build directories when changing compiler flags so stale binaries do
not hide failures. Report checks actually run and any skipped checks/gaps.
Inspect diffs, including untracked files, before committing. Preserve unrelated
changes. Keep commits focused; do not commit weights, virtual environments, build
outputs, or exploratory research. Ask before installing dependencies or downloading
large files. Never push/publish without explicit permission.
