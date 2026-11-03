# Lunelette

A learning-first Ministral 3 inference engine in C, with a path toward an
optimized, portable runtime.

**Status:** real-weight embedding milestone, not yet model inference. A bounded
safetensors reader and BF16 embedding lookup are implemented; no tokenizer,
transformer, or generation loop.

## Initial target

- Ministral 3 **3B**, text-only; eventually the **8B and 14B** variants.
- Development machine: Apple M5, 32 GB unified memory.
- Readable CPU correctness baseline first; profile before adding Apple-specific
  acceleration. Keep platform-specific code separate when it becomes necessary.
- Python is reference/testing tooling only, not a runtime dependency.

## Build and check

Requires a C11 compiler and Make; on macOS, Apple's command-line developer tools
provide both. No external runtime libraries beyond the C/math libraries.

```sh
make
./build/lunelette --help
make test
make sanitize
```

The first numerical primitive is a stable float32 softmax, with tests for known
outputs, in-place execution, large offsets, and invalid input. These are basic
unit tests—not evidence of model correctness.

## Layout

- `include/`: public C declarations
- `src/`: CLI and portable numerical code
- `tests/`: small deterministic C tests
- `tools/reference/`: reference-validation plan; Python harness not yet implemented
- `../ds4/`: existing sibling DwarfStar checkout, left unchanged and outside
  the Lunelette project

## Safetensors / embedding API

`include/safetensors.h` exposes `st_open`, `st_find`, `st_read`, `st_close`,
`st_bf16_to_f32`, and `st_embedding`. Open validates the entire header and file
layout without loading weight payloads. Find returns a borrowed tensor pointer;
read takes a byte offset relative to that tensor and returns raw little-endian
bytes. Embedding accepts a BF16 rank-2 tensor, a nonnegative in-range token,
and an output buffer of exactly the row width. It reads only that row and
converts BF16 bits to IEEE binary32 via `memcpy`, preserving signed zero,
subnormals, infinities and NaN payload bits (no arithmetic conversion).
Functions return 0 on success, -1 on failure; open provides a diagnostic.
Initialize/open a fresh handle, then close it, including on error. Handles are
not thread-safe. Do not modify public tensor descriptors or reopen live handles.

Deliberate limits (unsupported inputs fail, not silently reinterpret):

- BF16, F32, I64 only; at most 8 dimensions, 4,096 tensors, 1 MiB header.
- Nonempty unique tensor names, up to 255 printable ASCII bytes. JSON escapes
  for quote, slash, backslash and printable ASCII `\uXXXX` are decoded.
  Non-ASCII/control strings are unsupported, including metadata strings.
- Optional string-to-string metadata: at most 64 unique keys, 255-byte keys,
  4,095-byte values. Unknown tensor fields and duplicate fields are rejected.
- Unsigned decimal integer syntax only for shapes/offsets; dimensions <= INT64_MAX.
  Scalars and zero-sized tensors work. Shape products and byte sizes are checked
  for overflow. Offsets must exactly cover the payload, without gaps, overlaps,
  or unindexed trailing bytes; empty tensors must be on interval boundaries.
- Seekable files no larger than `LONG_MAX`; on 32-bit-long systems large model
  shards cannot be opened. Requires IEEE binary32 floats (compile-time checked).
- File must remain unchanged while open. No sharded-index parsing or tokenizer.

Standard C `fseek`/`fread` avoids POSIX mmap and whole-model RAM residency.
The implementation holds a bounded header and descriptor table plus one BF16
row during lookup; the caller owns the float output. Bounded quadratic metadata
validation favors clarity over large-header performance. No new dependencies.

## Optional real-weight validation

Default tests generate synthetic files; external weights are never required.
The optional comparator consumes the saved Python `input_ids` (I64 `[1,N]`)
and compares every embedding bit against `hidden_0` (F32 `[1,N,D]`), including
signed zero. It neither tokenizes nor runs Python or the transformer.

```sh
make integration \
  WEIGHTS=../models/Ministral-3-3B-Instruct-2512-BF16/model-00001-of-00002.safetensors \
  FIXTURES=../experiments/reference/fixtures
# Same five comparisons with ASan/UBSan (after make sanitize):
make BUILD=build/sanitize integration \
  WEIGHTS=../models/Ministral-3-3B-Instruct-2512-BF16/model-00001-of-00002.safetensors \
  FIXTURES=../experiments/reference/fixtures
# Or choose explicit individual fixtures:
./build/compare_embeddings WEIGHT_SHARD FIXTURE.safetensors [MORE_FIXTURES...]
```

The local checkpoint's `model.safetensors.index.json` maps
`language_model.model.embed_tokens.weight` to shard **00001** of two; its shape
is `[131072,3072]`, BF16 (805,306,368 bytes). Shard 00002 is not needed for
embeddings. Five local fixtures match exactly: short (6 tokens), multilingual
(24), code (22), chat (24), longer (150), each with width 3072. This validates
embedding lookup only, not checkpoint-wide compatibility or full model inference.

Synthetic tests cover supported dtypes, scalar/empty shapes, BF16 special bits,
invalid tokens/read bounds, duplicate/missing fields and names, malformed JSON,
overflow, rank limits, invalid offsets, overlap, and every truncation position
of a small valid file. Tests also exercise metadata/escape handling, parser
capacity boundaries, reordered intervals, and empty tensors at valid and interior
boundaries. `make test` generates tiny weights and matching/mismatching fixtures:
the comparator must accept the exact match and reject a one-bit F32 difference.
These tests remain active with `-DNDEBUG`. `make sanitize` runs them under ASan
and UBSan.

A post-open truncation/short-read test is intentionally omitted: file mutation
violates the unchanged-file contract, and stdio may retain previously buffered
payload bytes. Portable C11 has no reliable truncation/buffer-invalidation hook
for that scenario. Truncation before open is tested at every byte position;
read failures are checked by the reader and diagnosed by the comparator.

## Next milestones

Record immutable checkpoint/reference revisions and license details, then add
and compare transformer operations incrementally. Check cached and uncached
decoding separately; matching generated text alone is not a correctness test.

## DwarfStar as a reference

Lunelette is inspired by antirez's [DwarfStar](https://github.com/antirez/ds4):
a focused native engine rather than a universal model framework. Study and reuse
suitable pieces deliberately; do not inherit unrelated model or infrastructure
scope. Inspected only the safetensors database/loading slice in
`../ds4/gguf-tools/deepseek4-quantize.c` (roughly lines 515–665). Its separation
of tensor descriptors, data-base offsets, and lazy shard file access is useful.
The broader database, hash maps and threading are unnecessary here. The parser
and reader are original; no upstream code was reused or modified. Any future adapted
code must retain its required attribution and license notices.

The project's own distribution license has not yet been selected.
