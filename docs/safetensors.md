# Safetensors and embedding contracts

All shell commands in this guide run from the repository root, not `docs/`.
External weights and fixtures are optional, caller-supplied, and not bundled or
downloaded automatically. Paths in examples are repository-local placeholders.

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

- F8_E4M3 (one byte), BF16, F32, I64 only; at most 8 dimensions, 4,096 tensors, 1 MiB header.
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

`st_load_bf16` is a reusable bounded tensor-to-F32 helper: the caller owns the
float buffer, validates the desired shape, and passes the exact whole-tensor
count (not a capacity). It uses 64 KiB (65,536 bytes) of scratch space, validates dtype,
count and handle membership, and may partially modify output on I/O failure.
It allocates nothing and retains no state. See the header contract.
See [optional validation and reader test coverage](validation.md).

## FP8 numeric reads

`st_f8_e4m3_to_f32` implements the finite E4M3 variant: signed zero,
subnormals in units of 2^-9, finite extrema +/-448 and NaN bytes 0x7f/0xff.
Only the exact safetensors dtype spelling `F8_E4M3` is accepted.
`st_load_fp8` fills exactly the caller's FP32 destination using 64 KiB scratch,
rejecting NaNs and nonfinite reconstructed products. The positive finite FP32
scale is multiplied after decoding, without intermediate BF16 rounding. Invalid
arguments leave output unchanged; I/O/nonfinite failures may leave partial data.
The checkpoint-specific module owns companion names, scalar policy and shapes.
