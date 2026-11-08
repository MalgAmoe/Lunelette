# Tokenizer reference

## Integration contract

[include/tokenizer.h](../../include/tokenizer.h) is the public API. Open/encode/decode return boolean 1 on success and 0 on failure, unlike the numerical engine APIs’ 0/-1 convention. Compile and link **both** `src/tokenizer.c` and `src/tokenizer_json.c`; Unicode data is compiled in. There are no runtime dependencies beyond C11 and the standard library.

```sh
make BUILD=build/tokenizer build/tokenizer/test_tokenizer
(cd build/tokenizer && ./test_tokenizer)
```

The model-independent tests generate their own complete synthetic vocabulary. They use an exclusively created `test_tokenizer_fixture.tmp` in the working directory and remove it on success. Existing files at that name are not overwritten. For optional differential validation, supply explicit paths:

```sh
build/tokenizer/test_tokenizer /path/to/tekken.json /path/to/oracle.bin
```

`open` loads `tekken.json`, not the Hugging Face `tokenizer.json` representation. It requires the exact supported v13 pattern/config, all 130072 ordinary ranks, and all 1000 special ranks. Ordinary output IDs are byte-vocabulary ranks + 1000. `token_str` is never used for ordinary decoding. Duplicate ranks, duplicate byte strings, missing byte coverage, noncanonical base64 and malformed JSON fail. Known framing spellings are validated against their explicit public IDs.

Encoding accepts ordinary **strict UTF-8** with explicit byte length; embedded NUL is supported. It adds no BOS/EOS, chat template, or default system message. Literal `[INST]`, `<s>`, and other special spellings remain ordinary text. The caller must explicitly insert control IDs, and must decide the supported chat layout. System framing IDs are 17/18; user instruction framing is 3/4; assistant content has no independent role token in this API. BOS/EOS are 1/2. This is not a claim to implement the checkpoint's full chat/tool/multimodal template.

Decoding concatenates raw bytes across token boundaries, including partial or invalid UTF-8, and adds a convenience NUL outside the returned length. It rejects all special IDs rather than guessing whether to skip/render them. The caller handles EOS and framing explicitly. Results are caller-owned (`free`); failures return NULL/zero outputs, never partial results. Immutable loaded tokenizers permit concurrent readers, but close must not race any operation.

## Algorithm and limits

The handwritten splitter implements the seven regex alternatives in order, including greedy optional-prefix backtracking, overlapping upper/lower letter classes and whitespace lookahead/backtracking. It is not an ASCII approximation. BPE first checks the whole piece, then repeatedly merges the lowest-rank adjacent pair, choosing the leftmost tie. The implementation uses an indexed binary min-heap, not a full pair-rank rescan after each merge. One node per original byte offset carries previous/next links, end offset, current heap position and pair rank. Ordering is rank first, original byte offset second, preserving leftmost ties. Merging extends the left node over its right neighbor, repairs links, marks the right node inactive at sentinel rank and updates the left and preceding pairs. Heap repair updates positions in both directions; there are no stale queued candidates. When no finite-ranked pair remains, traversal of surviving links emits ordinary ranks + 1000. Scratch is one checked allocation of `piece_bytes × (sizeof(tk_merge) + sizeof(size_t))`, freed per piece; whole-piece vocabulary hits bypass it.

Bounds are enforced, not inferred from input allocation:

- File: 32 MiB; JSON depth: 32; nodes: 1500000.

- Per object: 64 members (duplicate decoded keys rejected); array: 131072 elements.

- Decoded JSON string: 4096 bytes; numeric/literal span: 128 bytes.

- Ordinary token: 1024 decoded bytes; total vocabulary payload: 16 MiB.

- `lunelette_tokenizer_encode`: 4096 input bytes. `lunelette_tokenizer_encode_with_limit`: explicit input limit must be exactly `LUNELETTE_TOKENIZER_MAX_INPUT` (4096) or `LUNELETTE_TOKENIZER_LONG_INPUT` (32768); other limits reject. Long admission does not truncate token output or enlarge decode limits.

- Decode: 4096 IDs, at most 4 MiB output, regardless of encode profile.

- All size arithmetic is bounded before allocation. Loader memory is bounded by file bytes, node capacity (32-byte nodes on the tested 64-bit target), and the fixed vocabulary/hash/payload limits. A conservative 160 MiB budget covers temporary node-array reallocations and allocation overhead on that target. The encoder retains a 4097-character local workspace (65552 bytes on the tested 64-bit layout), even for long calls. Above 4096 input bytes it additionally allocates `(length + 1)` character records on the per-call heap. Output allocation is at most one uint32 ID per input byte (a minimal allocation for empty input). At 32768 bytes, character heap, result and worst-piece BPE scratch are respectively 524304, 131072 and 1572864 bytes on that layout. These are simultaneous bounded payloads, not a process-memory peak; not intended for tiny stacks.

JSON parsing is a separate private original implementation, not an exported or modified safetensors parser. It validates JSON number grammar, escapes, surrogate pairs and UTF-8 even in ignored fields, and rejects trailing data. Unknown fields are parsed/validated but not interpreted. This is intentionally a bounded checkpoint-specific format, not a general-purpose tokenizer loader.

## Evidence and provenance

Reference checkpoint: [mistralai/Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16/tree/b6d637bef2393152b3da2b2fde72eecdee30557e), revision `b6d637bef2393152b3da2b2fde72eecdee30557e`.

- `tekken.json` SHA256: `600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843`.

- `tokenizer.json` SHA256: `d5f6046775b112f0e2d456ee9dba450684ab964fe5c4e231599bdc6773028135`.

- Oracle: installed `mistral_common` 1.13.1 `Tekkenizer.encode`, `bos=False`, `eos=False`, with `tiktoken` 0.14.0, Python 3.12.11, CPU.

- The Hugging Face JSON's exact splitter pattern and **every** vocabulary byte mapping/ID were independently checked against Tekken. Its added-token and postprocessing behavior is intentionally not the ordinary-text oracle.

- Unicode data: installed `regex` **2026.9.29**, whose package metadata identifies Unicode **18.0.0**. This pins the generated properties, not an assertion that every possible Tekken implementation uses that Unicode revision. [tests/tokenizer_generate_unicode.py](../../tests/tokenizer_generate_unicode.py) exhaustively queries nine properties and emits 2479 compressed runs. It requires that exact development package version; normal builds/tests need no Python. No regex implementation code was copied. The source package's Apache-2.0/CNRI-Python notices are retained in [src/tokenizer_unicode_LICENSE.txt](../../src/tokenizer_unicode_LICENSE.txt). See also [regex](https://pypi.org/project/regex/2026.9.29/) and [mistral-common Tekkenizer](https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/tekken.py).

## Fixture and comparator contract

The optional differential fixture is external, not bundled runtime data. Its ordinary-text oracle is `mistral_common` 1.13.1 `Tekkenizer.encode(bos=False, eos=False)`, with `tiktoken` 0.14.0, Python 3.12.11 and CPU. The pinned fixture SHA256 is `1bdbb8654e10fd2f63df263133247c1b30a0ccf2cb40e3a5c25e7933d7d91c2a`; it contains 32469 records (32437 ordinary encode/roundtrip records and 32 raw-decode chunks covering all ordinary ranks). Coverage includes multilingual/code/numbers, literal special spellings, whitespace pairs, Unicode property transitions, seeded random text (`0x13B0E`) and every Unicode scalar in ASCII context. This describes fixture provenance, not current-run attestation or proof for all strings.

The comparator is tested with a known match, deliberate ID mismatch and length mismatch. Synthetic tests also cover leftmost BPE ties, whole-piece shortcut, lowercase-to-uppercase piece boundaries, raw split UTF-8, malformed UTF-8/JSON, vocabulary duplicates, base64 rejection, missing ranks, invalid IDs and limits.

Fixture format: four literal bytes `TKD1`, then a little-endian u32 record count; each record has little-endian u32 `mode`, byte length and ID count, then exactly that many raw bytes and little-endian u32 IDs. Mode 0 compares exact encoded IDs and then raw decoded bytes; mode 1 only compares raw decoded bytes, allowing non-UTF-8 token payloads. The reader requires record count < 1000000, mode <= 1, byte length <= 4 MiB, ID count <= 4096 and exact EOF without trailing data. Mode 0 still uses the default 4096-byte encoder; the format’s larger byte bound does not bypass API admission. A mismatch is failure, not permission to regenerate the oracle. The comparator checks both length and byte/ID content and frees all returned buffers.

## Maintained integration and boundaries

The Makefile owns `TOKENIZER_SOURCES` (`src/tokenizer.c`, `src/tokenizer_json.c`) and the compiled Unicode header. Tokenizer, allocation-failure and heap-BPE tests are part of `core-test`; no model, official tokenizer or Python package is needed for synthetic coverage. See [testing](../development/testing.md) for exact execution/dependency boundaries. Do not describe this as an integration-only prototype or manually regenerate Unicode to run normal builds.

[tests/test_tokenizer_bpe.c](../../tests/test_tokenizer_bpe.c) compares heap behavior to an independent frozen rescan rule, including non-monotonic ranks and adjacent repairs. Optional differential checking uses `test_tokenizer` with explicit compatible `tekken.json` and TKD1 paths. Unicode generation is a separate development operation using the exact pinned package, never a runtime step. Host sanitizers do not establish leak checking on platforms where LeakSanitizer is unavailable.

Service framing, incremental UTF-8 carry and special-ID handling belong to [service](service.md); the tokenizer itself neither manufactures chat history nor skips controls. The [fixed native tools](tools.md) have separate structural framing. Caller-owned output lifetimes and immutable-reader concurrency must be retained when integrating either encode API.
