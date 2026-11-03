# Bounded Tekken13 tokenizer prototype

## Integration contract

`include/tokenizer.h` is the public API. Compile and link **both**
`src/tokenizer.c` and `src/tokenizer_json.c`; Unicode data is compiled in.
There are no runtime dependencies beyond C11 and the standard library.

```sh
mkdir -p build/tokenizer
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -Iinclude \
  src/tokenizer.c src/tokenizer_json.c tests/test_tokenizer.c \
  -o build/tokenizer/test_tokenizer
build/tokenizer/test_tokenizer
```

The model-independent tests generate their own complete synthetic vocabulary.
They use an exclusively created `test_tokenizer_fixture.tmp` in the working
directory and remove it on success. Existing files at that name are not overwritten.
For optional differential validation, supply explicit paths:

```sh
build/tokenizer/test_tokenizer /path/to/tekken.json /path/to/oracle.bin
```

`open` loads `tekken.json`, not the Hugging Face `tokenizer.json` representation.
It requires the exact supported v13 pattern/config, all 130072 ordinary ranks,
and all 1000 special ranks. Ordinary output IDs are byte-vocabulary ranks + 1000.
`token_str` is never used for ordinary decoding. Duplicate ranks, duplicate byte
strings, missing byte coverage, noncanonical base64 and malformed JSON fail.
Known framing spellings are validated against their explicit public IDs.

Encoding accepts ordinary **strict UTF-8** with explicit byte length; embedded
NUL is supported. It adds no BOS/EOS, chat template, or default system message.
Literal `[INST]`, `<s>`, and other special spellings remain ordinary text. The
caller must explicitly insert control IDs, and must decide the supported chat
layout. System framing IDs are 17/18; user instruction framing is 3/4; assistant
content has no independent role token in this API. BOS/EOS are 1/2. This is not a
claim to implement the checkpoint's full chat/tool/multimodal template.

Decoding concatenates raw bytes across token boundaries, including partial or
invalid UTF-8, and adds a convenience NUL outside the returned length. It rejects
all special IDs rather than guessing whether to skip/render them. The caller
handles EOS and framing explicitly. Results are caller-owned (`free`); failures
return NULL/zero outputs, never partial results. Immutable loaded tokenizers
permit concurrent readers, but close must not race any operation.

## Algorithm and limits

The handwritten splitter implements the seven regex alternatives in order,
including greedy optional-prefix backtracking, overlapping upper/lower letter
classes and whitespace lookahead/backtracking. It is not an ASCII approximation.
BPE first checks the whole piece, then repeatedly merges the lowest-rank adjacent
pair, choosing the leftmost tie. This deliberately favors a readable baseline
over speed; pair ranks are rescanned after each merge.

Bounds are enforced, not inferred from input allocation:

- File: 32 MiB; JSON depth: 32; nodes: 1500000.
- Per object: 64 members (duplicate decoded keys rejected); array: 131072 elements.
- Decoded JSON string: 4096 bytes; numeric/literal span: 128 bytes.
- Ordinary token: 1024 decoded bytes; total vocabulary payload: 16 MiB.
- Encode: 4096 input bytes; decode: 4096 IDs, at most 4 MiB output.
- All size arithmetic is bounded before allocation. Loader memory is bounded by
  file bytes, node capacity (32-byte nodes on the tested 64-bit target), and the
  fixed vocabulary/hash/payload limits. A conservative 160 MiB budget covers
  temporary node-array reallocations and allocation overhead on that target.
  Encode uses about 100 KiB of stack on that target; not intended for tiny stacks.

JSON parsing is a separate private original implementation, not an exported or
modified safetensors parser. It validates JSON number grammar, escapes, surrogate
pairs and UTF-8 even in ignored fields, and rejects trailing data. Unknown fields
are parsed/validated but not interpreted. This is intentionally a bounded
checkpoint-specific format, not a general-purpose tokenizer loader.

## Evidence and provenance

Reference checkpoint:
[mistralai/Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16/tree/b6d637bef2393152b3da2b2fde72eecdee30557e),
revision `b6d637bef2393152b3da2b2fde72eecdee30557e`.

- `tekken.json` SHA256:
  `600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843`.
- `tokenizer.json` SHA256:
  `d5f6046775b112f0e2d456ee9dba450684ab964fe5c4e231599bdc6773028135`.
- Oracle: installed `mistral_common` 1.13.1 `Tekkenizer.encode`, `bos=False`,
  `eos=False`, with `tiktoken` 0.14.0, Python 3.12.11, CPU.
- The Hugging Face JSON's exact splitter pattern and **every** vocabulary byte
  mapping/ID were independently checked against Tekken. Its added-token and
  postprocessing behavior is intentionally not the ordinary-text oracle.
- Unicode data: installed `regex` **2026.9.29**, whose package metadata identifies
  Unicode **18.0.0**. This pins the generated properties, not an assertion that
  every possible Tekken implementation uses that Unicode revision.
  `tests/tokenizer_generate_unicode.py` exhaustively queries nine properties and
  emits 2479 compressed runs. It requires that exact development package version;
  normal builds/tests need no Python. No regex implementation code was copied.
  The source package's Apache-2.0/CNRI-Python notices are retained in
  `src/tokenizer_unicode_LICENSE.txt`. See also [regex](https://pypi.org/project/regex/2026.9.29/)
  and [mistral-common Tekkenizer](https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/tekken.py).

Differential evidence: **32469 records**: 32437 ordinary-text cases with exact
ID equality and byte roundtrip, plus raw-byte decoding of all 130072 ordinary
ranks in 32 chunks. Text covers multilingual/code/numbers, all special spellings,
all whitespace pairs, property-transition neighbors, seeded random sequences,
and every Unicode scalar embedded in an ASCII context. Seed: `0x13B0E`.
This is broad regression evidence, not an exhaustive proof for all strings.
Fixture SHA256:
`1bdbb8654e10fd2f63df263133247c1b30a0ccf2cb40e3a5c25e7933d7d91c2a`.

The comparator is tested with a known match, deliberate ID mismatch and length
mismatch. Synthetic tests also cover leftmost BPE ties, whole-piece shortcut,
lowercase-to-uppercase piece boundaries, raw split UTF-8, malformed UTF-8/JSON,
vocabulary duplicates, base64 rejection, missing ranks, invalid IDs and limits.

Fixture format: magic `TKD1`, little-endian u32 record count; each record has
u32 mode (0 encode+decode, 1 raw decode only), byte length, ID count, raw bytes,
then little-endian u32 IDs. Fixture generation and full provenance/results are
optional research artifacts, not bundled runtime data.

Verification: optimized and NDEBUG builds with `-Werror`, and combined
AddressSanitizer/UndefinedBehaviorSanitizer builds passed the full differential
suite. Apple AddressSanitizer rejected `detect_leaks=1`; checks were rerun with
`detect_leaks=0`. LeakSanitizer is therefore **not** claimed. Shared build files
were intentionally not changed; integrate these translation units/tests explicitly.
