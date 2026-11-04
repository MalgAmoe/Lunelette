# Reference validation artifact index

Independent Transformers CPU FP32 reference fixtures **were generated previously**
and used by the optional C comparators. Those external outputs, generation scripts,
environments and experiment records are not bundled in this directory. This is an
index, not a bundled Python harness or a claim of end-to-end reproducibility.
No installation, download or fixture generation is performed by default tests.

Run all documented commands from the repository root. Supply explicit paths to
weights and fixtures; repository-local ignored `models/` and `fixtures/` paths in
examples are placeholders, not required bundled assets.

## Maintained consumers and contracts

| Optional consumer | Schema, commands and evidence |
|---|---|
| `compare_embeddings`, `make integration` | [Embedding bit comparison](../../docs/validation.md#optional-real-weight-validation) |
| `compare_projections` | [Layer-0 norm and QKV](../../docs/validation.md#optional-layer-0-pre-attention-comparison) |
| `compare_attention` | [Isolated attention](../../docs/validation.md#isolated-attention-comparison) |
| `compare_block`, `make block-integration` | [First block](../../docs/validation.md#first-block-comparison) |
| `compare_forward`, `make forward-integration` | [Full forward](../../docs/validation.md#full-forward-comparison), [current hidden acceptance](../../docs/full-depth-validation.md) |
| `compare_generation` | [Fixed-ID per-step generation](../../docs/validation.md#fixed-id-generation-comparison) |
| `compare_cache` | [Cached versus uncached/reference](../../docs/validation.md#cache-comparison) |
| `compare_metal` | [Metal schema and verification](../../docs/metal.md#verification) |

## Required provenance for supplied artifacts

Record the immutable model and tokenizer revisions/files, inspected implementation
revision, Python/library versions and build configuration, device, execution dtype,
attention mode and thread count. Preserve exact prompts and token IDs, chat-template
settings, special-token handling, positions, cache mode and capture points.
Document tensor names, shapes, contiguous layout, dtype/endianness, fixture hashes,
engine/source/binary identity and exact comparison commands. Use independent
reference outputs, not C-derived expected values. Full-depth historical provenance
and frozen calibration records are described in the linked guides; their external
records are not made available by this index.

Default synthetic tests and comparator match/mismatch/nonfinite self-tests do not
replace real-weight comparisons. Bit equality applies to embedding expansion;
arithmetic tolerances depend on the capture point and precision. Quantized or
other-device execution needs separate evidence. Generate independent references
in a separate process from resident Metal weights to avoid memory pressure.
