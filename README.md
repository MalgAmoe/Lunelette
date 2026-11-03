# Lunelette

A learning-first Ministral 3 inference engine in C, with a path toward an
optimized, portable runtime.

**Status:** project scaffold, not yet an inference engine. No model loader,
tokenizer, transformer, or generation loop is implemented.

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

## Next milestones

1. Verify the exact Ministral 3 3B checkpoint, license, architecture, tokenizer,
   and compatible Transformers version. Record immutable model/code revisions.
2. Run a short fixed prompt in the trusted implementation and export token IDs,
   intermediate activations, and next-token logits.
3. Choose the initial weight-loading format after inspecting the actual files.
4. Implement and compare the C forward pass incrementally, then cached decoding.
5. Add generation, measure performance, then address quantization and acceleration.

Begin with unquantized weights and explicit numerical tolerances. Check cached
and uncached decoding separately. Do not use matching generated text as the only
correctness test.

## DwarfStar as a reference

Lunelette is inspired by antirez's [DwarfStar](https://github.com/antirez/ds4):
a focused native engine rather than a universal model framework. Study and reuse
suitable pieces deliberately; do not inherit unrelated model or infrastructure
scope. No DwarfStar source has been copied into this scaffold. Any future adapted
code must retain its required attribution and license notices.

The project's own distribution license has not yet been selected.
