# Frozen native regression

`make test` now includes `make metal-reference-test` on macOS. This is offline, model-free GPU testing; no Python, model or network is required. A machine without a Metal device prints SKIP with zero GPU cases, not PASS.

The project/norm candidate comes from the same embedded production shader as the runtime. The reference is the independent, hash-pinned static source in `tests/frozen/`; it is never copied from the candidate during build or test. The two libraries use the production compile options. All non-NaN output bits must agree, including zero signs and infinities; NaN classifications, not payloads, must agree. Inputs and output guards must remain untouched. Exactly representable integer dot products provide an additional analytical oracle. Sizes include SIMD boundaries and tails and the 3072-wide model norm. This is not full-model numerical validation, nor an alternative CPU engine.

The unchanged-arithmetic contract is bitwise native parity. There is no approved relaxed native contract for future precision changes. The existing independent Python comparison remains `2e-4 + 1e-4*abs(reference)` and is not relaxed by these tests. DS4's model-specific top-K overlap limits and 8.0 score bound are not appropriate limits for Ministral and have not been adopted.

## One-operation diagnostic helper

For external, contiguous little-endian FP32 residual rows and FP32 norm weights:

```sh
make metal-norm-diagnostic INPUT=residual.f32 NORM_WEIGHTS=norm-weight.f32 \
  OUTPUT=new-output.f32 COLUMNS=3072 ROWS=248
```

This dispatches only the maintained custom norm kernel, using eps 1e-5, safe math and precise functions. It validates lengths, input immutability and output guards; output creation is exclusive. It does **not** judge numerical agreement. No device returns SKIP/77 (Make reports failure), so an automation must not count it as a pass.

The standalone test executable also accepts an explicit candidate-source argument for deliberate fault injection:

```sh
build/test_metal_reference tests/frozen/metal-baseline-v1.metal faulty-candidate.metal
```

Only the candidate may be overridden; a damaged/missing frozen source fails.

## Full-model native logit regression

`make native-self-test` is portable and included in `core-test` (also sanitize and NDEBUG). `make build/compare_native` builds the optional macOS model runner. Neither command captures or updates a reference. The runner uses only public runtime APIs, with no production instrumentation or dependency additions.

Explicit capture into an already-created **new empty directory**:

```sh
build/compare_native capture MODEL.safetensors mps NEW_DIRECTORY \
  FIXTURES/prefill6.ids FIXTURES/prefill6.safetensors \
  FIXTURES/prefill24.ids FIXTURES/prefill24.safetensors \
  FIXTURES/prefill150.ids FIXTURES/prefill150.safetensors \
  FIXTURES/prefill248.ids FIXTURES/prefill248.safetensors --strategy=resident-fp32
```

Use `custom` for the separate custom baseline. Capture creates files exclusively; existing files fail, never overwrite. Freeze and hash the artifacts and provenance **before** invoking `check` in a fresh process with the same arguments and reference directory. `check` never writes a baseline and runs two reset cycles: 64 full vectors per mode. A third process can repeat the same check without recapture. Do not run two resident model processes concurrently. The unchanged production memory guard remains final; no retry, headroom change or fallback is permitted.

The eight `generated` IDs in each existing FP8 oracle fixture are used only for teacher forcing: prefill plus seven true cached decodes. Oracle logits are NOT copied. Native outputs alone form this regression reference. Missing or malformed IDs, steps, modes, context, shapes, formats and nonfinite vectors fail; no empty comparison or missing-file fallback succeeds. The complete trajectory must match. The binary format is fixed little-endian uint32 header/IDs and IEEE FP32 logits: magic `0x324e554c` (LUN2), strategy (0 resident / 1 arithmetic / 2 lookup), mode (0 MPS / 1 custom), context 256, prefix length, steps 8, vocabulary 131072, prompt IDs, eight forced IDs, eight full logit vectors. Each mode archives about 16 MiB, outside the engine repository.

Acceptance for **unchanged Ministral arithmetic** is finite values on both sides, exact bitwise logits (score tolerance zero), identical top 1 and full top 5/20/64 sets at every step. Ties sort by increasing vocabulary ID. Per-step output reports finite counts separately, exactness, both top 1 IDs, top-K intersection counts, max absolute error, RMS and cosine with double accumulation. Bitwise repeatability must be established in separate processes before accepting this contract. This is not DS4's 8.0 score limit. An FP16 or other precision-changing candidate needs a separately agreed numerical contract; this test does not validate one.

A provenance manager should hash the entire checkpoint **before construction**, verify reference manifest/hash and all artifact hashes, and record separate reference/candidate source snapshots, environment, command, tokens and shapes. Do not require candidate source hashes to equal frozen reference source hashes: future candidate changes are the purpose of this test. Direct C commands validate the binary structure and trajectory, not JSON/SHA provenance; optional external stdlib tooling can supply that archival layer without making Python a runtime or portable test dependency. Capture must remain an explicit action, not a check side effect. Never auto-update a failed model baseline.

Offline file validation (no model/device needed):

```sh
build/test_native --files REFERENCE.bin CANDIDATE.bin mps 6 --strategy=resident-fp32
```

The offline build shares the exact C metrics/format reader with the model runner. Its unit tests cover known swaps, ties against an independent double rank oracle, nonfinite values, same-argmax changes outside top 64 and malformed headers/truncated vectors. Full-file negative validation should additionally preserve logs for NaN, Inf, top 1 swap, tail change, missing step, truncation, mode/token/context/shape/ format corruption and manifest hash corruption. These are logit/artifact mutations, not engine KV perturbations, and must not be described as KV injection.

These baselines are usable regression evidence, accepted on repeatability and component tests, not an independent full-model mathematical oracle. The existing strict Python comparator, failure logs and thresholds remain unchanged. The known 128-comparison independent-reference panel gap (64 per mode) and longer-case numerical disagreements remain open; neither native parity nor generated text establishes general correctness. No footprint, performance or benchmark claim is made.

## Strategy identity and compatibility

Both comparators accept a final `--strategy=resident-fp32`, `--strategy=compact-fp8-arithmetic` or `--strategy=compact-fp8-lookup`. Omission deliberately means fixed resident FP32, independent of the product default. Native capture/check and offline `--files` share this policy. `make native-regression` passes resident explicitly; `STRATEGY=compact-fp8-arithmetic` or `STRATEGY=compact-fp8-lookup` overrides it. Use separate reference directories per strategy and prefill mode.

Explicit-strategy native captures use LUN2. Legacy capture invocations without a selector retain LUN1 and fixed resident selection so existing LUN1-only provenance managers remain usable; they cannot capture compact strategies. New callers should always select explicitly. LUN1 (`0x314e554c`) has no strategy word and is interpreted exclusively as resident; its remaining layout is unchanged. Both formats are readable offline, including mixed-version resident pairs. Unknown formats, invalid strategy words and requested/artifact strategy mismatches fail; check validates every artifact before model construction. Native acceptance remains same-strategy strict finite bitwise parity, never automatic cross-strategy acceptance. Runtime memory queries report actual strategy and must match the request before execution.

Strategy binding does not add checkpoint hashes to the binary: the external provenance manager still owns model identity, source identity, precision, layout and environment checks. Context remains 256 with the existing fixed trajectories; compact selector support is not new numerical evidence or long-context/service coverage. Independent-reference comparison supports all three strategies with unchanged tolerances and teacher forcing, not native cross-strategy approval.
