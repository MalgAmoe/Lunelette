# Provisional bounded full-depth validation

This is a test acceptance rule, not a production math change or universal accuracy
claim. It applies only to `compare_forward` hidden_13 (residual after layer 12)
and hidden_26 (final normalized hidden), pinned Ministral 3 3B BF16 checkpoint
expanded to FP32, batch one, uncached eager CPU text, positions starting at zero,
1–256 tokens. Real evidence reaches 150 tokens, not 256. Isolated projections,
attention and layer comparisons are unchanged. Last-token logits must still pass
`|a-b| <= 2e-4 + 1e-4 |b|` for **every** vocabulary element. Any nonfinite operand
rejects. Hidden strict elementwise failure counts remain visible diagnostics and
are not used as the full-depth pass/fail decision.

## Equations and frozen limits

For each token independently, let a be actual, b reference, width D=3072:

- relative RMS = `sqrt(sum((a-b)^2)/D) / max(sqrt(sum(b^2)/D), 1e-6)`;
  this is relative L2 above the explicit reference RMS floor.
- cosine distance = `max(0, 1 - dot(a,b)/(norm(a)*norm(b)))`;
  both exactly zero gives zero distance; exactly one zero rejects regardless of
  other metrics. Near-zero nonzero vectors use the same cosine equation.
- absolute component error = `max_j |a_j-b_j|`.

Sums use double precision; each row must satisfy **all** three limits. No sequence
averaging may dilute a bad token. Equality at a limit passes. Strict diagnostics
also include overall RMSE, maximum absolute/relative error and allowance ratio.

| Metric | Calibration maximum | Frozen limit (4x) |
|---|---:|---:|
| Relative RMS | 0.00033568723258481294 | 0.0013427489303392518 |
| Cosine distance | 5.5185654379563687e-8 | 2.2074261751825475e-7 |
| Absolute component error | 0.00420379638671875 | 0.016815185546875 |

The plan preceded new measurements. Limits were frozen at
**2026-11-03T19:45:33.801929+00:00**, before held-out reference outputs. Calibration
used only trusted PyTorch variation: official batched decoder Linear versus
row-wise `F.linear`, with unchanged weights and all other model math, on the
existing short (6), code (22), multilingual (24), and longer (150) token inputs.
Maxima pool both capture points and all rows. Fourfold headroom permits modest
unseen reduction-order amplification; it is an engineering envelope, **not** a
confidence bound or a measured worst-case guarantee. Predeclared minimum limits
were 1e-7, 1e-12, 1e-5 respectively, with a hard calibration stop above absolute
ceiling 0.05; none determined the actual limits. The independent absolute bound
helps reject sparse corruption even when relative RMS and cosine pass. No C
measurement or held-out output was used to select or retune thresholds.

Top-1 diagnostics select the smallest index on exact ties, report both tie counts
and largest-minus-second-largest margins (zero for tied maxima). Agreement means
each chosen index is also maximal in the other vector. These diagnostics do not
replace or add an acceptance threshold to strict logits.

## Provenance and evidence

Checkpoint: `mistralai/Ministral-3-3B-Instruct-2512-BF16`, revision
`b6d637bef2393152b3da2b2fde72eecdee30557e`. Reference: Python 3.12.11, torch
2.15.0, Transformers 5.0.0, safetensors 0.8.0; CPU float32, eager attention, four
threads, eval/inference mode, no cache, last-token logits. Official implementation:
[Transformers pinned source](https://github.com/huggingface/transformers/blob/08810b1e278938278c50153ee1edfd7a20a759da/src/transformers/models/ministral3/modeling_ministral3.py).
Fixtures are I64 `[1,N]` input IDs, F32 `[1,N,3072]` hidden states, F32
`[1,131072]` last logits, contiguous batch/token/channel layout.

External experiment artifacts retain the predeclared plan, calibration script and
SHA256, timestamped frozen JSON, per-prompt control metrics, exact text and IDs,
tokenizer settings, package/build configuration, model provenance, fixture hashes,
engine source/binary hashes and exact run commands. They are deliberately not
runtime dependencies. To reproduce the method, run the two PyTorch call shapes
on calibration inputs, freeze limits before generating new held-out fixtures,
then invoke the self-contained comparator with explicitly supplied shards and
fixtures, as documented in the [validation guide](validation.md#full-forward-comparison). No Python, downloads or fixtures are required
by `make test`.

Held-outs were distinct from the five known fixtures: structured inventory JSON
(54 tokens), whitespace/Unicode normalization (44), and 12-record retrieval (136).
Encoding used `add_special_tokens=True`, no chat template, no truncation. Normal
C full-forward passed 8/8 sequences: both hidden checkpoints (16/16) and strict
logits (8/8); all top-1 IDs agreed, without ties. The longer fixture still has
**1220/460800 strict hidden_26 failures** (maxabs 0.00101947784424, relative RMS
9.40727243325e-5, cosine distance 4.42064784778e-9). No other strict elementwise
failures were observed. Held-out max relative RMS was 2.68303502218e-5 at hidden_13
retrieval; max cosine distance 3.56968454795e-10 at the same point. These are
bounded empirical results, not general certification.

Comparator self-tests reject NaN/positive and negative Inf on either side,
one-zero/nonzero rows, sign flips, adjacent channel permutation, sparse spike,
and vector scaling, including one bad row among three good rows. A 0.02 sparse
spike on a unit-magnitude 3072-wide row passes RMS/cosine but fails the independent
absolute bound. Known matches, all-zero pairs and close near-zero pairs pass.
These mutations test comparator sensitivity only; they do not prove broad model
correctness. Normal, ASan/UBSan and NDEBUG synthetic suites are required alongside
optional real comparisons. Remaining gaps include unseen input distribution,
real 256-token boundary, other hardware/backends, cached execution, and any claim
of universal threshold suitability. No production reduction/math was changed in
this validation task.
