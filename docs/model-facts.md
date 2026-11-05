# Fixed model facts and runtime policy

`model_facts.h` / `model_facts.c` describe the fixed
Ministral-3-3B-Instruct-2512 text checkpoint: dimensions, logical weight names
and shapes, tied embeddings, norm epsilon and rotary parameters. They are not
a config loader or a multiple-model interface. Facts follow this checkpoint's
`text_config` (including its YaRN parameters), not assumptions about 8B/14B.
The block uses pre-RMSNorm, causal GQA, split-half YaRN with position-dependent
query scaling, SiLU gate/up multiplication, and residual additions. The output
head is tied to the embedding matrix. The resident runtime retains its existing numerical contract; this cleanup
does not alter shader math or reduction behavior.

`runtime_limits.h` separately specifies the current 256-token execution policy.
The model position bound is 262144; exposing that fact does not implement a
262144-token forward or cache. Tokenizer vocabulary/encoding restrictions remain
owned by the tokenizer, even where a numeric limit happens to match.

`metal_weights.c` validates the official single-file HF mixed checkpoint:
182 F8_E4M3 projections, 53 BF16 norms and BF16 embedding/tied head, exact shapes,
and two positive finite rank-0 BF16 companion scales per projection. Unknown text
names fail; vision/projector metadata is parsed but not executed or architecture-validated.
FP8 values are widened and multiplied by widened weight_scale_inv in FP32.
Activation scales are validated but unused. This is not W8A8 execution.

The former CPU forward/cache implementation and validator were retired after
`4d4f8f8`; [historical contracts](cpu-runtime.md) and numerical provenance remain
available. Independent frozen PyTorch outputs, not a duplicate C executor, are
the numerical oracle. No generic tensor system or backend dispatch is introduced.
