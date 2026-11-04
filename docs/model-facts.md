# Fixed model facts and the CPU reference

`model_facts.h` / `model_facts.c` describe the fixed
Ministral-3-3B-Instruct-2512 text checkpoint: dimensions, logical weight names
and shapes, tied embeddings, norm epsilon and rotary parameters. They are not
a config loader or a multiple-model interface. Facts follow this checkpoint's
`text_config` (including its YaRN parameters), not assumptions about 8B/14B.
The block uses pre-RMSNorm, causal GQA, split-half YaRN with position-dependent
query scaling, SiLU gate/up multiplication, and residual additions. The output
head is tied to the embedding matrix. Existing reference math defines the exact
rounding/reduction behavior; this cleanup does not alter it.

`runtime_limits.h` separately specifies the current 256-token execution policy.
The model position bound is 262144; exposing that fact does not implement a
262144-token forward or cache. Tokenizer vocabulary/encoding restrictions remain
owned by the tokenizer, even where a numeric limit happens to match.

`cpu_weights.c` is private CPU safetensors validation: required BF16 names must
occur exactly once across the two supplied shards, with the expected shapes.
Unrelated vision tensors are ignored. Both full forward and cache use this same
validation, before execution. Lookup order, diagnostics and transactional output
and cache-prefix guarantees are unchanged.

The current forward/cache APIs are CPU/reference-specific: borrowed seekable
shards, FP32 activations, CPU workspaces, and FP32 K/V storage. They are not
universal backend boundaries. Logical weights do not imply `float *`, expanded
runtime weights, file residency, or a particular device. Future execution code
may own weight storage, precision, packing/layout, residency and cache/scratch
state. It may fuse operations without materializing these reference intermediates,
provided the chosen numerical contract is verified. This leaves CPU/GPU choices
open; it does not finalize future interfaces, introduce dispatch, or prescribe
a generic tensor system.
