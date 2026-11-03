# Reference validation (planned)

Use the existing Hugging Face Transformers implementation rather than building
another Python inference engine. Exact checkpoint support and dependency versions
must be verified before creating a pinned environment; none are installed here.

The first harness should run a short, fixed text input, without sampling, in eval
mode. Record:

- Model repository and immutable revision, config, tokenizer revision and files.
- Python, Transformers, tensor library versions, device and execution dtype.
- Exact prompt, chat-template settings, special-token handling and input token IDs.
- Selected intermediate activations and full next-token logits.
- Position information and outputs for both uncached and KV-cached decoding.

Export fixtures in a simple documented format readable by C. Include tensor
names, shapes, layout, dtype and endianness, plus provenance and comparison
metrics. Choose absolute/relative tolerances based on reference precision and
operation; do not assume bitwise equality across CPU and GPU implementations.

Start with unquantized weights. Validate quantized execution separately against
that baseline. Keep downloaded model weights under the ignored models/ directory.
Run reference and C inference sequentially if memory pressure requires it.

No reference outputs have been generated yet. The existing softmax unit tests
are hand-specified mathematical cases, not Transformers-exported fixtures.
