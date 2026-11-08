# Retired CPU runtime: historical contracts

Standalone CPU inference and its APIs were retired after `4d4f8f8`. The former two-shard BF16 CLI, forward/cache APIs and scalar-model execution descriptions are historical, not supported invocation instructions. CPU host parsing, tokenization, generation policy and YaRN preparation remain required; this is not retirement of host work or host tests.

Use the [Metal runtime](metal.md), [service contract](service.md), [testing guide](testing.md) and [current validation](validation.md). Supported storage includes both compact reconstruction choices and explicit full FP32 residency; no second C inference engine is required for diagnostics.

## Retrieve the exact original

The exact pre-closeout document is in Git baseline `989efb62c84433e1e2cccba23f2730722d4ac40c` at `docs/cpu-runtime.md`:

```sh
git show 989efb62c84433e1e2cccba23f2730722d4ac40c:docs/cpu-runtime.md
```

This requires that Git object to be available in your checkout; a source-only distribution or shallow clone may not contain it. Exact before-images were also preserved in a local development archive outside the repository, with hashes and original link bases. That archive and the external experiment payloads are **not shipped** here; Git retrieval supplies the document, not model files, fixtures or campaign artifacts.

## Historical section links

These compatibility anchors refer to sections of the exact original above, not current invocation instructions.

<a id="cpu-runtime-and-cli-contracts"></a>
<a id="bounded-text-cli"></a>
<a id="rotary-embeddings-and-attention-scalar-baseline"></a>
<a id="complete-first-transformer-block"></a>
<a id="complete-uncached-3b-text-forward"></a>
<a id="greedy-generation-from-fixed-ids"></a>
<a id="bounded-kv-cache"></a>
