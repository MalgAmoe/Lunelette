# Retired CPU/BF16 validation: historical evidence

The former two-shard CPU/BF16 commands and detailed measurements are historical, not commands for the maintained single-file mixed FP8/BF16 loader. They did not establish cached/long-context Metal correctness. The retired longer fixture retained **1,220 / 460,800 strict final-hidden failures**; a separately calibrated hidden-state envelope did not erase them or change the strict logit criterion.

Use [current validation](validation.md), [native regression](native-regression.md) and the [numerical evidence review](numerical-closeout-review.md). Independent numerical acceptance remains **FAIL/open**. Frozen independent references, strict finite checks and unchanged tolerances remain required; historical BF16 results are not the current checkpoint's oracle. Detailed schemas, reduction investigations, measurements and unrun-check gaps remain in the exact original below.

## Retrieve the exact original

The exact pre-closeout document is in Git baseline `989efb62c84433e1e2cccba23f2730722d4ac40c` at `docs/validation-history.md`:

```sh
git show 989efb62c84433e1e2cccba23f2730722d4ac40c:docs/validation-history.md
```

This requires that Git object to be available in your checkout; a source-only distribution or shallow clone may not contain it. Exact before-images were also preserved in a local development archive outside the repository, with hashes and original link bases. That archive and the external experiment payloads are **not shipped** here; Git retrieval supplies the document, not model files, fixtures or campaign artifacts.

## Historical section links

These compatibility anchors refer to sections of the exact original above, not current invocation instructions.

<a id="historical-cpubf16-validation-archive"></a>
<a id="historical-material-below--not-current-invocation-instructions"></a>
<a id="reading-the-evidence"></a>
<a id="optional-real-weight-validation"></a>
<a id="optional-layer-0-pre-attention-comparison"></a>
<a id="isolated-attention-comparison"></a>
<a id="first-block-comparison"></a>
<a id="full-forward-comparison"></a>
<a id="reduction-drift-investigation"></a>
<a id="historical-strict-full-forward-results-and-remaining-gap"></a>
<a id="fixed-id-generation-comparison"></a>
<a id="cache-comparison"></a>
