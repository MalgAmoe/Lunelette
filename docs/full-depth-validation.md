# Retired BF16 full-depth validation: historical criteria

The former per-token RMS/cosine/absolute envelope applied only to the retired uncached BF16 CPU `compare_forward` hidden states. It is not an acceptance rule for the current mixed FP8/BF16 Metal runtime. Historical coverage reached 150 tokens; the longer fixture retained **1,220 / 460,800 strict hidden-state failures** despite passing that separate envelope. Passing logits and top-1 agreement did not erase the discrepancy.

Use [current validation](validation.md) and the [numerical evidence review](numerical-closeout-review.md). **Independent numerical acceptance remains FAIL/open**, with finite operands and the unchanged bound `abs(error) <= 2e-4 + 1e-4 * abs(reference)`. Native parity is a separate provisional-control check, not an independent oracle. Original equations, frozen limits, calibration and held-out provenance remain available through the original below; they must not be promoted to current tolerances.

## Retrieve the exact original

The exact pre-closeout document is in Git baseline `989efb62c84433e1e2cccba23f2730722d4ac40c` at `docs/full-depth-validation.md`:

```sh
git show 989efb62c84433e1e2cccba23f2730722d4ac40c:docs/full-depth-validation.md
```

This requires that Git object to be available in your checkout; a source-only distribution or shallow clone may not contain it. Exact before-images were also preserved in a local development archive outside the repository, with hashes and original link bases. That archive and the external experiment payloads are **not shipped** here; Git retrieval supplies the document, not model files, fixtures or campaign artifacts.

## Historical section links

These compatibility anchors refer to sections of the exact original above, not current invocation instructions.

<a id="provisional-bounded-full-depth-validation"></a>
<a id="equations-and-frozen-limits"></a>
<a id="provenance-and-evidence"></a>
