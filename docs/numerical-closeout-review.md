# Numerical evidence review for closeout

This is an evidence-only review, not a new inference campaign, tolerance change or
acceptance decision. The existing [M3 disposition](real-workload-m3.md) already
allows a **provisional engineering control** while retaining **independent
numerical acceptance FAIL/open**. Neither that disposition nor this review
certifies an independently validated baseline.

## What the existing evidence establishes

| Evidence | Verified meaning | Boundary |
|---|---|---|
| M3 failing tensor sidecars |32 files across two modes and two reset cycles | Includes intermediate tensors; not32 distinct failing logit vectors |
| Full-logit failures |Five distinct mode/state vectors,41 failing coordinates;10 vectors/82 coordinates with reset duplicates | MPS prefix248 cached steps3/4/5; custom steps3/4 |
| Largest absolute logit differences |Approximately0.000525 MPS and0.000384 custom | Finite, but outside the frozen criterion at some coordinates |
| Teacher-forced choices |128/128 logged argmax matches over32 distinct context/step states | Modes/resets do not create independent prompt coverage; not task-quality acceptance |
| M4/M5 independent panels |10/64 captured cases fail per strategy/mode panel;192 tensor checks per panel | A failing case may contain an intermediate failure, not a logit failure |
| Historical stability |M4/M5 normalized-metric equality is recorded; failed sidecar sets/hashes match M3 | Stronger than equal failure counts, but source- and schedule-bound |

The preserved independent criterion remains:

```text
finite(actual) and finite(reference)
abs(actual - reference) <= 2e-4 + 1e-4 * abs(reference)
```

The closeout review independently rehashed all32 M3 failing sidecars and128 M4/M5
copies (four panels per milestone,16 files each), checked their exact names/hash
sets against the retained records and M3, and verified fixture hashes. It directly
recomputed finite checks, strict mismatch counts, maxima and argmax on all10
retained failing-logit arrays. Original logs supplied the other observed argmax
results; passing native arrays were not invented. No model was loaded or executed.

This reconfirms existing failures. It does **not** run the latest binary against
an independent oracle or extend short-context results to4352 tokens. The later
[useful-workload results](useful-workload-panel.md) are genuine task/state evidence,
not a replacement for the numerical gate.

## What localization already ruled down—not out

At prefix248 prefill, the first residual threshold crossings were block20 MPS
and block16 custom, zero-based. Tested matched-input operations and entire CPU
blocks passed the strict local comparisons. Substituting the independent CPU block
on the native input retained the original threshold crossing. Inherited input
differences dominated at those coordinates.

This weakens a defect hypothesis localized to those blocks. It does not establish
that every upstream operation is correct, prove benign rounding as the sole cause,
explain all later cached-step failures, or justify a wider tolerance. See the
source-bound account in [M3](real-workload-m3.md) and subsequent unchangedness gates
in [M4](real-workload-m4.md) and [M5](real-workload-m5.md).

The historical output-impact panel had no perturbation-scale near ties. That does
not cover untested near-tie prompts, sampling or longer trajectories. Arithmetic,
JSON formatting, recall, citation and capitalization failures have no demonstrated
causal attribution to these numerical discrepancies. Do not blame either the model
or engine solely from those task outputs.

## Decision needed—not an automatic new experiment

[Spec §13B](real-workload-spec.md#b-current-checkpoint-numerical-baseline) permits
an operational service while the numerical work is pending, with results labelled
diagnostic. Its resolve-or-explicitly-disposition language must not be read as
permission to turn a documented failure into a pass. The current explicit
disposition deliberately withholds independently validated-baseline status.

Two different closeout claims are therefore possible:

1. **Operational diagnostic test system:** consider acceptance with the existing
   limited disposition and visible FAIL/open numerical status. This is a user
   acceptance decision, not made by this document.
2. **Independently validated comparison baseline:** not established. Further work
   or a substantively justified, separately agreed disposition would be needed;
   renaming the limited disposition or repeating the same tests is insufficient.

No new discrepancy or discriminating implementation-defect hypothesis emerged
from this review. Do not repeat reproduction, first-crossing replay or the old task
panel merely to pursue a green result. No new tracing experiment is recommended
now. If full numerical acceptance is required, first agree a separate study with
its exact question, checkpoint/path/input coverage, unchanged primary criterion,
resource bounds, stopping condition and decision consequences. New nonfinite,
state/shape/weight errors or a demonstrably worse regression would instead justify
a focused defect investigation.

## Limits retained

- [Native regression](native-regression.md) is not an independent mathematical oracle.
- Independent long-context correctness remains unverified; [M6](real-workload-m6.md)
  supplies bounded native/state/KV evidence.
- No fixture regeneration, tolerance relaxation, prompt repair, new model process,
  dependency installation or runtime edit occurred in this review.
- Stored metric-equality reports were inspected; this review independently rechecked
  raw failed arrays, not every historical passing tensor or every old diagnostic.
