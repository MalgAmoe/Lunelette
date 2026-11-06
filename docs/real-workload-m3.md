# M3 disposition: layered validation and provisional control

M3's bounded investigation, session checks and task evaluation are complete.
**Independent numerical acceptance remains FAIL/open.** The resident-FP32 engine
at `c79c292042b066a15586b7fd4ce16184d9a93c3f` is a provisional engineering control,
not an independently validated mathematical oracle or a task-quality endorsement.
At this disposition's original publication M4 had not started. The subsequent
[M4 closeout](real-workload-m4.md) records completed bounded implementation and
measurements; this historical M3 evidence is not a new numerical acceptance.

## Correction to the framing of M3

The [native regression strategy](native-regression.md) already established a
usable unchanged-arithmetic regression baseline before M3. It deliberately
separated native repeatability from independent PyTorch agreement. M3 did not
create that distinction or require erasing every known independent discrepancy
before engineering comparisons could proceed.

Treating the unresolved PyTorch gate as a blanket blocker to all optimization
work was too broad. Conversely, native parity cannot make that gate pass. The
[original specification](real-workload-spec.md), including section 13B, remains
unchanged: this is a limited provisional-control disposition, not a declaration
of an independently validated comparison baseline.

## Four distinct evidence layers

| Layer | Evidence and purpose | Boundary |
|---|---|---|
| Isolated operators | DS4-adapted frozen shader comparisons, analytical inputs, canaries and deliberate-fault checks; 192 project/norm GPU cases in the reviewed adoption campaign | Covers tested operators/inputs, not full-model mathematical correctness |
| Native regression and state | Existing frozen full-vocabulary references: 64 comparisons per mode, prefixes 6/24/150/248, prefill plus seven cached steps, two resets; M3 adds 14 service/direct full-logit pairs per mode and continuation/reset checks | Finite bitwise parity for unchanged arithmetic; reference provenance and mode matter |
| Independent checkpoint reference | Strict finite criterion `abs(error) <= 2e-4 + 1e-4*abs(reference)`; known mismatches reproduced and retained | Remains failed; no tolerance relaxation, benign-rounding proof or oracle acceptance |
| Free-running task behavior | Seven frozen examples, two repeats in each mode: three examples pass consistently; IDs/text/stops repeat | Small short-task panel, not broad quality certification or full-logit regression |

These are separate checks, not substitutes. The historical DS4-adoption runs and
M3 runs have their own build identities; do not relabel old captures as newly
captured at c79c292. Source-bound archived references remain valid regression
artifacts under their declared scope.

## What M3 added

- All 32 reproduced failing-vector sidecars matched historical failures bitwise.
- At prefix 248 prefill, first strict residual crossings were block 20 (MPS) and
  block 16 (custom), zero-based. Tested same-input operations and whole blocks pass;
  the independent CPU block on native inputs retains the crossing. Inherited
  differences dominate there; upstream defects are not universally ruled out.
- Retained independent logit panel: five distinct failing mode/state vectors,
  41 failing coordinates before reset duplicates; largest absolute logit errors
  approximately 0.000525 MPS and 0.000384 custom. All 128 observed token choices agree,
  with no perturbation-scale near ties. These are 32 distinct trajectory states,
  not 128 independent examples, and not free-running task-quality evidence.
- Real tasks: lookup, classification and missing-information handling pass.
  Arithmetic violates integer-only output and exhausts its 64-token budget; both
  JSON tasks add fences; two-turn recall violates exact output. No post-hoc
  rescoring. All 32 requests complete operationally: 28 EOS, four budget stops.
  Official tokenizer checks confirm framing, ledgers and streamed/buffered text.

M3 session/task harnesses and detailed raw captures remain external experiment
artifacts, not newly bundled maintained tests. The original native regression
suite is maintained; no unimplemented test promotion is implied here.

## M4 entry conditions

1. Reuse the established frozen native suite and operator checks. Do not recapture
   or replace valid goldens merely to make a candidate pass. MPS and custom remain
   separate controls; compare matching checkpoint, tokenizer, framing, precision,
   context, authoritative IDs and execution shapes.
2. Extend source-bound full-logit/intermediate captures only for workloads or paths
   not already covered. M3's new real-task prompts have IDs/text records but no
   full-logit captures; add those if used for numerical candidate comparison.
3. Require finite bitwise native parity where arithmetic is unchanged. A deliberate
   precision/order change needs a separately agreed, preregistered comparison
   contract; there is no automatically approved relaxed native threshold.
4. Report control/candidate independent-reference results separately using the
   unchanged strict criterion and preserved failures. Native agreement must not
   be presented as independent correctness.
5. Keep task outcomes separate again: retain the three-of-seven baseline and all
   failures. Changed trajectories or improved-looking output need review, not
   silent redefinition of success or numerical equivalence.
6. Compare performance only for matched work and report absolute latency/memory.
   Faster failed answers are not faster successful tasks. Named buffers, process
   samples, driver accounting and peaks are different quantities.

Further numerical tracing requires a concrete falsifiable defect hypothesis or a
separately scoped acceptance study, not an automatic recursive search. A new
nonfinite/state/weight/shape error or unexplained regression remains a blocker to
that candidate. This disposition authorizes no candidate implementation, commit,
precision change, longer context or alteration of the original specification.
