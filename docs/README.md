# Documentation index

Start with the repository [quickstart](../README.md). Current defaults and measured recommendations are different; no optimization trial silently promotes a default. Independent numerical acceptance remains FAIL/open.

## Use the current system

- [Native service contract](service.md): startup, framing, HTTP, state, limits and metrics.
- [Interactive adapter](interactive-workload.md): persistent NDJSON sessions and controls.
- [Workload scenarios](workload-scenarios.md): repeatable client workloads.
- [Native tool loop](native-tool-loop.md): fixed calculator, lookup and observation profiles.
- [Metal runtime](metal.md): engine contracts, CLI, memory and precision.
- [Useful-workload panel](useful-workload-panel.md): bounded configuration recommendation.
- [FP8 lookup](fp8-lookup.md): optional single-token alternative, not a universal winner.
- [Build and testing](testing.md): distinguish host, GPU, socket and real-weight checks.

## Status and numerical evidence

- [Coverage/status](real-workload-status.md) and [closeout checklist](spec-closeout-checklist.md).
- [Numerical closeout review](numerical-closeout-review.md): unresolved independent findings.
- [Validation](validation.md): maintained consumers and current acceptance contracts.
- [Native regression](native-regression.md): provisional native control contracts.
- [Controlled replay](controlled-replay.md): full-logit capture, not task-quality evidence.
- [Long-context runtime tests](long-context-testing.md) and [HTTP checks](long-context-http-testing.md).
- [Reference artifact index](../tools/reference/README.md): externally supplied fixtures.

## Technical reference

- [Model facts](model-facts.md), [safetensors contracts](safetensors.md).
- [Builder specification](real-workload-spec.md): original requirements and later research amendment; consult status for outcomes, rather than treating every research candidate as pending integration.

<a id="milestones-and-history"></a>

## Retained rationale, controls and evidence

These mixed-purpose documents stay in the repository because they explain active diagnostic controls, design invariants or meaningful validation limitations as well as source-bound results. They are not a sequence of tasks to rerun. Current contracts live in the [runtime](metal.md), [service](service.md), [testing](testing.md) and [validation](validation.md) guides; historical defaults and results do not supersede those contracts. Both compact choices and explicit full FP32 residency remain supported.

- [M1 diagnostic contract](real-workload-m1.md): deterministic model-free foundation.
- [M2](real-workload-m2.md): native service.
- [M3](real-workload-m3.md): numerical disposition and task failures.
- [M4](real-workload-m4.md): compact storage.
- [M5](real-workload-m5.md): prefix reuse and checkpoints.
- [M6](real-workload-m6.md): long context and chunked ingestion.
## Historical compatibility pointers

The following pages retain their paths and original section anchors but no longer duplicate detailed retired commands or implementation chronology. Each gives an exact `git show` command using baseline `989efb62c84433e1e2cccba23f2730722d4ac40c` and its original repository-relative path. The baseline must be present in local Git history. Exact local archival copies with hashes and original link bases were preserved outside the engine; they are not shipped or required to use this repository. Neither Git documentation nor these pointers bundle external experiment payloads.

- [M1 implementation plan](real-workload-plan.md): completed sequence; active control explained above.
- [Application-JSON tool loop](tool-loop-v1.md): retired protocol and failed observation, not native tools.
- [CPU/BF16 validation](validation-history.md): retired measurements and strict discrepancy.
- [CPU runtime](cpu-runtime.md): retired standalone execution, not CPU host work.
- [BF16 full-depth criteria](full-depth-validation.md): historical envelope, not current-checkpoint acceptance.

No controls, source snapshots, model files, builds, manifests or receipts were retired or rewritten by this documentation closeout. Current-source validation inventories include documentation: edited bytes intentionally invalidate earlier build-source receipts; create new receipts for new builds rather than rewriting historical provenance.
