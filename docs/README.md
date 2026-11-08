# Documentation index

Start with the repository [quickstart](../README.md). Current defaults and measured
recommendations are different; no optimization trial silently promotes a default.
Independent numerical acceptance remains FAIL/open.

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
- [Builder specification](real-workload-spec.md): original requirements and later research amendment;
  consult status for outcomes, rather than treating every research candidate as pending integration.

## Milestones and history

These documents preserve decisions, failures and source-bound evidence. They are
not a sequence of additional tasks to rerun. Current contracts live in the
[runtime](metal.md), [service](service.md), [testing](testing.md) and
[validation](validation.md) guides; milestone results remain source-bound evidence.

- [M1 plan](real-workload-plan.md), [M1 protocol](real-workload-m1.md): deterministic foundation.
- [M2](real-workload-m2.md): native service.
- [M3](real-workload-m3.md): numerical disposition and task failures.
- [M4](real-workload-m4.md): compact storage.
- [M5](real-workload-m5.md): prefix reuse and checkpoints.
- [M6](real-workload-m6.md): long context and chunked ingestion.
- [Tool loop v1](tool-loop-v1.md): earlier bounded protocol.
- [CPU/BF16 validation archive](validation-history.md): preserved retired commands and findings.
- [CPU runtime](cpu-runtime.md): retired standalone CPU implementation.
- [Full-depth validation history](full-depth-validation.md): historical thresholds and evidence;
  not current-checkpoint numerical acceptance.

Keep historical links and raw evidence intact when reorganizing documentation.
