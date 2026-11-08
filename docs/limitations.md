# Supported scope and limitations

Lunelette is an **operational diagnostic test system**, not a production-ready or independently validated numerical baseline. The implementation supports the interfaces below; evidence remains bounded by its recorded source, model, inputs and schedules. Loading a model successfully is not numerical validation. No comparator runs at startup, and `/info` evidence labels/documentation paths do not attest to the supplied model or a new binary.

## Supported scope

| Area | Supported | Not implied / unsupported |
|---|---|---|
| Runtime | One resident C Metal/MPS implementation, macOS 15+, Apple Silicon Apple7+ | Standalone CPU inference, generic backend framework, arbitrary devices |
| Model | Official single-file HF-layout mixed FP8/BF16 Ministral 3 3B, text only | BF16-model fallback, shards, 8B/14B, vision or arbitrary configs |
| Projection storage | Compact arithmetic (normal default), compact lookup, expanded FP32 | Native FP8 arithmetic, W8A8, integrated experimental MPP/fused-attention candidates |
| Service profiles | C256/B256, C4352/B256, C4352/B128 | Sliding window, truncation, dynamic resizing, model position bound as allocated capacity |
| CLI | One system/user turn, 256 total tokens, buffered output | Full service capacity or streaming/multi-turn CLI state |
| Concurrency | One session/worker/job, bounded HTTP clients and explicit busy rejection | Batching, simultaneous inference, hidden job queue |
| Checkpoints | Plain chat, explicit reuse on/off, in-process boundary and endpoint logits | Tool-mode checkpoints, durable sessions, arbitrary stale-state rollback |
| Tools | Fixed calculator, frozen stock lookup and recorded observations | Arbitrary plugins, shell/network execution, live sensors, parallel tool calls |
| Deployment | Loopback diagnostic service and optional local clients | OpenAI compatibility, remote/authenticated deployment, UI, vector database or embedded agent framework |

Defaults, C enum-zero compatibility and comparator exceptions are documented in [running](guides/running.md). Device support is a runtime admission condition, not proof of validation on every supported device or memory size. Development evidence used Apple M5/32 GiB. No automatic fallback or guard weakening is permitted.

<a id="independent-numerical-evidence"></a>

## Independent numerical acceptance: FAIL/open

The independent reconstructed-FP32 PyTorch/Hugging Face comparison has retained strict failures under finite checks and `abs(actual - reference) <= 2e-4 + 1e-4 * abs(reference)`. The native engine is a **provisional engineering control**, not an independent mathematical oracle. Frozen native agreement and unchanged token choices do not close this gate. See [validation policy](development/validation.md) for commands, exact-versus-changed-schedule contracts and treatment of known mismatches.

The retained short-context service investigation at `c79c292042b066a15586b7fd4ce16184d9a93c3f` used prefixes 6/24/150/248, cached steps and reset repeats. Its logit failures include five distinct mode/state vectors (MPS prefix 248 cached steps 3/4/5; custom steps 3/4), with maximum absolute differences about 0.000525 and 0.000384 respectively. Intermediate mismatches also exist. Matching argmaxes in that bounded panel do not cover near ties, arbitrary prompts or long trajectories. These are source-bound observations, not fresh results for the current checkout.

Local matched-input operator/block checks did not establish a localized implementation defect or prove benign rounding as the only cause. Known unchanged discrepancies may coexist with authorized engineering under the preserved policy; new/materially worse discrepancies require assessment. Do not widen tolerances, refresh goldens, repair prompts or repeat until a failure disappears. Further tracing needs a bounded hypothesis and scope, not an automatic campaign.

## Compact arithmetic evidence

The compact implementation's published bounded evidence is associated with `9961dec5eed19726a786da348ed6ba9a93efa457`: finite reconstruction/domain checks, ordered stage reuse, native full-logit/state comparisons and service/direct checks. The supported implementation preserves FP8→FP32 reconstruction, widened-BF16 scaling and declared reduction order; [engine reference](reference/engine.md) explains this contract.

That evidence supports a storage/execution regression control, **not independent acceptance**. Unchanged outputs do not establish task quality, and old latency/memory observations are not current speed guarantees. Full FP32 remains supported and provisional, not a better mathematical oracle merely because its weights are expanded.

## Compact lookup evidence

Lookup changes only single-token packed-byte reconstruction; scaling, reduction, multi-token conversion and tied head remain as documented in [engine reference](reference/engine.md#single-token-arithmetic-and-lookup). Its finite signed-zero-aware table check and bounded arithmetic/lookup full-logit replay comparisons establish their declared same-schedule regression relationship, not an independent reference result or universal performance winner.

The original lookup integration is recorded at `d88541d56fd7790ba70d705f211f22b9167e1030`; subsequent source changes require their own tests and identities. Free-greedy workload panels are not full-logit comparisons. Use [controlled replay](development/replay.md) when explicitly authorized to collect matched trajectories; capture alone is not a comparison.

## Checkpoint evidence

The prefix-checkpoint integration documented by `289afb1e2a581f51fbe15774e4e9f765be884d56` has associated evidence including finite native state/logit checks, exact reset/restore/rebranch relationships and separately bounded changed-partition comparisons. These validate particular schedules, not every possible cancellation/failure path or an independent mathematical baseline. Model-free adapter/HTTP tests and real-model observations are distinct.

The [service contract](reference/service.md#checkpointrestore-and-recovery) owns current semantics: save materializes pending input, ON restore reuses retained KV/logits, OFF restore preserves an authoritative replay recipe, and cancellation after mutation is not rollback. All tool profiles reject checkpoints. State-invalid/nonfinite/memory errors stop affected validation regardless of previously passed checks.

## Long-context evidence

The long-context integration at `4d06d326acccfd8201fe38c7d45c1f3b783346a5` has bounded native/state/KV and HTTP evidence for configured chunk/capacity schedules. **Independent long-context correctness remains unverified.** Short-context independent fixtures cannot certify 4352-token execution. The seven-cell runtime suite uses MPS; a separate bounded custom/resident B256 continuation was reported, not a full custom-configuration sweep. The reduced B128 model schedule did not establish full-4K B128 coverage.

The maintained [long-context suites](development/long-context.md) expose particular fixed cells and controls, not an exhaustive configuration sweep. Conditional EOS-ready/incomplete-UTF-8 branches not reached in a particular native run must remain recorded as unreached; deterministic branch tests and later ordinary EOS responses cannot retroactively change that run. Evidence manifests containing documentation hashes intentionally drift when documentation changes; generate new receipts for new builds rather than rewrite old ones.

## Task quality and performance

Protocol success, task success, numerical correctness and performance are separate. Retained real-model tasks include arithmetic/formatting, citation, capitalization and unsupported-assertion failures. Fixed-tool execution can succeed while the final answer fails its declared criterion. No causal attribution to model versus engine follows from those outputs alone. The narrow task panels do not establish general reliability.

[Workload scoring](guides/workloads.md) preserves exact answers and EOS/ready conditions. A failed answer is a valid evaluation result, not permission to rescore, retry silently or claim faster successful-task throughput. Forced replay has no task-success score.

Performance claims require matched executed work, complete sessions, explicit startup/warm-cache conditions, variability and balanced order. Component improvements cannot be added into whole-model speedups. Named GPU buffers, admission headroom, sampled process footprint, driver allocations and physical peaks are different quantities; driver memory is unmeasured/null. There is no validated energy, DRAM-traffic, occupancy or Neural Accelerator attribution.

## Evidence availability

Revision references above identify integrations/dispositions, not a claim that every capture came from that exact clean commit: some measurements used frozen pre-reporting source snapshots. Exact capture manifests remain authoritative for measured identity. Small frozen controls and their provenance are maintained with the tests. Models, full numerical fixtures and detailed campaign receipts are optional external artifacts, not bundled or downloaded. Their absence is a verification gap, not a pass. Current use and acceptance rules remain fully described in this repository; historical source revisions identify evidence scope and are not prerequisites for reading these contracts.
