# Documentation

Start with the [quickstart](../README.md). These pages describe the supported engine and maintained diagnostic tools, not a sequence of implementation milestones. **Independent numerical acceptance remains FAIL/open**; [limitations](limitations.md) separates supported behavior from verified scope.

## Run it

- [Build and run](guides/running.md): required assets, CLI/server flags, defaults and storage/context choices.
- [Interactive client](guides/interactive.md): persistent NDJSON commands, bounds, cancellation and cleanup.
- [Workloads and scoring](guides/workloads.md): scenario versions, operations, repetitions and protocol versus task success.

## Understand it

- [Engine](reference/engine.md): model architecture, weight formats, arithmetic, memory and C ownership.
- [Service](reference/service.md): HTTP, framing, state, checkpoints, cancellation and metric definitions.
- [Fixed tools](reference/tools.md): native tool framing, calculator, frozen-document lookup and recorded observations.
- [Tokenizer](reference/tokenizer.md): Tekken API, heap-based BPE, byte/token bounds and Unicode provenance.

## Change and verify it

- [Testing](development/testing.md): host/GPU/socket/resource boundaries and exact entrypoints.
- [Validation](development/validation.md): fixture formats, comparators, numerical and task/performance acceptance rules.
- [Controlled replay](development/replay.md): forced trajectories and full-logit captures, never task-quality evidence.
- [Long-context diagnostics](development/long-context.md): supported optional fixed suites and their coverage limits.
- [Project standards](../AGENTS.md): coding, review, safety and evidence policy.

Current instructions are self-contained. Optional models, tokenizer assets and generated references are supplied separately. Frozen controls keep their minimal provenance beside their tests. Historical plans, trials and full run receipts are not required to use or understand this checkout; source-bound evidence does not certify a later build.
