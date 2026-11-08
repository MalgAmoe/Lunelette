# Real-workload specification coverage

The agreed **operational diagnostic test-system scope is delivered** against the
[builder specification](real-workload-spec.md), including the final
[interactive agent adapter](interactive-workload.md). Persistent service,
long-context ingestion, prefix controls, bounded native tools and adaptive sessions
have bounded evidence. **Independent numerical acceptance remains FAIL/open.**
This is not an independently validated engine or a general model-quality claim.
Historical evidence belongs to its recorded source/build, not automatically to
every later binary.

For acceptance decisions rather than another campaign, see the
[spec closeout checklist](spec-closeout-checklist.md) and
[evidence-only numerical review](numerical-closeout-review.md). The checklist
distinguishes operational delivery from validated-baseline status, and narrow
coverage from new requirements that the original spec does not impose.

## Compact optimization research: not integrated

[Specification §15](real-workload-spec.md#15-follow-up-phase-compact-mpp-and-fused-attention)
defines experiments for improved staged/direct compact MPP projections and fused
online-softmax attention. Compact residency is the optimization target; expanded
resident FP32 is an optional reference only. Bounded expansion scratch is allowed
with all costs reported.

Two separately archived component screens are complete: the staged cooperative
MPP candidate and the first fused online-attention candidate passed their bounded
numerical checks but failed their prospective speed criteria. Neither advances to
full-engine validation as a speed candidate. Attention used synthetic FP32 inputs,
not full-model inference; its scratch-elimination design is not a measured engine
memory saving.

Direct compact-to-tile MPP subsequently compiled and passed a bounded layer13 down-
projection screen after the original API/syntax failure (retained separately).
It met the prospective speed criterion at 128 rows, but not 256. A subsequent
unchanged-kernel screen covered all seven block projections in layers 0/13/25:
315 numerical cases passed. MLP projections at 16/128 rows and attention output
at 128 are the strongest selective-hybrid candidates; Q128 is mixed against the
per-process criterion, K/V regress, and no 256-row cell clears the speed gate.
Inputs were mostly synthetic; neither whole-workload gains nor engine memory
savings have been established.

The subsequent isolated full-model hybrid trial stopped at its native numerical
gate: one coordinate in the first failing checked cached-key vector exceeded the
unchanged tolerance (error0.00027505, allowance0.00022344). The two checked full-logit
vectors and native reset/restore consistency passed, but did not override that
failure. Control256, HTTP validation and complete-request timing were withheld;
there is no whole-workload speed result. The conversion stage remained allocated.
No candidate has been integrated, and no retry or additional sweep follows automatically.
This phase does not reopen operational delivery or change defaults or independent
FAIL/open status.

## Interactive closeout

The [bounded native adapter session](interactive-workload.md#bounded-native-verification)
used one load and dynamically chosen follow-ups, retained an actual five-event
summary, and exercised live generation inspect/cancel, reset, checkpoint/restore
and clean shutdown. Two of three prospective task checks passed; the longer summary
had correct facts but four bullets instead of three. That failure was preserved.
Fixed tools, plain-chat checkpoints and deferred optimization limits remain explicit;
no independent numerical gate was relaxed.

## Requirement map

| Spec area | Implemented / retained evidence | Limit or missing part |
|---|---|---|
| §§2–4 lifecycle/API | One model/session/job, loopback, explicit configuration/readiness, retained requests, cancellation, reset/close; real resident service and model-free transport tests. [M1](real-workload-m1.md), [M2](real-workload-m2.md) | No general platform/concurrency claim; those are not initial requirements. |
| §5 conversation/pending state | Official native chat and V13 tool framing, generated-ID authority, committed/pending accounting, partial/reset behavior. [Tool guide](native-tool-loop.md) | Tool-mode checkpoints disabled; partial calls require reset, never execution. |
| §6 prefix controls | Explicit on/off/legacy, materialized checkpoints, restore/rebranch, pending EOS and reset checks; matched real-model comparisons. [M5](real-workload-m5.md) | [Version2 scenarios](workload-scenarios.md) expose explicit reuse and asynchronous checkpoint/restore. One new-client controlled-replay ON/OFF pair completed two sessions per load and matched eight full finite vectors bitwise; this is native consistency, not independent correctness or task quality. |
| §7 streaming/failures | UTF-8 buffering, terminal metadata, bounded queues, request-owned disconnect/cancel, worker serialization; real and model-free checks. [M2](real-workload-m2.md), [M6](real-workload-m6.md) | Long HTTP natural-EOS-ready and incomplete-UTF8 conditional branches were not reached in that campaign; other tests are not those same observations. |
| §8 tools | Bounded native calculator; frozen document lookup; saved observations through the same lookup mechanism. Validated complete call/result/same-session continuation. [Tool guide](native-tool-loop.md) | Three fixed profiles, not arbitrary registration or parallel tools; no live sensor access. Real calculator, lookup→calculator and saved-observation loops completed. Calculator scores pass; stock citation brackets and observation label capitalization fail their frozen exact-output criteria. |
| §9 context/ingestion | Default256 and opt-in4352, block128/256, checked admission/allocation, chunk cancellation, near-capacity/overflow and real runtime/KV/HTTP evidence. [M6](real-workload-m6.md) | Independent long-context numerical correctness unverified. No silent truncation, fallback or guard weakening. |
| §10 strategies | Resident FP32, compact FP8 arithmetic and opt-in [lookup scalar decoding](fp8-lookup.md), with staged conversion+MPS/custom multi-token projection. [M4 control](real-workload-m4.md) | Shape-specific small-token tiles and paired gate/up projection fusion remain unimplemented. One same-binary arithmetic/lookup replay pair matched eight full finite vectors bitwise (two prefix positions repeated four times). That replay has no speedup or task-quality claim. A later [bounded useful-workload panel](useful-workload-panel.md) passed40/40 compact checks at chunks128/256. Resident256 initially failed availability admission; a separately authorized follow-up passed20/20 resident checks at both chunks. All six bounded configurations are now covered across separate runs, not a general strategy benchmark. Document lookup is unrelated. |
| §11 measurements | Server metrics, client timing, task/check results, bounded tool durations, work counts, source/model identity, named allocations and sampled footprint. [M5](real-workload-m5.md), [M6](real-workload-m6.md) | Samples are not peaks; no energy/general performance acceptance. Legacy captures retain their original metric definitions. Faster failing/truncated tasks are not faster successful work. |
| §12 reusable scenarios | Versioned lifecycle/generation/cancel/pause scenarios, bounded native tool loops, exact-answer score steps; independent matched-work replay tooling. | Version2 adds bounded repetitions, explicit reuse and checkpoint recipes under one model load. [Native controlled replay](controlled-replay.md) adds explicit fixed trajectories and full-vector captures/comparison, bounded to16 selections per process; forced outputs never score task success. One real ON/OFF pair matched all eight captured vectors per policy; no broad benchmark claim. `controlled_protocol_only` is scripted protocol testing, not numerical replay or task quality. |
| §13 gates | Model-free protocol tests; bounded native numerical/state contracts; distinct task scores and failures. [M3](real-workload-m3.md), [M5](real-workload-m5.md), [M6](real-workload-m6.md) | Strict independent-reference mismatches remain unresolved. Native self-consistency is not independent correctness. Operational service ≠ validated numerical baseline. |
| §14 delivery/report | Service can load once, execute multiple requests or native tool continuations, collect evidence and reset while retaining weights. | Follow-up optimization candidates remain unimplemented; broader quality and tool-loop strategy comparisons are unmeasured, not automatically new initial gates. Real M7 tool runs used compact/MPS/legacy only. Final acceptance must state supported scope and numerical limitations. |

## Five initial workload families

| Family | Existing coverage | Remaining distinction |
|---|---|---|
| Intermittent log/sensor work | Pauses/resident repeated requests already exist. New `recorded-sensors-v1.json` supplies three scored synthetic recorded-reading tasks: two requests without an explicit pause, then a one-second pause before the third. | These are user-message classification/extraction tasks, not general log summarization. A separate real saved-observation lookup completes correctly but fails strict final label casing (`Pump-A` versus `pump-A`). Neither path accesses live sensors. See run status below. |
| Repeated structured tasks | M3–M5 real structured-task panel and repeated policy/strategy runs; strict score 3/7 examples, failures retained. | Checkpoint timing schedules include truncated and failing responses; no successful structured-task-throughput claim. |
| Document questions | M6 document lookup returns LARCH; matched useful-work panel passes 2/3 distinct tasks under each policy. M7 lookup retrieves a cited frozen passage. | M6 arithmetic returns 33 instead of43. A later [bounded panel](useful-workload-panel.md) scored470/2198-token archives across two compact strategies and chunks128/256, with exact answers in both repetitions. A separately authorized resident follow-up also passed at both chunks, preserving the earlier startup refusal. These related synthetic checks and temporally separated timings are not broad retrieval-quality or performance acceptance. |
| Tool-assisted investigation | Calculator 19+24 succeeds. Lookup→subtract31−12 uses correct tools/results and returns19 with the correct passage ID. | Exact final required square brackets; actual angle brackets fail strict task scoring. No retries/repair or changed criterion. |
| Growing conversation | Bounded real multi-turn memory and replay; M6 responses17/9/replay9 pass. | A later [bounded panel](useful-workload-panel.md) passed three free-greedy memory turns at36/498/1685-token histories across both compact strategies/chunks and subsequently resident at both chunks. Replay is not another independent task; the new outputs vary only from3 to7 characters, not a broad response-length quality sweep. |

The original 32-job M6 campaign retained 11 jobs; 21 were deliberately suspended
for seven focused cells and HTTP coverage. Finishing that Cartesian campaign is
**not** the default next step. Earlier compiler-limit, timeout/cleanup, admission,
protocol and wrong-answer failures remain preserved.

## Exact-answer scoring and recorded-reading scenario

`tools/real_workload.py` adds a client-only version-1 operation:

```json
{"op":"score","id":"previous-generation-id","text":"exact expected answer"}
```

It is allowed only in `free_greedy`, immediately after the matching `generate` or
`continue`; expected UTF-8 text is bounded to4096 bytes. `expect` remains a protocol
assertion and is not accepted on `score`. Use `score`, not `expect.text`, to record
free-running answer quality without relabelling a wrong answer as protocol failure.
A score passes only with exact text and a completed/EOS/ready turn. A valid budget
stop is a task failure, not a fabricated completed answer. Each score is retained
under `task_checks`; total task success combines these with any tool-task scores.
With no task scores it remains null. Scoring never triggers an extra generation,
retry or request. Existing transport/runtime failure handling is not redefined.

`tools/scenarios/recorded-sensors-v1.json` keeps weights resident across one session,
two resets and three independent inputs. Each contains explicitly synthetic
recorded pump readings and requests the hottest device, integer temperature and
ALERT at >=80, otherwise OK. Fixed expected outputs are `pump-A 84 ALERT`,
`pump-B 75 OK`, `pump-B 89 ALERT`. Each generation is capped at32 tokens,96 maximum
in total. Official tokenizer checking gives67 framed input IDs each; each turn
fits context256 even at its output cap. Reset avoids depending on an earlier
answer or partial response. The one-second pause is an explicit small schedule,
not a claim of representative production inter-arrival statistics.

Model-free client tests cover scoring/refusal, strict quality aggregation, no extra
work and the reset/pause sequence. Run them without weights:

```sh
python3 -m unittest discover -s tests -p 'test_*workload*.py'
python3 -O -m unittest discover -s tests -p 'test_*workload*.py'
```

One unchanged real-model run subsequently passed **3/3 reading checks**, with
protocol and scenario task success true: exact outputs matched those listed above.
It used compact FP8 arithmetic/MPS, context256/chunk256, legacy reuse, one model
load and the same session across two resets. The responses generated9/8/9 tokens
(26 total) and all ended naturally. The recorded pause was1.005 seconds; official
tokenizer replay matched each reset-turn input, output and complete ledger exactly.
Client/server exited zero; no retries or prompt changes. This is one synthetic
recorded-reading scenario, not an observation tool, general summary-quality claim,
representative intermittent benchmark or cross-strategy performance comparison.

## Remaining scope and acceptance decisions

The saved-observation lookup profile implements the third deterministic tool-data
category through existing machinery; the earlier user-input scenario is not its
substitute. No live sensor subsystem was added.

1. Version2 controls now have a [bounded scored document/history comparison](useful-workload-panel.md)
   across compact arithmetic/lookup and chunks128/256, plus a separately authorized
   resident follow-up at both chunks. All six bounded configurations have successful
   results; the original resident refusal remains preserved. The final interactive
   session adds a genuinely aggregating85-token summary, with its formatting failure
   retained. Broad reliability studies and sweeps are not mandatory initial gates. Do not retry automatically, resurrect the
   suspended broad campaign or treat replay as quality.
2. Evaluate execution candidates without conflating numerical consistency with
   speed or successful-task throughput. Lookup scalar execution is now maintained;
   tiles/fusion remain distinct follow-up candidates. Historical experiments do
   not promote those follow-ups into maintained code.
3. Keep numerical-baseline disposition and task quality separate. No automatic
   independent-reference investigation, tolerance change or retry-until-pass follows
   from this audit. Further model checks should answer a specific uncovered question.
4. Tool support/version1 scoring were committed as `5932c5e`; reviewed version2
   controls, native controlled replay and lookup were committed with approval as
   `d88541d`. No defaults were promoted and nothing was pushed. Do not interpret
   operational diagnostic delivery as independent numerical or blanket quality acceptance.
