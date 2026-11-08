# Real-workload specification: closeout checklist

This checks the [original specification](real-workload-spec.md); it does not amend
it. The subsequently agreed operational diagnostic delivery scope is now complete,
including the [interactive adapter and bounded adaptive session](interactive-workload.md).
This is not independent numerical acceptance. The operational definition of done has been
demonstrated: one loaded service can run multi-request/tool tasks, stream results,
record measurements, reset/reuse state and repeat a workload under identified
strategies with state, numerical-regression and task-scoring checks.

**An independently validated numerical baseline has not been established.**
The existing limited diagnostic disposition is documented, not newly granted.
The agreed closeout claim is an operational diagnostic test system, with fixed tool
profiles, plain-chat-only checkpoints and deferred optimization candidates. The
adaptive session adds a five-event summary and a longer response; its formatting
failure remains a failure. No automatic additional campaign follows.

## Reading the statuses

- **Delivered / bounded evidence:** implemented and supported by the cited,
  source-bound reports. Not a fresh rerun of every test on every later binary.
- **Open validation:** an acceptance claim is not established.
- **Coverage decision:** evidence exists, but intended representativeness needs
  agreement; the spec does not prescribe a statistical sample size.
- **Follow-up / excluded:** not an unimplemented initial requirement in disguise.

## Requirement-by-requirement review

| Spec requirement | Status and evidence | Closeout criterion or remaining decision |
|---|---|---|
| §1 complete-task evaluation | Delivered: scored requests, tools, reused histories, timings and memory observations. [Coverage](real-workload-status.md) | Accept only claims supported by each recorded task; do not infer general accuracy. |
| §2 architecture/concurrency | Delivered: C/Metal/MPS service, loopback, one active job, serialized engine calls, explicit busy rejection; Python only for tooling. [M1](real-workload-m1.md), [M2](real-workload-m2.md) | Required state/error tests pass within supported scope; no generic framework or concurrent inference required. |
| §3 lifecycle/readiness/identity | Delivered: explicit configuration, one load, readiness and numerical status separately, identities, clean shutdown. [M2](real-workload-m2.md), [panel](useful-workload-panel.md) | No fallback or admission bypass. A resource refusal means an unavailable cell, not a failed answer or a throughput measurement. |
| §4 minimal API/bounds/errors | Delivered: session/request/control APIs, IDs, duplicate protection, bounded parsing/history/output, structured state/recovery errors. [M1](real-workload-m1.md), [M6](real-workload-m6.md) | Preserve model-free rejection/recovery evidence; no OpenAI compatibility claim. Startup usage text has a minor omission of newer options, not an observed API failure. |
| §5 states/framing/pending tokens | Delivered: ready/generating/partial/needs-reset/closed, pinned tokenizer identity/version, authoritative IDs, pending-token accounting. [M5](real-workload-m5.md), [native tools](native-tool-loop.md) | Keep literal-token and partial/continuation tests; installed official-package contents are not immutable pins. |
| §6 prefix save/restore and OFF control | Delivered for plain chat: materialized snapshots, endpoint logits, invalidation, retained versus recomputed authoritative inputs. [M5](real-workload-m5.md), [scenarios](workload-scenarios.md) | Tool-mode checkpoints explicitly return501. Confirm this supported-scope restriction is acceptable; if combined tool/checkpoint support is required, it is additional implementation work, not a passed combination. |
| §7 streaming/cancel/disconnect | Delivered: UTF-8 buffering, final text/IDs, bounded queues, safe-boundary cancellation, retained failure/recovery status. [M2](real-workload-m2.md), [M6](real-workload-m6.md) | Deterministic model-free coverage remains distinct from live observations. Unreached conditional branches in an older HTTP campaign must not be retroactively marked exercised by newer ordinary EOS runs. |
| §8 deterministic tool loop | Delivered: native V13 calculator, frozen-document lookup and recorded observations through lookup; complete-call validation and bounded continuation. [Tool guide](native-tool-loop.md) | Initial fixed profiles only. No arbitrary registration, parallel tool calls or live sensors claimed. Protocol success and strict final-answer failure remain separate. |
| §9 context/chunk ingestion | Delivered native/state evidence for256 and4352 capacity, chunks128/256, near-capacity/overflow and cancellation. [M6](real-workload-m6.md) | Independent long-context correctness is unverified; do not equate allocation/KV/trajectory tests with an independent oracle. |
| §10 initial three strategies | Delivered: resident FP32, compact arithmetic and compact lookup; actual policy/precision/accounting reported. [M4](real-workload-m4.md), [lookup](fp8-lookup.md), [panel](useful-workload-panel.md) | Six document/history configurations completed across separate campaigns,60/60 frozen checks. Timing separation and two repetitions limit ranking confidence. |
| §10 tiles and gate/up fusion | Follow-up candidates, not implemented. | Spec explicitly labels these follow-up/separate experiments; do not require them for initial operational closure or silently call them delivered. |
| §11 measurements | Delivered where applicable: loading versus resident work, first token versus actual text, completion, pauses/tools/cancel, work-v2 and memory definitions. [M5](real-workload-m5.md), [M6](real-workload-m6.md) | Report full sessions alongside request timing; include conversion, retain failures/conditioning. Samples are not peaks; absent driver observations remain absent. |
| §12 reusable/versioned client | Delivered: ordinary scenarios, exact scores, bounded repetitions/reuse/checkpoints, separate controlled replay and the [interactive adapter](interactive-workload.md). [Scenarios](workload-scenarios.md), [replay](controlled-replay.md) | Replay task score stays null. No requirement to implement every tool/control combination or arbitrary agent framework has been silently inferred. Supported combinations must remain explicit. |
| §13A model-free gates | Delivered source-bound parser/state/identity/UTF-8/cancel/disconnect/slow-client/recovery/context tests. [M1](real-workload-m1.md), [M6](real-workload-m6.md) | Relevant regression tiers apply to future code changes; this documentation review does not claim a fresh complete-suite run. |
| §13B independent baseline | Open validation; explicit provisional-control disposition already exists. [Numerical review](numerical-closeout-review.md), [M3](real-workload-m3.md) | Operational diagnostic delivery retains this limitation. Independently validated-baseline status is not established. No tolerance widening or failure relabelling. |
| §13C state/strategy gates | Bounded fresh/resident, reset/reuse, streamed/buffered, continuation, full-vector/intermediate, chunk/capacity/recovery evidence. [Native regression](native-regression.md), [M4](real-workload-m4.md), [M5](real-workload-m5.md), [M6](real-workload-m6.md) | Source/schedule-specific evidence, not all-path validation. Native parity is not §13B acceptance. |
| §13D task criteria | Delivered: frozen exact-output/tool criteria, genuine outputs and separately reported failures. [Coverage](real-workload-status.md) | A task failure is a valid evaluation result. It blocks claiming that task succeeded, not automatically completion of the evaluation system. Do not repair prompts or demand100% accuracy as a new spec requirement. |
| §14 delivery/report | Core operational definition demonstrated; results, identities, failures, recommendations and limits recorded. [Panel](useful-workload-panel.md) | Agreed operational diagnostic delivery is complete; open numerical status, supported-scope restrictions and narrow workload coverage remain. |

## Five initial workload families

| Family | What has actually been demonstrated | Coverage decision |
|---|---|---|
| Intermittent log/sensor summaries | Earlier classification/extraction and saved-observation tool work; the interactive session adds an adaptive five-event/two-pump summary after a recorded pause. | Summary facts passed human review, but four bullets failed the required three. One example, not general summary reliability or live sensors. |
| Repeated structured tasks | M3–M5 frozen tasks and repeated policy/strategy schedules; successes and failures retained. | Criteria and measurements exist. No promise of universally correct JSON/arithmetic; failing examples are not grounds for retry-until-pass. |
| Document questions |470/2198-token synthetic archives, three strategies, chunks128/256, two repetitions; exact outputs matched. | Bounded family coverage is present. No extra Cartesian campaign or broad retrieval benchmark is required by the stated initial spec. |
| Tool-assisted investigation | Native lookup→calculator→final answer, plus calculator and observation profiles. | Tool execution works; stock citation brackets and observation capitalization fail frozen final criteria. Accept as measured failures, not successful-task throughput. |
| Growing conversation | Three real memory turns at36/498/1685 input tokens across the six configurations; outputs differ from3 to7 characters. | Literally varying response lengths are covered, but narrowly. The interactive follow-up adds an85-token aggregate answer with its format failure retained. The spec sets no mandatory output-length minimum or broad sweep. |

The later authorized interactive session addresses these two narrow areas together:
increasing history, a recorded pause and a longer fact-checked multi-event summary.
Follow-ups were chosen after observing actual answers/stats; their criteria were
recorded before each scored generation. The four-versus-three-bullet failure was
retained. This is adaptive exploration under one strategy, not a frozen comparative
benchmark, and does not establish cross-strategy performance.

## Do not add moving completion requirements

The initial spec does **not** require100% pretrained-model accuracy, a statistical
reliability study, an energy benchmark, exact peak-memory instrumentation, arbitrary
tool plugins, all combinations of features, or a broad response-length sweep.
Concurrent inference, batching, remote deployment/authentication, UI, durable
sessions, vector databases and embedded agent frameworks are explicitly excluded.

Small-token tiles/fusion remain follow-ups. The original suspended32-job campaign
need not be revived. Neither a failed model answer nor an unavailable memory
configuration should be hidden, but neither is automatically a service defect.

## Historical decisions posed before interactive delivery

1. **Acceptance claim:** operational diagnostic test system with the existing
   numerical disposition, or independently validated baseline? Only the first
   is currently supportable; final acceptance is the user's decision.
2. **Initial examples:** accept narrow recorded-reading/short-answer coverage, or
   require the one combined summary/longer-response scenario above?
3. **Supported scope:** accept the documented fixed tool profiles/plain-chat-only
   checkpoints and deferred optimization candidates, or explicitly expand scope?

These decisions were subsequently bounded to operational diagnostic delivery plus
one interactive adapter session; see the [current status](real-workload-status.md).
The table's older campaign observations remain historical evidence, not claims
that the interactive run exercised every older conditional branch.

No new numerical defect was established by the evidence-only review. There is no
basis for another automatic tracing or optimization campaign. If full numerical
acceptance is required, agree that study separately before model execution.
