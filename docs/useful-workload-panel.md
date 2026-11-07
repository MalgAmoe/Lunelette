# Bounded document/history strategy panel

## Recommendation

For the measured document/history workload, the conservative configuration is
**compact FP8 arithmetic, MPS, context4352, chunk256, reuseON**. Chunk256 reduced
observed summed generation-call latency by about13–14% versus128, at identical
outputs/model-token totals and105.625MiB more named scratch allocation.

Lookup remains a valid opt-in alternative, but its overall advantage was only
0.9% at256 and2.4% at128; the direction reversed between repetitions. Its reported
decode component was consistently lower, not enough to establish a whole-workload
winner here. No defaults were changed. These are diagnostic observations, not
production performance acceptance.

## Source and incomplete resident comparison

Scenario controls, native replay and lookup were committed as
`d88541d56fd7790ba70d705f211f22b9167e1030`. A fresh binary from that clean commit
and208 source files were pinned. Model SHA256:
`728f1826cd0e38191ca7b1379e81f78cf0555c6ffd95882aabd2404632346f86`;
Tokenizer SHA256:
`600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843`.

The original six-cell panel stopped before its first task: resident FP32/chunk256
required availability **>16.02GiB**, but the guard's last estimate was15.94GiB.
Client1/server2, owned group gone, no readiness or generation requests. This is
resource admission refusal, not task-quality failure. Resident128 was not tried;
its admissibility and both resident configurations' latency remain unmeasured.

A separately frozen continuation ran only the four originally planned compact
cells, without retrying resident, changing the guard, evicting processes or
repairing prompts. Same scenario bytes/source/binary, context4352, MPS, reuseON;
two repetitions in fresh sessions per model load. Each process exited
client/server0 and its owned group was gone. Original failure/skipped-cell records
remain unchanged. The full three-strategy comparison is **not complete**.

## Exact scores and executed work

All **40/40 exact EOS/ready checks** passed in the compact continuation. These are
five related synthetic checks × two repetitions × four configurations, not40
independent tasks or a general reliability estimate:

| Check | Input tokens | Exact answer | Generated tokens | Model tokens |
|---|---:|---|---:|---:|
| Short archive |470|731|4|473|
| Longer archive |2198|731|4|2201|
| Initial memory |36|417|4|39|
| History second turn |498|902|4|462|
| History third turn |1685|417 902|8|1191|

Documents share the same fact/question at two sizes. History uses genuine prior
model outputs, not substituted answers; response lengths vary only modestly.
Every configuration generated48 tokens and executed8732 model tokens across its
two repetitions. Model tokens sum prefill_committed+append_committed+
decode_committed; pending counters are subsets. No recompute or logit restore.

Independent inspection matched official user framing, authoritative generated IDs,
complete ledgers, remaining capacity, pending EOS, retained metadata and every
score/work receipt. At each chunk, arithmetic/lookup full trajectories and work
receipts matched exactly. Across chunks, trajectories/model-token totals matched;
partition-call counts intentionally differed. No logits or independent numerical
reference were evaluated in this free-greedy panel.

## Diagnostic latency and memory

Times sum five generation-call completions per repetition, excluding loading and
preceding message POSTs. Both repetitions count; neither was discarded as warmup.
Full repetition durations including orchestration were retained separately.

| Strategy | Chunk | Exact checks | Rep1 seconds | Rep2 seconds | Mean seconds | Named buffers GiB |
|---|---:|---:|---:|---:|---:|---:|
| Compact arithmetic |128|10/10|23.277|21.950|22.613|5.391|
| Compact lookup |128|10/10|21.945|22.181|22.063|5.391|
| Compact arithmetic |256|10/10|19.772|19.079|19.426|5.495|
| Compact lookup |256|10/10|18.859|19.650|19.255|5.495|

The2198-token archive averaged9.69–9.72s at256 versus11.20–11.23s at128. The last
history turn averaged5.81s at256 versus6.56–6.66s at128. Ingestion dominates this
short-answer workload.

Mean reported decode-step time was84.01ms lookup versus92.07ms arithmetic at128,
and85.39ms versus91.67ms at256. The reduction appeared in both repetitions at both
chunks, but does not establish complete decode-heavy workload throughput.

Named buffers were5,788,922,368 bytes at128 and5,899,678,720 at256, identical
between compact strategies. Maximum **observed footprint samples** ranged
5.506–5.622GiB; these are not peaks. No driver peak or energy measurement.

## Verification and limits

The frozen original plan allowed six loads/60 generation requests/900s overall;
the continuation used four successful loads/40 requests with600s overall bound,
120s per scenario and unchanged64-token response caps. No automatic retries,
output extensions or scoring relaxation occurred. Runtime/protocol/limit failures
stop the fixed schedule; quality-only failures may continue planned repetitions.

The external evidence bundle retains original and continuation plans, raw client
NDJSON/server logs/results, source/binary/scenario pins, per-request CSV and an
independent analysis. No-model supervisor tests passed normal/-O, including
malformed partial-result preservation and cleanup errors. Independent audit tests
passed normal/-O on a matching copy and deliberately corrupted score, work and
ledger copies. Original artifacts were not altered.

This adds bounded compact document-size×chunk and growing-history coverage, not a
broad response-length sweep or full strategy comparison. Fixed order, two
non-independent repetitions and no thermal control cannot settle small timing
differences. Strict independent-reference discrepancies and historical wrong
answers remain unresolved. Native consistency is not independent correctness;
no peak-memory, energy, general quality or performance acceptance follows.
