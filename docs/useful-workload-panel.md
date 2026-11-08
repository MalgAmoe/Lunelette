# Bounded document/history strategy panel

## Recommendation

For the measured document/history workload, the conservative configuration is **compact FP8 arithmetic, MPS, context 4352, chunk 256, reuse ON**. Chunk 256 reduced observed summed generation-call latency by about 13–14% versus 128, at identical outputs/model-token totals and 105.625 MiB more named scratch allocation.

Lookup remains a valid opt-in alternative, but its overall advantage was only 0.9% at 256 and 2.4% at 128; the direction reversed between repetitions. Its reported decode component was consistently lower, not enough to establish a whole-workload winner here. No defaults were changed. These are diagnostic observations, not production performance acceptance.

A subsequently authorized resident follow-up completed both chunks. Resident 256 is a latency-first alternative when memory permits: 18.402 s mean versus 19.426 s compact arithmetic, but 13.844 GiB named buffers versus 5.495 GiB and slower loading. Keep the compact recommendation for memory-conscious use; see the follow-up below.

## Source and incomplete resident comparison

Scenario controls, native replay and lookup were committed as `d88541d56fd7790ba70d705f211f22b9167e1030`. A fresh binary from that clean commit and 208 source files were pinned. Model SHA256: `728f1826cd0e38191ca7b1379e81f78cf0555c6ffd95882aabd2404632346f86`; Tokenizer SHA256: `600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843`.

The original six-cell panel stopped before its first task: resident FP32/chunk 256 required availability **>16.02 GiB**, but the guard's last estimate was 15.94 GiB. Client 1/server 2, owned group gone, no readiness or generation requests. This is resource admission refusal, not task-quality failure. Resident 128 was not tried in that original campaign. Both configurations later completed in the separately authorized follow-up below; the original refusal remains preserved.

A separately frozen continuation ran only the four originally planned compact cells, without retrying resident, changing the guard, evicting processes or repairing prompts. Same scenario bytes/source/binary, context 4352, MPS, reuse ON; two repetitions in fresh sessions per model load. Each process exited client/server 0 and its owned group was gone. Original failure/skipped-cell records remain unchanged. At that stage the three-strategy comparison was incomplete.

## Exact scores and executed work

All **40/40 exact EOS/ready checks** passed in the compact continuation. These are five related synthetic checks × two repetitions × four configurations, not 40 independent tasks or a general reliability estimate:

| Check | Input tokens | Exact answer | Generated tokens | Model tokens |
|---|---:|---|---:|---:|
| Short archive |470|731|4|473|
| Longer archive |2198|731|4|2201|
| Initial memory |36|417|4|39|
| History second turn |498|902|4|462|
| History third turn |1685|417 902|8|1191|

Documents share the same fact/question at two sizes. History uses genuine prior model outputs, not substituted answers; response lengths vary only modestly. Every configuration generated 48 tokens and executed 8732 model tokens across its two repetitions. Model tokens sum prefill_committed+append_committed+ decode_committed; pending counters are subsets. No recompute or logit restore.

Independent inspection matched official user framing, authoritative generated IDs, complete ledgers, remaining capacity, pending EOS, retained metadata and every score/work receipt. At each chunk, arithmetic/lookup full trajectories and work receipts matched exactly. Across chunks, trajectories/model-token totals matched; partition-call counts intentionally differed. No logits or independent numerical reference were evaluated in this free-greedy panel.

## Diagnostic latency and memory

Times sum five generation-call completions per repetition, excluding loading and preceding message POSTs. Both repetitions count; neither was discarded as warmup. Full repetition durations including orchestration were retained separately.

| Strategy | Chunk | Exact checks | Rep 1 seconds | Rep 2 seconds | Mean seconds | Named buffers GiB |
|---|---:|---:|---:|---:|---:|---:|
| Compact arithmetic |128|10/10|23.277|21.950|22.613|5.391|
| Compact lookup |128|10/10|21.945|22.181|22.063|5.391|
| Compact arithmetic |256|10/10|19.772|19.079|19.426|5.495|
| Compact lookup |256|10/10|18.859|19.650|19.255|5.495|

The 2198-token archive averaged 9.69–9.72 s at 256 versus 11.20–11.23 s at 128. The last history turn averaged 5.81 s at 256 versus 6.56–6.66 s at 128. Ingestion dominates this short-answer workload.

Mean reported decode-step time was 84.01 ms lookup versus 92.07 ms arithmetic at 128, and 85.39 ms versus 91.67 ms at 256. The reduction appeared in both repetitions at both chunks, but does not establish complete decode-heavy workload throughput.

Named buffers were 5,788,922,368 bytes at 128 and 5,899,678,720 at 256, identical between compact strategies. Maximum **observed footprint samples** ranged 5.506–5.622 GiB; these are not peaks. No driver peak or energy measurement.

## Verification and limits

The frozen original plan allowed six loads/60 generation requests/900 s overall; the continuation used four successful loads/40 requests with 600 s overall bound, 120 s per scenario and unchanged 64-token response caps. No automatic retries, output extensions or scoring relaxation occurred. Runtime/protocol/limit failures stop the fixed schedule; quality-only failures may continue planned repetitions.

The external evidence bundle retains original and continuation plans, raw client NDJSON/server logs/results, source/binary/scenario pins, per-request CSV and an independent analysis. No-model supervisor tests passed normal/-O, including malformed partial-result preservation and cleanup errors. Independent audit tests passed normal/-O on a matching copy and deliberately corrupted score, work and ledger copies. Original artifacts were not altered.

This initial panel added bounded compact document-size×chunk and growing-history coverage. The subsequent resident follow-up below completes that small configuration matrix across separate runs, not a broad response-length or strategy benchmark. Fixed order, two non-independent repetitions and no thermal control cannot settle small timing differences. Strict independent-reference discrepancies and historical wrong answers remain unresolved. Native consistency is not independent correctness; no peak-memory, energy, general quality or performance acceptance follows.

## Subsequently authorized resident follow-up

One new attempt per chunk, 128 then 256, completed under the unchanged memory guard, using the exact retained d88541d binary and frozen scenario. HEAD749bf6a added only documentation; runtime source and binary identity were checked separately. Neither resident configuration was automatically retried. Each loaded once, used two sessions and exited client/server 0 with its owned group gone.

All 20 resident exact EOS/ready checks passed. Together with 40 compact checks, all six configurations now have successful bounded results across separately timed runs—not 60 independent tasks. Complete trajectories and work-v2 counters matched both compact controls at each chunk: 48 generated/8732 model tokens per cell. These counters do not imply identical GPU operations across storage strategies.

| Resident chunk | Checks | Rep 1 generation seconds | Rep 2 generation seconds | Mean seconds | Named buffers GiB | Load seconds |
|---|---:|---:|---:|---:|---:|---:|
|128|10/10|19.734|18.509|19.121|13.741|7.230|
|256|10/10|18.964|17.840|18.402|13.844|7.381|

At 256, resident mean generation latency was 5.3% below arithmetic and 4.4% below lookup, with 8.350 GiB extra named buffers. Its longer archive averaged 8.615 s and last history turn 5.505 s. It did not win every case: the short archive averaged 2.140 s, including a retained 2.750 s first-repetition request versus 1.530 s in the second. No first-use measurement was discarded or assigned an unverified cause.

Mean resident decode-step time was 148.79 ms at 128 and 148.72 ms at 256, slower than both compact strategies. Its overall advantage is specific to this ingestion-heavy, short-answer mix; it does not establish superior long-response throughput. Loading was also slower. Maximum observed resident footprint samples were 13.876/13.975 GiB, not peaks. The earlier memory refusal remains relevant to operational headroom.

The follow-up ran later, not interleaved with compact controls. Two repetitions, temporal separation and uncontrolled thermals limit ranking confidence. Main independently checked all 20 official input/output ledgers, scores, state, work and timing invariants, plus source/binary identity and cleanup. No-model audit checks accepted a known match and rejected corrupted work/ledger copies while retaining strict wrong-answer failure. No independent numerical or default promotion follows.
