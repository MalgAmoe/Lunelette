# Native controlled replay

Controlled replay executes the **real native materialization path**, but selects
caller-supplied token IDs instead of using the model's greedy choice. It is an
opt-in diagnostic mode, never task-quality evidence. Default greedy service and
existing tool profiles are unchanged. Tool profiles cannot be combined with replay.

Use `tools/scenarios/replay-v2.json` with the [version2 client](workload-scenarios.md).
It declares `mode:"controlled_replay"`, ON reuse, two repetitions and these steps:
create, message, checkpoint, replay, restore, replay, close. Each replay supplies
`[53156,2]`; the complete scenario has eight forced selection steps and no scores.
Change only the declared policy to `off` for a separately launched comparison.
Run processes sequentially; retain source/build/model/tokenizer identities.

The client owns an exclusive `replay-logits.bin` in its fresh output directory.
It selects server startup `--evaluation-mode controlled-replay --replay-logits FILE`.
Both options are required together. The server refuses existing capture paths,
even empty files; failed evidence is not removed or overwritten. The filename is
never supplied over HTTP. `/info` reports generator `native_controlled_replay`,
evaluation mode `controlled_replay` and both replay limits16.

## Admission and runtime behavior

A scenario replay operation is:

```json
{"op":"replay","id":"first","tokens":[53156,2]}
```

The existing `/sessions/N/generate` route receives `id`, the derived `budget`, and
`replay_tokens` as a canonical comma-separated decimal **string**, for example
`"53156,2"`. The flat JSON parser is not broadened to arbitrary nested data.
Ordinary IDs1000..131071 and a final EOS2 are allowed;1..16 IDs must exactly match
the budget. Other special IDs, leading zeroes, whitespace, floats, duplicate fields,
missing sequences and default-mode replay requests are rejected before admission.

At most16 selection slots are reserved for the **whole process**, including all
repetitions. Cancellation/error does not reclaim reservations; reset/session close
must not permit capture growth beyond the bound. Context and request-history checks
still apply. The client checks its entire declared schedule before starting a model.
Replay scenarios reject ordinary generate/continue/cancel, tools and score steps;
checkpoint/restore, reset, pauses and repetitions remain available. The HTTP
cancellation machinery remains active; interrupted captures are failure evidence.

For every step the adapter performs normal model materialization, computes and
validates the actual finite greedy choice, captures full logits, then marks the
forced token as authoritative emitted/pending state. The next materialization must
consume that forced ID—not the discarded greedy choice. EOS, partial continuation,
checkpoint restoration and OFF recomputation retain their existing contracts.
Replay events and retained request metadata explicitly label the evaluation mode
and planned IDs. The client checks request/session identity, emitted IDs, before
state, committed/pending counts and the complete authoritative ledger.

## Capture contract

Version1 file format is explicitly little-endian:

-16-byte file header:8-byte `LNRPLY1` plus NUL, uint32 vocabulary131072, uint32 limit16.
- Each record: four uint32 values—zero-based selection index, prefix length,
  forced ID, actual greedy ID—then the full prefix as uint32 IDs, then131072 IEEE
  binary32 logits. No native struct padding or pointers are serialized.

The prefix is the actual runtime input before selection. Records are **attempted
selections**: cancellation can arrive after capture but before token publication.
Only successful completed scenarios are correlated against all emitted IDs and
prefixes after shutdown. Failed, partial or write-error captures remain preserved,
not relabelled as completed trajectories. Writes/flushes fail closed; capture
failure remains sticky across reset. At most16 records, less than9MiB at context4352.

Capture adds host I/O. Its timings are diagnostic, not inference-speed acceptance.
The scenario wall limit excludes startup/cleanup; retain an outer process/output/
free-space supervisor for real experiments. Memory samples are not peak measurements.

## Full-vector comparison

```sh
python3 tools/replay_logits.py reference/replay-logits.bin candidate/replay-logits.bin
# Only for an explicitly changed prefill-partition contract:
python3 tools/replay_logits.py reference/replay-logits.bin candidate/replay-logits.bin \
  --changed-partition
```

The parser rejects malformed/oversized/truncated/nonfinite data and checks each
record's first-tie greedy argmax. Comparisons require nonempty equal-length
trajectories with identical full prefixes and forced IDs. The default compares all
finite float32 bits, including signed zero. The explicit changed-partition mode uses
only the fixed rule `abs(error) <= 2e-4 + 1e-4 * abs(reference)`; it is not permission
to relax a failed same-schedule result. Mismatches exit nonzero. Relative error at
zero reference is reported as null rather than non-standard JSON infinity.

Files do not themselves attest model/code/strategy identity. Bind comparisons to
the corresponding pinned launch inputs and `/info` metadata. This is a native
artifact comparator, **not an independent numerical oracle**. Known independent
reference discrepancies and long-context independent-validation gaps remain.
`task_success` is always null; forced text never becomes a successful model answer.

## Tests

```sh
make test
python3 -m unittest discover -s tests -p 'test_*workload*.py'
python3 -m unittest discover -s tests -p 'test_replay_logits.py'
# Optional macOS native adapter/socket tests with CPU runtime stubs:
make workload-replay-test TOKENIZER=/path/to/official/tekken.json
```

Tests force a token different from the stub greedy choice, verify its consumption
by subsequent runtime input, full binary captures, EOS/partial/cancellation,
exclusive file creation, capture failures/caps, ON/OFF restore, metadata rejection,
empty/mismatched/nonfinite comparisons and deliberately changed logits. The actual
CLI/socket test runs two sessions per process and eight captures per policy. Stub
results are not real-weight numerical or task-quality evidence.

## Bounded real-weight observation

One reviewed sequential ON/OFF pair subsequently completed with compact FP8
arithmetic/MPS, context256/chunk256. Each process loaded the model once, ran two
sessions and captured eight full131072-element vectors. All eight matched vectors
were finite and **bitwise identical** between policies. Official V13 tokenization
matched the seven-token input and the complete forced ledgers. Actual greedy IDs
alternated1065/18562, distinct from the forced53156/2 trajectory.

Each reply consumed its forced non-EOS token and left EOS pending exactly once;
ON restore recovered the saved cache/logits, OFF restored logical state and rebuilt
its seven-token prefix. Both clients/servers exited zero and process groups were
gone. No retries, trajectory changes or tolerance relaxation. Task success remained
null. This is bounded native state/vector consistency, not independent numerical
correctness, free-running answer quality or a performance comparison.
