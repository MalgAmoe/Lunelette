# Compact FP8 lookup scalar candidate

`compact-fp8-lookup` is an **opt-in execution strategy**, not document lookup.
Select it with service `--strategy`, workload-client `--native-strategy`, or
single-turn CLI `--metal-strategy`. Resident FP32 remains the default; the existing
`compact-fp8-arithmetic` strategy remains available unchanged.

For single-token block projections only, the new Metal kernel indexes a256-entry
unscaled FP32 constant table using the packed E4M3FN byte, multiplies by the widened
BF16 scale in FP32, then uses the same Kahan lane loop and SIMD reduction as the
arithmetic kernel. All254 finite code values are exact, including signed zero.
The two NaN codes remain rejected by unchanged host payload validation; their
unreachable table entries are not permission to accept invalid weights.

The table consists of exact hexadecimal literals, derived from fraction/exponent
bits: `fraction * 2^-9` for exponent0, otherwise `(8+fraction) * 2^(exponent-10)`,
with the sign bit applied. It is a1KiB shader constant, not a per-weight table or a
new explicitly allocated MTLBuffer. Pipeline/compiler storage is not reported as
an exact driver-memory allocation or peak.

## Unchanged paths and accounting

-182 FP8 projections stay packed; BF16-derived norms, embedding and tied head stay
  FP32. Activations and KV remain FP32.
- Multi-token projections use the original arithmetic conversion into the same
  stable FP32 stage, followed by the selected MPS/custom path. No lookup conversion,
  small-token tile, fusion, precision change or fallback is introduced.
- Weight bytes4,637,454,336 and conversion stage113,246,208 match arithmetic compact
  storage. Scratch/KV sizes, context/chunk admission and the memory guard are unchanged.
- Default strategy and reuse policy are unchanged. Unknown strategy enums still
  fail admission; numeric enum2 is now lookup,3 is invalid.

`/info` reports the actual selected strategy and `single_token_block_kernel`.
Lookup's evidence link points here, not to M4's arithmetic-only validation.
No strategy is promoted merely because it loads or produces matching tokens.

## Verification contract

Same-schedule finite full vectors must match arithmetic decoding **bitwise**.
No tolerance relaxation follows from a failure. Table-source auditing checks all
literal bit patterns, including negative zero. GPU tests compare full scalar
projection outputs over all finite codes, awkward widths/tails and multiple valid
scales; production-seam tests exercise both strategies at counts1/2/6 with MPS and
custom execution, stage/output guards and cache/reset behavior. Host tests check
identical named-buffer accounting, invalid enums and CLI/native selection.

```sh
make test
python3 -m unittest discover -s tests -p 'test_fp8_lookup.py'
# Optional supplied tokenizer; no weights/downloads:
make service-long-test TOKENIZER=/path/to/official/tekken.json
```

For a bounded real comparison, use [controlled replay](controlled-replay.md) with
identical source/binary, model, tokenizer, prompt, trajectories, context/chunk,
prefill and reuse settings, changing only arithmetic versus lookup strategy. Pin
the artifacts and compare full vectors. Forced outputs do not score task success;
capture I/O timings are not a performance benchmark. Lookup may be slower.
Independent-reference discrepancies remain unresolved; native consistency is not
independent correctness. Long-context and broad cross-workload acceptance require
separate evidence. No energy or peak-memory claim is made.

## Bounded real-weight observation

One sequential same-binary arithmetic/lookup pair completed with MPS,
context256/chunk256 and reuseON. Each process loaded once, used two sessions and
captured eight full finite131072-element vectors. **All eight paired vectors were
bitwise identical**, including the scalar decode after the forced token. These
are two prefix positions repeated four times, not eight independent tasks.

Official tokenizer replay, full input/forced ledgers, checkpoint/pending state,
actual strategy/kernel reporting, identical named-buffer sizes and clean shutdown
were independently checked. No retries or tolerance relaxation. Task success
remained null. This establishes bounded native consistency only; there is no
speedup, successful-task throughput or independent-reference acceptance claim
from that replay.

A later [free-greedy document/history panel](useful-workload-panel.md) passed40/40
compact checks across arithmetic/lookup, chunks128/256 and two repetitions.
Actual trajectories and same-chunk work receipts matched. Lookup's reported
decode component was lower, but its overall advantage was only0.9% at256 and2.4%
at128, reversing direction between repetitions; no whole-workload winner or
default promotion follows. Chunk256 showed the clearer useful-workload benefit.
Resident256 was refused by the unchanged availability guard before tasks, and
resident128 was not attempted in that original campaign. A separately authorized
follow-up subsequently passed20/20 resident checks at both chunks with the same
binary and frozen tasks, completing the bounded configuration coverage across
separate runs. Resident256 had lower mean generation latency but substantially more
named memory and slower decode/loading; this does not promote a universal strategy
winner. Independent numerical discrepancies remain unresolved.
