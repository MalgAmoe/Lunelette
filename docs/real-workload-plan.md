# Real-workload M1 execution plan (prototype v1)

> Historical M1 implementation plan, not the current delivery status. See the
> [coverage report](real-workload-status.md) for delivered capabilities and the
> [specification](real-workload-spec.md) for subsequent amendments.

M1 delivered only a model-free foundation, not the real service or full specification.

## Sequence
1. Inspect existing JSON parser; reuse strict string decoding and primitives.
2. Implement bounded loopback HTTP poll transport and serialized generator worker.
3. Implement test-ledger session accounting, request tombstones, UTF-8 streaming.
4. Add stdlib scenario runner, versioned fixtures, C and live protocol tests.
5. Run portable core, sanitizers, NDEBUG and protocol checks; inspect final diff.

## Prototype contract
Test executable only: build/test_service_server. No production server or model
fallback. diagnostic_test_generator, conversation_protocol=test-ledger-v1,
model_loaded=false, runtime_ready=false, controlled_protocol_only. Capacity 256
logical tokens, including pending output; input UTF-8 bytes are synthetic tokens.
No official conversation framing, inference, tools, checkpoints or performance
claims. Unsupported planned operations return 501, never no-op success.

| State | inspect | message | generate/continue | cancel | reset | close |
|---|---|---|---|---|---|---|
| ready | yes | yes | yes | request-specific | yes | yes |
| generating | yes | 409 | 409 | immediate acknowledgement | 409 | 409 |
| partial | yes | 409 | yes | request-specific | yes | yes |
| needs_reset | yes | 409 | 409 | request-specific | yes | yes |
| closed | 410 | 410 | 410 | request-specific | 410 | 410 |

One active session. Session IDs monotonically increase; older IDs return 410.
Generation admission atomically reserves one of 256 lifetime request IDs (ASCII
1..64); no eviction/replay, even after reset. Rejected admissions do not reserve
IDs. Existing IDs return 409 with existing status; capacity exhaustion requires
restart. Results bounded by fixed fixture output/context and fixed event ring.
Budget exhaustion is partial, EOS is ready; cancellation/disconnect/error requires
reset. Pending UTF-8 survives continuation; invalid input/output fails safely.

HTTP/1.1, Content-Length, no transfer encodings, close per response; NDJSON stream.
Bounded clients, header/body limits and deadlines. One poll thread owns sockets;
one worker owns generator calls. Cancellation is checked at safe tick boundaries.
Worker cleanup is joined before reset/destroy. Signal handler only sets a flag;
poll wakes within 50ms. SIGPIPE ignored; send failures cancel streams.

## vNext (not M1)
Resident Metal adapter, authoritative pinned tokenizer/framing, numerical gates,
checkpoint/recompute, tools, compact strategies, long context and real workloads.
All absent capabilities must be explicitly unsupported. Unknown measurements are
null; synthetic timings are not model performance. Task success remains null.

## Evidence
Record commands and outcomes outside engine under workspace evidence. No model
runs, dependencies, downloads, commits, pushes or delegated agents.
