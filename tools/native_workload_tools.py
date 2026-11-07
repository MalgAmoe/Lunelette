"""Bounded native calculator/investigation/observation client; never decode or replay tokens."""
import time
import json
import hashlib
from pathlib import Path

from workload_tools import ProtocolError, LimitError, integer, validate_operation as custom_validate

CORPUS_SHA256 = "670215ba689b16ce7fbc3444c67a728682d039b7ad0fd830d917bd876b432258"
CORPUS_PATH = Path(__file__).resolve().parent / "fixtures/stock-documents-v1.json"

OBSERVATION_CORPUS_SHA256 = "136b7bcb4e89246fa7756854ce7b67264f6f32356faed379999f8098a74d53a0"
OBSERVATION_CORPUS_PATH = Path(__file__).resolve().parent / "fixtures/observations-v1.json"

def load_corpus(observation=False):
    """Read only the fixed bundled collection; validate every record before use."""
    path = OBSERVATION_CORPUS_PATH if observation else CORPUS_PATH
    expected_hash = OBSERVATION_CORPUS_SHA256 if observation else CORPUS_SHA256
    document_ids = ("pump-readings",) if observation else ("larch-stock", "cedar-stock")
    with path.open("rb") as stream:
        raw = stream.read(8193)
    if len(raw) > 8192 or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ProtocolError("observation corpus drift" if observation else "stock corpus drift")
    value = json.loads(raw)
    if (set(value) != {"version", "documents"} or type(value["version"]) is not int
            or value["version"] != 1 or type(value["documents"]) is not list
            or len(value["documents"]) != len(document_ids)):
        raise ProtocolError("invalid corpus collection")
    records = {}
    for record in value["documents"]:
        if (type(record) is not dict or set(record) != {"document_id", "passage_id", "text"}
                or type(record["document_id"]) is not str
                or record["document_id"] not in document_ids
                or record["document_id"] in records
                or record["passage_id"] != record["document_id"].upper() + ":P1"
                or type(record["text"]) is not str
                or not 1 <= len(record["text"].encode("utf-8")) <= 1024
                or len(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")) > 2048):
            raise ProtocolError("invalid corpus record")
        records[record["document_id"]] = record
    return records



def validate_operation(op):
    investigation = op.get("op") == "native_investigation_loop"
    observation = op.get("op") == "native_observation_loop"
    if "answer" not in op:
        raise ValueError("missing fixed answer")
    custom_validate({**op, "op": "tool_loop", "answer": 43} if investigation or observation else {**op, "op": "tool_loop"})
    if op["op"] not in ("native_tool_loop", "native_investigation_loop", "native_observation_loop"):
        raise ValueError("expected native_tool_loop")
    for key, upper in (("budget", 128), ("max_calls", 1 if observation else 2),
                       ("max_generated", 256 if observation else 384)):
        if op[key] > upper:
            raise ValueError(f"native {key} exceeds fixed limit")
    if op["answer"] != ("pump-A 84 ALERT" if observation else "19 [LARCH-STOCK:P1]" if investigation else 43):
        raise ValueError("invalid fixed native task score")


def capability(service, investigation=False, observation=False):
    if (not getattr(service, "native", False) or service.info.get("tools") is not True
            or service.info.get("conversation_protocol") != ("native-observation-v13" if observation else "native-investigation-v13" if investigation else "native-calculator-v13")):
        raise ProtocolError("matching native tool profile capability required")


def tokens(value):
    if (not isinstance(value, list) or not value
            or any(type(t) is not int or not 0 <= t <= 131071 for t in value)):
        raise ProtocolError("invalid native token IDs")
    return value


def ledger(service, value, session, state):
    if not isinstance(value, dict):
        raise ProtocolError("missing native ledger")
    ids = tokens(value.get("token_ids"))
    capacity = service.info.get("context_capacity")
    if type(capacity) is not int or len(ids) > capacity:
        raise ProtocolError("invalid context capacity")
    for key, expected in {"session_id": session, "state": state,
                          "logical_tokens": len(ids), "remaining": capacity - len(ids)}.items():
        if type(value.get(key)) is not type(expected) or value.get(key) != expected:
            raise ProtocolError(f"invalid ledger {key}")
    committed, pending = value.get("committed_tokens"), value.get("pending_tokens")
    if (type(committed) is not int or type(pending) is not int or committed < 0
            or pending not in (0, 1) or committed + pending > len(ids)):
        raise ProtocolError("invalid ledger cache accounting")
    return ids


def arguments(call, request_id, investigation=False, observation=False):
    if (type(call) is not dict or set(call) != {"id", "name", "arguments"}
            or type(call["id"]) is not str or call["id"] != request_id
            or type(call["name"]) is not str
            or call["name"] not in (("lookup",) if observation else ("calculator", "lookup") if investigation else ("calculator",))):
        raise ProtocolError("invalid tool call metadata")
    args = call["arguments"]
    if call["name"] == "lookup":
        if (type(args) is not dict or set(args) != {"document_id"}
                or type(args["document_id"]) is not str
                or args["document_id"] not in (("pump-readings",) if observation else ("larch-stock", "cedar-stock"))):
            raise ProtocolError("invalid lookup schema")
        return args
    if (type(args) is not dict or set(args) != {"operation", "a", "b"}
            or args["operation"] not in ("add", "subtract", "multiply")):
        raise ProtocolError("invalid calculator schema")
    integer(args["a"])
    integer(args["b"])
    return args


def validate_response(service, session, request_id, result, budget, prefix, investigation=False, observation=False):
    if not isinstance(result, dict):
        raise ProtocolError("missing terminal")
    request = result.get("request_metadata")
    if not isinstance(request, dict):
        raise ProtocolError("missing retained request")
    if (result.get("event") in ("error", "cancelled")
            and result.get("stop_reason") in ("generator_failure", "cancelled")):
        raise RuntimeError(f"native generation failed: {result['stop_reason']}")
    call = result.get("stop_reason") == "tool_call"
    state = "awaiting_tool" if call else "ready"
    if (result.get("event") != "completed" or request.get("status") != "completed"
            or request.get("operation") != "generate"
            or result.get("stop_reason") != ("tool_call" if call else "eos")):
        raise ProtocolError("generation must end naturally")
    for item in (result, request):
        for key, expected in {"id": request_id, "session_id": session,
                              "session_valid": True, "recovery": "none", "state": state}.items():
            if type(item.get(key)) is not type(expected) or item.get(key) != expected:
                raise ProtocolError(f"invalid identity/state {key}")
    for key in ("generated_tokens", "stop_reason", "logical_tokens", "committed_tokens",
                "pending_tokens", "text", "tool_call", "metrics"):
        if type(result.get(key)) is not type(request.get(key)) or result.get(key) != request.get(key):
            raise ProtocolError(f"terminal/retained mismatch {key}")
    count = result.get("generated_tokens")
    if type(count) is not int or not 1 <= count <= budget:
        raise ProtocolError("invalid generated count")
    ids = tokens(request.get("token_ids"))
    full = ledger(service, result.get("session_metadata"), session, state)
    if len(ids) != count or full != prefix + ids or ids[-1] != 2:
        raise ProtocolError("generated suffix mismatch")
    for key, expected in {"logical_tokens": len(full), "committed_tokens": len(full) - 1,
                          "pending_tokens": 1}.items():
        for item in (result, result["session_metadata"]):
            if type(item.get(key)) is not int or item[key] != expected:
                raise ProtocolError("terminal cache accounting mismatch")
    ordinary = lambda seq: bool(seq) and all(1000 <= t <= 131071 for t in seq)
    if call:
        if ids[0] != 9 or ids.count(32) != 1:
            raise ProtocolError("invalid native call framing")
        split = ids.index(32)
        if not ordinary(ids[1:split]) or not ordinary(ids[split + 1:-1]):
            raise ProtocolError("invalid native call body")
        if (result.get("text") != "" or result.get("terminal_text") != ""
                or type(result.get("text_delta_count")) is not int
                or result["text_delta_count"] != 0):
            raise ProtocolError("native call emitted text")
        arguments(result.get("tool_call"), request_id, investigation, observation)
        arguments(request.get("tool_call"), request_id, investigation, observation)
    elif ("tool_call" in result or "tool_call" in request
          or not isinstance(result.get("text"), str)
          or any(not 1000 <= t <= 131071 for t in ids[:-1])):
        raise ProtocolError("invalid final response")
    return full


def run_loop(service, session, op, reserve_id):
    validate_operation(op)
    started = time.monotonic()
    previous = getattr(service, "absolute_deadline", None)
    service.absolute_deadline = min(previous, started + op["wall_seconds"]) if previous is not None else started + op["wall_seconds"]
    investigation = op["op"] == "native_investigation_loop"
    observation = op["op"] == "native_observation_loop"
    calls = generated = 0
    operations = []
    durations = []
    corpus = None
    report = {"protocol_success": False, "task_success": None, "failure_kind": None,
              "native_tools": True, "diagnostic_only": True, "tool_loop_success": False}

    def check():
        if time.monotonic() >= service.absolute_deadline:
            raise LimitError("absolute wall deadline exceeded")

    try:
        capability(service, investigation, observation)
        if investigation or observation:
            corpus = load_corpus(observation=observation)
            corpus_hash = OBSERVATION_CORPUS_SHA256 if observation else CORPUS_SHA256
            report.update(corpus_version=1, corpus_sha256=corpus_hash)
            service.record("observation_corpus" if observation else "investigation_corpus",
                           version=1, sha256=corpus_hash)
        if type(session) is not int or session <= 0:
            raise ProtocolError("active session required")
        check()
        status, initial = service.request("GET", f"/sessions/{session}")
        check()
        if status != 200:
            raise RuntimeError("cannot inspect input ledger")
        prefix = ledger(service, initial, session, "ready")
        if prefix[:2] != [1, 5] or 6 not in prefix or prefix[-1] not in (4, 8):
            raise ProtocolError("native tool declaration/input framing required")
        while True:
            check()
            budget = min(op["budget"], op["max_generated"] - generated)
            if budget <= 0:
                raise LimitError("total generated token limit")
            request_id = f"{op['id']}-{calls}"
            reserve_id(request_id)
            result = service.generate(session, request_id, budget)
            count = result.get("generated_tokens") if isinstance(result, dict) else None
            if (type(count) is int and 1 <= count <= budget and result.get("id") == request_id
                    and type(result.get("session_id")) is int and result["session_id"] == session):
                generated += count
            check()
            prefix = validate_response(service, session, request_id, result, budget, prefix, investigation, observation)
            if result["stop_reason"] == "eos":
                sequence_ok = operations == [("lookup", {"document_id": "larch-stock"}),
                                              ("calculator", {"operation": "subtract", "a": 31, "b": 12})]
                if observation:
                    sequence_ok = operations == [("lookup", {"document_id": "pump-readings"})]
                success = (sequence_ok and result["text"] == op["answer"]) if investigation or observation else (calls > 0 and result["text"].strip() == "43")
                report.update(protocol_success=True, task_success=success,
                              answer=result["text"], tool_loop_success=sequence_ok if investigation or observation else calls > 0)
                break
            if calls >= op["max_calls"]:
                raise LimitError("tool call limit")
            args = arguments(result["tool_call"], request_id, investigation, observation)
            check()
            tool_started = time.monotonic()
            name = result["tool_call"]["name"]
            if name == "lookup":
                answer = corpus[args["document_id"]]
                content = json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            else:
                a, b = args["a"], args["b"]
                answer = {"add": lambda: a + b, "subtract": lambda: a - b,
                          "multiply": lambda: a * b}[args["operation"]]()
                integer(answer, 10**12)
                content = str(answer)
            duration = time.monotonic() - tool_started
            durations.append(duration)
            operations.append((name, args.copy()))
            service.record("native_tool_operation", name=name, arguments=args,
                           tool_seconds=duration, tool_call_id=request_id)
            calls += 1
            check()
            status, reply = service.request("POST", f"/sessions/{session}/messages",
                                           {"role": "tool", "tool_call_id": request_id,
                                            "content": content})
            check()
            if status != 200:
                raise RuntimeError("tool result rejected")
            full = ledger(service, reply, session, "ready")
            suffix = full[len(prefix):]
            if (full[:len(prefix)] != prefix or len(suffix) < 3 or suffix[0] != 7
                    or suffix[-1] != 8 or any(not 1000 <= t <= 131071 for t in suffix[1:-1])
                    or reply.get("committed_tokens") != len(prefix) - 1
                    or reply.get("pending_tokens") != 1):
                raise ProtocolError("tool result ledger not preserved/framed")
            prefix = full
    except (ProtocolError, LimitError) as error:
        report.update(failure_kind="protocol" if isinstance(error, ProtocolError) else "limit", error=str(error))
    except Exception as error:
        report.update(failure_kind="limit" if isinstance(error, TimeoutError) and time.monotonic() >= service.absolute_deadline else "runtime", error=repr(error))
    finally:
        service.absolute_deadline = previous
    report.update(tool_calls=calls, tool_used=calls > 0, generated_tokens=generated,
                  tool_seconds=sum(durations), tool_call_seconds=durations,
                  tool_operations=operations, duration_seconds=time.monotonic() - started)
    service.record("native_tool_loop_result", **report)
    return report
