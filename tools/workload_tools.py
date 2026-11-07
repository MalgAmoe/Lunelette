"""Bounded application-level calculator protocol; no native tool framing."""
import json
import re
import time


class ProtocolError(ValueError):
    pass


class LimitError(RuntimeError):
    pass


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ProtocolError(f"duplicate field: {key}")
        result[key] = value
    return result


def integer(value, bound=1000000):
    if type(value) is not int or abs(value) > bound:
        raise ProtocolError("expected bounded integer")
    return value


def reject_number(value):
    raise ProtocolError("only bounded integers are supported")


def parse(text):
    try:
        if not isinstance(text, str) or len(text.encode("utf-8")) > 2048:
            raise ProtocolError("response exceeds 2048 bytes")
    except UnicodeError as error:
        raise ProtocolError("invalid Unicode response") from error

    def parse_int(value):
        if len(value.lstrip("-")) > 13:
            raise ProtocolError("integer too long")
        return integer(int(value), 1000000000000)
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_int=parse_int,
                           parse_float=reject_number, parse_constant=reject_number)
    except (ValueError, RecursionError) as error:
        raise ProtocolError(str(error)) from error
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise ProtocolError("expected application protocol version 1")
    kind = value.get("type")
    fields = {"tool": {"version", "type", "id", "name", "arguments"},
              "final": {"version", "type", "answer"}}
    if not isinstance(kind, str) or kind not in fields or set(value) != fields[kind]:
        raise ProtocolError("unknown response type or fields")
    if kind == "final":
        integer(value["answer"], 1000000000000)
    else:
        if not isinstance(value["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", value["id"]):
            raise ProtocolError("invalid tool call id")
        if value["name"] != "calculator":
            raise ProtocolError("tool not allowlisted")
        args = value["arguments"]
        if not isinstance(args, dict) or set(args) != {"operation", "a", "b"}:
            raise ProtocolError("invalid calculator arguments")
        if args["operation"] not in ("add", "subtract", "multiply"):
            raise ProtocolError("operation not allowlisted")
        integer(args["a"])
        integer(args["b"])
    return value


def validate_operation(op):
    fields = {"op", "id", "budget", "max_calls", "max_generated", "wall_seconds", "answer"}
    if set(op) != fields:
        raise ValueError("invalid tool_loop fields")
    if not isinstance(op["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", op["id"]):
        raise ValueError("invalid tool_loop id")
    for name, upper in (("budget", 256), ("max_calls", 16), ("max_generated", 4096),
                        ("wall_seconds", 120)):
        if type(op[name]) is not int or not 1 <= op[name] <= upper:
            raise ValueError(f"invalid {name}")
    integer(op["answer"], 1000000000000)


def validate_response(service, session, request_id, result, budget):
    """Fail closed before interpreting application JSON or appending a result."""
    if not isinstance(result, dict):
        raise RuntimeError("invalid terminal response")
    if (result.get("id") != request_id or type(result.get("session_id")) is not int
            or result["session_id"] != session or result.get("session_valid") is not True
            or result.get("recovery") != "none"):
        raise RuntimeError("invalid terminal identity or recovery state")
    count = result.get("generated_tokens")
    if type(count) is not int or not 1 <= count <= budget:
        raise RuntimeError("invalid generated token accounting")
    if result.get("event") != "completed":
        raise RuntimeError("generation did not complete")
    if result.get("stop_reason") != "eos" or result.get("state") != "ready":
        raise ProtocolError("tool protocol requires natural EOS and ready state")
    if not getattr(service, "native", False):
        ledger = result.get("session_metadata")
        if ledger is not None and (not isinstance(ledger, dict) or ledger.get("state") != "ready"):
            raise RuntimeError("session ledger is not ready")
        return count
    request = result.get("request_metadata")
    ledger = result.get("session_metadata")
    if not isinstance(request, dict) or not isinstance(ledger, dict):
        raise RuntimeError("missing native metadata")
    if (request.get("status") != "completed" or request.get("operation") != "generate"
            or request.get("id") != request_id
            or type(request.get("session_id")) is not int or request["session_id"] != session
            or request.get("session_valid") is not True or request.get("recovery") != "none"
            or type(ledger.get("session_id")) is not int or ledger["session_id"] != session):
        raise RuntimeError("invalid native request identity or status")
    for key in ("generated_tokens", "stop_reason", "state", "logical_tokens",
                "committed_tokens", "pending_tokens"):
        if type(request.get(key)) is not type(result.get(key)) or request.get(key) != result.get(key):
            raise RuntimeError(f"native terminal/request mismatch: {key}")
    ids = request.get("token_ids")
    tokens = ledger.get("token_ids")
    for sequence in (ids, tokens):
        if (not isinstance(sequence, list) or not sequence
                or any(type(token) is not int or not 0 <= token <= 131071 for token in sequence)):
            raise RuntimeError("invalid native token IDs")
    if len(ids) != count or ids[-1] != 2 or 2 in ids[:-1] or tokens[-count:] != ids:
        raise RuntimeError("invalid native generated suffix or EOS")
    capacity = service.info.get("context_capacity")
    if type(capacity) is not int or not len(tokens) <= capacity:
        raise RuntimeError("invalid native context capacity")
    expected = {"logical_tokens": len(tokens), "committed_tokens": len(tokens) - 1,
                "pending_tokens": 1, "remaining": capacity - len(tokens), "state": "ready"}
    for key, value in expected.items():
        if type(ledger.get(key)) is not type(value) or ledger.get(key) != value:
            raise RuntimeError(f"invalid native ledger: {key}")
        if key != "remaining" and (type(result.get(key)) is not type(value) or result.get(key) != value):
            raise RuntimeError(f"native terminal/ledger mismatch: {key}")
    return count


def run_loop(service, session, op, reserve_id):
    """Never replay assistant text or automatically repair a mutated session."""
    validate_operation(op)
    started = time.monotonic()
    deadline = started + op["wall_seconds"]
    previous = getattr(service, "absolute_deadline", None)
    service.absolute_deadline = min(previous, deadline) if previous is not None else deadline
    calls, generated, tool_seconds = 0, 0, 0.0
    seen = set()
    report = {"protocol_success": False, "task_success": None, "failure_kind": None,
              "diagnostic_only": True, "native_tools": False}

    def check_time():
        if time.monotonic() >= service.absolute_deadline:
            raise LimitError("tool loop wall deadline exceeded")

    try:
        if type(session) is not int or session <= 0:
            raise RuntimeError("tool loop requires an active integer session")
        while True:
            check_time()
            remaining = op["max_generated"] - generated
            if remaining <= 0:
                raise LimitError("total generated token limit")
            budget = min(op["budget"], remaining)
            request_id = f"{op['id']}-{len(seen)}"
            reserve_id(request_id)
            result = service.generate(session, request_id, budget)
            service.record("tool_loop_response", response=result)
            check_time()
            # Retain bounded reported work even when a budget stop fails the
            # application protocol. Never count another request's response.
            count = result.get("generated_tokens") if isinstance(result, dict) else None
            if (type(count) is int and 1 <= count <= budget
                    and result.get("id") == request_id
                    and type(result.get("session_id")) is int
                    and result["session_id"] == session):
                generated += count
            validate_response(service, session, request_id, result, budget)
            value = parse(result.get("text"))
            if value["type"] == "final":
                report.update(protocol_success=True, task_success=value["answer"] == op["answer"],
                              answer=value["answer"])
                break
            if value["id"] in seen:
                raise ProtocolError("duplicate tool call id")
            if calls >= op["max_calls"]:
                raise LimitError("tool call limit")
            seen.add(value["id"])
            check_time()
            tool_start = time.monotonic()
            args = value["arguments"]
            a, b = args["a"], args["b"]
            operation = args["operation"]
            if operation == "add":
                answer = a + b
            elif operation == "subtract":
                answer = a - b
            else:
                answer = a * b
            integer(answer, 1000000000000)
            tool_seconds += time.monotonic() - tool_start
            calls += 1
            check_time()
            content = "APPLICATION_TOOL_RESULT_V1 " + json.dumps(
                {"version": 1, "id": value["id"], "name": "calculator", "result": answer},
                sort_keys=True, separators=(",", ":"))
            service.record("tool_result", content=content, tool_seconds=tool_seconds)
            status, reply = service.request("POST", f"/sessions/{session}/messages",
                                            {"role": "user", "content": content})
            check_time()
            if status != 200:
                raise RuntimeError(f"tool result append failed: {reply}")
    except (ProtocolError, LimitError) as error:
        report.update(failure_kind="protocol" if isinstance(error, ProtocolError) else "limit",
                      error=str(error))
    except TimeoutError as error:
        expired = time.monotonic() >= service.absolute_deadline
        report.update(failure_kind="limit" if expired else "runtime", error=repr(error))
    except Exception as error:
        report.update(failure_kind="runtime", error=repr(error))
    finally:
        service.absolute_deadline = previous
    report.update(tool_calls=calls, generated_tokens=generated, tool_seconds=tool_seconds,
                  duration_seconds=time.monotonic() - started)
    service.record("tool_loop_result", **report)
    return report
