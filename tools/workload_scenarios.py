"""Bounded version-two admission and asynchronous checkpoint verification.

One service process owns all repetitions; each repetition owns a fresh session.
A checkpoint alias denotes the single published server checkpoint, never a token
replay or a client-side KV snapshot. Failures stop execution without retry.
"""
import copy
import re
import time

from replay_logits import replay_tokens

NAME = re.compile(r"[A-Za-z0-9_-]{1,40}\Z", re.ASCII)
TOOLS = {"native_tool_loop", "native_investigation_loop", "native_observation_loop"}
WORK = ("prefill_attempted prefill_committed append_attempted append_committed "
        "pending_attempted pending_committed prefill_calls append_calls decode_calls "
        "decode_attempted decode_committed logit_restores recompute_attempted "
        "recompute_committed").split()


class DeadlineExceeded(TimeoutError):
    """An explicit application/request budget expired, not transport inactivity."""


class ProtocolError(RuntimeError):
    """An admitted response does not satisfy the native protocol contract."""


class ControlFailure(RuntimeError):
    """A refused or unsuccessful control; no rollback is assumed."""


def validate(value):
    fields = {"version", "mode", "prefix_reuse", "repetitions", "wall_seconds", "operations"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid version 2 fields")
    if type(value["version"]) is not int or value["version"] != 2:
        raise ValueError("expected version 2")
    if value["mode"] not in ("free_greedy", "controlled_replay") or value["prefix_reuse"] not in ("legacy", "on", "off"):
        raise ValueError("invalid version 2 policy/mode")
    for key, maximum in (("repetitions", 16), ("wall_seconds", 120)):
        if type(value[key]) is not int or not 1 <= value[key] <= maximum:
            raise ValueError("invalid " + key)
    ops = value["operations"]
    if not isinstance(ops, list) or not 2 <= len(ops) <= 256:
        raise ValueError("invalid operations bound")
    if len(ops) * value["repetitions"] > 4096:
        raise ValueError("expanded operation limit")
    if any(not isinstance(op, dict) or not isinstance(op.get("op"), str) for op in ops):
        raise ValueError("operation must be object with string op")
    if ops[0]["op"] != "create" or ops[-1]["op"] != "close":
        raise ValueError("create first and close final required")
    profiles = {o["op"] for o in ops} & TOOLS
    controls = any(o["op"] in ("checkpoint", "restore") for o in ops)
    has_tools = any(o["op"] in TOOLS | {"tool_loop"} for o in ops)
    if len(profiles) > 1 or (controls and (has_tools or value["prefix_reuse"] == "legacy")):
        raise ValueError("unsupported control/profile policy")
    planned = 0
    alias = None
    identities = set()
    for i, op in enumerate(ops):
        action = op["op"]
        if value["mode"] == "controlled_replay":
            if action not in {"create", "message", "reset", "pause", "checkpoint", "restore", "close", "replay"}:
                raise ValueError("operation prohibited in controlled replay")
        elif action == "replay":
            raise ValueError("replay requires controlled_replay")
        if action == "replay":
            if set(op) != {"op", "id", "tokens"}:
                raise ValueError("invalid replay fields")
            planned += len(replay_tokens(op["tokens"]))
        if action in ("create", "close") and i not in (0, len(ops) - 1):
            raise ValueError("extra create/close")
        needs_id = action in {"generate", "continue", "cancel", "checkpoint", "restore",
                              "score", "tool_loop", "replay"} | TOOLS
        if needs_id or "id" in op:
            if not isinstance(op.get("id"), str) or not NAME.fullmatch(op["id"]):
                raise ValueError("invalid template identity")
        if action in ("checkpoint", "restore"):
            name = op.get("checkpoint")
            if (set(op) != {"op", "id", "checkpoint"} or not isinstance(name, str)
                    or not NAME.fullmatch(name)):
                raise ValueError("invalid checkpoint operation")
            if action == "restore" and alias != name:
                raise ValueError("stale checkpoint alias")
            alias = name
        if action in ("reset", "close"):
            alias = None
        ids = []
        if action in ("generate", "continue", "cancel", "checkpoint", "restore", "replay"):
            ids = [op["id"]]
        elif action in TOOLS | {"tool_loop"}:
            if type(op.get("max_calls")) is not int or not 1 <= op["max_calls"] <= 16:
                raise ValueError("invalid max_calls")
            ids = [f"{op['id']}-{n}" for n in range(op["max_calls"] + 1)]
        for identity in ids:
            if identity in identities:
                raise ValueError("expanded request identity collision")
            identities.add(identity)
    # Reserve even calls that might not happen; never depend on a model choosing
    # an early finish to fit the non-evicting server request history.
    if value["mode"] == "controlled_replay" and not planned:
        raise ValueError("controlled replay requires a nonempty trajectory")
    if planned * value["repetitions"] > 16:
        raise ValueError("replay process capture limit")
    if len(identities) * value["repetitions"] > 256:
        raise ValueError("request history limit")


def expanded(value, repetition):
    result = copy.deepcopy(value)
    for op in result["operations"]:
        if "id" in op:
            op["id"] = f"r{repetition}-" + op["id"]
    return result


def require(condition, detail):
    if not condition:
        raise ProtocolError(detail)


def equal(actual, expected, detail):
    # JSON bool is not a token count or numeric identity, despite Python equality.
    require(type(actual) is type(expected) and actual == expected, detail)


def checkpoint_id(value):
    require(isinstance(value, str) and 1 <= len(value) <= 20
            and re.fullmatch(r"[1-9][0-9]*", value)
            and int(value) <= 18446744073709551615, "invalid checkpoint identity")


def ledger(value, session, capacity, states=("ready", "partial")):
    require(isinstance(value, dict), "invalid session metadata")
    equal(value.get("session_id"), session, value)
    require(value.get("state") in states, value)
    for key in ("logical_tokens", "committed_tokens", "pending_tokens", "remaining"):
        require(type(value.get(key)) is int and 0 <= value[key] <= capacity, value)
    tokens = value.get("token_ids")
    require(isinstance(tokens, list) and len(tokens) == value["logical_tokens"]
            and all(type(t) is int and 0 <= t < 131072 for t in tokens), value)
    require(value["logical_tokens"] + value["remaining"] == capacity
            and value["pending_tokens"] <= 1
            and value["committed_tokens"] + value["pending_tokens"] <= len(tokens), value)


def control(service, session, operation, identity, saved=None):
    require(operation in ("checkpoint", "restore"), "invalid control operation")
    require(type(session) is int and session > 0, "invalid session identity")
    require(service.info.get("prefix_reuse") in ("on", "off")
            and service.info.get("checkpoints") is True, "checkpoints not supported")
    capacity = service.info.get("context_capacity")
    require(type(capacity) is int and capacity in (256, 4352), "invalid context capacity")
    if operation == "restore":
        require(isinstance(saved, dict), "no published checkpoint")
        checkpoint_id(saved.get("checkpoint_id"))
        ledger(saved.get("ledger"), session, capacity)

    def request(method, path, body=None, expected=200):
        status, result = service.request(method, path, body)
        if status != expected:
            raise ControlFailure((method, path, status, result))
        require(isinstance(result, dict), "invalid response object")
        return result

    old = getattr(service, "absolute_deadline", None)
    deadline = time.monotonic() + service.request_seconds
    service.absolute_deadline = min(old, deadline) if old is not None else deadline
    try:
        before = request("GET", f"/sessions/{session}")
        ledger(before, session, capacity)
        body = {"id": identity}
        if operation == "restore":
            body["checkpoint_id"] = saved["checkpoint_id"]
        admission = request("POST", f"/sessions/{session}/{operation}", body, 202)
        for key, expected in {"id": identity, "operation": operation, "session_id": session,
                              "status": "running", "state": "control"}.items():
            equal(admission.get(key), expected, admission)
        ledger(admission, session, capacity, ("control",))
        for key in ("logical_tokens", "committed_tokens", "pending_tokens", "token_ids"):
            equal(admission.get(key), before[key], admission)
        while True:
            service.bounded_timeout(1)
            terminal = request("GET", f"/requests/{identity}")
            for key, expected in {"id": identity, "operation": operation,
                                  "session_id": session}.items():
                equal(terminal.get(key), expected, terminal)
            if terminal.get("status") in ("error", "cancelled"):
                raise ControlFailure(terminal)
            if terminal.get("status") == "completed":
                break
            equal(terminal.get("status"), "running", terminal)
            time.sleep(service.bounded_timeout(.01))
        for key, expected in {"session_valid": True, "recovery": "none", "generated_tokens": 0,
                              "token_ids": [], "stop_reason": "none"}.items():
            equal(terminal.get(key), expected, terminal)
        checkpoint = terminal.get("checkpoint_id")
        checkpoint_id(checkpoint)
        for key in ("state", "logical_tokens", "committed_tokens", "pending_tokens"):
            equal(terminal.get("before_" + key), before[key], terminal)
        state = request("GET", f"/sessions/{session}")
        ledger(state, session, capacity)
        source = before if operation == "checkpoint" else saved["ledger"]
        on = service.info["prefix_reuse"] == "on"
        expected = {"token_ids": source["token_ids"], "logical_tokens": source["logical_tokens"],
                    "state": source["state"], "pending_tokens": 0,
                    "committed_tokens": source["logical_tokens"]
                    if operation == "checkpoint" or on else 0}
        for key, value in expected.items():
            equal(state.get(key), value, state)
            if key != "token_ids":
                equal(terminal.get(key), value, terminal)
        metrics = terminal.get("metrics")
        require(isinstance(metrics, dict) and isinstance(metrics.get("work"), dict), terminal)
        work = metrics["work"]
        for key in WORK:
            require(type(work.get(key)) is int and work[key] >= 0, terminal)
        equal(terminal.get("cache_restored"), operation == "restore" and on, terminal)
        if operation == "restore":
            equal(checkpoint, saved["checkpoint_id"], terminal)
            for key in WORK:
                equal(work[key], int(on and key == "logit_restores"), terminal)
        service.bounded_timeout(1)
        return {"checkpoint_id": checkpoint, "ledger": state, "terminal": terminal,
                "admission": admission, "before": before}
    finally:
        service.absolute_deadline = old
