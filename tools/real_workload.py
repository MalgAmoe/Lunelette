#!/usr/bin/env python3
"""Versioned diagnostic workload runner. Explicit native or model-free launch."""
import argparse
import http.client
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import time
import threading

from workload_tools import run_loop, validate_operation
import native_workload_tools
import workload_scenarios
import replay_logits


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key}")
        result[key] = value
    return result


def load_scenario(path, context=256):
    value = json.loads(Path(path).read_text(), object_pairs_hook=strict_object)
    return validate_scenario(value, context)


def validate_scenario(value, context=256):
    if not isinstance(value, dict):
        raise ValueError("scenario must be an object")
    v2 = type(value.get("version")) is int and value["version"] == 2
    if v2:
        workload_scenarios.validate(value)
    if not v2 and (set(value) != {"version", "mode", "operations"} or type(value["version"]) is not int or value["version"] != 1):
        raise ValueError("expected version 1 scenario schema")
    if value["mode"] not in ({"free_greedy", "controlled_replay"} if v2 else {"controlled_protocol_only", "free_greedy"}):
        raise ValueError("unsupported scenario mode")
    if not isinstance(value["operations"], list) or len(value["operations"]) > 256:
        raise ValueError("operations must be bounded list")
    fields = {"create": set(), "message": {"content"}, "generate": {"id", "budget"},
              "continue": {"id", "budget"}, "cancel": {"id", "budget"},
              "pause": {"seconds"}, "reset": set(), "close": set(),
              "score": {"id", "text"},
              "tool_loop": {"id", "budget", "max_calls", "max_generated", "wall_seconds", "answer"}}
    if v2:
        fields.update(checkpoint={"id", "checkpoint"}, restore={"id", "checkpoint"}, replay={"id", "tokens"})
    fields["native_tool_loop"] = fields["tool_loop"]
    fields["native_investigation_loop"] = fields["tool_loop"]
    fields["native_observation_loop"] = fields["tool_loop"]
    for index, op in enumerate(value["operations"]):
        if not isinstance(op, dict) or not isinstance(op.get("op"), str) or op["op"] not in fields:
            raise ValueError("unknown operation")
        if op["op"] in {"native_tool_loop", "native_investigation_loop", "native_observation_loop"}:
            native_workload_tools.validate_operation(op)
        if op["op"] == "tool_loop":
            validate_operation(op)
        required = fields[op["op"]]
        if not required <= op.keys() or set(op) - required - {"op", "expect"}:
            raise ValueError("invalid operation fields")
        if op["op"] == "score":
            previous = value["operations"][index - 1] if index else {}
            if (value["mode"] != "free_greedy" or "expect" in op
                    or not isinstance(op["text"], str)
                    or len(op["text"].encode("utf-8")) > 4096
                    or previous.get("op") not in {"generate", "continue"}
                    or previous.get("id") != op["id"]):
                raise ValueError("score requires immediate matching generation in free_greedy")
        if "expect" in op:
            if not isinstance(op["expect"], dict) or set(op["expect"]) - {
                    "text", "state", "logical_tokens", "committed_tokens", "pending_tokens", "event"}:
                raise ValueError("invalid expectation")
            if "event" in op["expect"] and (op["op"] not in {"generate", "continue", "cancel"} or
                    not isinstance(op["expect"]["event"], str) or
                    op["expect"]["event"] not in {"completed", "cancelled", "error"}):
                raise ValueError("expected event must be a terminal generation event")
        if op["op"] == "pause":
            if type(op["seconds"]) not in (int, float) or not 0 <= op["seconds"] <= 5:
                raise ValueError("pause outside 0..5 seconds")
        if "budget" in op and (type(op["budget"]) is not int or not 1 <= op["budget"] <= 256):
            raise ValueError("invalid budget")
        if "id" in op and (not isinstance(op["id"], str) or not 1 <= len(op["id"]) <= 64):
            raise ValueError("invalid id")
        if "content" in op and (not isinstance(op["content"], str) or len(op["content"].encode()) > ((32768 if context == 4352 else 4096) if value["mode"] in {"free_greedy", "controlled_replay"} else 256)):
            raise ValueError("invalid content")
    return value


class Service:
    """Owns only the child it launches. Every request/arrival/failure is recorded."""
    def __init__(self, executable, output, tick_ms=50, native_args=None, request_seconds=120,
                 expected_strategy=None, evaluation_mode="free_greedy"):
        if type(request_seconds) not in (int, float) or not 1 <= request_seconds <= 120:
            raise ValueError("request_seconds outside 1..120")
        if expected_strategy is not None and (native_args is None or expected_strategy not in
                {"resident-fp32", "compact-fp8-arithmetic", "compact-fp8-lookup"}):
            raise ValueError("expected strategy requires a supported native launch")
        if evaluation_mode not in {"free_greedy", "controlled_replay"} or (evaluation_mode == "controlled_replay" and native_args is None):
            raise ValueError("invalid evaluation mode/launch")
        if native_args and any(a in {"--evaluation-mode", "--replay-logits"} for a in native_args):
            raise ValueError("evaluation/capture arguments are owned by Service")
        if evaluation_mode == "controlled_replay" and "--tools" in native_args:
            raise ValueError("replay prohibits tools")
        self.evaluation_mode = evaluation_mode
        self.native = native_args is not None
        self.request_seconds = request_seconds if self.native else 15
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = (self.output / "client.ndjson").open("x")
        self.stderr = (self.output / "server.stderr").open("x")
        try:
            arguments = [str(Path(executable).resolve()), "--port", "0"]
            arguments += native_args if self.native else ["--tick-ms", str(tick_ms)]
            if evaluation_mode == "controlled_replay":
                self.replay_path = self.output.resolve() / "replay-logits.bin"
                arguments += ["--evaluation-mode", "controlled-replay", "--replay-logits", str(self.replay_path)]
            self.process = subprocess.Popen(arguments,
                                            stdout=self.stderr, stderr=subprocess.PIPE)
        except BaseException as error:
            self.record("failure", phase="startup", error=repr(error))
            self.stderr.close()
            self.log.close()
            raise
        self.port = None
        self.stderr_thread = None
        self.stderr_error = None
        self.stderr_stop = threading.Event()
        line = bytearray()
        try:
            deadline = time.monotonic() + (180 if self.native else 3)
            startup_bytes = 0
            while self.port is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stderr], [], [], remaining)[0]:
                    raise TimeoutError("no startup announcement")
                chunk = os.read(self.process.stderr.fileno(), 1)
                if not chunk:
                    raise RuntimeError("server exited before readiness; see server.stderr")
                startup_bytes += 1
                if len(line) >= 4096 or startup_bytes > 65536:
                    raise RuntimeError("startup output exceeds bounds")
                line.extend(chunk)
                if chunk != b"\n":
                    continue
                complete_line = bytes(line)
                line.clear()
                self.stderr.write(complete_line.decode(errors="replace"))
                self.stderr.flush()
                try:
                    announcement = json.loads(complete_line)
                except ValueError:
                    if not self.native:
                        raise
                else:
                    if announcement.get("event") == "listening":
                        self.port = announcement["port"]
            if self.native:
                self.stderr_thread = threading.Thread(target=self.drain_stderr, daemon=True)
                self.stderr_thread.start()
            health = self.request("GET", "/health")[1]
            info = self.request("GET", "/info")[1]
            if self.native:
                if not health.get("model_loaded") or not health.get("runtime_ready") or info.get("generator") != ("native_controlled_replay" if evaluation_mode == "controlled_replay" else "native_greedy"):
                    raise RuntimeError("native runtime not ready")
                if info.get("numerical_validation") != "diagnostic_known_reference_mismatches":
                    raise RuntimeError("unexpected numerical status")
                if expected_strategy is not None and info.get("execution_strategy") != expected_strategy:
                    raise RuntimeError("native execution strategy mismatch")
            elif not health.get("operational_test_service") or health.get("model_loaded") is not False:
                raise RuntimeError("unexpected readiness")
            if not self.native and (info.get("generator") != "diagnostic_test_generator" or info.get("runtime_ready") is not False):
                raise RuntimeError("not the diagnostic protocol service")
            if evaluation_mode == "controlled_replay":
                for key, expected in {"evaluation_mode": evaluation_mode, "replay_max_tokens": 16, "replay_capture_limit": 16}.items():
                    workload_scenarios.equal(info.get(key), expected, "replay info mismatch")
            self.info = info
        except BaseException as error:
            if line:
                self.stderr.write(line.decode(errors="replace"))
                self.stderr.flush()
            self.record("failure", phase="startup", error=repr(error))
            try:
                self.close()
            except BaseException as cleanup_error:
                # Preserve the cause of startup failure, not just cleanup's error.
                print(f"real-workload: startup cleanup: {cleanup_error}", file=sys.stderr)
            raise

    def drain_stderr(self):
        # Only this thread writes the diagnostic file after native readiness.
        # Draining while the child runs prevents a full stderr pipe from blocking
        # the serialized model worker (and its cancellation boundary).
        try:
            trailing_bytes = 0
            while True:
                ready = select.select([self.process.stderr], [], [], .1)[0]
                if not ready:
                    if self.stderr_stop.is_set():
                        return
                    continue
                chunk = os.read(self.process.stderr.fileno(), 4096)
                if not chunk:
                    return
                self.stderr.write(chunk.decode(errors="replace"))
                self.stderr.flush()
                if self.stderr_stop.is_set():
                    trailing_bytes += len(chunk)
                    if trailing_bytes >= 65536:
                        raise RuntimeError("diagnostic output exceeds post-shutdown drain bound")
        except Exception as error:
            self.stderr_error = repr(error)

    def record(self, kind, **values):
        self.log.write(json.dumps({"kind": kind, "monotonic": time.monotonic(), **values},
                                  ensure_ascii=False) + "\n")
        self.log.flush()

    def bounded_timeout(self, seconds):
        deadline = getattr(self, "absolute_deadline", None)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise workload_scenarios.DeadlineExceeded("application wall deadline exceeded")
            return min(seconds, remaining)
        return seconds

    def connect(self):
        return http.client.HTTPConnection("127.0.0.1", self.port,
                                          timeout=self.bounded_timeout(4))

    def request(self, method, path, body=None):
        connection = self.connect()
        payload = b"" if body is None else json.dumps(body).encode()
        sent = time.monotonic()
        raw = bytearray()
        self.record("send", method=method, path=path, body=body)
        try:
            connection.request(method, path, payload, {"Content-Length": str(len(payload))})
            transport_socket = getattr(connection, "sock", None)
            response = connection.getresponse()
            # read() can be held alive by trickling bytes: its socket timeout is
            # inactivity-based. Bound each available-data read and total bytes.
            while True:
                timeout = self.bounded_timeout(4)
                # Python 3.12 closes a length-delimited response's socket as
                # soon as read1 consumes its last byte, before another read.
                if getattr(response, "isclosed", lambda: False)():
                    break
                if transport_socket is not None:
                    transport_socket.settimeout(timeout)
                chunk = response.read1(4096)
                self.bounded_timeout(4)
                if not chunk:
                    break
                if len(raw) + len(chunk) > 65536:
                    raise workload_scenarios.ProtocolError("response exceeds 65536 bytes")
                raw.extend(chunk)
            result = json.loads(raw, object_pairs_hook=strict_object)
            self.record("response", status=response.status, raw=raw.decode(), elapsed=time.monotonic() - sent)
            return response.status, result
        except BaseException as error:
            self.record("failure", error=repr(error), path=path, partial_raw_hex=raw.hex())
            deadline = getattr(self, "absolute_deadline", None)
            if (isinstance(error, (TimeoutError, socket.timeout)) and deadline is not None
                    and time.monotonic() >= deadline):
                raise workload_scenarios.DeadlineExceeded("application wall deadline exceeded") from error
            raise
        finally:
            connection.close()

    def generate(self, session, request_id, budget, cancel=False, replay_tokens=None):
        replay = replay_tokens is not None
        if replay != (getattr(self, "evaluation_mode", "free_greedy") == "controlled_replay"):
            raise ValueError("generation evaluation mode mismatch")
        payload = {"id": request_id, "budget": budget}
        before = None
        if replay:
            replay_logits.replay_tokens(replay_tokens)
            budget = len(replay_tokens)
            payload.update(budget=budget, replay_tokens=",".join(map(str, replay_tokens)))
            status, before = self.request("GET", f"/sessions/{session}")
            workload_scenarios.require(status == 200, "cannot inspect replay prefix")
            workload_scenarios.ledger(before, session, self.info["context_capacity"])
        connection = self.connect()
        if getattr(self, "native", False):
            connection.timeout = self.bounded_timeout(self.request_seconds)
        path = f"/sessions/{session}/generate"
        body = json.dumps(payload).encode()
        sent = time.monotonic()
        self.record("send", method="POST", path=path, body=body.decode())
        first_text = None
        events, text, raw_line = [], "", bytearray()
        terminal_seen = False
        cancel_sent = cancel_ack_seconds = cancel_completion_seconds = None
        # Fixed test generator reaches EOS within ten one-second ticks. This
        # is a diagnostic transport deadline, not a real-model latency limit.
        deadline = sent + getattr(self, "request_seconds", 15)
        if getattr(self, "absolute_deadline", None) is not None:
            deadline = min(deadline, self.absolute_deadline)
        try:
            connection.request("POST", path, body)
            transport_socket = getattr(connection, "sock", None)
            response = connection.getresponse()
            if response.status != 200 or response.getheader("Content-Type") != "application/x-ndjson":
                raise RuntimeError((response.status, response.read()))
            if cancel:
                cancel_sent = time.monotonic()
                status, ack = self.request("POST", f"/requests/{request_id}/cancel", {})
                cancel_ack_seconds = time.monotonic() - cancel_sent
                if status != 200 or not ack["cancel_requested"]:
                    raise RuntimeError("cancellation not acknowledged")
            # read1 observes actual available payload, not readline's newline wait.
            while True:
                if transport_socket:
                    remaining = max(0.001, deadline - time.monotonic())
                    transport_socket.settimeout(remaining if getattr(self, "native", False)
                                                else min(4.0, remaining))
                chunk = response.read1(1024)
                arrived = time.monotonic()
                if arrived >= deadline:
                    raise workload_scenarios.DeadlineExceeded("diagnostic generation deadline exceeded")
                if not chunk:
                    break
                self.record("arrival", raw_hex=chunk.hex(), since_send=arrived - sent)
                for byte in chunk:
                    raw_line.append(byte)
                    if len(raw_line) > 8192:
                        raise ValueError("stream event exceeds byte limit")
                    if byte != 10:
                        continue
                    event = json.loads(raw_line.decode("utf-8"), object_pairs_hook=strict_object)
                    raw_line.clear()
                    if not isinstance(event, dict) or event.get("id") != request_id:
                        raise ValueError("stream request identity mismatch")
                    if replay:
                        replay_logits.replay_tokens(event.get("replay_tokens"))
                        workload_scenarios.equal(event.get("session_id"), session, event)
                        workload_scenarios.equal(event.get("evaluation_mode"), "controlled_replay", event)
                        workload_scenarios.equal(event.get("replay_tokens"), replay_tokens, event)
                        count = event.get("generated_tokens")
                        workload_scenarios.require(type(count) is int and 0 <= count <= budget, event)
                        if events:
                            workload_scenarios.require(count >= events[-1]["generated_tokens"], event)
                        else:
                            workload_scenarios.equal(count, 0, event)
                    kind = event.get("event")
                    if terminal_seen or len(events) >= budget + 2:
                        raise ValueError("events after terminal or event count exceeded")
                    if not events:
                        if kind != "started":
                            raise ValueError("stream must begin with started")
                    elif kind not in {"text_delta", "completed", "cancelled", "error"}:
                        raise ValueError("invalid stream event order or type")
                    if not isinstance(event.get("text"), str):
                        raise ValueError("stream text must be a string")
                    event["text"].encode("utf-8", errors="strict")
                    terminal_seen = kind in {"completed", "cancelled", "error"}
                    if terminal_seen and cancel_sent is not None:
                        cancel_completion_seconds = arrived - cancel_sent
                    events.append(event)
                    if kind == "text_delta":
                        if event["text"] and first_text is None:
                            first_text = arrived - sent
                        text += event["text"]
            if raw_line or not events or events[-1]["event"] not in {"completed", "cancelled", "error"}:
                raise RuntimeError("missing complete terminal event")
            result = {**events[-1], "text": text,
                      "text_delta_count": sum(e["event"] == "text_delta" for e in events),
                      "terminal_text": events[-1]["text"], "first_actual_text_seconds": first_text,
                      "completion_seconds": time.monotonic() - sent,
                      "cancel_ack_seconds": cancel_ack_seconds,
                      "cancel_completion_seconds": cancel_completion_seconds}
            if getattr(self, "native", False):
                status, retained = self.request("GET", f"/requests/{request_id}")
                if status != 200:
                    raise RuntimeError("cannot inspect native request")
                if result["event"] == "completed" and retained["text"] != text:
                    raise ValueError("streamed/retained text mismatch")
                if result.get("metrics") != retained.get("metrics"):
                    raise ValueError("terminal/retained metrics mismatch")
                result["request_metadata"] = retained
                status, ledger = self.request("GET", f"/sessions/{session}")
                if status != 200:
                    raise RuntimeError("cannot inspect native ledger")
                result["session_metadata"] = ledger
                if replay:
                    if result["event"] in {"error", "cancelled"}:
                        raise workload_scenarios.ControlFailure(result)
                    replay_logits.replay_tokens(retained.get("replay_tokens"))
                    for key, expected in {"id": request_id, "session_id": session,
                                          "operation": "generate"}.items():
                        workload_scenarios.equal(retained.get(key), expected, retained)
                    for key in ("state", "logical_tokens", "committed_tokens", "pending_tokens"):
                        for source in (result, retained):
                            workload_scenarios.equal(source.get("before_" + key), before[key], source)
                    for key, expected in {"evaluation_mode": "controlled_replay", "replay_tokens": replay_tokens}.items():
                        workload_scenarios.equal(retained.get(key), expected, retained)
                    ids = retained.get("token_ids")
                    workload_scenarios.require(isinstance(ids, list) and all(type(t) is int for t in ids), retained)
                    workload_scenarios.equal(ids, replay_tokens[:len(ids)], "replay emitted IDs mismatch")
                    workload_scenarios.equal(result.get("generated_tokens"), len(ids), "terminal replay count mismatch")
                    result["token_ids"] = ids
                    workload_scenarios.equal(retained.get("status"), result["event"], retained)
                    workload_scenarios.equal(retained.get("generated_tokens"), len(ids), retained)
                    workload_scenarios.ledger(ledger, session, self.info["context_capacity"])
                    workload_scenarios.equal(ledger["token_ids"], before["token_ids"] + ids, "replay ledger mismatch")
                    workload_scenarios.equal(ids, replay_tokens, "incomplete completed replay")
                    ended = replay_tokens[-1] == 2
                    expected_state = "ready" if ended else "partial"
                    expected_counts = {"state": expected_state,
                                       "logical_tokens": len(before["token_ids"]) + len(ids),
                                       "committed_tokens": len(before["token_ids"]) + len(ids) - 1,
                                       "pending_tokens": 1}
                    for source in (result, retained, ledger):
                        for key, expected in expected_counts.items():
                            workload_scenarios.equal(source.get(key), expected, source)
                    for source in (result, retained):
                        for key, expected in {"stop_reason": "eos" if ended else "budget",
                                              "session_valid": True, "recovery": "none"}.items():
                            workload_scenarios.equal(source.get(key), expected, source)
                    result["before_prefix_ids"] = before["token_ids"]
                    result["diagnostic_forced_output"] = True
            self.record("stream_result", **result)
            return result
        except BaseException as error:
            self.record("failure", error=repr(error), path=path)
            if (isinstance(error, (TimeoutError, socket.timeout))
                    and not isinstance(error, workload_scenarios.DeadlineExceeded)
                    and time.monotonic() >= deadline):
                raise workload_scenarios.DeadlineExceeded("generation deadline exceeded") from error
            raise
        finally:
            connection.close()

    def control(self, session, operation, identity, saved=None):
        return workload_scenarios.control(self, session, operation, identity, saved)

    def close(self):
        timed_out = False
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=120 if getattr(self, "native", False) else 3)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.record("failure", error="owned child shutdown timed out; killing")
            self.process.kill()
            self.process.wait(timeout=3)
        reader = getattr(self, "stderr_thread", None)
        if reader:
            self.stderr_stop.set()
            reader.join(timeout=3)
            if reader.is_alive():
                self.record("failure", error="native diagnostic reader did not finish")
                return False
            if self.stderr_error:
                self.record("failure", error=self.stderr_error)
                timed_out = True
        if self.process.stderr:
            if not reader:
                self.stderr.write(self.process.stderr.read().decode(errors="replace"))
            self.process.stderr.close()
        self.record("shutdown", returncode=self.process.returncode)
        if self.process.returncode != 0:
            self.record("failure", error="owned child exited unsuccessfully")
        self.stderr.close()
        self.log.close()
        return not timed_out and self.process.returncode == 0


def run_once(service, scenario):
    if (scenario["mode"] in {"free_greedy", "controlled_replay"}) != getattr(service, "native", False):
        raise ValueError("scenario mode must match explicit native/test launch")
    session = None
    start = time.monotonic()
    request_ids = set()
    tool_reports = []
    task_checks = []
    previous_generation = None
    saved = None
    controls = []
    replay_steps = []
    service.operation_report = {"task_checks": task_checks, "tool_loops": tool_reports, "controls": controls, "replay_steps": replay_steps}

    def reserve_id(request_id):
        if request_id in request_ids:
            raise ValueError("duplicate request id")
        request_ids.add(request_id)

    for op in scenario["operations"]:
        action = op["op"]
        if scenario.get("version", 1) == 2:
            service.bounded_timeout(1)
        service.record("operation", operation=op)
        if action in {"checkpoint", "restore"}:
            reserve_id(op["id"])
            receipt = service.control(session, action, op["id"], saved)
            controls.append(receipt)
            if action == "checkpoint":
                saved = receipt
            result = receipt["terminal"]
        elif action == "replay":
            reserve_id(op["id"])
            result = service.generate(session, op["id"], len(op["tokens"]), replay_tokens=op["tokens"])
            replay_steps.append({"id": op["id"], "replay_tokens": op["tokens"],
                                 "actual_ids": result.get("token_ids"),
                                 "before_prefix_ids": result.get("before_prefix_ids"),
                                 "result": result})
            workload_scenarios.equal(result.get("event"), "completed", result)
        elif action == "score":
            completed_turn = (previous_generation.get("event") == "completed"
                              and previous_generation.get("stop_reason") == "eos"
                              and previous_generation.get("state") == "ready")
            result = {"request_id": op["id"], "expected_text": op["text"],
                      "actual_text": previous_generation["text"],
                      "completed_turn": completed_turn,
                      "success": completed_turn and previous_generation["text"] == op["text"]}
            task_checks.append(result)
            service.record("task_check", **result)
        elif action in {"tool_loop", "native_tool_loop", "native_investigation_loop", "native_observation_loop"}:
            loop = native_workload_tools.run_loop if action != "tool_loop" else run_loop
            result = loop(service, session, op, reserve_id)
            tool_reports.append(result)
            if not result["protocol_success"]:
                return {**result, "mode": scenario["mode"], "tool_loops": tool_reports,
                        "task_checks": task_checks,
                        "performance_evidence": False}
        elif action == "pause":
            time.sleep(service.bounded_timeout(op["seconds"]) if scenario.get("version", 1) == 2 else op["seconds"])
            if scenario.get("version", 1) == 2:
                service.bounded_timeout(1)
            result = {}
        elif action in {"generate", "continue", "cancel"}:
            reserve_id(op["id"])
            result = service.generate(session, op["id"], op["budget"], action == "cancel")
            previous_generation = result
            expected_terminal = op.get("expect", {}).get(
                "event", "cancelled" if action == "cancel" else "completed")
            if (scenario.get("version", 1) == 2 and result.get("event") == "error"
                    and expected_terminal != "error"):
                raise workload_scenarios.ControlFailure(result)
            if result.get("event") != expected_terminal:
                raise ValueError(f"{action}: expected terminal {expected_terminal!r}, "
                                 f"got {result.get('event')!r}")
        else:
            method, path, body = "POST", "/sessions", {}
            if action == "message":
                path, body = f"/sessions/{session}/messages", {"role": "user", "content": op["content"]}
            elif action == "reset":
                path = f"/sessions/{session}/reset"
            elif action == "close":
                method, path = "DELETE", f"/sessions/{session}"
            status, result = service.request(method, path, body)
            if status != 200:
                raise RuntimeError(result)
            if action == "create":
                session = result["session_id"]
            if action in {"create", "reset", "close"}:
                saved = None
        for key, expected in op.get("expect", {}).items():
            if result.get(key) != expected:
                raise ValueError(f"{action}: {key}: expected {expected!r}, got {result.get(key)!r}")
    return {"protocol_success": True,
            "task_success": (all(r["task_success"] for r in tool_reports)
                             and all(r["success"] for r in task_checks))
                            if tool_reports or task_checks else None,
            "task_checks": task_checks,
            "controls": controls, "replay_steps": replay_steps, "tool_loops": tool_reports, "mode": scenario["mode"],
            "duration_seconds": time.monotonic() - start, "performance_evidence": False}


def run(service, scenario):
    if scenario.get("version", 1) == 1:
        return run_once(service, scenario)
    validate_scenario(scenario, getattr(service, "info", {}).get("context_capacity", 256))
    if not getattr(service, "native", False):
        raise ValueError("version 2 requires explicit native launch")
    if getattr(service, "evaluation_mode", "free_greedy") != scenario["mode"]:
        raise ValueError("scenario evaluation mode mismatch")
    if service.info.get("prefix_reuse") != scenario["prefix_reuse"]:
        raise ValueError("scenario prefix policy differs from running service")
    if (any(op["op"] in {"checkpoint", "restore"} for op in scenario["operations"])
            and service.info.get("checkpoints") is not True):
        raise ValueError("running service does not support checkpoints")
    report = {"protocol_success": False, "task_success": None, "repetitions": [],
              "completed_repetitions": 0, "planned_repetitions": scenario["repetitions"],
              "prefix_reuse": scenario["prefix_reuse"], "mode": scenario["mode"],
              "wall_seconds": scenario["wall_seconds"], "performance_evidence": False}
    old = getattr(service, "absolute_deadline", None)
    start = time.monotonic()
    deadline = start + scenario["wall_seconds"]
    service.absolute_deadline = min(old, deadline) if old is not None else deadline
    try:
        for number in range(1, scenario["repetitions"] + 1):
            entry = {"repetition": number, "started": time.monotonic(), "completed": False}
            report["repetitions"].append(entry)
            service.record("repetition_start", repetition=number)
            try:
                entry["result"] = run_once(service, workload_scenarios.expanded(scenario, number))
                entry["completed"] = entry["result"]["protocol_success"]
                if not entry["completed"]:
                    report["failure_kind"] = entry["result"].get("failure_kind", "protocol")
                    break
                report["completed_repetitions"] += 1
            except BaseException as error:
                entry["error"] = repr(error)
                report["failure_kind"] = (
                    "limit" if isinstance(error, workload_scenarios.DeadlineExceeded) else
                    "protocol" if isinstance(error, (ValueError, workload_scenarios.ProtocolError))
                    else "runtime")
                entry["failure_kind"] = report["failure_kind"]
                entry["partial_result"] = getattr(service, "operation_report", {})
                service.record("failure", error=repr(error), repetition=number)
                if not isinstance(error, Exception):
                    service.scenario_report = report
                    raise
                break
            finally:
                entry["elapsed_seconds"] = time.monotonic() - entry["started"]
                service.record("repetition_end", **entry)
        report["protocol_success"] = report["completed_repetitions"] == scenario["repetitions"]
        scores = []
        for entry in report["repetitions"]:
            evidence = entry.get("result", entry.get("partial_result", {}))
            scores.extend(c["success"] for c in evidence.get("task_checks", []))
            scores.extend(c["task_success"] for c in evidence.get("tool_loops", []) if c.get("task_success") is not None)
        report["task_success"] = all(scores) if scores and report["protocol_success"] else None
        return report
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        service.absolute_deadline = old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True, help="new directory, never overwritten")
    parser.add_argument("--native-prefix-reuse", choices=["legacy", "on", "off"])
    parser.add_argument("--native-model")
    parser.add_argument("--native-tokenizer")
    parser.add_argument("--native-prefill", choices=["mps", "custom"], default="mps")
    parser.add_argument("--native-strategy", choices=["resident-fp32", "compact-fp8-arithmetic", "compact-fp8-lookup"],
                        help="native storage strategy (default: resident-fp32); requires model/tokenizer")
    parser.add_argument("--request-seconds", type=float, default=120)
    parser.add_argument("--native-context", type=int, choices=[256, 4352], default=256)
    parser.add_argument("--native-chunk", type=int, choices=[128, 256], default=256)
    args = parser.parse_args()
    try:
        scenario = load_scenario(args.scenario, args.native_context)
        if args.native_context == 256 and args.native_chunk != 256:
            raise ValueError("context256 requires chunk256")
        native_args = None
        if args.native_model or args.native_tokenizer or args.native_strategy:
            if not args.native_model or not args.native_tokenizer:
                raise ValueError("native model and tokenizer both required")
            native_args = ["--model", args.native_model, "--tokenizer", args.native_tokenizer,
                           "--strategy", args.native_strategy or "resident-fp32", "--context", str(args.native_context), "--prefill-chunk", str(args.native_chunk),
                           "--prefill", args.native_prefill]
        policy = scenario.get("prefix_reuse", args.native_prefix_reuse or "legacy")
        if args.native_prefix_reuse is not None and (native_args is None or args.native_prefix_reuse != policy):
            raise ValueError("conflicting prefix policy or test launch")
        if native_args is not None and policy != "legacy":
            native_args += ["--prefix-reuse", policy]
        if (scenario["mode"] in {"free_greedy", "controlled_replay"}) != (native_args is not None):
            raise ValueError("scenario mode must match explicit native/test launch")
        profiles = {op["op"] for op in scenario["operations"]
                    if op["op"] in {"native_tool_loop", "native_investigation_loop", "native_observation_loop"}}
        if len(profiles) > 1:
            raise ValueError("mixed native tool profiles")
        if profiles:
            if native_args is None:
                raise ValueError("native tool loop requires native launch")
            native_args += ["--tools", "observation-v13" if "native_observation_loop" in profiles else
                           "investigation-v13" if "native_investigation_loop" in profiles else "calculator-v13"]
        if native_args is None:
            service = Service(args.server, args.output)
        else:
            service = Service(args.server, args.output, native_args=native_args,
                              request_seconds=args.request_seconds,
                              expected_strategy=args.native_strategy or "resident-fp32",
                              evaluation_mode=scenario["mode"])
    except Exception as error:
        print(f"real-workload: {error}", file=sys.stderr)
        return 1
    if native_args is not None and (service.info.get("context_capacity") != args.native_context or
                                    service.info.get("prefill_chunk_size") != args.native_chunk):
        service.close()
        print("real-workload: native limits differ from requested profile", file=sys.stderr)
        return 1
    if native_args is not None and service.info.get("prefix_reuse") != policy:
        service.close()
        print("real-workload: prefix reuse mismatch", file=sys.stderr)
        return 1
    exit_status = 0
    result = {"protocol_success": False, "task_success": None}
    if native_args is not None:
        result["execution_strategy"] = service.info["execution_strategy"]
    try:
        result.update(run(service, scenario))
        if not result["protocol_success"]:
            exit_status = 1
    except BaseException as error:
        result.update(getattr(service, "scenario_report", {}))
        service.record("failure", error=repr(error))
        result["error"] = repr(error)
        exit_status = 1
        print(f"real-workload: {error}", file=sys.stderr)
    finally:
        try:
            shutdown_ok = service.close()
        except Exception as error:
            shutdown_ok = False
            result["shutdown_error"] = repr(error)
            print(f"real-workload: shutdown: {error}", file=sys.stderr)
        if not shutdown_ok:
            print("real-workload: owned child shutdown failed", file=sys.stderr)
            result["protocol_success"] = False
            result["shutdown_failure"] = True
            exit_status = 1
        if scenario["mode"] == "controlled_replay":
            result["task_success"] = None
            result["performance_evidence"] = False
            result["capture_status"] = "incomplete_failure_evidence"
            if shutdown_ok and result.get("protocol_success"):
                try:
                    steps = [step for entry in result["repetitions"] for step in entry["result"]["replay_steps"]]
                    result["capture_verification"] = replay_logits.verify(service.replay_path, steps)
                    result["capture_status"] = "completed_verified"
                except Exception as error:
                    result["capture_error"] = repr(error)
                    result["protocol_success"] = False
                    exit_status = 1
        (Path(args.output) / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    return exit_status


if __name__ == "__main__":
    sys.exit(main())
