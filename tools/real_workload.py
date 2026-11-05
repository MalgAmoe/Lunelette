#!/usr/bin/env python3
"""Version-1 controlled protocol runner. Standard library only; no model claims."""
import argparse
import http.client
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key}")
        result[key] = value
    return result


def load_scenario(path):
    value = json.loads(Path(path).read_text(), object_pairs_hook=strict_object)
    if set(value) != {"version", "mode", "operations"} or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("expected version 1 scenario schema")
    if value["mode"] != "controlled_protocol_only":
        raise ValueError("free_greedy unsupported by diagnostic generator")
    if not isinstance(value["operations"], list) or len(value["operations"]) > 256:
        raise ValueError("operations must be bounded list")
    fields = {"create": set(), "message": {"content"}, "generate": {"id", "budget"},
              "continue": {"id", "budget"}, "cancel": {"id", "budget"},
              "pause": {"seconds"}, "reset": set(), "close": set()}
    for op in value["operations"]:
        if not isinstance(op, dict) or op.get("op") not in fields:
            raise ValueError("unknown operation")
        required = fields[op["op"]]
        if not required <= op.keys() or set(op) - required - {"op", "expect"}:
            raise ValueError("invalid operation fields")
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
        if "content" in op and (not isinstance(op["content"], str) or len(op["content"].encode()) > 256):
            raise ValueError("invalid content")
    return value


class Service:
    """Owns only the child it launches. Every request/arrival/failure is recorded."""
    def __init__(self, executable, output, tick_ms=50):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = (self.output / "client.ndjson").open("x")
        self.stderr = (self.output / "server.stderr").open("x")
        try:
            self.process = subprocess.Popen([str(Path(executable).resolve()), "--port", "0",
                                             "--tick-ms", str(tick_ms)],
                                            stdout=self.stderr, stderr=subprocess.PIPE)
        except BaseException as error:
            self.record("failure", phase="startup", error=repr(error))
            self.stderr.close()
            self.log.close()
            raise
        self.port = None
        try:
            deadline, line = time.monotonic() + 3, bytearray()
            while not line.endswith(b"\n"):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stderr], [], [], remaining)[0]:
                    raise TimeoutError("no startup announcement")
                chunk = os.read(self.process.stderr.fileno(), 1)
                if not chunk or len(line) >= 4096:
                    raise RuntimeError("invalid startup announcement")
                line.extend(chunk)
            self.stderr.write(line.decode())
            self.stderr.flush()
            self.port = json.loads(line)["port"]
            health = self.request("GET", "/health")[1]
            info = self.request("GET", "/info")[1]
            if not health.get("operational_test_service") or health.get("model_loaded") is not False:
                raise RuntimeError("unexpected readiness")
            if info.get("generator") != "diagnostic_test_generator" or info.get("runtime_ready") is not False:
                raise RuntimeError("not the diagnostic protocol service")
        except BaseException as error:
            self.record("failure", phase="startup", error=repr(error))
            try:
                self.close()
            except BaseException as cleanup_error:
                # Preserve the cause of startup failure, not just cleanup's error.
                print(f"real-workload: startup cleanup: {cleanup_error}", file=sys.stderr)
            raise

    def record(self, kind, **values):
        self.log.write(json.dumps({"kind": kind, "monotonic": time.monotonic(), **values},
                                  ensure_ascii=False) + "\n")
        self.log.flush()

    def connect(self):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=4)

    def request(self, method, path, body=None):
        connection = self.connect()
        payload = b"" if body is None else json.dumps(body).encode()
        sent = time.monotonic()
        self.record("send", method=method, path=path, body=body)
        try:
            connection.request(method, path, payload, {"Content-Length": str(len(payload))})
            response = connection.getresponse()
            raw = response.read()
            result = json.loads(raw)
            self.record("response", status=response.status, raw=raw.decode(), elapsed=time.monotonic() - sent)
            return response.status, result
        except BaseException as error:
            self.record("failure", error=repr(error), path=path)
            raise
        finally:
            connection.close()

    def generate(self, session, request_id, budget, cancel=False):
        connection = self.connect()
        path = f"/sessions/{session}/generate"
        body = json.dumps({"id": request_id, "budget": budget}).encode()
        sent = time.monotonic()
        self.record("send", method="POST", path=path, body=body.decode())
        first_text = None
        events, text, raw_line = [], "", bytearray()
        terminal_seen = False
        # Fixed test generator reaches EOS within ten one-second ticks. This
        # is a diagnostic transport deadline, not a real-model latency limit.
        deadline = sent + 15.0
        try:
            connection.request("POST", path, body)
            response = connection.getresponse()
            if response.status != 200 or response.getheader("Content-Type") != "application/x-ndjson":
                raise RuntimeError((response.status, response.read()))
            if cancel:
                status, ack = self.request("POST", f"/requests/{request_id}/cancel", {})
                if status != 200 or not ack["cancel_requested"]:
                    raise RuntimeError("cancellation not acknowledged")
            # read1 observes actual available payload, not readline's newline wait.
            while True:
                chunk = response.read1(1024)
                arrived = time.monotonic()
                if arrived >= deadline:
                    raise TimeoutError("diagnostic generation deadline exceeded")
                if not chunk:
                    break
                self.record("arrival", raw_hex=chunk.hex(), since_send=arrived - sent)
                for byte in chunk:
                    raw_line.append(byte)
                    if len(raw_line) > 1024:
                        raise ValueError("stream event exceeds byte limit")
                    if byte != 10:
                        continue
                    event = json.loads(raw_line.decode("utf-8"), object_pairs_hook=strict_object)
                    raw_line.clear()
                    if not isinstance(event, dict) or event.get("id") != request_id:
                        raise ValueError("stream request identity mismatch")
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
                    events.append(event)
                    if kind == "text_delta":
                        if event["text"] and first_text is None:
                            first_text = arrived - sent
                        text += event["text"]
            if raw_line or not events or events[-1]["event"] not in {"completed", "cancelled", "error"}:
                raise RuntimeError("missing complete terminal event")
            result = {**events[-1], "text": text, "first_actual_text_seconds": first_text,
                      "completion_seconds": time.monotonic() - sent}
            self.record("stream_result", **result)
            return result
        except BaseException as error:
            self.record("failure", error=repr(error), path=path)
            raise
        finally:
            connection.close()

    def close(self):
        timed_out = False
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.record("failure", error="owned child shutdown timed out; killing")
            self.process.kill()
            self.process.wait(timeout=3)
        if self.process.stderr:
            self.stderr.write(self.process.stderr.read().decode(errors="replace"))
            self.process.stderr.close()
        self.record("shutdown", returncode=self.process.returncode)
        if self.process.returncode != 0:
            self.record("failure", error="owned child exited unsuccessfully")
        self.stderr.close()
        self.log.close()
        return not timed_out and self.process.returncode == 0


def run(service, scenario):
    session = None
    start = time.monotonic()
    for op in scenario["operations"]:
        action = op["op"]
        service.record("operation", operation=op)
        if action == "pause":
            time.sleep(op["seconds"])
            result = {}
        elif action in {"generate", "continue", "cancel"}:
            result = service.generate(session, op["id"], op["budget"], action == "cancel")
            expected_terminal = op.get("expect", {}).get(
                "event", "cancelled" if action == "cancel" else "completed")
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
        for key, expected in op.get("expect", {}).items():
            if result.get(key) != expected:
                raise ValueError(f"{action}: {key}: expected {expected!r}, got {result.get(key)!r}")
    return {"protocol_success": True, "task_success": None, "mode": "controlled_protocol_only",
            "duration_seconds": time.monotonic() - start, "performance_evidence": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True, help="new directory, never overwritten")
    args = parser.parse_args()
    try:
        scenario = load_scenario(args.scenario)
        service = Service(args.server, args.output)
    except Exception as error:
        print(f"real-workload: {error}", file=sys.stderr)
        return 1
    exit_status = 0
    result = {"protocol_success": False, "task_success": None}
    try:
        result = run(service, scenario)
    except BaseException as error:
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
        (Path(args.output) / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    return exit_status


if __name__ == "__main__":
    sys.exit(main())
