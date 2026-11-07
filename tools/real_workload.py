#!/usr/bin/env python3
"""Version-1 diagnostic workload runner. Explicit native or model-free launch."""
import argparse
import http.client
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import threading

from workload_tools import run_loop, validate_operation
import native_workload_tools


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key}")
        result[key] = value
    return result


def load_scenario(path, context=256):
    value = json.loads(Path(path).read_text(), object_pairs_hook=strict_object)
    if set(value) != {"version", "mode", "operations"} or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("expected version 1 scenario schema")
    if value["mode"] not in {"controlled_protocol_only", "free_greedy"}:
        raise ValueError("unsupported scenario mode")
    if not isinstance(value["operations"], list) or len(value["operations"]) > 256:
        raise ValueError("operations must be bounded list")
    fields = {"create": set(), "message": {"content"}, "generate": {"id", "budget"},
              "continue": {"id", "budget"}, "cancel": {"id", "budget"},
              "pause": {"seconds"}, "reset": set(), "close": set(),
              "score": {"id", "text"},
              "tool_loop": {"id", "budget", "max_calls", "max_generated", "wall_seconds", "answer"}}
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
        if "content" in op and (not isinstance(op["content"], str) or len(op["content"].encode()) > ((32768 if context == 4352 else 4096) if value["mode"] == "free_greedy" else 256)):
            raise ValueError("invalid content")
    return value


class Service:
    """Owns only the child it launches. Every request/arrival/failure is recorded."""
    def __init__(self, executable, output, tick_ms=50, native_args=None, request_seconds=120,
                 expected_strategy=None):
        if type(request_seconds) not in (int, float) or not 1 <= request_seconds <= 120:
            raise ValueError("request_seconds outside 1..120")
        if expected_strategy is not None and (native_args is None or expected_strategy not in
                {"resident-fp32", "compact-fp8-arithmetic"}):
            raise ValueError("expected strategy requires a supported native launch")
        self.native = native_args is not None
        self.request_seconds = request_seconds if self.native else 15
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = (self.output / "client.ndjson").open("x")
        self.stderr = (self.output / "server.stderr").open("x")
        try:
            arguments = [str(Path(executable).resolve()), "--port", "0"]
            arguments += native_args if self.native else ["--tick-ms", str(tick_ms)]
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
                if not health.get("model_loaded") or not health.get("runtime_ready") or info.get("generator") != "native_greedy":
                    raise RuntimeError("native runtime not ready")
                if info.get("numerical_validation") != "diagnostic_known_reference_mismatches":
                    raise RuntimeError("unexpected numerical status")
                if expected_strategy is not None and info.get("execution_strategy") != expected_strategy:
                    raise RuntimeError("native execution strategy mismatch")
            elif not health.get("operational_test_service") or health.get("model_loaded") is not False:
                raise RuntimeError("unexpected readiness")
            if not self.native and (info.get("generator") != "diagnostic_test_generator" or info.get("runtime_ready") is not False):
                raise RuntimeError("not the diagnostic protocol service")
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
                raise TimeoutError("application wall deadline exceeded")
            return min(seconds, remaining)
        return seconds

    def connect(self):
        return http.client.HTTPConnection("127.0.0.1", self.port,
                                          timeout=self.bounded_timeout(4))

    def request(self, method, path, body=None):
        connection = self.connect()
        payload = b"" if body is None else json.dumps(body).encode()
        sent = time.monotonic()
        self.record("send", method=method, path=path, body=body)
        try:
            connection.request(method, path, payload, {"Content-Length": str(len(payload))})
            transport_socket = getattr(connection, "sock", None)
            response = connection.getresponse()
            if transport_socket is not None:
                transport_socket.settimeout(self.bounded_timeout(4))
            raw = response.read()
            self.bounded_timeout(4)
            result = json.loads(raw, object_pairs_hook=strict_object)
            self.record("response", status=response.status, raw=raw.decode(), elapsed=time.monotonic() - sent)
            return response.status, result
        except BaseException as error:
            self.record("failure", error=repr(error), path=path)
            raise
        finally:
            connection.close()

    def generate(self, session, request_id, budget, cancel=False):
        connection = self.connect()
        if getattr(self, "native", False):
            connection.timeout = self.bounded_timeout(self.request_seconds)
        path = f"/sessions/{session}/generate"
        body = json.dumps({"id": request_id, "budget": budget}).encode()
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
                    raise TimeoutError("diagnostic generation deadline exceeded")
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


def run(service, scenario):
    if (scenario["mode"] == "free_greedy") != getattr(service, "native", False):
        raise ValueError("scenario mode must match explicit native/test launch")
    session = None
    start = time.monotonic()
    request_ids = set()
    tool_reports = []
    task_checks = []
    previous_generation = None

    def reserve_id(request_id):
        if request_id in request_ids:
            raise ValueError("duplicate request id")
        request_ids.add(request_id)

    for op in scenario["operations"]:
        action = op["op"]
        service.record("operation", operation=op)
        if action == "score":
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
            time.sleep(op["seconds"])
            result = {}
        elif action in {"generate", "continue", "cancel"}:
            reserve_id(op["id"])
            result = service.generate(session, op["id"], op["budget"], action == "cancel")
            previous_generation = result
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
    return {"protocol_success": True,
            "task_success": (all(r["task_success"] for r in tool_reports)
                             and all(r["success"] for r in task_checks))
                            if tool_reports or task_checks else None,
            "task_checks": task_checks,
            "tool_loops": tool_reports, "mode": scenario["mode"],
            "duration_seconds": time.monotonic() - start, "performance_evidence": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True, help="new directory, never overwritten")
    parser.add_argument("--native-model")
    parser.add_argument("--native-tokenizer")
    parser.add_argument("--native-prefill", choices=["mps", "custom"], default="mps")
    parser.add_argument("--native-strategy", choices=["resident-fp32", "compact-fp8-arithmetic"],
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
        if (scenario["mode"] == "free_greedy") != (native_args is not None):
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
                              expected_strategy=args.native_strategy or "resident-fp32")
    except Exception as error:
        print(f"real-workload: {error}", file=sys.stderr)
        return 1
    if native_args is not None and (service.info.get("context_capacity") != args.native_context or
                                    service.info.get("prefill_chunk_size") != args.native_chunk):
        service.close()
        print("real-workload: native limits differ from requested profile", file=sys.stderr)
        return 1
    exit_status = 0
    result = {"protocol_success": False, "task_success": None}
    if native_args is not None:
        result["execution_strategy"] = service.info["execution_strategy"]
    try:
        result.update(run(service, scenario))
        if not result["protocol_success"]:
            exit_status = 1
    except Exception as error:
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
