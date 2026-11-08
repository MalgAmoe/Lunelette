"""Owned diagnostic service transport and process lifecycle; no CLI or scenario runner.

Callbacks and retained-state validation remain synchronous. Only the launched child
is owned; callers own scenario policy, scoring and command-line configuration.
"""
import copy
import contextlib
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

import workload_scenarios
import replay_logits


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key}")
        result[key] = value
    return result


class AdmissionError(ValueError):
    """Request rejected before a generation stream was admitted."""


class Service:
    """Owns only the child it launches. Every request/arrival/failure is recorded."""
    def __init__(self, executable, output, tick_ms=50, native_args=None, request_seconds=120,
                 expected_strategy=None, evaluation_mode="free_greedy", absolute_deadline=None, log_bytes=None,
                 evidence_config=None):
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
        self.absolute_deadline = absolute_deadline
        self.log_bytes = log_bytes
        self.log_lock = threading.Lock()
        self.evaluation_mode = evaluation_mode
        self.native = native_args is not None
        self.request_seconds = request_seconds if self.native else 15
        config_text = None if evidence_config is None else json.dumps(evidence_config, indent=2) + "\n"
        if config_text is not None and len(config_text.encode()) > 65536:
            raise ValueError("evidence config exceeds 65536 bytes")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = (self.output / "client.ndjson").open("x")
        try:
            self.stderr = (self.output / "server.stderr").open("x")
        except BaseException:
            self.log.close()
            raise
        try:
            if config_text is not None:
                with (self.output / "config.json").open("x") as config_file:
                    config_file.write(config_text)
            arguments = [str(Path(executable).resolve()), "--port", "0"]
            arguments += native_args if self.native else ["--tick-ms", str(tick_ms)]
            if evaluation_mode == "controlled_replay":
                self.replay_path = self.output.resolve() / "replay-logits.bin"
                arguments += ["--evaluation-mode", "controlled-replay", "--replay-logits", str(self.replay_path)]
            self.process = subprocess.Popen(arguments,
                                            stdout=self.stderr if log_bytes is None else subprocess.PIPE,
                                            stderr=subprocess.PIPE if log_bytes is None else subprocess.STDOUT)
            if log_bytes is not None:
                self.process.stderr = self.process.stdout
        except BaseException as error:
            try:
                self.record("failure", phase="startup", error=repr(error))
            except Exception:
                pass
            finally:
                for stream in (self.stderr, self.log):
                    try:
                        stream.close()
                    except Exception:
                        pass
            raise
        self.port = None
        self.stderr_thread = None
        self.stderr_error = None
        self.stderr_stop = threading.Event()
        line = bytearray()
        try:
            deadline = time.monotonic() + (180 if self.native else 3)
            if absolute_deadline is not None:
                deadline = min(deadline, absolute_deadline)
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
                self.write_diagnostic(complete_line.decode(errors="replace"))
                self.stderr.flush()
                try:
                    announcement = json.loads(complete_line)
                except ValueError:
                    if not self.native:
                        raise
                else:
                    if announcement.get("event") == "listening":
                        self.port = announcement["port"]
            if self.native or log_bytes is not None:
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
            try:
                if line:
                    self.write_diagnostic(line.decode(errors="replace"))
                    self.stderr.flush()
            except Exception:
                pass
            try:
                self.record("failure", phase="startup", error=repr(error))
            except Exception:
                pass
            finally:
                try:
                    self.close()
                except BaseException as cleanup_error:
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
                decoded = chunk.decode(errors="replace")
                self.write_diagnostic(decoded)
                self.stderr.flush()
                if self.stderr_stop.is_set():
                    trailing_bytes += len(chunk)
                    if trailing_bytes >= 65536:
                        raise RuntimeError("diagnostic output exceeds post-shutdown drain bound")
        except Exception as error:
            self.stderr_error = repr(error)

    def write_diagnostic(self, text):
        limit = getattr(self, "log_bytes", None)
        if limit is not None and self.stderr.tell() + len(text.encode()) > limit:
            self.stderr_error = "diagnostic log byte limit"
            raise RuntimeError(self.stderr_error)
        self.stderr.write(text)

    def record(self, kind, **values):
        line = json.dumps({"kind": kind, "monotonic": time.monotonic(), **values},
                          ensure_ascii=False) + "\n"
        with getattr(self, "log_lock", contextlib.nullcontext()):
            if getattr(self, "log_bytes", None) is not None and self.log.tell() + len(line.encode()) > self.log_bytes:
                self.stderr_error = "client log byte limit"
                raise RuntimeError(self.stderr_error)
            self.log.write(line)
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
        try:
            self.record("send", method=method, path=path, body=body)
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

    def generate(self, session, request_id, budget, cancel=False, replay_tokens=None, on_event=None):
        """Stream validated events to optional synchronous callback; exceptions abort.

        Events precede final retained-result checks. The default remains buffered.
        """
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
            self.record("send", method="POST", path=path, body=body.decode())
            connection.request("POST", path, body)
            transport_socket = getattr(connection, "sock", None)
            response = connection.getresponse()
            if response.status in {400, 404, 409, 422}:
                detail = response.read(65536)
                raise AdmissionError(f"generation admission rejected ({response.status}): {detail!r}")
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
                    if on_event is not None:
                        on_event(copy.deepcopy(event))
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
        failed = False

        def evidence(kind, **values):
            nonlocal failed
            try:
                self.record(kind, **values)
            except Exception:
                failed = True

        try:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=3 if getattr(self, "log_bytes", None) is not None
                                  else (120 if getattr(self, "native", False) else 3))
            except subprocess.TimeoutExpired:
                failed = True
                self.process.kill()
                self.process.wait(timeout=3)
                evidence("failure", error="owned child shutdown timed out; killed")
            reader = getattr(self, "stderr_thread", None)
            if reader:
                self.stderr_stop.set()
                reader.join(timeout=3)
                if reader.is_alive():
                    failed = True
                    evidence("failure", error="native diagnostic reader did not finish")
                if self.stderr_error:
                    failed = True
                    evidence("failure", error=self.stderr_error)
            if self.process.stderr:
                if not reader:
                    while True:
                        chunk = self.process.stderr.read(4096)
                        if not chunk:
                            break
                        try:
                            self.write_diagnostic(chunk.decode(errors="replace"))
                        except Exception:
                            failed = True
                            break
                self.process.stderr.close()
            evidence("shutdown", returncode=self.process.returncode)
            if self.process.returncode != 0:
                evidence("failure", error="owned child exited unsuccessfully")
        finally:
            # Evidence failures must never prevent closing the remaining resources.
            for stream in (self.process.stderr, self.stderr, self.log):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        failed = True
        return not failed and not getattr(self, "stderr_error", None) and self.process.returncode == 0
