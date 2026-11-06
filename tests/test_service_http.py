#!/usr/bin/env python3
"""Live HTTP regressions; intentionally separate from model-free native core-test."""
import http.client
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import contextlib
import io
import subprocess
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import real_workload
from real_workload import Service, load_scenario, run


def check(value):
    if not value:
        raise RuntimeError("protocol check failed")


def cleanup(service):
    """Never replace the foreground failure with a cleanup assertion."""
    original = sys.exc_info()[1]
    try:
        check(service.close())
    except BaseException as error:
        if original is None:
            raise
        print(f"cleanup failed while preserving {original!r}: {error!r}", file=sys.stderr)
        if hasattr(original, "add_note"):
            original.add_note(f"cleanup failure: {error!r}")


def runner_faults(directory):
    startup_output = Path(directory) / "failed-startup"
    try:
        Service(Path(directory) / "nonexistent-executable", startup_output)
    except OSError:
        pass
    else:
        raise RuntimeError("missing executable unexpectedly started")
    startup_records = [json.loads(line) for line in
                       (startup_output / "client.ndjson").read_text().splitlines()]
    check(any(r["kind"] == "failure" and r.get("phase") == "startup"
              for r in startup_records))
    # Native limits must not inherit the fake byte-token budget.
    native_scenario = Path(directory) / "native-long-message.json"
    native_scenario.write_text(json.dumps({"version": 1, "mode": "free_greedy",
        "operations": [{"op": "message", "content": "hello " * 100}]}))
    check(load_scenario(native_scenario)["mode"] == "free_greedy")
    # A caller error must be rejected before creating a resident process.
    with patch.object(real_workload, "Service") as launch, patch.object(sys, "argv", [
            "runner", "--server", "unused", "--scenario", str(native_scenario),
            "--output", str(Path(directory) / "wrong-mode")]), contextlib.redirect_stderr(io.StringIO()):
        check(real_workload.main() == 1)
        check(not launch.called)
    # Client selection is passed explicitly, checked against /info and recorded.
    for selected in (None, "resident-fp32", "compact-fp8-arithmetic"):
        output = Path(directory) / ("select-" + str(selected))
        expected = selected or "resident-fp32"
        argv = ["runner", "--server", "unused", "--scenario", str(native_scenario),
                "--output", str(output), "--native-model", "model", "--native-tokenizer", "tokenizer"]
        if selected:
            argv += ["--native-strategy", selected]
        with patch.object(real_workload, "Service") as launch, patch.object(sys, "argv", argv), \
                patch.object(real_workload, "run", return_value={"protocol_success": True}):
            output.mkdir()
            launch.return_value.close.return_value = True
            launch.return_value.info = {"execution_strategy": expected}
            check(real_workload.main() == 0)
            keywords = launch.call_args.kwargs
            check(keywords["expected_strategy"] == expected)
            arguments = keywords["native_args"]
            check(arguments[arguments.index("--strategy") + 1] == expected)
            check(json.loads((output / "result.json").read_text())["execution_strategy"] == expected)
    failed_output = Path(directory) / "native-scenario-failure"
    failed_output.mkdir()
    with patch.object(real_workload, "Service") as launch, patch.object(sys, "argv", [
            "runner", "--server", "unused", "--scenario", str(native_scenario),
            "--output", str(failed_output), "--native-model", "model", "--native-tokenizer", "tokenizer",
            "--native-strategy", "compact-fp8-arithmetic"]), \
            patch.object(real_workload, "run", side_effect=RuntimeError("scenario failure")), \
            contextlib.redirect_stderr(io.StringIO()):
        launch.return_value.info = {"execution_strategy": "compact-fp8-arithmetic"}
        launch.return_value.close.return_value = True
        check(real_workload.main() == 1)
        launch.return_value.close.assert_called_once()
        report = json.loads((failed_output / "result.json").read_text())
        check(report["execution_strategy"] == "compact-fp8-arithmetic")
        check(report["protocol_success"] is False and "scenario failure" in report["error"])
    for selected in ("resident-fp32", "compact-fp8-arithmetic", "invalid"):
        with patch.object(real_workload, "Service") as launch, patch.object(sys, "argv", [
                "runner", "--server", "unused", "--scenario", str(native_scenario),
                "--output", str(Path(directory) / "reject-strategy"), "--native-strategy", selected]), \
                contextlib.redirect_stderr(io.StringIO()):
            if selected == "invalid":
                try:
                    real_workload.main()
                except SystemExit as error:
                    check(error.code == 2)
                else:
                    raise RuntimeError("invalid strategy accepted")
            else:
                check(real_workload.main() == 1)
            check(not launch.called)
    # Preserve an unterminated startup diagnostic on EOF.
    child = Path(directory) / "native-startup-child"
    child.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write('unfinished diagnostic')\n")
    child.chmod(0o700)
    partial = Path(directory) / "unterminated-startup"
    try:
        Service(child, partial, native_args=[])
    except RuntimeError:
        pass
    else:
        raise RuntimeError("unterminated startup unexpectedly succeeded")
    check("unfinished diagnostic" in (partial / "server.stderr").read_text())
    # Drain native stderr during execution, rather than only after waiting for
    # shutdown (a full pipe can otherwise block a worker indefinitely).
    marker = Path(directory) / "stderr-written"
    child.write_text(f"#!{sys.executable}\n" +
        "import os, signal, time\n" +
        "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n" +
        "os.write(2, b'{\"event\":\"listening\",\"port\":1234}\\n')\n" +
        "data = b'x' * 1048576\n" +
        "while data:\n    data = data[os.write(2, data):]\n" +
        f"open({str(marker)!r}, 'w').write('done')\n" +
        "while True: time.sleep(.01)\n")
    with patch.object(Service, "request", side_effect=[
            (200, {"model_loaded": True, "runtime_ready": True}),
            (200, {"generator": "native_greedy", "numerical_validation": "diagnostic_known_reference_mismatches"})]):
        service = Service(child, Path(directory) / "stderr-drain", native_args=[])
    try:
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        check(marker.exists())
    finally:
        cleanup(service)
    check((Path(directory) / "stderr-drain/server.stderr").stat().st_size > 1048576)
    compact, resident = "compact-fp8-arithmetic", "resident-fp32"
    for index, (expected, actual) in enumerate([(compact, compact), (compact, resident),
            (compact, None), (compact, "absent"), (resident, resident), (resident, compact)]):
        output = Path(directory) / f"identity-{index}"
        with patch.object(Service, "request", side_effect=[
                (200, {"model_loaded": True, "runtime_ready": True}),
                (200, {"generator": "native_greedy",
                       "numerical_validation": "diagnostic_known_reference_mismatches",
                       **({} if actual == "absent" else {"execution_strategy": actual})})]):
            try:
                owned = Service(child, output, native_args=["--strategy", expected],
                                expected_strategy=expected)
            except RuntimeError as error:
                check(actual != expected and "strategy mismatch" in str(error))
            else:
                try:
                    check(actual == expected)
                finally:
                    cleanup(owned)
        records = [json.loads(line) for line in (output / "client.ndjson").read_text().splitlines()]
        check(any(r["kind"] == "shutdown" and r["returncode"] == 0 for r in records))
    scenario_path = Path(directory) / "fault.json"
    scenario = {"version": 1, "mode": "controlled_protocol_only",
                "operations": [{"op": "generate", "id": "fault", "budget": 1}]}

    class StubService:
        terminal = "completed"
        shutdown = True

        def __init__(self, executable, output):
            self.output = Path(output)
            self.output.mkdir()

        def generate(self, *args):
            event = {"event": self.terminal, "text": ""}
            self.record("arrival", raw=event)
            return event

        def record(self, kind, **values):
            with (self.output / "client.ndjson").open("a") as log:
                log.write(json.dumps({"kind": kind, **values}) + "\n")

        def close(self):
            self.record("shutdown", success=self.shutdown)
            if self.shutdown == "raise":
                raise RuntimeError("controlled shutdown exception")
            return self.shutdown

    for action, event, expected, shutdown, success in [
            ("generate", "error", None, True, False),
            ("generate", "cancelled", None, True, False),
            ("continue", "error", None, True, False),
            ("continue", "cancelled", None, True, False),
            ("cancel", "completed", None, True, False),
            ("generate", "error", "error", True, True),
            ("generate", "cancelled", "cancelled", True, True),
            ("generate", "completed", None, False, False),
            ("generate", "error", None, False, False),
            ("generate", "error", None, "raise", False)]:
        op = scenario["operations"][0]
        op["op"] = action
        op.pop("expect", None)
        if expected:
            op["expect"] = {"event": expected}
        scenario_path.write_text(json.dumps(scenario))
        output = Path(directory) / f"fault-{len(list(Path(directory).glob('fault-*')))}"
        StubService.terminal, StubService.shutdown = event, shutdown
        with patch.object(real_workload, "Service", StubService), patch.object(
                sys, "argv", ["runner", "--server", "unused", "--scenario", str(scenario_path),
                              "--output", str(output)]), contextlib.redirect_stderr(io.StringIO()):
            status = real_workload.main()
        report = json.loads((output / "result.json").read_text())
        check((status == 0) == success and report["protocol_success"] == success)
        log = [json.loads(line) for line in (output / "client.ndjson").read_text().splitlines()]
        check(sum(entry["kind"] == "arrival" for entry in log) == 1)
        check(next(entry for entry in log if entry["kind"] == "arrival")["raw"]["event"] == event)
        if shutdown is not True:
            check(report["shutdown_failure"])
        if event == "error" and expected is None:
            check("expected terminal" in report["error"])
    for invalid in ["text_delta", 1, None, [], {}]:
        scenario["operations"][0]["expect"] = {"event": invalid}
        scenario_path.write_text(json.dumps(scenario))
        try:
            load_scenario(scenario_path)
        except ValueError:
            pass
        else:
            raise RuntimeError("invalid terminal expectation accepted")

    # Exercise real close with controlled process stubs; no unrelated child exists.
    class Child:
        stderr = io.BytesIO(b"controlled shutdown diagnostic")
        returncode = None
        killed = False

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout):
            if timeout_case and not self.killed:
                raise subprocess.TimeoutExpired("owned stub", timeout)
            self.returncode = -9 if self.killed else 7

        def kill(self):
            self.killed = True

    for timeout_case in (False, True):
        service = Service.__new__(Service)
        service.process = Child()
        service.process.stderr = io.BytesIO(b"diagnostic")
        service.log, service.stderr = io.StringIO(), io.StringIO()
        records = []
        service.record = lambda kind, **values: records.append({"kind": kind, **values})
        check(not service.close())
        check(service.process.killed == timeout_case)
        check(any(record["kind"] == "failure" for record in records))
    # Cleanup failure must be observable without masking the first assertion.
    stub = StubService.__new__(StubService)
    stub.close = lambda: False
    with contextlib.redirect_stderr(io.StringIO()) as diagnostic:
        try:
            try:
                raise ValueError("first failure")
            finally:
                cleanup(stub)
        except ValueError as error:
            check(str(error) == "first failure")
        else:
            raise RuntimeError("foreground failure lost")
    check("cleanup failed" in diagnostic.getvalue())
    try:
        cleanup(stub)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("cleanup failure ignored")


def stream_faults():
    """Run the real stream decoder against controlled wire event sequences."""
    started = {"event": "started", "id": "probe", "text": ""}
    delta = {"event": "text_delta", "id": "probe", "text": "ok"}
    completed = {"event": "completed", "id": "probe", "text": ""}

    class Response:
        status = 200

        def __init__(self, events):
            self.chunks = iter([json.dumps(e).encode() + b"\n" for e in events])

        def getheader(self, name):
            return "application/x-ndjson"

        def read1(self, size):
            return next(self.chunks, b"")

    class Connection:
        def __init__(self, events):
            self.response = Response(events)

        def request(self, *args):
            pass

        def getresponse(self):
            return self.response

        def close(self):
            pass

    for events, valid in [
            ([started, delta, completed], True),
            ([completed], False),
            ([started, {**delta, "id": "other"}, completed], False),
            ([started, started, completed], False),
            ([started, completed, delta, completed], False),
            ([started, {**delta, "event": "unknown"}, completed], False),
            ([started, {**delta, "text": "x" * 8192}, completed], False)]:
        service = Service.__new__(Service)
        service.connect = lambda: Connection(events)
        service.record = lambda *args, **kwargs: None
        try:
            result = service.generate(1, "probe", 3)
        except (ValueError, RuntimeError):
            check(not valid)
        else:
            check(valid)
            check(result["text"] == "ok")
    service = Service.__new__(Service)
    service.connect = lambda: Connection([started, delta, completed])
    service.record = lambda *args, **kwargs: None
    with patch.object(real_workload.time, "monotonic", side_effect=[0.0, 16.0]):
        try:
            service.generate(1, "probe", 3)
        except TimeoutError:
            pass
        else:
            raise RuntimeError("total generation deadline ignored")


def raw(port, payload):
    with socket.create_connection(("127.0.0.1", port), timeout=4) as sock:
        sock.sendall(payload)
        result = b""
        while True:
            data = sock.recv(8192)
            if not data:
                return result
            result += data


def terminal(service, request_id):
    for _ in range(100):
        status, result = service.request("GET", f"/requests/{request_id}")
        if status == 200 and result["status"] != "generating":
            return result
        time.sleep(.02)
    raise TimeoutError("request never terminated")


def main():
    executable = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="lunelette-service-") as directory:
        runner_faults(directory)
        stream_faults()
        service = Service(executable, Path(directory) / "live", tick_ms=100)
        try:
            for suffix in [b"Content-Length: 0\r\nContent-Length: 0", b"Content-Length: -1",
                           b"Content-Length: +1", b"Content-Length: 999999999999999999999",
                           b"Transfer-Encoding: chunked\r\nContent-Length: 0",
                           b"Content-Length: x", b"Content-Length: 1.0"]:
                response = raw(service.port, b"POST /sessions HTTP/1.1\r\nHost: localhost\r\n" + suffix + b"\r\n\r\n")
                check(response.startswith(b"HTTP/1.1 400"))
            response = raw(service.port, b"POST /sessions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{")
            check(not response)  # incomplete body expires without admission
            for body in [b'{"x":1}', b'{"x":1,"x":2}', b'{"x":"\\u0000"}', b'[]', b'{}x']:
                response = raw(service.port, b"POST /sessions HTTP/1.1\r\nHost: x\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                check(response.startswith(b"HTTP/1.1 400"))
            # NUL truncates header scanning: deadline closes without admission.
            check(not raw(service.port, b"POST /sessions HTTP/1.1\r\nHost: x\x00y\r\nContent-Length: 2\r\n\r\n{}"))
            for payload in [
                    b"POST /sessions HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\n\r\n{\x00}",
                    b"POST /sessions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
                    b"POST /sessions HTTP/1.1\r\nHost: x\r\nContent-Length: 8193\r\n\r\n"]:
                check(raw(service.port, payload).startswith(b"HTTP/1.1 400"))
            header = b"GET /health HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\nX-Pad: "
            for length, status in [(4096, 200), (4097, 400)]:
                payload = header + b"x" * (length - len(header) - 4) + b"\r\n\r\n"
                check(raw(service.port, payload).startswith(f"HTTP/1.1 {status}".encode()))
            for length in (8192, 8193):
                # At the accepted body limit use valid padded JSON; above it reject framing.
                payload = (b"POST /boundary HTTP/1.1\r\nHost: x\r\nContent-Length: " +
                           str(length).encode() + b"\r\n\r\n" + b"{}" + b" " * (length - 2))
                check(raw(service.port, payload).startswith(
                    f"HTTP/1.1 {501 if length == 8192 else 400}".encode()))
            check(service.request("POST", "/sessions", {})[0] == 200)
            # Omitted policy remains legacy: controls are unsupported, never a
            # silently recomputed imitation of a retained-cache checkpoint.
            check(service.request("POST", "/sessions/1/checkpoint", {"id": "save"})[0] == 501)
            check(service.request("POST", "/sessions/1/restore",
                                  {"id": "restore", "checkpoint_id": "1"})[0] == 501)
            check(service.request("DELETE", "/sessions/1/generate", {"id":"wrong-method", "budget":1})[0] == 501)
            check(service.request("GET", "/health", {"unknown":1})[0] == 400)
            check(service.request("POST", "/sessions/1/generate", {"id":"bad-budget", "budget":1.5})[0] == 400)
            check(service.request("POST", "/sessions/1/generate", {"id":"bad-budget", "budget":True})[0] == 400)
            # Partial multi-byte continuation and duplicate replay after reset.
            first = service.generate(1, "first", 3)
            status, retained = service.request("GET", "/requests/first")
            check(retained["operation"] == "generate")
            check(retained["cache_restored"] is False)
            check(retained["metrics"]["metrics_version"] == 2)
            check(retained["metrics"]["work_version"] == 1)
            check(status == 200 and first["metrics"] == retained["metrics"])
            check(first["metrics"]["test_only"] is True)
            check(first["metrics"]["process_phys_footprint_before_bytes"] is None)
            samples = first["metrics"]["decode_step_duration_us"]
            check(sum(samples) == first["metrics"]["decode_total_us"])
            check(len(samples) == first["metrics"]["decode_steps"])
            check(all(sample >= 0 for sample in samples))
            check(first["text"] == "A" and first["pending_tokens"] == 1)
            check(service.request("POST", "/sessions/1/messages", {"role":"user", "content":"x"})[0] == 409)
            second = service.generate(1, "second", 7)
            check(second["text"] == "€😀Z" and second["state"] == "ready")
            check(service.request("GET", "/requests/second")[1]["text"] == second["text"])
            check(service.request("GET", "/info")[1]["fixture_text"] == first["text"] + second["text"])
            check(service.request("POST", "/sessions/1/reset", {})[0] == 200)
            check(service.request("POST", "/sessions/1/generate", {"id":"first", "budget":1})[0] == 409)
            cancelled = service.generate(1, "timed-cancel", 100, cancel=True)
            check(cancelled["event"] == "cancelled")
            check(cancelled["cancel_ack_seconds"] >= 0)
            check(cancelled["cancel_completion_seconds"] >= cancelled["cancel_ack_seconds"])
            status, retained = service.request("GET", "/requests/timed-cancel")
            check(status == 200 and retained["metrics"] == cancelled["metrics"])
            check(retained["metrics"]["cancel_source"] == "explicit")
            check(retained["metrics"]["cancel_latency_ms"] >= 0)
            check(service.request("POST", "/sessions/1/reset", {})[0] == 200)
            # Blocked native tick must not block status/cancel/reset rejection.
            connection = service.connect()
            connection.request("POST", "/sessions/1/generate", json.dumps({"id":"busy", "budget":100}))
            response = connection.getresponse()
            check(response.status == 200)
            start = time.monotonic()
            check(service.request("GET", "/sessions/1")[1]["state"] == "generating")
            check(service.request("POST", "/sessions/1/reset", {})[0] == 409)
            check(service.request("DELETE", "/sessions/1", {})[0] == 409)
            status, busy = service.request("POST", "/sessions/1/generate", {"id":"retryable", "budget":1})
            check(status == 409 and busy["error"] == "busy_stream")
            check(busy["session_valid"] is True and busy["state"] == "generating")
            check(busy["session_id"] == 1 and busy["request_id"] == "")
            check(busy["recovery"] == "wait_for_request_completion")
            check(service.request("POST", "/requests/busy/cancel", {})[1]["cancel_requested"])
            check(time.monotonic() - start < .5)
            events = [json.loads(line) for line in response.read().splitlines()]
            check(events[-1]["event"] == "cancelled" and events[-1]["state"] == "needs_reset")
            connection.close()
            malformed = raw(service.port, b"GET /health HTTP/1.1\r\nContent-Length: x\r\n\r\n")
            failure = json.loads(malformed.split(b"\r\n\r\n", 1)[1])
            check(failure["error"] == "invalid_http" and failure["session_valid"] is False)
            check(failure["state"] == "needs_reset" and failure["recovery"] == "reset")
            check(failure["session_id"] == 1 and failure["request_id"] == "")
            check(service.request("POST", "/sessions/1/reset", {})[0] == 200)
            check(service.generate(1, "retryable", 1)["text"] == "A")
            check(service.request("POST", "/sessions/1/reset", {})[0] == 200)
            connection = service.connect()
            connection.request("POST", "/sessions/1/generate", json.dumps({"id":"disconnect", "budget":100}))
            response = connection.getresponse()
            response.close()
            connection.close()
            check(terminal(service, "disconnect")["status"] == "cancelled")
            check(service.request("POST", "/sessions/1/reset", {})[0] == 200)
            # Exactly 256 synthetic input bytes; no implicit truncation.
            check(service.request("POST", "/sessions/1/messages", {"role":"user", "content":"x" * 256})[0] == 200)
            check(service.request("POST", "/sessions/1/generate", {"id":"full", "budget":1})[0] == 422)
            check(service.request("POST", "/sessions/1/messages", {"role":"user", "content":"x"})[0] == 422)
            check(service.request("POST", "/sessions/1/checkpoint", {})[0] == 501)
            check(service.request("DELETE", "/sessions/1", {})[0] == 200)
            check(service.request("GET", "/sessions/1")[0] == 410)
        finally:
            cleanup(service)
        for name in ("partial-v1.json", "cancel-reset-v1.json"):
            service = Service(executable, Path(directory) / name, tick_ms=100)
            try:
                result = run(service, load_scenario(Path(__file__).resolve().parents[1] / "tools/scenarios" / name))
                check(result["protocol_success"] and result["task_success"] is None)
            finally:
                cleanup(service)
        service = Service(executable, Path(directory) / "shutdown", tick_ms=1000)
        closed = False
        connection = None
        response = None
        try:
            connection = service.connect()
            connection.request("POST", "/sessions", "{}")
            check(connection.getresponse().status == 200)
            connection.close()
            connection = service.connect()
            connection.request("POST", "/sessions/1/generate", json.dumps({"id":"shutdown", "budget":10}))
            response = connection.getresponse()
            check(response.status == 200)
            start = time.monotonic()
            check(service.request("GET", "/sessions/1")[1]["state"] == "generating")
            check(service.request("POST", "/requests/shutdown/cancel", {})[1]["cancel_requested"])
            check(time.monotonic() - start < .5)
            start = time.monotonic()
            closed = True
            cleanup(service)
            check(time.monotonic() - start < 1.5)
            response.close()
            connection.close()
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
            if not closed:
                cleanup(service)
    print("live HTTP parsing/state/streaming/cancellation/disconnect/scenario/shutdown tests passed")


if __name__ == "__main__":
    main()
