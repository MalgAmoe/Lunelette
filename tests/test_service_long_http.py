#!/usr/bin/env python3
"""Long native HTTP with actual adapter + CPU API stubs; no weights or model."""
import json
from pathlib import Path
import sys
import tempfile
import socket
import threading
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import real_workload
from real_workload import Service


def check(value, detail="check failed"):
    if not value:
        raise RuntimeError(detail)


def exercise(server, tokenizer, block, directory):
    service = Service(server, directory, native_args=[
        "--model", tokenizer, "--tokenizer", tokenizer, "--strategy", "resident-fp32",
        "--context", "4352", "--prefill-chunk", str(block), "--prefill", "custom",
        "--prefix-reuse", "on"])
    measurements = {"block": block, "preprocessing_http_ms": [], "concurrent_health_ms": []}

    def request(method, path, body=None, expected=200):
        status, result = service.request(method, path, body)
        check(status == expected, (path, status, result))
        return result

    def raw(body, expected, declared=None):
        connection = service.connect()
        try:
            connection.request("POST", "/sessions/1/messages", body,
                               {"Content-Length": str(len(body) if declared is None else declared)})
            response = connection.getresponse()
            data = response.read()
            check(response.status == expected, (response.status, data))
        finally:
            connection.close()

    try:
        check(service.info["context_capacity"] == 4352)
        check(service.info["long_context_validation"] == "external_evidence_only")
        check(service.info["prefill_chunk_size"] == block)
        request("POST", "/sessions", {})
        # Force all 16 input owners live together, not just the active request.
        held = []
        try:
            for _ in range(15):
                connection = socket.create_connection(("127.0.0.1", service.port), timeout=3)
                connection.sendall(b"GET /health HTTP/1.1\r\n")
                held.append(connection)
            request("GET", "/health")
        finally:
            for connection in held:
                connection.close()
        # Entire HTTP preprocessing includes socket receive, parser, framing/BPE,
        # ledger serialization and response. Separate health connection samples
        # remaining socket-owner blocking; this is not just isolated BPE timing.
        patterns = [" " * 32768, "x" * 32768, "!" * 32768,
                    "\x01" * 32768, "é" * 16384, "1234567890" * 3276 + "12345678"]
        measurements["pattern_order"] = ["spaces", "letters", "punctuation",
                                         "escaped_controls", "multibyte", "digits"] * 3
        for text in patterns * 3:
            request("POST", "/sessions/1/reset", {})
            ready = threading.Event()
            failure = []
            def preprocess():
                connection = service.connect()
                payload = json.dumps({"role": "user", "content": text}).encode()
                tick = time.perf_counter()
                try:
                    connection.request("POST", "/sessions/1/messages", payload)
                    ready.set()
                    response = connection.getresponse()
                    result = json.loads(response.read())
                    expected = 200 if text == " " * 32768 else 422
                    check(response.status == expected, (response.status, result))
                    measurements["preprocessing_http_ms"].append((time.perf_counter() - tick) * 1000)
                except BaseException as error:
                    failure.append(error)
                finally:
                    ready.set()
                    connection.close()
            thread = threading.Thread(target=preprocess)
            thread.start()
            check(ready.wait(3))
            tick = time.perf_counter()
            request("GET", "/health")
            measurements["concurrent_health_ms"].append((time.perf_counter() - tick) * 1000)
            thread.join(4)
            check(not thread.is_alive() and not failure, failure)
            ledger = request("GET", "/sessions/1")
            check(ledger["logical_tokens"] == (515 if text == " " * 32768 else 0), ledger)
        request("POST", "/sessions/1/reset", {})
        request("POST", "/sessions/1/messages", {"role": "user", "content": " " * 32768})
        raw(json.dumps({"role": "user", "content": " " * 32769}).encode(), 400)
        raw(b'{"role":"user","content":"x","extra":' + b'[' * 5 + b'0' + b']' * 5 + b'}', 400)
        raw(b'{}' + b' ' * 262142, 400)  # exact body bound, invalid schema
        raw(b'', 400, declared=262145)
        result = service.generate(1, "long", 1)
        check(result["event"] == "completed", result)
        retained = request("GET", "/requests/long")
        work = retained["metrics"]["work"]
        check(retained["metrics"]["metrics_version"] == 3)
        check(retained["metrics"]["work_version"] == 2)
        check(work["prefill_calls"] == 1 and work["append_calls"] == (515 - 1) // block, work)
        check(work["prefill_committed"] + work["append_committed"] == 515, work)
        request("POST", "/sessions/1/checkpoint", {"id": "save"}, 202)
        for _ in range(500):
            saved = request("GET", "/requests/save")
            if saved["status"] == "completed":
                break
            time.sleep(.002)
        check(saved["status"] == "completed", saved)
        check(saved["metrics"]["work"]["pending_committed"] == 1, saved)
        return measurements
    finally:
        check(service.close(), "shutdown")


def client_profile(server, tokenizer, block, directory):
    scenario = directory / "scenario.json"
    document = {"version": 1, "mode": "free_greedy", "operations": [
        {"op": "create"},
        {"op": "message", "content": " " * 32768,
         "expect": {"logical_tokens": 515}},
        {"op": "generate", "id": "cli-long", "budget": 1}]}
    scenario.write_text(json.dumps(document))
    check(real_workload.load_scenario(scenario, 4352) == document)
    try:
        real_workload.load_scenario(scenario)
    except ValueError:
        pass
    else:
        raise RuntimeError("default client accepted long input")
    output = directory / ("cli-" + str(block))
    arguments = ["real_workload", "--server", server, "--scenario", str(scenario),
                 "--output", str(output), "--native-model", tokenizer,
                 "--native-tokenizer", tokenizer, "--native-context", "4352",
                 "--native-chunk", str(block), "--native-prefill", "custom"]
    with patch.object(sys, "argv", arguments):
        check(real_workload.main() == 0, "complete long CLI path")
    check(json.loads((output / "result.json").read_text())["protocol_success"])
    for text in (" " * 32769, "é" * 16384 + "x"):
        document["operations"][1]["content"] = text
        scenario.write_text(json.dumps(document))
        try:
            real_workload.load_scenario(scenario, 4352)
        except ValueError:
            pass
        else:
            raise RuntimeError("client accepted oversized decoded UTF8 input")


if __name__ == "__main__":
    check(len(sys.argv) == 3)
    with tempfile.TemporaryDirectory(prefix="lunelette-long-http-") as directory:
        rows = [exercise(sys.argv[1], str(Path(sys.argv[2]).resolve()), block,
                         Path(directory) / str(block)) for block in (128, 256)]
        for block in (128, 256):
            client_profile(sys.argv[1], str(Path(sys.argv[2]).resolve()), block, Path(directory))
        print(json.dumps({"http_responsiveness": rows}, indent=2))
