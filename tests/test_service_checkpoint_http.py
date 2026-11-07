#!/usr/bin/env python3
"""Real localhost native checkpoint protocol with CPU stubs, not model validation.

Usage: test_service_checkpoint_http.py SERVER OFFICIAL_TOKENIZER_JSON
All checks remain active under python -O. No downloads or Python model runtime.
"""
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from real_workload import Service


def check(value, detail="protocol check failed"):
    if not value:
        raise RuntimeError(detail)


def exercise(executable, tokenizer, policy, directory):
    service = Service(executable, directory, native_args=[
        "--model", tokenizer, "--tokenizer", tokenizer,
        "--strategy", "resident-fp32", "--context", "256",
        "--prefill-chunk", "256", "--prefill", "custom",
        "--prefix-reuse", policy])
    admitted = set()

    def request(method, path, body=None, expected=200):
        status, result = service.request(method, path, body)
        check(status == expected, (method, path, status, result))
        return result

    def poll(identity, expected="completed"):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = request("GET", f"/requests/{identity}")
            if result["status"] in ("completed", "cancelled", "error"):
                check(result["status"] == expected, result)
                return result
            time.sleep(.002)
        raise TimeoutError(identity)

    def control(operation, identity, checkpoint=None, cancel=False):
        body = {"id": identity}
        if checkpoint is not None:
            body["checkpoint_id"] = checkpoint
        request("POST", f"/sessions/1/{operation}", body, 202)
        admitted.add(identity)
        # Admission connection has already closed. Controls must survive it.
        pending = request("GET", f"/requests/{identity}")
        check(pending["status"] not in ("completed", "cancelled", "error"), pending)
        check(not pending["cancel_requested"])
        if cancel:
            check(request("POST", f"/requests/{identity}/cancel", {})["cancel_requested"])
        return poll(identity, "cancelled" if cancel else "completed")

    def generate(identity, budget=1):
        result = service.generate(1, identity, budget)
        admitted.add(identity)
        check(result["event"] == "completed", result)
        return request("GET", f"/requests/{identity}")

    try:
        request("POST", "/sessions", {})
        request("POST", "/sessions/1/messages", {"role": "user", "content": "Hi"})
        # Retain the completed stream's client object through admission. The
        # server may already have closed its socket; deterministic late-server-
        # close coverage is in test_service_transport.c.
        old = service.connect()
        try:
            old.request("POST", "/sessions/1/generate", json.dumps({"id": "old", "budget": 1}))
            response = old.getresponse()
            check(response.status == 200)
            events = [json.loads(line) for line in response.read().splitlines()]
            check(events[-1]["event"] == "completed")
            admitted.add("old")
            request("POST", "/sessions/1/checkpoint", {"id": "save"}, 202)
            admitted.add("save")
            response.close()
        finally:
            old.close()
        saved = poll("save")
        check(not saved["cancel_requested"])
        checkpoint = saved["checkpoint_id"]
        logical = saved["logical_tokens"]
        generate("branch", 2)
        restored = control("restore", "restore", checkpoint)
        check(restored["logical_tokens"] == logical)
        check(restored["committed_tokens"] == (logical if policy == "on" else 0))
        check(restored["pending_tokens"] == 0)
        check(restored["cache_restored"] == (policy == "on"))
        work = restored["metrics"]["work"]
        check(work["logit_restores"] == (1 if policy == "on" else 0))
        check(all(work[name] == 0 for name in ("prefill_committed", "append_committed",
                  "decode_committed", "prefill_calls", "append_calls", "decode_calls")))
        # A post-completion cancel cannot undo the published restore.
        check(not request("POST", "/requests/restore/cancel", {})["cancel_requested"])
        state = request("GET", "/sessions/1")
        check(state["logical_tokens"] == logical)
        check(state["committed_tokens"] == (logical if policy == "on" else 0))
        continued = generate("continued")
        work = continued["metrics"]["work"]
        check(work["prefill_committed"] == (logical if policy == "off" else 0), continued)
        check(work["prefill_calls"] == (1 if policy == "off" else 0))
        boundary = generate("boundary")
        check(boundary["metrics"]["work"]["prefill_calls"] == (1 if policy == "off" else 0))
        if policy == "off":
            check(boundary["metrics"]["work"]["prefill_committed"] == logical + 1)
        cancelled = control("restore", "cancel-restore", checkpoint, cancel=True)
        check(cancelled["state"] == "needs_reset")
        check(cancelled["metrics"]["cancel_source"] == "explicit")
        request("POST", "/sessions/1/restore", {"id": "invalidated", "checkpoint_id": checkpoint}, 409)
        request("POST", "/sessions/1/reset", {})
        request("POST", "/sessions/1/messages", {"role": "user", "content": "Hi"})
        control("checkpoint", "recovered")
        cancelled = control("checkpoint", "cancel-save", cancel=True)
        check(cancelled["state"] == "needs_reset")
        request("POST", "/sessions/1/reset", {})
        request("POST", "/sessions/1/messages", {"role": "user", "content": "Hi"})
        # Generate and control IDs share the same bounded history, across reset.
        request("POST", "/sessions/1/checkpoint", {"id": "old"}, 409)
        request("POST", "/sessions/1/generate", {"id": "save", "budget": 1}, 409)
        while len(admitted) < 256:
            identity = f"capacity-{len(admitted)}"
            control("checkpoint", identity)
        request("POST", "/sessions/1/checkpoint", {"id": "overflow"}, 503)
        request("POST", "/sessions/1/generate", {"id": "overflow", "budget": 1}, 503)
        request("POST", "/sessions/1/reset", {})
        request("POST", "/sessions/1/checkpoint", {"id": "save"}, 409)
    finally:
        original = sys.exc_info()[1]
        try:
            check(service.close(), "server shutdown failed")
        except BaseException:
            if original is None:
                raise
            print("cleanup failed after foreground failure", file=sys.stderr)


def main():
    check(len(sys.argv) == 3, __doc__)
    tokenizer = str(Path(sys.argv[2]).resolve())
    with tempfile.TemporaryDirectory(prefix="lunelette-checkpoint-") as directory:
        for policy in ("on", "off"):
            exercise(sys.argv[1], tokenizer, policy, Path(directory) / policy)
    print("native stub localhost checkpoint on/off integration passed (not model validation)")


if __name__ == "__main__":
    main()
