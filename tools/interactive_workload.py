#!/usr/bin/env python3
"""Bounded persistent NDJSON client. No model execution unless explicitly configured."""
import argparse
import json
import os
import select
import signal
import stat
import sys
import threading
import time

from real_workload import AdmissionError, Service, strict_object
import native_workload_tools

LINE_BYTES = 262144
MAX_COMMANDS = 256
OUTPUT_BYTES = 4 * 1024 * 1024
LOG_BYTES = 8 * 1024 * 1024


def command(raw, message_bytes=4096):
    value = json.loads(raw, object_pairs_hook=strict_object)
    fields = {"create": set(), "message": {"content"}, "generate": {"budget"},
              "continue": {"budget"}, "inspect": set(), "reset": set(),
              "checkpoint": set(), "restore": set(), "cancel": set(), "quit": set(),
              "tools": {"budget", "max_calls", "max_generated", "wall_seconds", "answer", "profile"}}
    if not isinstance(value, dict) or value.get("op") not in fields:
        raise ValueError("unknown command")
    if set(value) != {"id", "op"} | fields[value["op"]]:
        raise ValueError("invalid fields")
    identity = value["id"]
    if not isinstance(identity, str) or not 1 <= len(identity) <= 48 or not all(
            c.isascii() and (c.isalnum() or c in "-_") for c in identity):
        raise ValueError("invalid id")
    if "budget" in value and (type(value["budget"]) is not int or not 1 <= value["budget"] <= 256):
        raise ValueError("budget outside 1..256")
    if "content" in value and (not isinstance(value["content"], str)
            or len(value["content"].encode()) > message_bytes):
        raise ValueError(f"content exceeds {message_bytes} bytes")
    if value["op"] == "tools":
        if value["profile"] not in {"calculator", "investigation", "observation"}:
            raise ValueError("invalid tool profile")
        native_workload_tools.validate_operation(tool_operation(value))
    return value


def tool_operation(value):
    names = {"calculator": "native_tool_loop", "investigation": "native_investigation_loop",
             "observation": "native_observation_loop"}
    return {**{k: v for k, v in value.items() if k != "profile"}, "op": names[value["profile"]]}


class Output:
    """Nonblocking bounded writes; a slow consumer is a fatal transport failure."""
    def __init__(self, fd, deadline):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("stdout must be a pipe, socket, or terminal, not a regular file")
        self.fd, self.deadline = fd, deadline
        self.lock = threading.Lock()
        self.total = 0
        os.set_blocking(fd, False)

    def emit(self, value):
        data = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with self.lock:
            self.total += len(data)
            if self.total > OUTPUT_BYTES:
                raise RuntimeError("stdout byte limit")
            end = min(self.deadline, time.monotonic() + .25)
            while data:
                remaining = end - time.monotonic()
                if remaining <= 0 or not select.select([], [self.fd], [], remaining)[1]:
                    raise TimeoutError("stdout consumer stalled")
                try:
                    n = os.write(self.fd, data)
                    data = data[n:]
                except BlockingIOError:
                    continue


class Adapter:
    def __init__(self, service, emit, budget):
        self.service, self.emit, self.remaining = service, emit, budget
        self.session = self.saved = self.worker = self.active = None
        self.ids = set()
        self.ids_lock = threading.Lock()
        self.fatal = None
        self.state_lock = threading.RLock()
        self.cancel_sent = set()
        self.cancelled = threading.Event()
        self.completed = threading.Event()
        self.failed = False

    def cancel_active(self):
        # Latch intent before looking at the active ID. Never hold this lock
        # across generate: a started callback may run on the worker thread.
        with self.state_lock:
            self.cancelled.set()
            if self.active is None:
                return 200, {"cancel_pending": True}
            return self.send_cancel(self.active)

    def send_cancel(self, identity):
        if identity in self.cancel_sent:
            return 200, {"cancel_pending": True}
        status, result = self.service.request("POST", f"/requests/{identity}/cancel", {})
        if status == 200:
            self.cancel_sent.add(identity)
        return status, result

    def started_event(self, identity, event):
        if event["event"] == "started":
            with self.state_lock:
                if self.cancelled.is_set():
                    self.send_cancel(identity)

    def reserve(self, identity):
        with self.ids_lock:
            if identity in self.ids:
                raise ValueError("id already used")
            self.ids.add(identity)

    def final(self, identity, **values):
        if threading.current_thread() is self.worker:
            self.completed.set()
        self.emit({"id": identity, "event": "final", **values})

    def dispatch(self, op):
        identity, action = op["id"], op["op"]
        self.reserve(identity)
        if self.worker is not None and self.completed.is_set():
            self.worker.join(timeout=1)
            if self.worker.is_alive():
                raise RuntimeError("completion worker did not finish")
        busy = self.worker is not None and self.worker.is_alive()
        if busy and action not in {"inspect", "cancel", "quit"}:
            raise ValueError("busy; command rejected, not queued")
        if action == "quit":
            self.final(identity, ok=True)
            return False
        if action == "create":
            if self.session is not None:
                raise ValueError("one session only; use reset")
            status, result = self.service.request("POST", "/sessions", {})
            if status != 200:
                if status in {400, 404, 409, 422}:
                    raise ValueError(f"admission rejected ({status}): {result}")
                raise RuntimeError(result)
            self.session = result["session_id"]
        else:
            if self.session is None:
                raise ValueError("create a session first")
            path = f"/sessions/{self.session}"
            if action in {"generate", "continue", "tools"}:
                if action == "tools":
                    native_workload_tools.capability(self.service,
                        investigation=op["profile"] == "investigation",
                        observation=op["profile"] == "observation")
                budget = op["max_generated"] if action == "tools" else op["budget"]
                if budget > self.remaining:
                    raise ValueError("total generated budget exhausted")
                self.remaining -= budget  # Reserved, never refunded, including errors/cancel.
                with self.state_lock:
                    self.active = None if action == "tools" else identity
                    self.cancelled.clear()
                    self.cancel_sent.clear()
                self.completed.clear()

                def generate():
                    try:
                        if action == "tools":
                            adapter = self
                            class ToolService:
                                def __getattr__(self, key):
                                    return getattr(adapter.service, key)

                                def __setattr__(self, key, value):
                                    setattr(adapter.service, key, value)

                                def request(self, *args, **kwargs):
                                    with adapter.state_lock:
                                        if adapter.cancelled.is_set():
                                            raise AdmissionError("tool loop cancelled; no further tool messages")
                                        return adapter.service.request(*args, **kwargs)

                                def generate(self, session, request_id, count):
                                    with adapter.state_lock:
                                        if adapter.cancelled.is_set():
                                            raise AdmissionError("tool loop cancelled before admission")
                                        adapter.active = request_id
                                    def event_callback(event):
                                        adapter.started_event(request_id, event)
                                        if event["event"] == "text_delta":
                                            adapter.emit({"id": identity, "request_id": request_id,
                                                          "event": "text", "text": event["text"]})
                                    try:
                                        return adapter.service.generate(session, request_id, count,
                                                                        on_event=event_callback)
                                    finally:
                                        with adapter.state_lock:
                                            adapter.active = None
                            result = native_workload_tools.run_loop(ToolService(), self.session,
                                tool_operation(op), self.reserve)
                            if not result["protocol_success"]:
                                self.failed = True
                            self.final(identity, ok=result["protocol_success"], stats=result)
                            return
                        result = self.service.generate(self.session, identity, budget,
                            on_event=lambda event: (self.started_event(identity, event),
                                self.emit({"id": identity, "event": "text", "text": event["text"]})
                                if event["event"] == "text_delta" else None))
                        if result["event"] == "error":
                            self.failed = True
                        self.final(identity, ok=result["event"] != "error", stats={
                            k: result.get(k) for k in ("event", "generated_tokens", "stop_reason",
                                "state", "completion_seconds", "first_actual_text_seconds", "metrics")})
                    except AdmissionError as error:
                        try:
                            self.final(identity, ok=False, error=str(error))
                        except Exception as output_error:
                            self.fatal = output_error
                            try:
                                self.service.record("failure", phase="adapter_worker",
                                                    error=repr(output_error))
                            except Exception:
                                pass
                    except Exception as error:
                        self.fatal = error
                        try:
                            self.service.record("failure", phase="adapter_worker", error=repr(error))
                        except Exception:
                            pass
                self.worker = threading.Thread(target=generate, daemon=True)
                self.worker.start()
                return True
            if action == "cancel":
                if not busy:
                    raise ValueError("no active generation")
                status, result = self.cancel_active()
            elif action == "inspect":
                status, result = self.service.request("GET", path)
            elif action in {"checkpoint", "restore"}:
                if (self.service.info.get("checkpoints") is not True
                        or self.service.info.get("prefix_reuse") not in {"on", "off"}):
                    raise ValueError("checkpoints not supported")
                if action == "restore" and self.saved is None:
                    raise ValueError("no published checkpoint")
                result = self.service.control(self.session, action, identity, self.saved)
                if action == "checkpoint":
                    self.saved = result
                status = 200
            else:
                body = {"role": "user", "content": op["content"]} if action == "message" else {}
                status, result = self.service.request("POST", path + (
                    "/messages" if action == "message" else "/reset"), body)
                if action == "reset" and status == 200:
                    self.saved = None
            if status != 200:
                if status in {400, 404, 409, 422}:
                    raise ValueError(f"admission rejected ({status}): {result}")
                raise RuntimeError(result)
        self.final(identity, ok=True, result=result)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-only", action="store_true", help="existing no-weight protocol server only")
    parser.add_argument("--native-model")
    parser.add_argument("--native-tokenizer")
    parser.add_argument("--native-strategy", default="resident-fp32", choices=["resident-fp32", "compact-fp8-arithmetic", "compact-fp8-lookup"])
    parser.add_argument("--native-prefill", default="mps", choices=["mps", "custom"])
    parser.add_argument("--native-prefix-reuse", default="legacy", choices=["legacy", "on", "off"])
    parser.add_argument("--native-context", type=int, default=256, choices=[256, 4352])
    parser.add_argument("--native-chunk", type=int, default=256, choices=[128, 256])
    parser.add_argument("--tools", choices=["calculator", "investigation", "observation"])
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--total-budget", type=int, default=1024)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800 or not 1 <= args.total_budget <= 4096:
        parser.error("seconds 1..1800; total-budget 1..4096")
    if args.test_only == bool(args.native_model or args.native_tokenizer) or (not args.test_only and not (args.native_model and args.native_tokenizer)):
        parser.error("choose --test-only OR both native model/tokenizer")
    if args.tools and args.test_only:
        parser.error("tools require native launch")
    if args.native_context == 256 and args.native_chunk != 256:
        parser.error("chunk128 requires context4352")
    deadline = time.monotonic() + args.seconds
    service = adapter = None
    def stop(signum, frame):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError("termination signal")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        config = json.dumps(vars(args), indent=2) + "\n"
        if len(config.encode()) > 65536:
            raise ValueError("config exceeds 65536 bytes")
        output = Output(sys.stdout.fileno(), deadline)
        native = None if args.test_only else ["--model", args.native_model, "--tokenizer", args.native_tokenizer,
            "--strategy", args.native_strategy, "--context", str(args.native_context), "--prefill-chunk", str(args.native_chunk),
            "--prefill", args.native_prefill]
        if native is not None and args.native_prefix_reuse != "legacy":
            native += ["--prefix-reuse", args.native_prefix_reuse]
        if args.tools:
            native += ["--tools", args.tools + "-v13"]
        service = Service(args.server, args.output, native_args=native,
            absolute_deadline=deadline, log_bytes=LOG_BYTES, evidence_config=vars(args),
            expected_strategy=None if args.test_only else args.native_strategy)
        if not args.test_only:
            expected = {"context_capacity": args.native_context, "prefill_chunk_size": args.native_chunk,
                        "prefix_reuse": args.native_prefix_reuse, "prefill": args.native_prefill,
                        "tools": bool(args.tools)}
            if args.tools:
                expected["conversation_protocol"] = "native-" + args.tools + "-v13"
            for key, value in expected.items():
                if service.info.get(key) != value:
                    raise RuntimeError(f"native /info {key} mismatch: expected {value!r}")
        adapter = Adapter(service, output.emit, args.total_budget)
        output.emit({"event": "ready", "info": service.info})
        pending = bytearray()
        count = 0
        running = True
        while running:
            if time.monotonic() >= deadline:
                raise TimeoutError("lifetime exceeded")
            if service.process.poll() is not None:
                raise RuntimeError("owned child exited unexpectedly")
            if adapter.fatal or service.stderr_error:
                raise RuntimeError(adapter.fatal or service.stderr_error)
            if not select.select([sys.stdin.fileno()], [], [], .05)[0]:
                continue
            chunk = os.read(sys.stdin.fileno(), 1)  # Never prefetch/queue competing commands.
            if not chunk:
                if pending:
                    raise ValueError("EOF within command")
                break
            pending.extend(chunk)
            if len(pending) > LINE_BYTES:
                raise ValueError("stdin line byte limit")
            if chunk != b"\n":
                continue
            count += 1
            if count > MAX_COMMANDS:
                raise ValueError("command count limit")
            raw = bytes(pending)
            pending.clear()
            op = command(raw, 32768 if args.native_context == 4352 else 4096)
            service.record("interactive_command", command=op)
            try:
                running = adapter.dispatch(op)
            except ValueError as error:
                output.emit({"id": op["id"], "event": "final", "ok": False, "error": str(error)})
        if adapter.worker and adapter.worker.is_alive():
            adapter.cancel_active()
            adapter.worker.join(timeout=3)
            if adapter.worker.is_alive():
                raise RuntimeError("active generation did not finish after shutdown cancel")
        if adapter.fatal or adapter.failed:
            raise RuntimeError(adapter.fatal or "runtime terminal error")
        return 0
    except Exception as error:
        if service is not None:
            try:
                service.record("failure", phase="adapter", error=repr(error))
            except Exception:
                pass
        print(f"interactive-workload: {error}", file=sys.stderr)
        return 1
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        if service is not None:
            # Interrupt transport/model work before joining the reader. Never wait on stdout.
            if service.process.poll() is None:
                service.process.terminate()
            try:
                service.process.wait(timeout=3)
            except Exception:
                service.process.kill()
                service.process.wait(timeout=3)
            if adapter and adapter.worker:
                adapter.worker.join(timeout=5)
            cleanup_ok = service.close()
            if not cleanup_ok or (adapter and adapter.worker and adapter.worker.is_alive()):
                print("interactive-workload: cleanup failed", file=sys.stderr)
                return 1
            if adapter and (adapter.failed or adapter.fatal):
                return 1


if __name__ == "__main__":
    sys.exit(main())
