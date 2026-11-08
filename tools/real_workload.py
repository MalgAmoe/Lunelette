#!/usr/bin/env python3
"""Versioned diagnostic workload runner. Explicit native or model-free launch."""
import argparse
import json
from pathlib import Path
import sys
import time

# Compatibility exports for archived clients; transport mocks target workload_client.
from workload_client import AdmissionError, Service, strict_object

import native_workload_tools
import workload_scenarios
import replay_logits


def load_scenario(path, context=256):
    value = json.loads(Path(path).read_text(), object_pairs_hook=strict_object)
    return validate_scenario(value, context)


def validate_scenario(value, context=256):
    if not isinstance(value, dict):
        raise ValueError("scenario must be an object")
    workload_scenarios.reject_retired_operations(value)
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
              "native_tool_loop": {"id", "budget", "max_calls", "max_generated", "wall_seconds", "answer"}}
    if v2:
        fields.update(checkpoint={"id", "checkpoint"}, restore={"id", "checkpoint"}, replay={"id", "tokens"})
    fields["native_investigation_loop"] = fields["native_tool_loop"]
    fields["native_observation_loop"] = fields["native_tool_loop"]
    for index, op in enumerate(value["operations"]):
        if not isinstance(op, dict) or not isinstance(op.get("op"), str) or op["op"] not in fields:
            raise ValueError("unknown operation")
        if op["op"] in {"native_tool_loop", "native_investigation_loop", "native_observation_loop"}:
            native_workload_tools.validate_operation(op)
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


def run_once(service, scenario):
    workload_scenarios.reject_retired_operations(scenario)
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
        elif action in {"native_tool_loop", "native_investigation_loop", "native_observation_loop"}:
            result = native_workload_tools.run_loop(service, session, op, reserve_id)
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
    workload_scenarios.reject_retired_operations(scenario)
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
                        help="native storage strategy (default: compact-fp8-arithmetic); requires model/tokenizer")
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
                           "--strategy", args.native_strategy or "compact-fp8-arithmetic", "--context", str(args.native_context), "--prefill-chunk", str(args.native_chunk),
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
                              expected_strategy=args.native_strategy or "compact-fp8-arithmetic",
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
