#!/usr/bin/env python3
"""Model-free tool-loop regressions; unittest checks remain active under -O."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from real_workload import Service, load_scenario, run
from workload_tools import ProtocolError, parse, run_loop

OP = {"op": "tool_loop", "id": "calc", "budget": 100, "max_calls": 2,
      "max_generated": 300, "wall_seconds": 10, "answer": 43}


def tool(call_id="a", operation="add", a=19, b=23):
    return {"version": 1, "type": "tool", "id": call_id, "name": "calculator",
            "arguments": {"operation": operation, "a": a, "b": b}}


def response(value, **changes):
    return {"event": "completed", "stop_reason": "eos", "state": "ready",
            "generated_tokens": 20, "text": json.dumps(value), **changes}


FINAL = {"version": 1, "type": "final", "answer": 43}


class Fake:
    native = False
    absolute_deadline = None

    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages, self.ids, self.records = [], [], []
        self.status = 200

    def generate(self, session, request_id, budget, cancel=False):
        self.ids.append((session, request_id, budget))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        reply = copy.deepcopy(reply)
        reply.setdefault("id", request_id)
        reply.setdefault("session_id", session)
        reply.setdefault("session_valid", True)
        reply.setdefault("recovery", "none")
        if self.native:
            count = reply["generated_tokens"]
            ids = [10] * (count - 1) + [2]
            tokens = [1, 3] + ids
            for key, value in {"logical_tokens": len(tokens),
                               "committed_tokens": len(tokens) - 1, "pending_tokens": 1}.items():
                reply.setdefault(key, value)
            reply.setdefault("request_metadata", {
                **{key: reply[key] for key in ("id", "session_id", "session_valid", "recovery",
                    "generated_tokens", "stop_reason", "state", "logical_tokens",
                    "committed_tokens", "pending_tokens")},
                "status": "completed", "operation": "generate", "token_ids": ids})
            reply.setdefault("session_metadata", {"session_id": session, "state": "ready",
                "logical_tokens": len(tokens), "committed_tokens": len(tokens) - 1,
                "pending_tokens": 1, "remaining": 256 - len(tokens), "token_ids": tokens})
        return reply

    def request(self, method, path, body):
        self.messages.append((method, path, body))
        return self.status, {"session_id": 7}

    def record(self, kind, **values):
        self.records.append((kind, values))


def scenario(op=None):
    return {"version": 1, "mode": "controlled_protocol_only", "operations": [
        {"op": "create"}, copy.deepcopy(op or OP)]}


class ToolTests(unittest.TestCase):
    def execute(self, replies, op=None):
        service = Fake(replies)
        result = run(service, scenario(op))
        self.assertIsNone(service.absolute_deadline)
        return service, result

    def test_multi_turn_and_score(self):
        service, result = self.execute([response(tool()), response(tool("b", "add", 42, 1)),
                                        response(FINAL)])
        self.assertTrue(result["protocol_success"])
        self.assertTrue(result["task_success"])
        self.assertEqual([item[1] for item in service.ids], ["calc-0", "calc-1", "calc-2"])
        self.assertEqual([item[0] for item in service.ids], [7, 7, 7])
        for method, path, body in service.messages[1:]:
            self.assertEqual((method, path, body["role"]), ("POST", "/sessions/7/messages", "user"))
            self.assertTrue(body["content"].startswith("APPLICATION_TOOL_RESULT_V1 "))
        self.assertIn('"result":42', service.messages[1][2]["content"])
        self.assertEqual(result["tool_loops"][0]["generated_tokens"], 60)
        _, wrong = self.execute([response({**FINAL, "answer": 42})])
        self.assertTrue(wrong["protocol_success"])
        self.assertFalse(wrong["task_success"])

    def test_parser_rejections(self):
        bad = ['{', '[]', json.dumps(tool()) + ' trailing',
               '{"version":1,"version":1,"type":"final","answer":43}',
               '{"version":1,"type":"final","answer":NaN}',
               '{"version":1,"type":"final","answer":Infinity}',
               '{"version":1,"type":"final","answer":1e999}',
               '{"version":1,"type":"final","answer":12345678901234567890}',
               json.dumps({**FINAL, "answer": True}), json.dumps({**FINAL, "version": True}),
               json.dumps({**FINAL, "answer": "43"}), json.dumps({**FINAL, "extra": 1}),
               json.dumps({**tool(), "name": "shell"}),
               json.dumps(tool(a=True)), json.dumps(tool(a=1000001)),
               json.dumps(tool(operation="eval")), json.dumps(tool("x" * 33)),
               json.dumps({**tool(), "arguments": {"operation": "add", "a": 1, "b": 2, "x": 0}}),
               json.dumps({**FINAL, "answer": "x" * 2049}),
               '{"version":1,"type":"tool","id":"x","name":"calculator",'
               '"arguments":{"operation":"add","a":1,"a":2,"b":3}}']
        for text in bad:
            with self.subTest(text=text[:100]):
                with self.assertRaises(ProtocolError):
                    parse(text)
                service, result = self.execute([response(FINAL, text=text)])
                self.assertEqual(result["failure_kind"], "protocol")
                self.assertIsNone(result["task_success"])
                self.assertEqual(len(service.messages), 1)
        for operation, a, b in [("add", -2, 3), ("subtract", -2, 3),
                                 ("multiply", 1000000, 1000000)]:
            parsed = parse(json.dumps(tool(operation=operation, a=a, b=b)))
            self.assertEqual(parsed["arguments"]["a"], a)

    def test_no_execution_without_natural_eos(self):
        for changes in ({"stop_reason": "budget", "state": "partial"},
                        {"state": "partial"}, {"stop_reason": "budget"},
                        {"text": '{'}, {"event": "cancelled"}, {"event": "error"},
                        {"generated_tokens": True}, {"session_metadata": {"state": "needs_reset"}}):
            service, report = self.execute([response(tool(), **changes)])
            self.assertFalse(report["protocol_success"])
            self.assertEqual(len(service.messages), 1)

    def test_failed_protocol_retains_reported_work(self):
        service, report = self.execute([response(tool(), stop_reason="budget", state="partial")])
        self.assertEqual(report["generated_tokens"], 20)
        self.assertEqual(report["tool_calls"], 0)
        self.assertEqual(len(service.messages), 1)

    def test_limits_duplicates_runtime_recovery(self):
        service, result = self.execute([response(tool()), response(tool())])
        self.assertEqual(result["failure_kind"], "protocol")
        self.assertEqual(len(service.messages), 2)
        op = {**OP, "max_calls": 1}
        service, result = self.execute([response(tool()), response(tool("b"))], op)
        self.assertEqual(result["failure_kind"], "limit")
        self.assertEqual(len(service.messages), 2)
        _, result = self.execute([response(tool(), generated_tokens=20)], {**OP, "max_generated": 20})
        self.assertEqual(result["failure_kind"], "limit")
        service, result = self.execute([TimeoutError("wire timeout")])
        self.assertEqual(result["failure_kind"], "runtime")
        service.replies = iter([response(FINAL)])
        status, _ = service.request("POST", "/sessions/7/reset", {})
        self.assertEqual(status, 200)
        reserved = set()
        recovered = run_loop(service, 7, {**OP, "id": "recovered"}, reserved.add)
        self.assertTrue(recovered["task_success"])
        self.assertEqual(service.messages[-1][1], "/sessions/7/reset")
        self.assertEqual(service.ids[-1][:2], (7, "recovered-0"))
        service = Fake([response(tool())])
        with patch("workload_tools.time.monotonic", side_effect=[0, 0, 0, 11, 11]):
            report = run(service, scenario())
        self.assertEqual(report["failure_kind"], "limit")
        self.assertEqual(len(service.messages), 1)
        with self.assertRaises(ValueError):
            run(Fake([response(FINAL)]), {"mode": "controlled_protocol_only", "operations": [
                {"op": "generate", "id": "calc-0", "budget": 100},
                {"op": "generate", "id": "calc-0", "budget": 100}]})
        _, report = self.execute([response(tool()), response(FINAL)], {**OP, "budget": 10})
        self.assertEqual(report["failure_kind"], "runtime")

    def test_calculation_append_and_request_collision(self):
        for operation, a, b, expected in [("subtract", -2, 3, -5),
                                          ("multiply", 1000000, 1000000, 1000000000000)]:
            service, result = self.execute([response(tool(operation=operation, a=a, b=b)),
                                            response({**FINAL, "answer": expected})],
                                           {**OP, "answer": expected})
            self.assertTrue(result["task_success"])
            payload = service.messages[-1][2]["content"].split(" ", 1)[1]
            self.assertEqual(json.loads(payload)["result"], expected)
        service = Fake([response(tool())])
        service.status = 422
        report = run_loop(service, 7, OP, set().add)
        self.assertEqual(report["failure_kind"], "runtime")
        self.assertEqual(len(service.ids), 1)
        service = Fake([response(FINAL)])
        report = run(service, {"mode": "controlled_protocol_only", "operations": [
            {"op": "generate", "id": "calc-0", "budget": 100}, OP]})
        self.assertEqual(report["failure_kind"], "runtime")
        self.assertEqual(len(service.ids), 1)
        service, report = self.execute([response(FINAL, generated_tokens=7)],
                                       {**OP, "max_generated": 7})
        self.assertTrue(report["task_success"])
        self.assertEqual(service.ids[0][2], 7)

    def test_native_coherence_matrix(self):
        fixture = Fake([response(FINAL)])
        fixture.native = True
        fixture.info = {"context_capacity": 256}
        good = fixture.generate(7, "calc-0", 100)
        for value in (tool(), FINAL):
            baseline = {**copy.deepcopy(good), "text": json.dumps(value)}
            final = copy.deepcopy(good)
            final["id"] = "calc-1"
            final["request_metadata"]["id"] = "calc-1"
            service = Fake([baseline, final])
            service.native = True
            service.info = fixture.info
            report = run_loop(service, 7, OP, set().add)
            self.assertTrue(report["protocol_success"])
            mutations = [(None, "id", "wrong"), (None, "session_id", True),
                         (None, "session_valid", False), (None, "recovery", "reset")]
            for key, bad in (("status", "running"), ("operation", "continue"),
                             ("id", "wrong"), ("session_id", 8), ("session_valid", False),
                             ("recovery", "reset"), ("generated_tokens", 19),
                             ("stop_reason", "budget"), ("state", "partial"),
                             ("logical_tokens", 23), ("committed_tokens", 20),
                             ("pending_tokens", True), ("token_ids", []),
                             ("token_ids", [10] * 19 + [True]),
                             ("token_ids", [10] * 19 + [131072]),
                             ("token_ids", [10] * 19 + [-1]),
                             ("token_ids", [2] + [10] * 18 + [2]),
                             ("token_ids", [10] * 20)):
                mutations.append(("request_metadata", key, bad))
            for key, bad in (("session_id", 8), ("state", "partial"),
                             ("logical_tokens", 21), ("committed_tokens", 22),
                             ("pending_tokens", 0), ("remaining", 1),
                             ("token_ids", [1, 3] + [11] * 19 + [2]),
                             ("token_ids", [True] + good["session_metadata"]["token_ids"][1:])):
                mutations.append(("session_metadata", key, bad))
            mutations.extend([(None, "request_metadata", None),
                              (None, "session_metadata", [] )])
            for parent, key, bad in mutations:
                with self.subTest(kind=value["type"], parent=parent, key=key, bad=bad):
                    broken = copy.deepcopy(baseline)
                    (broken[parent] if parent else broken)[key] = bad
                    service = Fake([broken])
                    service.native = True
                    service.info = fixture.info
                    report = run_loop(service, 7, OP, set().add)
                    self.assertFalse(report["protocol_success"])
                    self.assertIsNone(report["task_success"])
                    self.assertEqual(service.messages, [])

    def test_identity_session_and_interrupts(self):
        for session in (None, True, "7", 0, -1):
            service = Fake([response(FINAL)])
            report = run_loop(service, session, OP, set().add)
            self.assertEqual(report["failure_kind"], "runtime")
            self.assertEqual(service.ids, [])
        for value in (tool(), FINAL):
            for changes in ({"id": "wrong"}, {"session_id": True},
                            {"session_valid": False}, {"recovery": "reset"}):
                service, report = self.execute([response(value, **changes)])
                self.assertFalse(report["protocol_success"])
                self.assertEqual(len(service.messages), 1)
        service = Fake([])
        with patch.object(service, "generate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_loop(service, 7, OP, set().add)
        self.assertIsNone(service.absolute_deadline)
        service = Fake([TimeoutError("expired")])
        with patch("workload_tools.time.monotonic", side_effect=[0, 0, 11, 11]):
            report = run_loop(service, 7, OP, set().add)
        self.assertEqual(report["failure_kind"], "limit")

    def test_scenario_and_timeout(self):
        path = Path(__file__).resolve().parents[1] / "tools/scenarios/calculator-v1.json"
        loaded = load_scenario(path)
        self.assertEqual(loaded["operations"][2]["answer"], 43)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for changes in ({"max_calls": True}, {"wall_seconds": float("nan")},
                            {"max_generated": 10000}, {"id": "x" * 49}, {"extra": 1}):
                path.write_text(json.dumps(scenario({**OP, **changes})))
                with self.assertRaises(ValueError):
                    load_scenario(path)
        service = Service.__new__(Service)
        service.absolute_deadline = 12
        with patch("real_workload.time.monotonic", return_value=10):
            self.assertEqual(service.bounded_timeout(120), 2)
        with patch("real_workload.time.monotonic", return_value=12):
            with self.assertRaises(TimeoutError):
                service.bounded_timeout(4)


if __name__ == "__main__":
    unittest.main()
