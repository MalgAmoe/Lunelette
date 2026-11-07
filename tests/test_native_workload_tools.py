"""Deterministic fake API tests: no tokenizer, weights, or model execution."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import native_workload_tools as native
from real_workload import load_scenario

OP = {"op": "native_tool_loop", "id": "calc", "budget": 128, "max_calls": 2,
      "max_generated": 384, "wall_seconds": 120, "answer": 43}


class Fake:
    native = True
    absolute_deadline = None

    def __init__(self):
        self.info = {"tools": True, "conversation_protocol": "native-calculator-v13",
                     "context_capacity": 4352}
        self.ids = [1, 5, 1003, 6, 3, 1000, 4]
        self.committed = 0
        self.pending = 0
        self.messages = []
        self.turn = 0
        self.mutate = lambda value: None
        self.result_mutate = lambda value: None
        self.budgets = []
        self.direct = False

    def snapshot(self, state="ready"):
        return {"session_id": 7, "state": state, "token_ids": self.ids[:],
                "logical_tokens": len(self.ids), "remaining": 4352 - len(self.ids),
                "committed_tokens": self.committed, "pending_tokens": self.pending}

    def request(self, method, path, body=None):
        if method == "POST":
            self.messages.append(body)
            self.ids += [7, 1043, 8]
        value = self.snapshot()
        if method == "POST":
            self.result_mutate(value)
        return 200, value

    def generate(self, session, request_id, budget):
        self.budgets.append(budget)
        call = self.turn == 0 and not self.direct
        ids = [9, 1001, 32, 1002, 2] if call else [1043, 2]
        self.ids += ids
        self.committed, self.pending = len(self.ids) - 1, 1
        state = "awaiting_tool" if call else "ready"
        value = {"id": request_id, "session_id": session, "event": "completed",
                 "session_valid": True, "recovery": "none", "state": state,
                 "stop_reason": "tool_call" if call else "eos", "generated_tokens": len(ids),
                 "text": "" if call else " 43\n", "text_delta_count": 0 if call else 1,
                 "terminal_text": "", "logical_tokens": len(self.ids),
                 "committed_tokens": self.committed, "pending_tokens": 1}
        if call:
            value["tool_call"] = {"id": request_id, "name": "calculator",
                                  "arguments": {"operation": "add", "a": 19, "b": 24}}
        value["request_metadata"] = {**copy.deepcopy(value), "status": "completed",
                                      "operation": "generate", "token_ids": ids}
        value["session_metadata"] = self.snapshot(state)
        self.turn += 1
        self.mutate(value)
        return value

    def record(self, *args, **kwargs):
        pass


class Tests(unittest.TestCase):
    def execute(self, service=None, op=None):
        service = service or Fake()
        report = native.run_loop(service, 7, op or OP, set().add)
        self.assertIsNone(service.absolute_deadline)
        return service, report

    def test_positive_and_accounting(self):
        service, report = self.execute()
        self.assertTrue(report["protocol_success"])
        self.assertTrue(report["task_success"])
        self.assertEqual(report["generated_tokens"], 7)
        self.assertEqual(service.messages, [{"role": "tool", "tool_call_id": "calc-0", "content": "43"}])
        service = Fake()
        service.direct = True
        _, report = self.execute(service)
        self.assertTrue(report["protocol_success"])
        self.assertFalse(report["task_success"])
        self.assertFalse(report["tool_used"])

    def test_malformed_executes_zero(self):
        mutations = [(None, "id", "wrong"), (None, "session_id", True),
                     (None, "session_valid", False), (None, "recovery", "reset"),
                     (None, "stop_reason", "truncated_tool_call"), (None, "event", "error"),
                     (None, "text_delta_count", 1), (None, "terminal_text", "bad"),
                     (None, "generated_tokens", True), (None, "text", "bad"),
                     ("request_metadata", "status", "running"),
                     ("request_metadata", "token_ids", [9, 999, 32, 1002, 2]),
                     ("request_metadata", "token_ids", [9, 1001, 32, 32, 2]),
                     ("request_metadata", "tool_call", None),
                     ("session_metadata", "remaining", 1),
                     ("session_metadata", "committed_tokens", 0),
                     ("session_metadata", "token_ids", [1, 9, 1001, 32, 1002, 2])]
        for parent, key, bad in mutations:
            with self.subTest(parent=parent, key=key, bad=bad):
                service = Fake()
                def mutate(value):
                    (value[parent] if parent else value)[key] = bad
                service.mutate = mutate
                _, report = self.execute(service)
                self.assertFalse(report["protocol_success"])
                self.assertEqual(report["tool_calls"], 0)
                self.assertEqual(service.messages, [])

    def test_schema_and_bounds(self):
        for key, bad in [("a", True), ("a", 1.0), ("a", 1000001), ("b", -1000001),
                         ("operation", "divide"), ("extra", 1)]:
            service = Fake()
            def mutate(value):
                for item in (value, value["request_metadata"]):
                    item["tool_call"]["arguments"][key] = bad
            service.mutate = mutate
            _, report = self.execute(service)
            self.assertEqual(report["tool_calls"], 0)
            self.assertEqual(report["failure_kind"], "protocol")
        for a in (-1000000, 1000000):
            self.assertEqual(native.arguments({"id": "x", "name": "calculator", "arguments":
                {"operation": "multiply", "a": a, "b": a}}, "x")["a"], a)
        for key, bad in [("budget", 129), ("max_calls", 3), ("max_generated", 385),
                         ("wall_seconds", 121), ("answer", 42)]:
            with self.assertRaises(ValueError):
                native.validate_operation({**OP, key: bad})

    def test_capability_limits_deadline_and_result(self):
        service = Fake()
        service.info["tools"] = False
        _, report = self.execute(service)
        self.assertEqual(service.budgets, [])
        self.assertEqual(report["failure_kind"], "protocol")
        service, report = self.execute(op={**OP, "max_generated": 5})
        self.assertEqual(report["failure_kind"], "limit")
        self.assertEqual(report["generated_tokens"], 5)
        self.assertEqual(service.budgets, [5])
        service = Fake()
        service.absolute_deadline = -1
        report = native.run_loop(service, 7, OP, set().add)
        self.assertEqual(report["failure_kind"], "limit")
        self.assertEqual(service.absolute_deadline, -1)
        self.assertEqual(service.budgets, [])
        for key, bad in [("token_ids", [7, 1043, 8]), ("pending_tokens", 0),
                         ("state", "awaiting_tool")]:
            service = Fake()
            service.result_mutate = lambda value: value.update({key: bad})
            _, report = self.execute(service)
            self.assertFalse(report["protocol_success"])
            self.assertEqual(len(service.budgets), 1)
        service = Fake()
        with patch.object(service, "generate", side_effect=TimeoutError("wire")):
            _, report = self.execute(service)
        self.assertEqual(report["failure_kind"], "runtime")

    def test_wrong_final_and_inflight_deadline(self):
        service = Fake()
        def wrong(value):
            if value["stop_reason"] == "eos":
                value["text"] = value["request_metadata"]["text"] = "42"
        service.mutate = wrong
        _, report = self.execute(service)
        self.assertTrue(report["protocol_success"])
        self.assertFalse(report["task_success"])
        service = Fake()
        clock = [0]
        def expire(value):
            clock[0] = 120
        service.mutate = expire
        with patch("native_workload_tools.time.monotonic", side_effect=lambda: clock[0]):
            _, report = self.execute(service)
        self.assertEqual(report["failure_kind"], "limit")
        self.assertEqual(service.messages, [])
        self.assertEqual(report["tool_calls"], 0)
        self.assertEqual(report["generated_tokens"], 5)

    def test_runtime_and_missing_declaration(self):
        service = Fake()
        service.ids = [1, 3, 1000, 4]
        _, report = self.execute(service)
        self.assertEqual(report["failure_kind"], "protocol")
        self.assertEqual(service.budgets, [])
        for reason, event in (("generator_failure", "error"), ("cancelled", "cancelled")):
            service = Fake()
            def fail(value):
                value.update(event=event, stop_reason=reason)
                value["request_metadata"].update(status=event, stop_reason=reason)
            service.mutate = fail
            _, report = self.execute(service)
            self.assertEqual(report["failure_kind"], "runtime")
            self.assertEqual(report["generated_tokens"], 5)
            self.assertEqual(service.messages, [])

    def test_second_call_limit(self):
        service = Fake()
        original = service.generate
        def repeated(*args):
            service.turn = 0
            return original(*args)
        service.generate = repeated
        _, report = self.execute(service)
        self.assertEqual(report["failure_kind"], "limit")
        self.assertEqual(report["tool_calls"], 2)
        self.assertEqual(report["generated_tokens"], 15)
        self.assertEqual(len(service.messages), 2)

    def test_scenario(self):
        scenario = load_scenario(Path(__file__).resolve().parents[1] / "tools/scenarios/calculator-v13.json")
        self.assertEqual(scenario["operations"][2]["op"], "native_tool_loop")


if __name__ == "__main__":
    unittest.main()
