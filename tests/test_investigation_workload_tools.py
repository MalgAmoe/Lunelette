"""Model-free investigation profile, using the same authoritative fake ledger."""
import copy
import json
import unittest
from unittest.mock import patch
from test_native_workload_tools import Fake, OP, native, load_scenario
from pathlib import Path

INV = {**OP, "op": "native_investigation_loop", "answer": "19 [LARCH-STOCK:P1]"}


class Investigation(Fake):
    def __init__(self):
        super().__init__()
        self.info["conversation_protocol"] = "native-investigation-v13"
        self.calls = [("lookup", {"document_id": "larch-stock"}),
                      ("calculator", {"operation": "subtract", "a": 31, "b": 12})]
        self.final = INV["answer"]
        self.modify = lambda value: None
        self.records = []

    def record(self, event, **values):
        self.records.append((event, values))

    def generate(self, session, request_id, budget):
        turn = len(self.budgets)
        self.turn = 0 if turn < len(self.calls) else 1
        value = super().generate(session, request_id, budget)
        if turn < len(self.calls):
            name, args = self.calls[turn]
            value["tool_call"] = {"id": request_id, "name": name, "arguments": args.copy()}
            value["request_metadata"]["tool_call"] = copy.deepcopy(value["tool_call"])
        else:
            value["text"] = value["request_metadata"]["text"] = self.final
        self.modify(value)
        return value


class Tests(unittest.TestCase):
    def run_fake(self, service=None, op=None):
        service = service or Investigation()
        report = native.run_loop(service, 7, op or INV, set().add)
        self.assertIsNone(service.absolute_deadline)
        return service, report

    def test_positive(self):
        service, report = self.run_fake()
        self.assertTrue(report["task_success"])
        self.assertTrue(report["protocol_success"])
        self.assertEqual((report["tool_calls"], len(service.budgets)), (2, 3))
        self.assertEqual(report["generated_tokens"], 12)
        self.assertEqual(report["tool_seconds"], sum(report["tool_call_seconds"]))
        self.assertEqual(report["corpus_sha256"], native.CORPUS_SHA256)
        self.assertEqual(service.records[0][0], "investigation_corpus")
        self.assertEqual(service.messages[0], {"role": "tool", "tool_call_id": "calc-0",
            "content": json.dumps(native.load_corpus()["larch-stock"], ensure_ascii=False,
                                  sort_keys=True, separators=(",", ":"))})
        self.assertEqual(service.messages[1]["content"], "19")
        scenario = load_scenario(Path(__file__).resolve().parents[1] / "tools/scenarios/investigation-v13.json")
        self.assertEqual(scenario["operations"][2]["op"], "native_investigation_loop")

    def test_schema_fail_closed(self):
        bad_calls = [("lookup", {}), ("lookup", {"document_id": "other"}),
                     ("lookup", {"document_id": True}), ("lookup", {"document_id": 1.0}),
                     ("lookup", {"document_id": []}),
                     ("lookup", {"document_id": "larch-stock", "path": "/tmp/x"}),
                     ("calculator", {"operation": "subtract", "a": True, "b": 12}),
                     ("calculator", {"operation": "subtract", "a": 31.0, "b": 12}),
                     ("shell", {})]
        for call in bad_calls:
            with self.subTest(call=call):
                service = Investigation()
                service.calls[0] = call
                _, report = self.run_fake(service)
                self.assertEqual(report["failure_kind"], "protocol")
                self.assertEqual(service.messages, [])
                self.assertEqual(report["tool_calls"], 0)

    def test_bad_metadata_and_profiles(self):
        for key, value in [("stop_reason", "truncated_tool_call"), ("session_id", True),
                           ("generated_tokens", 5.0), ("text_delta_count", True)]:
            service = Investigation()
            service.modify = lambda result: result.update({key: value})
            _, report = self.run_fake(service)
            self.assertEqual(report["failure_kind"], "protocol")
            self.assertEqual(service.messages, [])
        for key, value in [("tools", 1), ("conversation_protocol", "native-calculator-v13"),
                           ("conversation_protocol", "unknown")]:
            service = Investigation()
            service.info[key] = value
            _, report = self.run_fake(service)
            self.assertEqual(report["failure_kind"], "protocol")
            self.assertEqual(service.budgets, [])

    def test_quality_not_protocol(self):
        for variant in ("cedar", "reverse", "wrong_math", "wrong_answer", "direct"):
            service = Investigation()
            if variant == "cedar":
                service.calls[0] = ("lookup", {"document_id": "cedar-stock"})
            elif variant == "reverse":
                service.calls.reverse()
            elif variant == "wrong_math":
                service.calls[1] = ("calculator", {"operation": "add", "a": 31, "b": 12})
            elif variant == "direct":
                service.calls = []
            else:
                service.final = "18 [LARCH-STOCK:P1]"
            _, report = self.run_fake(service)
            self.assertTrue(report["protocol_success"])
            self.assertFalse(report["task_success"])
            self.assertIsNone(report["failure_kind"])

    def test_limits_deadline_and_cancellation(self):
        for limits in ({"max_calls": 1}, {"max_generated": 5}):
            _, report = self.run_fake(op={**INV, **limits})
            self.assertEqual(report["failure_kind"], "limit")
        clock = [0]
        service = Investigation()
        service.modify = lambda value: clock.__setitem__(0, 120)
        with patch.object(native.time, "monotonic", side_effect=lambda: clock[0]):
            _, report = self.run_fake(service)
        self.assertEqual(report["failure_kind"], "limit")
        self.assertEqual(service.messages, [])
        self.assertEqual(report["generated_tokens"], 5)
        service = Investigation()
        with patch.object(service, "generate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                native.run_loop(service, 7, INV, set().add)
        self.assertIsNone(service.absolute_deadline)

    def test_corpus_drift(self):
        service = Investigation()
        with patch.object(Path, "open") as opening:
            opening.return_value.__enter__.return_value.read.return_value = b'{"version":1}'
            _, report = self.run_fake(service)
        self.assertEqual(report["failure_kind"], "protocol")
        self.assertEqual(service.budgets, [])
        self.assertEqual(service.messages, [])


if __name__ == "__main__":
    unittest.main()
