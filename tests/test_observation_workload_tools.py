"""Model-free observation reuse checks, including optimized Python execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_investigation_workload_tools import Investigation
from test_native_workload_tools import native, load_scenario
import real_workload

SCENARIO = Path(__file__).resolve().parents[1] / "tools/scenarios/observation-v13.json"
OBS = load_scenario(SCENARIO)["operations"][2]


class Observation(Investigation):
    def __init__(self):
        super().__init__()
        self.info["conversation_protocol"] = "native-observation-v13"
        self.calls = [("lookup", {"document_id": "pump-readings"})]
        self.final = OBS["answer"]


class Tests(unittest.TestCase):
    def execute(self, service=None, op=None):
        service = service or Observation()
        report = native.run_loop(service, 7, op or OBS, set().add)
        self.assertIsNone(service.absolute_deadline)
        return service, report

    def test_success(self):
        service, report = self.execute()
        self.assertTrue(report["protocol_success"])
        self.assertTrue(report["task_success"])
        self.assertEqual(report["tool_calls"], 1)
        self.assertEqual(report["generated_tokens"], 7)
        self.assertEqual(service.budgets, [128, 128])
        self.assertEqual(report["tool_seconds"], sum(report["tool_call_seconds"]))
        self.assertEqual(report["corpus_version"], 1)
        self.assertEqual(report["corpus_sha256"], native.OBSERVATION_CORPUS_SHA256)
        self.assertEqual(service.records[0][0], "observation_corpus")
        record = native.load_corpus(observation=True)["pump-readings"]
        self.assertEqual(service.messages, [{"role": "tool", "tool_call_id": "observation-v13-0",
            "content": json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}])
        self.assertEqual(load_scenario(SCENARIO)["operations"][1]["content"],
            "Use lookup to read pump-readings. Report the hottest device, its integer temperature, and ALERT if temperature is at least 80 else OK. Answer exactly <device> <temperature> <status>.")

    def test_bad_calls_no_action(self):
        for name, args in [("calculator", {"operation": "add", "a": 1, "b": 2}),
                           ("shell", {}), ("lookup", {}),
                           ("lookup", {"document_id": "pump-readings", "extra": 1})] + [
                           ("lookup", {"document_id": value}) for value in
                           ("larch-stock", "cedar-stock", "other", True, 1.0, [])]:
            with self.subTest(name=name, args=args):
                service = Observation()
                service.calls = [(name, args)]
                _, report = self.execute(service)
                self.assertEqual(report["failure_kind"], "protocol")
                self.assertEqual(report["tool_calls"], 0)
                self.assertEqual(service.messages, [])
        call = {"id": "x", "name": "lookup", "arguments": {"document_id": "pump-readings"}}
        for investigation in (False, True):
            with self.assertRaises(native.ProtocolError):
                native.arguments(call, "x", investigation=investigation)

    def test_metadata_and_profiles(self):
        for key, value in [("stop_reason", "truncated_tool_call"), ("session_id", True),
                           ("text_delta_count", 1), ("generated_tokens", 5.0)]:
            service = Observation()
            service.modify = lambda result: result.update({key: value})
            _, report = self.execute(service)
            self.assertEqual(report["failure_kind"], "protocol")
            self.assertEqual(service.messages, [])
        for profile in ("native-calculator-v13", "native-investigation-v13", "unknown"):
            service = Observation()
            service.info["conversation_protocol"] = profile
            _, report = self.execute(service)
            self.assertEqual(report["failure_kind"], "protocol")
            self.assertEqual(service.budgets, [])

    def test_task_not_protocol(self):
        for direct, final in [(False, "pump-B 72 OK"), (False, "pump-A 84 ALERT\n"),
                              (True, OBS["answer"])]:
            service = Observation()
            service.final = final
            if direct:
                service.calls = []
            _, report = self.execute(service)
            self.assertTrue(report["protocol_success"])
            self.assertFalse(report["task_success"])
            self.assertIsNone(report["failure_kind"])

    def test_limits_deadline_interrupt(self):
        for key, value in [("max_calls", 2), ("max_generated", 257), ("budget", 129),
                           ("wall_seconds", 121)]:
            with self.assertRaises(ValueError):
                native.validate_operation({**OBS, key: value})
        service = Observation()
        service.calls *= 2
        _, report = self.execute(service)
        self.assertEqual(report["failure_kind"], "limit")
        self.assertEqual(report["tool_calls"], 1)
        _, report = self.execute(op={**OBS, "max_generated": 5})
        self.assertEqual(report["failure_kind"], "limit")
        clock = [0]
        service = Observation()
        service.modify = lambda result: clock.__setitem__(0, 120)
        with patch.object(native.time, "monotonic", side_effect=lambda: clock[0]):
            _, report = self.execute(service)
        self.assertEqual(report["failure_kind"], "limit")
        self.assertEqual(report["generated_tokens"], 5)
        self.assertEqual(report["tool_calls"], 0)
        self.assertEqual(service.messages, [])
        service = Observation()
        service.absolute_deadline = 1e20
        with patch.object(service, "generate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                native.run_loop(service, 7, OBS, set().add)
        self.assertEqual(service.absolute_deadline, 1e20)

    def test_corpus_drift(self):
        with patch.object(Path, "open") as opening:
            opening.return_value.__enter__.return_value.read.return_value = b'{"version":1}'
            service, report = self.execute()
        self.assertEqual(report["failure_kind"], "protocol")
        self.assertEqual(service.budgets, [])
        self.assertEqual(service.messages, [])

    def test_loader_and_mixed_no_launch(self):
        scenario = load_scenario(SCENARIO)
        for change in ({"max_calls": 2}, {"answer": "other"}, {"extra": True}):
            invalid = copy.deepcopy(scenario)
            invalid["operations"][2].update(change)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "scenario.json"
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_scenario(path)
        for name in ("calculator-v13.json", "investigation-v13.json"):
            mixed = copy.deepcopy(scenario)
            mixed["operations"].append(load_scenario(SCENARIO.parent / name)["operations"][2])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "scenario.json"
                path.write_text(json.dumps(mixed))
                argv = ["real_workload", "--server", "unused", "--scenario", str(path),
                        "--output", str(Path(directory) / "out"),
                        "--native-model", "unused", "--native-tokenizer", "unused"]
                with patch.object(sys, "argv", argv), patch.object(real_workload, "Service") as launch:
                    self.assertEqual(real_workload.main(), 1)
                    launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
