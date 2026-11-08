"""Retired application JSON must fail before launch or service mutation, even -O."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import real_workload as workload
import workload_scenarios


class Retirement(unittest.TestCase):
    def scenarios(self):
        # Include earlier valid operations: rejection must precede even create.
        for mode in ("controlled_protocol_only", "free_greedy"):
            yield {"version": 1, "mode": mode, "operations": [
                {"op": "create"}, {"op": "tool_loop"}, {"op": "close"}]}
        for mode in ("free_greedy", "controlled_replay"):
            yield {"version": 2, "mode": mode, "prefix_reuse": "on",
                   "repetitions": 1, "wall_seconds": 10, "operations": [
                       {"op": "create"}, {"op": "tool_loop"}, {"op": "close"}]}

    def test_load_and_direct_admission(self):
        for scenario in self.scenarios():
            original = copy.deepcopy(scenario)
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "retired.json"
                path.write_text(json.dumps(scenario))
                with self.assertRaisesRegex(ValueError, "tool_loop.*retired"):
                    workload.load_scenario(path)
                with self.assertRaisesRegex(ValueError, "tool_loop.*retired"):
                    workload.validate_scenario(scenario)
                if scenario["version"] == 2:
                    with self.assertRaisesRegex(ValueError, "tool_loop.*retired"):
                        workload_scenarios.validate(scenario)
                for execute in (workload.run, workload.run_once):
                    service = Mock()
                    before = vars(service).copy()
                    with self.assertRaisesRegex(ValueError, "tool_loop.*retired"):
                        execute(service, scenario)
                    self.assertEqual(vars(service), before)
                    self.assertEqual(service.mock_calls, [])
                self.assertEqual(scenario, original)

    def test_cli_refuses_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retired.json"
            path.write_text(json.dumps(next(self.scenarios())))
            with patch.object(sys, "argv", ["workload", "--server", "unused",
                    "--scenario", str(path), "--output", str(Path(directory) / "out")]), \
                    patch.object(workload, "Service") as launch:
                self.assertNotEqual(workload.main(), 0)
                launch.assert_not_called()

    def test_native_scenarios_remain_explicit(self):
        root = Path(__file__).resolve().parents[1] / "tools/scenarios"
        for name, operation in (("calculator-v13.json", "native_tool_loop"),
                                ("investigation-v13.json", "native_investigation_loop"),
                                ("observation-v13.json", "native_observation_loop")):
            scenario = workload.load_scenario(root / name)
            self.assertIn(operation, [op["op"] for op in scenario["operations"]])


if __name__ == "__main__":
    unittest.main()
