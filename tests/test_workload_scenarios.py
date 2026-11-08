"""Model-free scenario admission and deadline regressions; python -O safe."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from real_workload import load_scenario, run

SCENARIO = Path(__file__).resolve().parents[1] / "tools/scenarios/checkpoint-repeat-v2.json"


class Scenarios(unittest.TestCase):
    def load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(json.dumps(value))
            return load_scenario(path)

    def test_schema(self):
        value = load_scenario(SCENARIO)
        self.assertEqual(value["repetitions"], 2)
        for key, bad in (("repetitions", True), ("wall_seconds", 121),
                         ("prefix_reuse", "legacy"), ("mode", "controlled_protocol_only")):
            clone = copy.deepcopy(value)
            clone[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(clone)

    def test_alias_and_collisions(self):
        value = load_scenario(SCENARIO)
        for operation in ({"op": "reset"}, {"op": "checkpoint", "id": "other", "checkpoint": "new"},
                          {"op": "generate", "id": "save", "budget": 1}):
            clone = copy.deepcopy(value)
            clone["operations"].insert(5, operation)
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                self.load(clone)

    def test_v1(self):
        self.assertEqual(self.load({"version": 1, "mode": "controlled_protocol_only",
                                   "operations": []})["version"], 1)

    def test_duplicate_ordinary_request_id(self):
        from unittest.mock import Mock
        # The version-one direct runner still owns duplicate-ID rejection.
        service = Mock(native=False)
        service.generate.return_value = {"event": "completed"}
        scenario = {"mode": "controlled_protocol_only", "operations": [
            {"op": "generate", "id": "same", "budget": 1},
            {"op": "generate", "id": "same", "budget": 1}]}
        with self.assertRaisesRegex(ValueError, "duplicate request id"):
            run(service, scenario)
        service.generate.assert_called_once()

    def test_interrupt_deadline_restored(self):
        class Interrupted:
            native = True
            info = dict(prefix_reuse="on", checkpoints=True, context_capacity=256)
            absolute_deadline = 1e20
            def bounded_timeout(self, seconds):
                raise KeyboardInterrupt()
            def record(self, *args, **kwargs):
                pass
        service = Interrupted()
        with self.assertRaises(KeyboardInterrupt):
            run(service, load_scenario(SCENARIO))
        self.assertEqual(service.absolute_deadline, 1e20)
        self.assertEqual(service.scenario_report["completed_repetitions"], 0)



class ExtendedScenarios(Scenarios):
    def minimal(self, operations=None, repetitions=2):
        return dict(version=2, mode='free_greedy', prefix_reuse='on',
                    repetitions=repetitions, wall_seconds=10,
                    operations=[{'op': 'create'}, *(operations or []), {'op': 'close'}])

    def test_root_and_operation_types(self):
        for root in (None, [], 2, 'scenario', True):
            with self.subTest(root=root), self.assertRaises(ValueError):
                self.load(root)
        for op in ({'op': []}, {'op': {}}, {'op': None}, {'op': 'unknown'},
                   {'op': 'generate', 'budget': 1}, {'op': 'checkpoint', 'checkpoint': 'a'}):
            with self.subTest(op=op), self.assertRaises(ValueError):
                self.load(self.minimal([op]))

    def test_bounds_and_types(self):
        for key, values in {'version': [True, 2.0, 3], 'repetitions': [0, 17, True, 1.0],
                            'wall_seconds': [0, 121, True, '10'],
                            'operations': [None, {}, [], [{'op': 'create'}]]}.items():
            for bad in values:
                value = self.minimal()
                value[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    self.load(value)
        self.assertEqual(self.load(self.minimal(repetitions=16))['repetitions'], 16)
        self.load(self.minimal([{'op': 'reset'}] * 254))
        with self.assertRaises(ValueError):
            self.load(self.minimal([{'op': 'reset'}] * 255))

    def test_alias_replacement_and_reset(self):
        save = lambda identity, alias: dict(op='checkpoint', id=identity, checkpoint=alias)
        restore = lambda alias: dict(op='restore', id='restore', checkpoint=alias)
        self.load(self.minimal([save('a', 'first'), save('b', 'second'), restore('second')]))
        for middle in ([save('b', 'second')], [{'op': 'reset'}]):
            with self.subTest(middle=middle), self.assertRaises(ValueError):
                self.load(self.minimal([save('a', 'first'), *middle, restore('first')]))

    def test_history_cap_includes_repetitions(self):
        operations = [dict(op='generate', id=f'g{i}', budget=1) for i in range(16)]
        self.load(self.minimal(operations, 16))
        with self.assertRaises(ValueError):
            self.load(self.minimal(operations + [dict(op='generate', id='extra', budget=1)], 16))

    def test_tool_namespace_and_control_policy(self):
        tool = dict(op='native_tool_loop', id='tool', budget=16, max_calls=2,
                    max_generated=64, wall_seconds=10, answer=43)
        self.load(self.minimal([tool, dict(op='generate', id='ordinary', budget=1)]))
        for identity in ('tool-0', 'tool-1', 'tool-2'):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                self.load(self.minimal([tool, dict(op='generate', id=identity, budget=1)]))
        save = dict(op='checkpoint', id='save', checkpoint='base')
        with self.assertRaises(ValueError):
            self.load(self.minimal([tool, save]))
        value = self.minimal([save])
        value['prefix_reuse'] = 'legacy'
        with self.assertRaises(ValueError):
            self.load(value)

    def test_cli_refuses_before_launch(self):
        from unittest.mock import patch
        import real_workload
        for extra in ([], ['--native-prefix-reuse', 'on'],
                      ['--native-model', 'unused', '--native-tokenizer', 'unused',
                       '--native-prefix-reuse', 'off']):
            with self.subTest(extra=extra), patch.object(sys, 'argv',
                    ['workload', '--server', 'unused', '--scenario', str(SCENARIO),
                     '--output', 'unused', *extra]), patch.object(real_workload, 'Service') as launch:
                self.assertNotEqual(real_workload.main(), 0)
                launch.assert_not_called()

    def test_aggregation_and_deadlines(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import real_workload
        good = dict(protocol_success=True, task_checks=[{'success': True}])
        bad_quality = dict(protocol_success=True, task_checks=[{'success': False}])
        for outcomes, complete, quality in (([bad_quality, good], 2, False),
                                            ([good, RuntimeError('broken')], 1, None),
                                            ([dict(protocol_success=False)], 0, None),
                                            ([TimeoutError('limit')], 0, None)):
            service = SimpleNamespace(native=True, absolute_deadline=101,
                                      info=dict(prefix_reuse='on', checkpoints=True, context_capacity=256),
                                      record=lambda *a, **k: None,
                                      operation_report={'task_checks': [{'success': True}]})
            with self.subTest(outcomes=outcomes), patch.object(real_workload, 'run_once', side_effect=outcomes) as once, patch.object(real_workload.time, 'monotonic', return_value=100):
                result = run(service, self.minimal())
                self.assertEqual(result['completed_repetitions'], complete)
                self.assertIs(result['task_success'], quality)
                self.assertEqual(once.call_count, len(outcomes))
                self.assertEqual(service.absolute_deadline, 101)
                if complete != 2:
                    self.assertFalse(result['protocol_success'])
                    self.assertIn(result['failure_kind'], ('protocol', 'runtime', 'limit'))
                if isinstance(outcomes[-1], Exception):
                    self.assertEqual(result['repetitions'][-1]['partial_result'], service.operation_report)


if __name__ == "__main__":
    unittest.main()
