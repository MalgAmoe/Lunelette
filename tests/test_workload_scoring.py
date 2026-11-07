"""Model-free client scoring and intermittent recorded-input regression tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import real_workload as workload


class FakeService:
    native = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.actions = []
        self.records = []

    def record(self, kind, **values):
        self.records.append((kind, values))

    def request(self, method, path, body):
        self.actions.append((method, path, body))
        return 200, {'session_id': 'resident-session'}

    def generate(self, session, request_id, budget, cancel):
        self.actions.append(('generate', session, request_id, budget, cancel))
        return next(self.responses)


def response(text='answer', **changes):
    return dict({'text': text, 'event': 'completed', 'stop_reason': 'eos',
                 'state': 'ready'}, **changes)


def scenario():
    return {'version': 1, 'mode': 'free_greedy', 'operations': [
        {'op': 'create'}, {'op': 'generate', 'id': 'g', 'budget': 32},
        {'op': 'score', 'id': 'g', 'text': 'answer'}, {'op': 'close'}]}


class ScoringTests(unittest.TestCase):
    def load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scenario.json'
            path.write_text(json.dumps(value))
            return workload.load_scenario(path)

    def test_exact_wrong_partial_and_terminal_conditions(self):
        for result, success in [(response(), True), (response('wrong'), False),
                                (response(stop_reason='budget', state='partial'), False),
                                (response(state='partial'), False),
                                (response(stop_reason='budget'), False)]:
            with self.subTest(result=result):
                service = FakeService([result])
                report = workload.run(service, self.load(scenario()))
                self.assertTrue(report['protocol_success'])
                self.assertEqual(report['task_success'], success)
                check = report['task_checks'][0]
                self.assertEqual(check['request_id'], 'g')
                self.assertEqual(check['expected_text'], 'answer')
                self.assertEqual(check['actual_text'], result['text'])
                self.assertEqual(check['success'], success)
                self.assertEqual(len(service.actions), 3)
                self.assertEqual(sum(k == 'task_check' for k, _ in service.records), 1)

    def test_existing_expect_and_error_handling(self):
        value = scenario()
        value['operations'][1]['expect'] = {'text': 'expected'}
        with self.assertRaises(ValueError):
            workload.run(FakeService([response()]), self.load(value))
        with self.assertRaises(ValueError):
            workload.run(FakeService([response(event='error')]), self.load(scenario()))

    def test_no_task_and_continue(self):
        value = scenario()
        del value['operations'][2]
        report = workload.run(FakeService([response()]), self.load(value))
        self.assertIsNone(report['task_success'])
        self.assertEqual(report['task_checks'], [])
        value = scenario()
        value['operations'][1]['op'] = 'continue'
        self.assertTrue(workload.run(FakeService([response()]), self.load(value))['task_success'])

    def test_invalid_scores(self):
        variants = []
        for field, values in [('text', [None, 4, [], {}, 'é' * 2049]),
                              ('id', [None, 4, [], {}, '', 'other', 'x' * 65])]:
            for item in values:
                value = scenario()
                value['operations'][2][field] = item
                variants.append(value)
        for field in ['id', 'text']:
            value = scenario()
            del value['operations'][2][field]
            variants.append(value)
        for field in ['expect', 'extra']:
            value = scenario()
            value['operations'][2][field] = {}
            variants.append(value)
        value = scenario()
        value['mode'] = 'controlled_protocol_only'
        variants.append(value)
        for predecessor in [None, {'op': 'pause', 'seconds': 0},
                            {'op': 'cancel', 'id': 'g', 'budget': 32},
                            {'op': 'reset'}]:
            value = scenario()
            if predecessor is None:
                value['operations'] = value['operations'][2:]
            else:
                value['operations'].insert(2, predecessor)
            variants.append(value)
        value = scenario()
        value['operations'].insert(3, copy.deepcopy(value['operations'][2]))
        variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(value)
        value = scenario()
        value['operations'][2]['text'] = 'é' * 2048
        self.load(value)

    def test_multiple_scores_aggregate_without_retry(self):
        value = scenario()
        value['operations'][3:3] = [
            {'op': 'continue', 'id': 'g2', 'budget': 32},
            {'op': 'score', 'id': 'g2', 'text': 'answer'}]
        service = FakeService([response('wrong'), response()])
        report = workload.run(service, self.load(value))
        self.assertTrue(report['protocol_success'])
        self.assertFalse(report['task_success'])
        self.assertEqual([c['success'] for c in report['task_checks']], [False, True])
        self.assertEqual(sum(a[0] == 'generate' for a in service.actions), 2)

    def test_invalid_score_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            value = scenario()
            value['operations'][2]['id'] = 'dangling'
            path.write_text(json.dumps(value))
            with patch.object(sys, 'argv', ['real_workload', '--server', 'unused',
                                           '--scenario', str(path), '--output', 'unused']), \
                    patch.object(workload, 'Service') as launch, \
                    patch.object(sys, 'stderr'):
                self.assertEqual(workload.main(), 1)
            launch.assert_not_called()

    def test_tool_report_aggregation(self):
        for score_ok in [True, False]:
            for tool_ok in [True, False]:
                value = scenario()
                value['operations'].insert(3, {'op': 'native_tool_loop', 'id': 'tool'})
                service = FakeService([response('answer' if score_ok else 'wrong')])
                with patch.object(workload.native_workload_tools, 'run_loop',
                                  return_value={'protocol_success': True, 'task_success': tool_ok}):
                    report = workload.run(service, value)
                self.assertEqual(report['task_success'], score_ok and tool_ok)

    def test_recorded_scenario_resident_flow(self):
        path = Path(__file__).resolve().parents[1] / 'tools/scenarios/recorded-sensors-v1.json'
        value = workload.load_scenario(path)
        expected = ['pump-A 84 ALERT', 'pump-B 75 OK', 'pump-B 89 ALERT']
        service = FakeService([response(text) for text in expected])
        with patch.object(workload.time, 'sleep') as sleep:
            report = workload.run(service, value)
        sleep.assert_called_once_with(1)
        self.assertTrue(report['task_success'])
        self.assertEqual(len(report['task_checks']), 3)
        self.assertEqual([a[:2] for a in service.actions], [
            ('POST', '/sessions'), ('POST', '/sessions/resident-session/messages'),
            ('generate', 'resident-session'), ('POST', '/sessions/resident-session/reset'),
            ('POST', '/sessions/resident-session/messages'), ('generate', 'resident-session'),
            ('POST', '/sessions/resident-session/reset'),
            ('POST', '/sessions/resident-session/messages'), ('generate', 'resident-session'),
            ('DELETE', '/sessions/resident-session')])
        self.assertEqual(sum(op.get('budget', 0) for op in value['operations']), 96)
        for action in service.actions:
            if action[0] == 'generate':
                self.assertEqual(action[3:], (32, False))
            elif action[1].endswith('/messages'):
                self.assertEqual(action[2]['role'], 'user')
                self.assertIn('Synthetic recorded readings', action[2]['content'])
        pauses = [i for i, op in enumerate(value['operations']) if op['op'] == 'pause']
        self.assertEqual(pauses, [9])


if __name__ == '__main__':
    unittest.main()
