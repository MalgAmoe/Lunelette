"""Replay admission and fake native orchestration, safe under -O."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import real_workload as w

SAMPLE = Path(__file__).resolve().parents[1] / 'tools/scenarios/replay-v2.json'


class Replay(unittest.TestCase):
    def test_admission(self):
        original = w.load_scenario(SAMPLE)
        self.assertEqual(original['mode'], 'controlled_replay')
        for tokens in ([True], [3], [999], [131072], [2, 1000], [], ['1000'], [1000]*17):
            value = copy.deepcopy(original)
            value['operations'][3]['tokens'] = tokens
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                w.validate_scenario(value)
        for operation in ({'op': 'generate', 'id': 'g', 'budget': 1},
                          {'op': 'score', 'id': 'g', 'text': 'x'},
                          {'op': 'replay', 'id': True, 'tokens': [1000]},
                          {'op': 'replay', 'id': 'save', 'tokens': [1000]}):
            value = copy.deepcopy(original)
            value['operations'].insert(3, operation)
            with self.assertRaises(ValueError):
                w.validate_scenario(value)
        value = copy.deepcopy(original)
        value['repetitions'] = 5
        with self.assertRaises(ValueError):
            w.validate_scenario(value)
        value = copy.deepcopy(original)
        value['mode'] = 'free_greedy'
        with self.assertRaises(ValueError):
            w.validate_scenario(value)
        with self.assertRaises(ValueError):
            w.validate_scenario(dict(version=1, mode='controlled_replay', operations=[]))

    def test_forced_output_not_quality(self):
        class Fake:
            native = True
            evaluation_mode = 'controlled_replay'
            info = dict(context_capacity=256, prefix_reuse='on', checkpoints=True)
            def record(self, *args, **kwargs):
                pass
            def bounded_timeout(self, seconds):
                return seconds
            def request(self, *args):
                return 200, {'session_id': 1}
            def control(self, *args):
                return {'terminal': {}}
            def generate(self, session, identity, budget, replay_tokens=None):
                return dict(event='completed', token_ids=replay_tokens,
                            before_prefix_ids=[1, 1000], text='forced nonsense')
        result = w.run(Fake(), w.load_scenario(SAMPLE))
        self.assertTrue(result['protocol_success'])
        self.assertIsNone(result['task_success'])
        self.assertFalse(result['performance_evidence'])
        steps = [s for e in result['repetitions'] for s in e['result']['replay_steps']]
        self.assertEqual(sum(len(s['actual_ids']) for s in steps), 8)
        self.assertEqual(steps[0]['id'], 'r1-first')

    def exercise_wire(self, mutate=None):
        tokens = [53156, 2]
        def ledger(ids):
            return dict(session_id=1, state='ready', token_ids=ids,
                        logical_tokens=len(ids), committed_tokens=len(ids)-1,
                        pending_tokens=1, remaining=256-len(ids))
        before = ledger([1, 1000])
        terminal = dict(id='r1-first', session_id=1, event='completed', text='',
                        generated_tokens=2, state='ready', logical_tokens=4,
                        committed_tokens=3, pending_tokens=1, session_valid=True,
                        recovery='none', stop_reason='eos',
                        evaluation_mode='controlled_replay', replay_tokens=tokens, metrics={})
        for key in ('state', 'logical_tokens', 'committed_tokens', 'pending_tokens'):
            terminal['before_' + key] = before[key]
        retained = dict(terminal, status='completed', operation='generate', token_ids=tokens)
        if mutate:
            mutate(terminal, retained)
        class Wire:
            status = 200
            sock = None
            def __init__(self):
                self.data = (json.dumps(dict(terminal, event='started', generated_tokens=0, token_ids=[])) + '\n'
                             + json.dumps(terminal) + '\n').encode()
            def request(self, method, path, body):
                self.payload = json.loads(body)
            def getresponse(self):
                return self
            def getheader(self, name):
                return 'application/x-ndjson'
            def read1(self, size):
                data, self.data = self.data, b''
                return data
            def close(self):
                pass
        wire = Wire()
        service = object.__new__(w.Service)
        service.native = True
        service.evaluation_mode = 'controlled_replay'
        service.request_seconds = 120
        service.info = {'context_capacity': 256}
        service.connect = lambda: wire
        service.record = lambda *args, **kwargs: None
        responses = iter([(200, before), (200, retained), (200, ledger([1, 1000] + tokens))])
        service.request = lambda *args: next(responses)
        result = service.generate(1, 'r1-first', 99, replay_tokens=tokens)
        self.assertEqual(wire.payload, dict(id='r1-first', budget=2, replay_tokens='53156,2'))
        self.assertEqual(result['before_prefix_ids'], [1, 1000])
        self.assertTrue(result['diagnostic_forced_output'])

    def test_native_wire(self):
        self.exercise_wire()

    def test_invalid_native_metadata(self):
        for field, bad in [('session_id', True), ('state', 'partial'),
                           ('pending_tokens', True), ('stop_reason', 'budget'),
                           ('replay_tokens', [53156.0, 2]), ('before_logical_tokens', 1)]:
            with self.subTest(field=field), self.assertRaises((RuntimeError, ValueError)):
                self.exercise_wire(lambda terminal, retained: terminal.update({field: bad}))
        with self.assertRaises(RuntimeError):
            self.exercise_wire(lambda terminal, retained: retained.update(id='other'))

    def test_owned_startup_arguments(self):
        with self.assertRaises(ValueError):
            w.Service('unused', 'unused', native_args=['--tools', 'calculator-v13'], evaluation_mode='controlled_replay')
        with self.assertRaises(ValueError):
            w.Service('unused', 'unused', native_args=['--replay-logits', '/tmp/x'])


if __name__ == '__main__':
    unittest.main()
