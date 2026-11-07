import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from driver import Driver, Journal
from test_driver import Mock
import replay


class Tests(unittest.TestCase):
    def test_replay_and_corruptions(self):
        for branch in ('partial', 'eos', 'utf8'):
            with self.subTest(branch=branch), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                mock = Mock(branch, 256)
                original_request = mock.request
                def request(method, path, body=None, declared=None):
                    status, value = original_request(method, path, body, declared)
                    if path == '/info':
                        value.update(execution_strategy='compact-fp8-arithmetic', prefill='mps')
                    return status, value
                mock.request = request
                original_calls = mock.calls
                def calls(identity):
                    for row in original_calls(identity):
                        row.update(request_id=identity, status=0, finite=True)
                    return original_calls(identity)
                mock.calls = calls
                point = mock.memory_point
                mock.memory_point = lambda phase: dict(point(phase), kind='memory')
                journal = Journal(root/'journal.jsonl')
                journal.record('startup_guard_observations', {})
                try:
                    Driver(mock, journal, mock, 256, 'compact-fp8-arithmetic', 'mps').run()
                finally:
                    journal.close()
                data = [json.loads(line) for line in journal.path.read_text().splitlines()]
                observations = []
                for row in data:
                    if row['kind'] == 'actual_calls':
                        observations.extend(row['value'])
                    elif row['kind'] == 'memory_point':
                        observations.append({k:v for k,v in row['value'].items() if k != 'phase'})
                for row in observations:
                    row.setdefault('kind', 'call')
                # Include kind in driver receipts, exactly as the real observer does.
                for row in data:
                    if row['kind'] == 'actual_calls':
                        for call in row['value']:
                            call['kind'] = 'call'
                def write(rows, obs):
                    (root/'reference-environment.json').write_text(json.dumps(replay.environment())+'\n')
                    (root/'journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    (root/'observation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in obs))
                class Tokenizer:
                    def __init__(self, path):
                        pass
                    def frame(self, text, initial):
                        return mock.frame(text, initial)
                    def decode(self, ids):
                        return mock.decode(ids)
                    def reset_decode(self):
                        pass
                with patch.object(replay, 'Tokenizer', Tokenizer):
                    write(data, observations)
                    self.assertEqual(replay.validate(root, 'unused')['status'], 'pass')
                    for mutate in ('finite', 'call', 'state', 'order', 'coverage'):
                        bad, obs = copy.deepcopy(data), copy.deepcopy(observations)
                        if mutate == 'finite':
                            next(r for r in obs if r['kind'] == 'call')['finite'] = False
                        elif mutate == 'call':
                            obs.pop(next(i for i,r in enumerate(obs) if r['kind'] == 'call'))
                        elif mutate == 'state':
                            next(r['value']['reply'] for r in bad if r['kind'] == 'http' and
                                 r['value']['method'] == 'GET' and r['value']['path'].startswith('/sessions/'))['token_ids'] = [999]
                        elif mutate == 'order':
                            bad[1], bad[2] = bad[2], bad[1]
                        else:
                            next(r for r in bad if r['kind'] == 'coverage')['value'] = {}
                        write(bad, obs)
                        with self.subTest(corruption=mutate), self.assertRaises((ValueError, KeyError)):
                            replay.validate(root, 'unused')


if __name__ == '__main__':
    unittest.main()
