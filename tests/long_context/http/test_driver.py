import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from driver import Driver, Journal, PREP, SocketTransport, frozen, validate_state, validate_work


def work(calls):
    w = {f'{a}_{s}': 0 for a in ('prefill', 'append', 'decode')
         for s in ('attempted', 'committed', 'calls')}
    w.update({k: 0 for k in ('pending_attempted', 'pending_committed',
                            'recompute_attempted', 'recompute_committed', 'logit_restores')})
    for c in calls:
        a = c['api']
        if a == 'restore':
            w['logit_restores'] += 1
        else:
            w[a + '_calls'] += 1
            for s in ('attempted', 'committed'):
                w[a + '_' + s] += c[s]
        for key in ('pending_attempted', 'pending_committed'):
            w[key] += c.get(key, 0)
    return {'metrics_version': 3, 'work_version': 2, 'work': w}


class Mock:
    """Tiny protocol stub, no numerical/runtime correctness claim."""
    def __init__(self, branch, block=128):
        self.branch, self.block = branch, block
        self.ids, self.saved, self.history, self.trace = [], [], {}, {}
        self.phase, self.cursor, self.tail = 'ready', 0, False
        self.texts = frozen()
        self.points = []

    def snapshot(self):
        return dict(session_id=1, state=self.phase, token_ids=list(self.ids),
                    logical_tokens=len(self.ids), committed_tokens=self.cursor,
                    pending_tokens=int(self.ids != [] and self.ids[-1] in (1001, 2) and len(self.ids) > self.cursor),
                    remaining=4352-len(self.ids))

    def finish(self):
        calls = [c for row in self.trace.values() for c in row if c['api'] != 'restore']
        return dict(unattributed_calls=0, model_calls=len(calls),
                    model_tokens=sum(c['attempted'] for c in calls))

    def calls(self, identity):
        return self.trace[identity]

    def frame(self, text, initial):
        return json.loads((PREP / 'negative.json').read_text())['ids']

    def decode(self, ids):
        return {'text': '' if self.branch in ('utf8', 'eos') else 'x' * len(ids),
                'incomplete_utf8': self.branch == 'utf8'}

    def memory_point(self, phase):
        self.points.append(phase)
        return dict(phase=phase, method='task_info_TASK_VM_INFO_phys_footprint',
                    monotonic_ns=1, physical_bytes=10, named_buffers={'stub': 1},
                    guard_diagnostics={'stub_only': True})

    def materialize(self):
        calls = []
        while self.cursor < len(self.ids):
            n = min(self.block, len(self.ids)-self.cursor)
            calls.append(dict(api='prefill' if self.cursor == 0 else 'append',
                              attempted=n, committed=n, start=self.cursor,
                              end=self.cursor+n, finite=True))
            self.cursor += n
        return calls

    def enrich(self, r, before, operation):
        r.update(session_id=1, session_valid=True, recovery='none', operation=operation,
                 cache_restored=operation == 'restore')
        r.setdefault('token_ids', [])
        r.setdefault('generated_tokens', 0)
        r.setdefault('text', '')
        for key in ('state', 'logical_tokens', 'committed_tokens', 'pending_tokens'):
            r[key] = self.snapshot()[key]
            r['before_' + key] = before[key]

    def request(self, method, path, body=None, declared=None):
        before = self.snapshot()
        if path == '/info':
            return 200, dict(context_capacity=4352, prefill_chunk_size=self.block,
                prefix_reuse='on', execution_strategy='resident-fp32', prefill='custom',
                model_loaded=True, runtime_ready=True, model_load_count=1,
                model_sha256='728f1826cd0e38191ca7b1379e81f78cf0555c6ffd95882aabd2404632346f86',
                tokenizer_sha256=json.loads((PREP/'texts.json').read_text())['tokenizer_sha256'])
        if path.startswith('/requests/'):
            return 200, self.history[path.split('/')[-1]]
        if method == 'GET' or path == '/sessions':
            return 200, self.snapshot()
        if path.endswith('/reset'):
            self.ids, self.cursor, self.phase, self.tail = [], 0, 'ready', False
        elif path.endswith('/messages'):
            if declared is not None or len(body['content'].encode()) > 32768:
                return 400, {'error': 'invalid_json_or_fields'}
            found = next((r for r in self.texts.values() if r['text'] == body['content']), None)
            if found is None:
                return 422, {'error': 'framing_or_context_overflow'}
            self.ids += ([1] if not self.ids and not found['initial'] else []) + found['ids']
        elif path.endswith('/generate'):
            if self.phase == 'ready' and self.ids[-1:] == [2]:
                return 409, {'error': 'user_input_required'}
            calls = self.materialize()
            selected = [2] if self.branch == 'eos' else [1001] * body['budget']
            for i in range(len(selected)):
                if i:
                    calls += self.materialize()
                self.ids.append(selected[i])
            self.phase = 'ready' if self.branch == 'eos' else 'partial'
            self.tail = self.branch == 'utf8'
            r = dict(id=body['id'], status='completed', token_ids=selected,
                     generated_tokens=len(selected), stop_reason='eos' if self.branch == 'eos' else 'budget',
                     text=self.decode(selected)['text'], metrics=work(calls))
            self.enrich(r, before, 'generate')
            self.trace[body['id']], self.history[body['id']] = calls, r
            terminal = {k: v for k, v in r.items() if k not in ('status', 'token_ids', 'operation', 'checkpoint_id', 'cache_restored')}
            return 200, [dict(terminal, event='started', text=''),
                         dict(terminal, event='text_delta'), dict(terminal, event='completed', text='')]
        elif path.endswith('/checkpoint') or path.endswith('/restore'):
            if self.tail:
                return 409, {'error': 'checkpoint_boundary_required'}
            if path.endswith('/checkpoint'):
                calls = self.materialize()
                if calls:
                    calls[0].update(pending_attempted=before['pending_tokens'], pending_committed=before['pending_tokens'])
                self.saved = list(self.ids)
                self.saved_phase = self.phase
            else:
                self.ids, self.phase = list(self.saved), self.saved_phase
                self.cursor = len(self.ids)
                calls = [dict(api='restore', attempted=0, committed=0)]
            r = dict(id=body['id'], status='completed', checkpoint_id='1', metrics=work(calls))
            self.enrich(r, before, 'checkpoint' if path.endswith('/checkpoint') else 'restore')
            self.trace[body['id']], self.history[body['id']] = calls, r
            return 202, self.snapshot()
        return 200, self.snapshot()


class Tests(unittest.TestCase):
    def test_branches(self):
        for block in (128, 256):
            for branch in ('partial', 'utf8', 'eos'):
                with self.subTest(block=block, branch=branch), tempfile.TemporaryDirectory() as td:
                    mock = Mock(branch, block)
                    journal = Journal(Path(td)/'trace.jsonl')
                    try:
                        d = Driver(mock, journal, mock, block, 'resident-fp32', 'custom')
                        d.run()
                        self.assertLess(journal.size, 1024*1024)
                        self.assertIn('document-after-ingestion', mock.points)
                        self.assertEqual('document-after-restore' in mock.points, branch != 'utf8')
                    finally:
                        journal.close()

    def test_final_state_drift(self):
        for field, value in [('session_id', 2), ('pending_tokens', 0),
                             ('committed_tokens', 0), ('state', 'ready'),
                             ('session_valid', False), ('recovery', 'reset'),
                             ('generated_tokens', 99), ('id', 'other')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                mock = Mock('partial')
                original = mock.request
                def drift(method, path, body=None, declared=None):
                    status, reply = original(method, path, body, declared)
                    if path.startswith('/requests/'):
                        reply = dict(reply, **{field: value})
                    return status, reply
                mock.request = drift
                j = Journal(Path(td)/'trace')
                try:
                    d = Driver(mock, j, mock, 128, 'resident-fp32', 'custom')
                    d.sid = 1
                    d.message('short', initial=True)
                    with self.assertRaises(ValueError):
                        d.generate('drift', 1)
                finally:
                    j.close()

    def test_negative_receipts(self):
        c = [dict(api='decode', attempted=1, committed=1)]
        r = {'metrics': work(c)}
        validate_work(r, c)
        for field, value in [('decode_calls', True), ('decode_committed', 2), ('pending_committed', 1)]:
            bad = copy.deepcopy(r)
            bad['metrics']['work'][field] = value
            with self.assertRaises(ValueError):
                validate_work(bad, c)
        r['metrics']['metrics_version'] = 2
        with self.assertRaises(ValueError):
            validate_work(r, c)

    def test_ledger_and_artifact_stop(self):
        m = Mock('partial')
        m.ids = [1000]
        with self.assertRaises(ValueError):
            validate_state(m.snapshot(), [1001])
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td)/'x')
            try:
                j.size = 64*1024*1024
                with self.assertRaises(ValueError):
                    j.record('x', {})
                with self.assertRaises(FileExistsError):
                    Journal(Path(td)/'x')
            finally:
                j.close()

    def test_client_profile(self):
        engine = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('d_client', engine/'tools/real_workload.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scenario = dict(version=1, mode='free_greedy', operations=[dict(op='create'),
                        dict(op='message', content=' '*32768), dict(op='generate', id='d', budget=4)])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'scenario.json'
            p.write_text(json.dumps(scenario))
            self.assertEqual(module.load_scenario(p, 4352), scenario)
            with self.assertRaises(ValueError):
                module.load_scenario(p, 256)
            scenario['operations'][1]['content'] += ' '
            p.write_text(json.dumps(scenario))
            with self.assertRaises(ValueError):
                module.load_scenario(p, 4352)

    def test_stops(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            mock = Mock('partial')
            j = Journal(Path(td)/'failures')
            try:
                d = Driver(mock, j, mock, 128, 'resident-fp32', 'custom')
                d.sid = 1
                mock.request = lambda *args: (200, mock.snapshot()) if args[1].startswith('/sessions') and args[0] == 'GET' else ((200, {'status': 'running'}) if args[0] == 'GET' else (202, {}))
                with patch('driver.time.sleep'):
                    with self.assertRaises(TimeoutError):
                        d.control('checkpoint', 'timeout')
                self.assertEqual(d.polls, 64)
                d.requests = 512
                with self.assertRaises(ValueError):
                    d.request('GET', '/info')
                mock.memory_point = lambda phase: dict(phase=phase, method='RSS')
                with self.assertRaises(ValueError):
                    d.point('bad')
            finally:
                j.close()

    def test_transport_validation(self):
        for port in (True, 0, 65536):
            with self.assertRaises(ValueError):
                SocketTransport(port)
        self.assertEqual(SocketTransport(12345).port, 12345)  # no connection


if __name__ == '__main__':
    unittest.main()
