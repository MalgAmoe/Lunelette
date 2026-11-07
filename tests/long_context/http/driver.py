"""D production-socket schedule. Model validation NOT_RUN."""
import hashlib
import http.client
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
PREP = HERE / 'fixtures'
CAP = 64 * 1024 * 1024
PRODUCTION_LAUNCH_ENABLED = True  # Requires a reviewed source change, not an environment bypass.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def verify_pins():
    # Runtime source/product receipts are a separate launch gate. These immutable
    # pins preserve the original schedule inputs, not a historical engine checkout.
    pins = json.loads((HERE / 'original-sha256.json').read_text())
    for name, expected in pins.items():
        if name.startswith('fixtures/'):
            require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected,
                    'fixture pin mismatch: ' + name)


def frozen():
    data = json.loads((PREP / 'texts.json').read_text())
    rows = {r['name']: r for r in data['texts']}
    for name, row in rows.items():
        raw = (PREP / (name + '.txt')).read_bytes()
        require(len(raw) == row['bytes'], 'text bytes')
        require(hashlib.sha256(raw).hexdigest() == row['text_sha256'], 'text hash')
        require(len(row['ids']) == row['count'] and all(integer(i) and i < 131072
                for i in row['ids']), 'frozen IDs')
        row['text'] = raw.decode('utf-8')
    require(rows['spaces']['count'] == 515 and rows['document']['count'] == 4083,
            'frozen count changed')
    return rows


class Journal:
    """One exclusive bounded artifact; no vectors, weights, or hidden temp outputs."""
    def __init__(self, path):
        self.path = Path(path)
        require(sum(p.stat().st_size for p in self.path.parent.rglob('*') if p.is_file()) < CAP,
                'owned directory already at 64MiB cap')
        self.file = self.path.open('xb')
        self.size = 0

    def record(self, kind, value):
        raw = (json.dumps({'kind': kind, 'value': value}, allow_nan=False) + '\n').encode()
        require(self.size + len(raw) <= 2 * 1024 * 1024, 'journal 2MiB cap')
        owned = sum(p.stat().st_size for p in self.path.parent.rglob('*') if p.is_file())
        require(max(self.size, owned) + len(raw) <= CAP, '64MiB artifact stop')
        self.file.write(raw)
        self.file.flush()
        self.size += len(raw)

    def close(self):
        self.file.close()


class SocketTransport:
    """Connect only to a supervisor-owned localhost server; never start a process."""
    def __init__(self, port):
        require(type(port) is int and 0 < port < 65536, 'port')
        require(PRODUCTION_LAUNCH_ENABLED, 'CLOSED: production HTTP execution not authorized')
        self.port = port

    def request(self, method, path, body=None, declared=None):
        if method != 'POST' and declared is None:
            declared = 0
        raw = body if isinstance(body, bytes) else json.dumps(body or {}).encode()
        # The central owner enforces the total 1200s wall deadline, including
        # ingestion before the first token. Do not add an unreviewed 30s limit.
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=1200)
        try:
            connection.request(method, path, raw if method == 'POST' else None,
                               {'Content-Length': str(len(raw) if declared is None else declared)})
            response = connection.getresponse()
            payload = response.read(262145)
            require(len(payload) <= 262144, 'response cap')
            if path.endswith('/generate') and response.status == 200:
                value = [json.loads(line) for line in payload.splitlines()]
            else:
                value = json.loads(payload)
            return response.status, value
        finally:
            connection.close()


def validate_state(state, ids):
    require(isinstance(state['token_ids'], list) and
            all(integer(i) and i < 131072 for i in state['token_ids']) and
            state['token_ids'] == ids, 'authoritative ledger differs')
    for key in ('logical_tokens', 'committed_tokens', 'pending_tokens', 'remaining'):
        require(integer(state[key]), 'state integer')
    require(state['logical_tokens'] == len(ids) <= 4352, 'logical count')
    require(state['remaining'] == 4352 - len(ids), 'remaining')
    require(state['committed_tokens'] <= len(ids) and state['pending_tokens'] <= 1,
            'cursor/pending')
    require(state['state'] in ('ready', 'partial'), 'invalid session')


def validate_work(result, calls):
    m = result['metrics']
    require(type(m['metrics_version']) is int and m['metrics_version'] == 3 and
            type(m['work_version']) is int and m['work_version'] == 2, 'schema3/work2')
    w = m['work']
    names = [f'{api}_{suffix}' for api in ('prefill', 'append', 'decode')
             for suffix in ('attempted', 'committed', 'calls')]
    names += ['pending_attempted', 'pending_committed', 'recompute_attempted',
              'recompute_committed', 'logit_restores']
    require(set(w) == set(names) and all(integer(w[k]) for k in names), 'work fields')
    for api in ('prefill', 'append', 'decode'):
        observed = [c for c in calls if c['api'] == api]
        require(w[api + '_calls'] == len(observed), 'actual call count')
        for suffix in ('attempted', 'committed'):
            require(w[api + '_' + suffix] == sum(c[suffix] for c in observed),
                    'actual token sum')
        require(w[api + '_committed'] <= w[api + '_attempted'], 'commit subset')
    total = {s: sum(w[a + '_' + s] for a in ('prefill', 'append', 'decode'))
             for s in ('attempted', 'committed')}
    for subset in ('pending', 'recompute'):
        require(w[subset + '_committed'] <= w[subset + '_attempted'], 'subset order')
        for suffix in total:
            require(w[subset + '_' + suffix] == sum(c.get(subset + '_' + suffix, 0)
                    for c in calls) <= total[suffix], 'subset receipt')
    require(w['logit_restores'] == sum(c['api'] == 'restore' for c in calls), 'restore receipt')
    require(all(c['api'] in ('prefill', 'append', 'decode', 'restore') and
                integer(c['attempted']) and integer(c['committed']) and
                c['committed'] <= c['attempted'] for c in calls), 'call trace')


class Driver:
    """Injected evidence supplies official decode, real API trace, and physical points.

    These are mandatory handoff interfaces, not fabricated HTTP capabilities.
    """
    def __init__(self, transport, journal, evidence, block, strategy, mode):
        require(block in (128, 256) and strategy in
                ('resident-fp32', 'compact-fp8-arithmetic') and mode in ('mps', 'custom'), 'cell')
        self.t, self.j, self.e = transport, journal, evidence
        self.block, self.strategy, self.mode = block, strategy, mode
        self.texts = frozen()
        self.ids = []
        self.requests = 0
        self.polls = 0
        self.sid = None
        self.model_calls = 0
        self.model_tokens = 0
        self.started = time.monotonic()

    def request(self, method, path, body=None, expected=200, declared=None):
        require(time.monotonic() - self.started < 1200, '20-minute process deadline')
        self.requests += 1
        require(self.requests <= 512, 'HTTP call stop')
        status, value = self.t.request(method, path, body, declared)
        self.j.record('http', dict(method=method, path=path, body=body.decode() if
                      isinstance(body, bytes) else body, declared=declared, status=status, reply=value))
        require(status == expected, f'{path}: unexpected HTTP {status}')
        return value

    def path(self, suffix=''):
        return f'/sessions/{self.sid}' + suffix

    def state(self):
        value = self.request('GET', self.path())
        validate_state(value, self.ids)
        require(type(value['session_id']) is int and value['session_id'] == self.sid, 'session identity')
        return value

    def point(self, phase):
        p = self.e.memory_point(phase)
        require(p['phase'] == phase and p['method'] == 'task_info_TASK_VM_INFO_phys_footprint',
                'physical point method')
        require(integer(p['monotonic_ns']) and integer(p['physical_bytes']) and
                p['physical_bytes'] > 0, 'physical point unavailable')
        require(p['guard_diagnostics'] is not None and p['named_buffers'] and
                all(integer(v) for v in p['named_buffers'].values()), 'memory census/guard')
        self.j.record('memory_point', p)

    def reset(self):
        self.request('POST', self.path('/reset'), {})
        self.ids = []
        if hasattr(self.e, "reset_decode"):
            self.e.reset_decode()
        self.state()

    def message(self, name, initial=False):
        row = self.texts[name]
        ids = row['ids'] if not initial else ([1] + row['ids'] if not row['initial'] else row['ids'])
        self.request('POST', self.path('/messages'), {'role': 'user', 'content': row['text']})
        self.ids += ids
        self.state()

    def receipt(self, result):
        calls = self.e.calls(result['id'])
        validate_work(result, calls)
        for c in calls:
            if c['api'] in ('prefill', 'append', 'decode'):
                require(0 < c['attempted'] <= (1 if c['api'] == 'decode' else self.block),
                        'API block bound')
                require(c['finite'] is True and integer(c['start']) and integer(c['end']) and
                        c['end'] - c['start'] == c['committed'] and c['end'] <= 4352,
                        'nonfinite/invalid cursor')
                self.model_calls += 1
                self.model_tokens += c['attempted']
        require(self.model_calls <= (55 if self.block == 128 else 37) and
                self.model_tokens <= 4635, 'model work envelope exceeded')
        self.j.record('actual_calls', calls)

    def check_result(self, identity):
        result = self.request('GET', '/requests/' + identity)
        require(result['id'] == identity and result['session_id'] == self.sid, 'request identity')
        require(result['session_valid'] is True and result['recovery'] == 'none', 'request recovery')
        require(result['status'] == 'completed', 'request not completed')
        self.receipt(result)
        return result

    def generate(self, identity, budget):
        before = self.state()
        events = self.request('POST', self.path('/generate'), {'id': identity, 'budget': budget})
        require(events and events[-1]['event'] == 'completed', 'generation failure')
        r = self.check_result(identity)
        ids = r['token_ids']
        require(0 < len(ids) <= budget and all(integer(i) and i < 131072 for i in ids), 'budget/IDs')
        require(r['generated_tokens'] == len(ids), 'generated count')
        require(r['stop_reason'] in ('budget', 'eos'), 'unexpected stop')
        require((ids[-1] == 2) == (r['stop_reason'] == 'eos') and 2 not in ids[:-1], 'EOS ledger')
        require(r['operation'] == 'generate', 'generate operation')
        self.ids += ids
        after = self.state()
        require(after['pending_tokens'] == 1 and after['committed_tokens'] == len(self.ids)-1 and
                after['state'] == ('ready' if r['stop_reason'] == 'eos' else 'partial'),
                'generation final cursor/phase')
        self.final_result(r, before, after)
        require(events[0]['event'] == 'started' and
                all(e['event'] in ('started', 'text_delta', 'completed') for e in events) and
                sum(e['event'] == 'started' for e in events) == 1 and
                sum(e['event'] == 'completed' for e in events) == 1, 'stream sequence')
        for event in events:
            require(event['id'] == identity and event['session_id'] == self.sid and
                    event['session_valid'] is True and event['recovery'] == 'none', 'stream identity/recovery')
        self.final_result(events[-1], before, after)
        for key in ('generated_tokens', 'stop_reason', 'metrics'):
            require(events[-1][key] == r[key], 'terminal request drift')
        require(''.join(e['text'] for e in events if e['event'] == 'text_delta') == r['text'],
                'stream text drift')
        if getattr(self.e, 'diagnostic_model_free', False):
            require(r['metrics']['test_only'] is True and
                    all(v == 0 for v in r['metrics']['work'].values()), 'diagnostic work claims')
        # Official decode must report buffered tail separately, not use replacement chars.
        decoded = self.e.decode(ids)
        require(decoded['text'] == r['text'] and type(decoded['incomplete_utf8']) is bool,
                'official decode mismatch')
        return r, decoded

    def final_result(self, result, before, after):
        for key in ('state', 'logical_tokens', 'committed_tokens', 'pending_tokens'):
            require(result[key] == after[key] and type(result[key]) is type(after[key]),
                    'final state drift: ' + key)
            require(result['before_' + key] == before[key], 'before state drift: ' + key)

    def control(self, operation, identity, checkpoint=None):
        before = self.state()
        body = {'id': identity}
        if checkpoint is not None:
            body['checkpoint_id'] = checkpoint
        self.request('POST', self.path('/' + operation), body, 202)
        for _ in range(64):
            self.polls += 1
            r = self.request('GET', '/requests/' + identity)
            if r['status'] in ('completed', 'error', 'cancelled'):
                require(r['status'] == 'completed', 'control failure')
                require(r['id'] == identity and r['session_id'] == self.sid and
                        r['session_valid'] is True and r['recovery'] == 'none' and
                        r['operation'] == operation, 'control identity/recovery')
                require(r['token_ids'] == [] and r['generated_tokens'] == 0 and r['text'] == '',
                        'control generated output')
                self.receipt(r)
                after = self.state()
                self.final_result(r, before, after)
                require(after['state'] == before['state'] and after['pending_tokens'] == 0 and
                        after['committed_tokens'] == len(self.ids), 'control phase/cursor')
                w = r['metrics']['work']
                if operation == 'checkpoint':
                    require(w['pending_attempted'] == w['pending_committed'] == before['pending_tokens'],
                            'save pending receipt')
                    require(r['cache_restored'] is False, 'save cache restored')
                else:
                    require(r['cache_restored'] is True and r['checkpoint_id'] == checkpoint,
                            'restore checkpoint/cache identity')
                return r
            time.sleep(.01)
        raise TimeoutError('bounded control polls exhausted; no retry')

    def checkpoint_branch(self, label, generated, decoded):
        before = self.state()
        if decoded['incomplete_utf8']:
            r = self.request('POST', self.path('/checkpoint'), {'id': label + '-save'}, 409)
            require(r['error'] == 'checkpoint_boundary_required', 'UTF8 boundary rejection')
            require(self.state() == before, 'rejected save mutation')
            self.j.record('coverage', {'case': label, 'checkpoint': 'partial-incompleteUTF8'})
            return
        saved_ids = list(self.ids)
        saved = self.control('checkpoint', label + '-save')
        require(saved['checkpoint_id'] not in ('0', '', None), 'checkpoint identity')
        materialized = self.state()
        require(materialized['committed_tokens'] == len(self.ids) and
                materialized['pending_tokens'] == 0, 'save did not materialize pending')
        self.point(label + '-after-save')
        self.control('restore', label + '-restore', saved['checkpoint_id'])
        self.ids = saved_ids
        restored = self.state()
        require(restored['state'] == before['state'] and restored['pending_tokens'] == 0 and
                restored['committed_tokens'] == len(self.ids),
                'restore phase/pending')
        self.point(label + '-after-restore')
        if generated['stop_reason'] != 'eos':
            self.j.record('coverage', {'case': label, 'ready': 'partial-no-naturalEOS'})
        if generated['stop_reason'] == 'eos':
            require(restored['state'] == 'ready', 'natural EOS must be ready')
            self.request('POST', self.path('/generate'), {'id': label + '-bare', 'budget': 1}, 409)
            require(self.state() == restored, 'bare continuation mutated ready')
            self.message('short')
        else:
            require(restored['state'] == 'partial', 'non-EOS must remain partial')
        self.generate(label + '-replay', 1)
        self.j.record('coverage', {'case': label, 'ready_naturalEOS': generated['stop_reason'] == 'eos'})

    def rejected(self, body, status, declared=None):
        before = self.state()
        self.request('POST', self.path('/messages'), body, status, declared)
        require(self.state() == before, 'admission rejection changed state')

    def run(self):
        try:
            return self._run()
        except BaseException as error:
            self.j.record('failure', {'type': type(error).__name__, 'detail': str(error),
                                     'retry': False})
            raise

    def _run(self):
        verify_pins()
        info = self.request('GET', '/info')
        require(info['context_capacity'] == 4352 and info['prefill_chunk_size'] == self.block and
                info['prefix_reuse'] == 'on' and info['execution_strategy'] == self.strategy and
                info['prefill'] == self.mode and info['model_loaded'] is True and
                info['runtime_ready'] is True and type(info['model_load_count']) is int and
                info['model_load_count'] == 1 and info['model_sha256'] ==
                '728f1826cd0e38191ca7b1379e81f78cf0555c6ffd95882aabd2404632346f86', 'server profile')
        require(info['tokenizer_sha256'] == json.loads((PREP / 'texts.json').read_text())['tokenizer_sha256'],
                'tokenizer identity')
        self.point('load-ready')
        created = self.request('POST', '/sessions', {})
        self.sid = created['session_id']
        self.reset()
        # Empty valid ready state isolates tokenizer capacity rejection from phase rejection.
        oversized = self.texts['document']['text'] + self.texts['new-user']['text']
        negative = json.loads((PREP / 'negative.json').read_text())
        require(hashlib.sha256(oversized.encode()).hexdigest() == negative['text_sha256'] and
                self.e.frame(oversized, True) == negative['ids'] and
                len(negative['ids']) > 4352 and len(oversized.encode()) < 32768,
                'official capacity probe differs from frozen exact IDs')
        self.rejected({'role': 'user', 'content': oversized}, 422)
        self.rejected({'role': 'user', 'content': ' ' * 32769}, 400)
        self.rejected(b'', 400, 262145)
        self.message('spaces')
        self.generate('spaces-4', 4)
        self.point('spaces-after-ingestion')
        self.reset()
        self.message('document')
        r, decoded = self.generate('document-8', 8)
        self.point('document-after-ingestion')
        self.checkpoint_branch('document', r, decoded)
        self.reset()
        self.message('short', initial=True)
        r, decoded = self.generate('short-4', 4)
        self.point('short-after-ingestion')
        self.checkpoint_branch('short', r, decoded)
        summary = self.e.finish()
        require(summary['unattributed_calls'] == 0 and summary['model_calls'] == self.model_calls and
                summary['model_tokens'] == self.model_tokens, 'hidden/missing actual calls')
        self.j.record('trace_summary', summary)
        self.j.record('result', {'status': 'bounded HTTP smoke only', 'launch_gate': 'AUTHORIZED', 'model_validation': 'NOT_ESTABLISHED',
                                'requests': self.requests, 'polls': self.polls})

