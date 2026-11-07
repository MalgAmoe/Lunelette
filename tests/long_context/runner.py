"""Seven serial cells; default inspection does not open model inputs."""
import argparse
import hashlib
import importlib.util
import json
import copy
import math
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent



def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    sys.path.insert(0, str(HERE / 'long'))
    return load('suite_long_check', HERE / 'long/check.py')


# service/session.c metrics_json; docs/real-workload-m2.md measurement definitions.
MS_KEYS = frozenset(('framing_ms', 'prefill_ms', 'decode_ms', 'first_token_ms',
                     'total_ms', 'cancel_latency_ms', 'cancel_received_monotonic_ms',
                     'materialize_ms', 'snapshot_ms', 'restore_ms'))
US_KEYS = frozenset(('prefill_us', 'suffix_us', 'decode_total_us'))
NULL_KEYS = frozenset(('first_token_ms', 'cancel_latency_ms'))


def timing_metrics(metrics):
    for key in MS_KEYS | US_KEYS | {'decode_step_duration_us'}:
        if key not in metrics:
            continue
        value = metrics[key]
        if value is None and key in NULL_KEYS:
            continue  # Preserve the documented unavailable sentinel exactly.
        values = value if key == 'decode_step_duration_us' else [value]
        require(type(values) is list, 'timing array: '+key)
        for number in values:
            if key in US_KEYS or key == 'decode_step_duration_us':
                require(type(number) is int and 0 <= number <= 2**64-1, 'timing uint64: '+key)
            else:
                require(type(number) in (int, float) and math.isfinite(number) and number >= 0,
                        'timing finite nonnegative: '+key)
        metrics[key] = [0]*len(values) if key == 'decode_step_duration_us' else 0


def normalize(rows):
    # Only cross-lane interleaving is ignored; never sort state events.
    out = dict(routes=[], worker=[])
    for original in rows:
        r = copy.deepcopy(original)
        if r['type'] in ('resource_point', 'prefix_hashes'):
            continue
        for k in ('file', 'representative'):
            if k in r:
                r[k] = Path(r[k]).name
        if r['type'] == 'route' and isinstance(r.get('reply'), dict):
            metrics = r['reply'].get('metrics')
            if metrics is not None:
                require(type(metrics) is dict, 'metrics object')
                timing_metrics(metrics)
        out['routes' if r['type'] == 'route' else 'worker'].append(r)
    return out


def short_check(work, contract, vector, *, stub=False):
    rows = [json.loads(s) for s in (work/'stdout').read_text().splitlines()]
    require(normalize(rows) == contract['rows'], 'short structural coverage/contract drift')
    endpoints = [r for r in rows if r['type'] == 'endpoint']
    hashes = [r for r in rows if r['type'] == 'prefix_hashes']
    require(len(hashes) == len(endpoints) == len(contract['files']), 'short hash coverage')
    for endpoint, record, name in zip(endpoints, hashes, contract['files']):
        require(endpoint['file'] == record['file'] == str(work/name), 'endpoint exact path')
        require(record['kind'] == ('stub_input_state_not_KV' if stub else '52_retained_KV_buffers'), 'native KV evidence required')
        require(len(record['sha256']) == 52 and all(re.fullmatch('[0-9a-f]{64}', h) for h in record['sha256']), 'all52 hashes')
        data = (work/name).read_bytes()
        vector(data, data)
    verified = [r for r in rows if r['type'] in ('direct_verified','duplicate_verified')]
    require([r['file'] for r in verified] == [r['file'] for r in endpoints], 'every endpoint independently verified')
    by_file = {r['file']: r for r in hashes}
    for r in verified:
        if r['type'] == 'duplicate_verified':
            a, b = Path(r['file']), Path(r['representative'])
            require(b.parent == work and b.name in contract['files'], 'representative path')
            vector(a.read_bytes(), b.read_bytes())
            require(by_file[str(a)]['sha256'] == by_file[str(b)]['sha256'], 'duplicate KV mismatch')
    resources = [r for r in rows if r['type'] == 'resource_point']
    require(len(resources) == len(endpoints) and all(type(r['footprint_point']) is int and r['footprint_point'] > 0 for r in resources), 'resource point coverage')
    return dict(owner_schedule=next(r for r in rows if r['type']=='owner_schedule'),
                token_budget=next(r for r in rows if r['type']=='token_budget'), resource_points=resources)


class Callbacks:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.started = {}
        self.checker = helpers()
        self.contracts = json.loads((HERE/'contracts.json').read_text())

    def prepare(self, job, work):
        self.started[job['id']] = time.monotonic()
        self.supervisor.durable(work/'timing-start.json', dict(monotonic=self.started[job['id']], scope='prepare before final pre-spawn pin checks'))

    def validate(self, job, work, code):
        metrics = dict(accepted=False)
        try:
            metrics.update(self.checks(job, work, code))
            metrics['accepted'] = True
            return 'pass'
        finally:
            started = self.started[job['id']]
            ended = time.monotonic()
            metrics['timing'] = dict(contract='monotonic prepare-through-post-exit-validation; includes final pre-spawn pin checks, not GPU-only latency', started=started, ended=ended, elapsed_seconds=ended-started)
            self.supervisor.durable(work/'validation.json', metrics, 128*1024)

    def checks(self, job, work, code):
        require(code == 0, 'child exit: '+str(code))
        name = job['id']
        metrics = {}
        if name.startswith('short-'):
            key = 'cancel-on' if 'cancel' in name else ('normal-off' if name.endswith('off') else 'normal-on')
            metrics = short_check(work, self.contracts[key], self.checker.vector)
            if name == 'short-resident-on':
                for f in self.contracts[key]['files']:
                    metrics.setdefault('pairs', []).append(self.checker.vector((work/f).read_bytes(), (work.parent/'short-compact-on'/f).read_bytes()))
            if name == 'short-compact-off':
                # OFF changes API partition; use only frozen changed-schedule tolerance.
                for a,b in zip(self.contracts[key]['files'],self.contracts['normal-on']['files']):
                    metrics.setdefault('pairs', []).append(self.checker.vector((work/a).read_bytes(), (work.parent/'short-compact-on'/b).read_bytes(), 'changed-schedule-frozen-tolerance'))
        else:
            # Charged exclusive copy, not a hardlink or an edit of the old checker.
            data = (work/'stdout').read_bytes()
            require(len(data) <= 2*1024**2, 'stdout alias bound')
            with (work/'stdout.log').open('xb') as f:
                f.write(data)
            block = 128 if name.endswith('128') else 256
            metrics['within'] = self.checker.check(work, block)
            if name == 'long-compact256':
                control = work.parent/'long-resident256'
                self.checker.check(control,256)
                for op in self.checker.schedule(256)['operations']:
                    if op['api']=='snapshot':
                        metrics.setdefault('pairs',[]).append(self.checker.vector((work/op['file']).read_bytes(),(control/op['file']).read_bytes()))
            if name == 'long-compact128':
                control = work.parent/'long-compact256'
                self.checker.check(control,256)
                for a,b in (('prefix.f32','boundary257.f32'),('tail513.f32','tail513.f32')):
                    metrics.setdefault('pairs',[]).append(self.checker.vector((work/a).read_bytes(),(control/b).read_bytes(),'changed-schedule-frozen-tolerance'))
            metrics['actual_model_tokens'] = self.checker.schedule(block)['tokens']
            records = [s for s in data.decode().splitlines() if s.startswith('RESOURCE ')]
            labels = ['load-ready'] + [('ingestion-endpoint' if op['api']=='snapshot' else op['api']) for op in self.checker.schedule(block)['operations'] if op['api'] in ('snapshot','save','restore')]
            require(len(records)==len(labels), 'resource coverage')
            for record,label in zip(records,labels):
                require(re.fullmatch(r'RESOURCE '+label+r' footprint_point=[1-9][0-9]* weights=\d+ conversion=\d+ scratch=\d+ kv=\d+ total=\d+',record), 'resource record')
            metrics['resource_points'] = records
            metrics['actual_operations'] = len(self.checker.schedule(block)['operations'])
        return metrics


