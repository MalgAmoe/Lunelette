"""Offline exact schedule replay against captured HTTP and native evidence."""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver import Driver, require
from provider import Tokenizer, environment


def validate(work, tokenizer):
    work = Path(work)
    def rows(name, cap):
        path = work/name
        require(path.stat().st_size <= cap, name+' cap')
        def no_constant(value):
            raise ValueError('nonfinite JSON: '+value)
        return [json.loads(line, parse_constant=no_constant) for line in path.read_text().splitlines()]
    recorded_environment = rows('reference-environment.json', 16384)
    require(recorded_environment == [environment()], 'reference environment versions drift')
    journal = rows('journal.jsonl', 2*1024*1024)
    observations = rows('observation.jsonl', 4*1024*1024)
    require(not (work/'failure-logits.f32').exists(), 'failure vector present')
    require(journal and journal[0]['kind'] == 'startup_guard_observations', 'startup receipt')
    expected = journal[1:]
    calls = [r for r in observations if r['kind'] == 'call']
    require(all(r['kind'] in ('call', 'memory') for r in observations), 'observation kind')
    require(all(r.get('finite') is True and r.get('status') == 0 and r.get('request_id') for r in calls),
            'nonfinite/error/unattributed observation')
    require(calls == [c for r in expected if r['kind'] == 'actual_calls' for c in r['value']],
            'ordered native calls differ')

    class Journal:
        index = 0
        def record(self, kind, value):
            require(self.index < len(expected) and expected[self.index] == dict(kind=kind, value=value),
                    'replay journal mismatch at '+str(self.index))
            self.index += 1
    class Transport:
        records = [r['value'] for r in expected if r['kind'] == 'http']
        index = 0
        def request(self, method, path, body=None, declared=None):
            require(self.index < len(self.records), 'extra HTTP request')
            row = self.records[self.index]
            self.index += 1
            require((method, path, body.decode() if isinstance(body, bytes) else body, declared) ==
                    (row['method'], row['path'], row['body'], row['declared']), 'ordered HTTP differs')
            return row['status'], row['reply']
    class Evidence(Tokenizer):
        points = [r['value'] for r in expected if r['kind'] == 'memory_point']
        index = 0
        def calls(self, identity):
            return [c for c in calls if c['request_id'] == identity]
        def memory_point(self, phase):
            require(self.index < len(self.points), 'extra memory point')
            point = self.points[self.index]
            self.index += 1
            require(point['phase'] == phase, 'memory point order')
            require({k:v for k,v in point.items() if k != 'phase'} in observations,
                    'memory point missing from observer')
            return point
        def finish(self):
            model = [c for c in calls if c['api'] != 'restore']
            return dict(unattributed_calls=sum(not c['request_id'] for c in calls),
                        model_calls=len(model), model_tokens=sum(c['attempted'] for c in model))
    evidence, transport, output = Evidence(tokenizer), Transport(), Journal()
    Driver(transport, output, evidence, 256, 'compact-fp8-arithmetic', 'mps').run()
    require(output.index == len(expected) and transport.index == len(transport.records) and
            evidence.index == len(evidence.points), 'unconsumed replay evidence')
    return dict(scope='schedule/state/work only', status='pass')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('tokenizer', type=Path)
    parser.add_argument('--reference-python', type=Path,
                        help='explicit reference venv; launch with its resolved interpreter')
    args = parser.parse_args()
    if args.reference_python is not None:
        from reference_environment import bootstrap
        bootstrap(args.reference_python)
    print(json.dumps(validate(args.work, args.tokenizer)))
