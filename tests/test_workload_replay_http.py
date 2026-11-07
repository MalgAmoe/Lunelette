"""Actual native stub/HTTP/client capture: SERVER TOKENIZER; no real weights."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from real_workload import load_scenario
from replay_logits import parse, compare


def check(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def main():
    check(len(sys.argv) == 3, __doc__)
    server, tokenizer = (str(Path(p).resolve()) for p in sys.argv[1:])
    root = Path(__file__).resolve().parents[1]
    original = load_scenario(root / 'tools/scenarios/replay-v2.json')
    python = [sys.executable, '-B'] + (['-O'] if sys.flags.optimize else [])
    with tempfile.TemporaryDirectory(prefix='lunelette-replay-http-') as directory:
        directory = Path(directory)
        for policy in ('on', 'off'):
            value = copy.deepcopy(original)
            value['prefix_reuse'] = policy
            scenario = directory / (policy + '.json')
            scenario.write_text(json.dumps(value))
            output = directory / policy
            result = subprocess.run(python + [str(root / 'tools/real_workload.py'),
                '--server', server, '--scenario', str(scenario), '--output', str(output),
                '--native-model', tokenizer, '--native-tokenizer', tokenizer,
                '--native-prefill', 'custom'], capture_output=True, text=True, timeout=30)
            check(result.returncode == 0, (result.stderr, (output / 'result.json').read_text()))
            receipt = json.loads((output / 'result.json').read_text())
            check(receipt['protocol_success'] and receipt['completed_repetitions'] == 2, receipt)
            check(receipt['task_success'] is None and receipt['performance_evidence'] is False, receipt)
            check(receipt['capture_status'] == 'completed_verified', receipt)
            check(receipt['capture_verification']['verified_records'] == 8, receipt)
            records = parse(output / 'replay-logits.bin')
            check([r['forced_token'] for r in records] == [53156, 2] * 4, 'targets')
            check(all(r['greedy_token'] == 22177 for r in records), 'real stub greedy differs')
            steps = [s for rep in receipt['repetitions'] for s in rep['result']['replay_steps']]
            check([s['id'] for s in steps] == ['r1-first', 'r1-second', 'r2-first', 'r2-second'], steps)
            check(steps[0]['result']['session_id'] != steps[2]['result']['session_id'], steps)
            for step in steps:
                work = step['result']['metrics']['work']
                check(work['prefill_calls'] == (policy == 'off'), work)
                check(work['decode_committed'] == 1, work)
                check(step['result']['session_metadata']['token_ids'][-2:] == [53156, 2], step)
            events = [json.loads(line) for line in (output / 'client.ndjson').read_text().splitlines()]
            check(any(e['kind'] == 'shutdown' and e['returncode'] == 0 for e in events), 'shutdown')
            infos = [json.loads(e['raw']) for e in events if e['kind'] == 'response'
                     and '"generator"' in e.get('raw', '')]
            check(len(infos) == 1 and infos[0]['model_load_count'] == 1, infos)
        comparison = compare(directory / 'on/replay-logits.bin', directory / 'off/replay-logits.bin')
        check(comparison['match'] and comparison['records'] == 8, comparison)
        result = subprocess.run(python + [str(root / 'tools/replay_logits.py'),
            str(directory / 'on/replay-logits.bin'), str(directory / 'off/replay-logits.bin')],
            capture_output=True, text=True, timeout=30)
        check(result.returncode == 0 and json.loads(result.stdout)['match'], result.stderr)
    print('native stub replay ON/OFF: eight full vectors each, forced non-greedy IDs, '
          'CLI/socket/checkpoint/capture verified (not model validation)')


if __name__ == '__main__':
    main()
