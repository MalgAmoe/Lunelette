"""Offline LONG trace/full-vector gates. No inference and no automatic retries."""
import argparse
import math
from pathlib import Path
import re
import struct
from schedule import schedule, require


def vector(a, b, contract='bitwise'):
    require(contract in ('bitwise', 'changed-schedule-frozen-tolerance'), 'comparison contract')
    require(len(a) == len(b) == 524288, 'full131072 vector size')
    x, y = struct.unpack('<131072f', a), struct.unpack('<131072f', b)
    require(all(math.isfinite(v) for v in x+y), 'nonfinite logits')
    errors = [abs(u-v) for u,v in zip(x,y)]
    max_normalized = max(e/(2e-4+1e-4*abs(v)) for e,v in zip(errors,y))
    require(a == b if contract == 'bitwise' else max_normalized <= 1, 'logits mismatch: '+contract)
    return dict(contract=contract, max_abs=max(errors), max_normalized=max_normalized)


def trace(text, block):
    plan = schedule(block)
    calls = re.findall(r'^CALL index=(\d+) api=(\w+) start=(\d+) count=(\d+) ids=(.*)$', text, re.M)
    results = re.findall(r'^RESULT index=(\d+) status=(\d+) length=(\d+)$', text, re.M)
    require(len(calls) == len(results) == len(plan['operations']), 'missing/extra records')
    length = prefix = 0
    rejection_counts = []
    expected_kv = {}
    for i, (op, call, result) in enumerate(zip(plan['operations'], calls, results)):
        ids = ','.join(map(str,op['ids']))
        require(call == (str(i), op['api'], str(op['start']), str(len(op['ids'])), ids), 'off-by-one/call/ID drift')
        if op['api'] == 'reset':
            length = prefix = 0
        elif op['api'] in ('prefill','append'):
            require(length == op['start'], 'cursor drift')
            length += len(op['ids'])
        elif op['api'] == 'save':
            prefix = length
        elif op['api'] == 'restore':
            length = prefix
        elif op['api'] == 'snapshot':
            expected_kv[op['file']] = prefix or length
            if not prefix:
                rejection_counts.extend([0,block+1] + ([1,1,4353] if length == 4352 else []))
        require(result == (str(i),'0',str(length)), 'result/status/length drift')
    rejected = re.findall(r'^REJECT test=(\d+) count=(\d+) status=(-?\d+) diagnostic=(.*)$', text, re.M)
    require([int(r[1]) for r in rejected] == rejection_counts and all(int(r[2]) != 0 for r in rejected), 'rejection coverage/status')
    records = re.findall(r'^ENDKV file=(\S+) prefix=(\d+) buffer=(\d+) sha256=([0-9a-f]{64})$', text, re.M)
    kv = {}
    for file, n, k, digest in records:
        require(file in expected_kv and int(n) == expected_kv[file] and 0 <= int(k) < 52, 'KV record bounds')
        require((file,int(k)) not in kv, 'duplicate KV record')
        kv[file,int(k)] = digest
    require(len(kv) == len(expected_kv)*52, 'all52 KV coverage')
    for file in ('a.f32','b.f32','restore_a.f32','restore_b.f32'):
        require(all(kv[file,k] == kv['prefix.f32',k] for k in range(52)), 'retained-prefix KV corruption')
    for a,b,_ in plan['pairs']:
        if a.startswith(('restore', 'replay_prefix')):
            require(all(kv[a+'.f32',k] == kv[b+'.f32',k] for k in range(52)), 'restore KV mismatch')
    require('CLEANUP checkpoint_destroyed runtime_destroyed host_copy_freed status=0' in text, 'cleanup failure')
    return plan


def check(directory, block):
    directory = Path(directory)
    plan = trace((directory/'stdout.log').read_text(), block)
    # Validate every artifact, even those without a comparison pair.
    for op in plan['operations']:
        if op['api'] == 'snapshot':
            data = (directory/op['file']).read_bytes()
            vector(data,data)
    return [vector((directory/(a+'.f32')).read_bytes(), (directory/(b+'.f32')).read_bytes(), c)
            for a,b,c in plan['pairs']]

if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser()
    p.add_argument('directory')
    p.add_argument('--block', type=int, choices=(128,256), required=True)
    p.add_argument('--resident-control', help='B256 same-schedule compact/resident bitwise pair')
    p.add_argument('--b256-control', help='B128 changed-schedule 257/513 compact control')
    args = p.parse_args()
    result = check(args.directory,args.block)
    if args.resident_control:
        require(args.block == 256, 'resident pairing only B256')
        check(args.resident_control,256)
        for op in schedule(256)['operations']:
            if op['api'] == 'snapshot':
                result.append(vector((Path(args.directory)/op['file']).read_bytes(),
                                     (Path(args.resident_control)/op['file']).read_bytes()))
    if args.b256_control:
        require(args.block == 128, 'changed-schedule smoke only B128')
        check(args.b256_control,256)
        for candidate, control in (('prefix.f32','boundary257.f32'), ('tail513.f32','tail513.f32')):
            result.append(vector((Path(args.directory)/candidate).read_bytes(),
                                 (Path(args.b256_control)/control).read_bytes(),
                                 'changed-schedule-frozen-tolerance'))
    print(json.dumps(result,indent=2))
