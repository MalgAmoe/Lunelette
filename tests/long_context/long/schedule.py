"""Bounded deterministic LONG schedules. No model access."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise ValueError(message)

def schedule(block):
    require(block in (128, 256), 'block')
    ops = []
    pairs = []
    def emit(api, start=0, ids=(), file=''):
        ops.append(dict(api=api, start=start, ids=list(ids), file=file))
    def snapshot(name):
        emit('snapshot', file=name + '.f32')
    def fresh(ids, name):
        emit('reset')
        for p in range(0, len(ids), block):
            emit('prefill' if p == 0 else 'append', p, ids[p:p+block])
        snapshot(name)
    def tokens(n):
        return [1000 + (37*i) % 8000 for i in range(n)]
    for n in (block-1, block, block+1):
        fresh(tokens(n), 'boundary%d' % n)
    if block == 256:
        fresh(tokens(513), 'tail513')
    prefix = 4096 if block == 256 else 257
    base = tokens(prefix)
    fresh(base, 'prefix')
    emit('save')
    for label, token in (('a', 9101), ('b', 9102)):
        emit('append', prefix, [token])
        snapshot(label)
        emit('restore')
        snapshot('restore_' + label)
        pairs.append(('restore_' + label, 'prefix', 'bitwise'))
    # Independently RESET and replay the identical chunk schedule, not a restore control.
    for label, token in (('a', 9101), ('b', 9102)):
        fresh(base, 'replay_prefix_' + label)
        pairs.append(('replay_prefix_' + label, 'prefix', 'bitwise'))
        emit('append', prefix, [token])
        snapshot('replay_' + label)
        pairs.append((label, 'replay_' + label, 'bitwise'))
    fresh(tokens(4352 if block == 256 else 513), 'capacity' if block == 256 else 'tail513')
    total = sum(len(op['ids']) for op in ops)
    require(total <= (20000 if block == 256 else 2048), 'input token budget')
    return dict(block=block, capacity=4352, tokens=total, operations=ops, pairs=pairs)

def header(path):
    lines = ['/* LONG-only frozen schedules; generated model-free. */']
    for block in (128, 256):
        lines.append('static const operation ops%d[] = {' % block)
        for op in schedule(block)['operations']:
            lines.append('{"%s", "%s", %d, %d, {%s}},' %
                         (op['api'], op['file'], op['start'], len(op['ids']),
                          ','.join(map(str, op['ids'])) or '0'))
        lines.append('};')
    Path(path).write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    print(json.dumps([schedule(b) for b in (256, 128)], indent=2))
