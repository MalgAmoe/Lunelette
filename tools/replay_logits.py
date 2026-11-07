#!/usr/bin/env python3
"""Bounded native capture comparison; NOT an independent numerical oracle."""
import argparse
import json
import math
import struct
import sys

VOCAB = 131072
LIMIT = 16
CONTEXT = 4352
MAX_BYTES = 16 + LIMIT * (16 + 4 * CONTEXT + 4 * VOCAB)


def replay_tokens(tokens):
    if (not isinstance(tokens, list) or not 1 <= len(tokens) <= LIMIT
            or any(type(t) is not int or not (1000 <= t < VOCAB or
                       (t == 2 and i == len(tokens) - 1)) for i, t in enumerate(tokens))):
        raise ValueError("invalid replay tokens")
    return tokens


def parse(path):
    """Read a bounded version-one artifact, validating each vector's own argmax."""
    with open(path, 'rb') as source:
        if not 16 <= source.seek(0, 2) <= MAX_BYTES:
            raise ValueError('capture size outside bound')
        source.seek(0)
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or data[:16] != b'LNRPLY1\0' + struct.pack('<II', VOCAB, LIMIT):
        raise ValueError('invalid capture header/size')
    records = []
    offset = 16
    while offset < len(data):
        if len(records) == LIMIT or len(data) - offset < 16:
            raise ValueError('record limit/trailing bytes')
        index, count, forced, greedy = struct.unpack_from('<IIII', data, offset)
        offset += 16
        if index != len(records) or not 1 <= count <= CONTEXT or not (forced == 2 or 1000 <= forced < VOCAB) or not 0 <= greedy < VOCAB:
            raise ValueError('invalid record fields')
        size = 4 * (count + VOCAB)
        if len(data) - offset < size:
            raise ValueError('truncated capture')
        prefix = list(struct.unpack_from('<' + str(count) + 'I', data, offset))
        if any(t >= VOCAB for t in prefix):
            raise ValueError('invalid prefix token')
        offset += count * 4
        raw = data[offset:offset + VOCAB * 4]
        logits = struct.unpack('<' + str(VOCAB) + 'f', raw)
        offset += VOCAB * 4
        if not all(math.isfinite(v) for v in logits):
            raise ValueError('nonfinite logits')
        actual = max(range(VOCAB), key=logits.__getitem__)
        if greedy != actual:
            raise ValueError('incorrect first-tie greedy argmax')
        records.append(dict(index=index, prefix=prefix, forced_token=forced,
                            greedy_token=greedy, logits=logits, raw=raw))
    return records


def verify(path, steps):
    """Correlate completed requests with actual prefixes, never planned text."""
    records = parse(path)
    expected = []
    for step in steps:
        tokens = replay_tokens(step['replay_tokens'])
        actual = step['actual_ids']
        if not isinstance(actual, list) or any(type(t) is not int for t in actual) or actual != tokens:
            raise ValueError('incomplete replay trajectory')
        prefix = step['before_prefix_ids']
        if (not isinstance(prefix, list) or not 1 <= len(prefix) <= CONTEXT
                or any(type(t) is not int or not 0 <= t < VOCAB for t in prefix)):
            raise ValueError('invalid authoritative prefix')
        prefix = list(prefix)
        for token in actual:
            expected.append((list(prefix), token))
            prefix.append(token)
    if not expected or len(records) != len(expected):
        raise ValueError('capture record count mismatch')
    for record, (prefix, forced) in zip(records, expected):
        if record['prefix'] != prefix or record['forced_token'] != forced:
            raise ValueError('capture context/target mismatch')
    return {'verified_records': len(records), 'independent_numerical_oracle': False}


def compare(reference, candidate, changed_partition=False):
    left, right = parse(reference), parse(candidate)
    if not left or not right or len(left) != len(right):
        raise ValueError('record count mismatch')
    mismatches = 0
    max_absolute = max_relative = 0.0
    relative_undefined = False
    for a, b in zip(left, right):
        if a['prefix'] != b['prefix'] or a['forced_token'] != b['forced_token']:
            raise ValueError('context/forced target mismatch')
        for i, (x, y) in enumerate(zip(a['logits'], b['logits'])):
            error = abs(x - y)
            max_absolute = max(max_absolute, error)
            if x:
                max_relative = max(max_relative, error / abs(x))
            elif error:
                relative_undefined = True
            matches = error <= 2e-4 + 1e-4 * abs(x) if changed_partition else a['raw'][4*i:4*i+4] == b['raw'][4*i:4*i+4]
            mismatches += not matches
    return dict(match=mismatches == 0, mismatch_count=mismatches,
                max_absolute_error=max_absolute,
                max_relative_error=None if relative_undefined else max_relative,
                relative_error_undefined_at_zero=relative_undefined,
                records=len(left), independent_numerical_oracle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--changed-partition', action='store_true')
    args = parser.parse_args()
    try:
        result = compare(args.reference, args.candidate, args.changed_partition)
        print(json.dumps(result))
        return 0 if result['match'] else 1
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
