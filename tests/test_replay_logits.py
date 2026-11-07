"""Synthetic capture checks; no model or independent numerical oracle."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import replay_logits as r


def capture(prefix=(1, 1000), forced=53156, greedy=0, value=1.0):
    logits = [0.0] * r.VOCAB
    logits[greedy] = value
    return (b'LNRPLY1\0' + struct.pack('<II', r.VOCAB, 16)
            + struct.pack('<IIII', 0, len(prefix), forced, greedy)
            + struct.pack('<' + str(len(prefix)) + 'I', *prefix)
            + struct.pack('<' + str(r.VOCAB) + 'f', *logits))


class Capture(unittest.TestCase):
    def test_parse_compare_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            a.write_bytes(capture())
            b.write_bytes(capture())
            self.assertTrue(r.compare(a, b)['match'])
            self.assertEqual(r.verify(a, [dict(replay_tokens=[53156], actual_ids=[53156], before_prefix_ids=[1, 1000])])['verified_records'], 1)
            b.write_bytes(capture(value=1.00001))
            self.assertFalse(r.compare(a, b)['match'])
            self.assertTrue(r.compare(a, b, True)['match'])
            b.write_bytes(capture(value=2))
            self.assertFalse(r.compare(a, b, True)['match'])
            b.write_bytes(capture(prefix=(1, 1001)))
            with self.assertRaises(ValueError):
                r.compare(a, b)
            with self.assertRaises(ValueError):
                r.verify(a, [dict(replay_tokens=[1000], actual_ids=[1000], before_prefix_ids=[1, 1000])])

    def test_empty_and_signed_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            a.write_bytes(capture()[:16])
            b.write_bytes(capture()[:16])
            with self.assertRaises(ValueError):
                r.compare(a, b)
            with self.assertRaises(ValueError):
                r.verify(a, [])
            a.write_bytes(capture())
            data = bytearray(capture())
            offset = 16 + 16 + 2 * 4 + 4  # second logit, zero in the reference
            struct.pack_into('<f', data, offset, -0.0)
            b.write_bytes(data)
            self.assertFalse(r.compare(a, b)['match'])
            self.assertTrue(r.compare(a, b, True)['match'])
            struct.pack_into('<f', data, offset, .00001)
            b.write_bytes(data)
            import json
            result = r.compare(a, b, True)
            self.assertIsNone(result['max_relative_error'])
            json.dumps(result, allow_nan=False)

    def test_rejections(self):
        good = capture()
        cases = [good[:-1], good + b'x', b'BADMAGIC' + good[8:],
                 capture(value=float('nan')), capture(value=float('inf')),
                 capture(prefix=(131072,)), capture(forced=3), capture(greedy=1, value=0)]
        for data in cases:
            with self.subTest(size=len(data)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'capture'
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    r.parse(path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'large'
            with path.open('wb') as target:
                target.truncate(r.MAX_BYTES + 1)
            with self.assertRaises(ValueError):
                r.parse(path)


if __name__ == '__main__':
    unittest.main()
