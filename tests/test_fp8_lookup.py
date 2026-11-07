"""Audit literal unscaled table bits, including negative zero; not GPU evidence."""
import math
from pathlib import Path
import re
import struct
import unittest


class LookupTable(unittest.TestCase):
    def test_exact_finite_codes(self):
        source = (Path(__file__).resolve().parents[1] / 'src/metal_kernels.metal').read_text()
        body = source.split('constant float fp8_values[256] = {', 1)[1].split('};', 1)[0]
        body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
        literals = [x.strip() for x in body.split(',') if x.strip()]
        self.assertEqual(len(literals), 256)
        for code, literal in enumerate(literals):
            self.assertTrue(literal.endswith('f'))
            actual = float.fromhex(literal[:-1])
            if code in (127, 255):
                # Both NaN codes are rejected by unchanged host payload admission.
                self.assertEqual(actual, 0)
                continue
            exponent, fraction = (code >> 3) & 15, code & 7
            expected = math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(8 + fraction, exponent - 10)
            if code & 128:
                expected = -expected
            self.assertEqual(struct.pack('<f', actual), struct.pack('<f', expected), code)
        self.assertEqual(struct.pack('<f', float.fromhex(literals[128][:-1])), b'\0\0\0\x80')


if __name__ == '__main__':
    unittest.main()
