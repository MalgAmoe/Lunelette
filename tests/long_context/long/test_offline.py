"""File-backed checker smoke with synthetic data only; temporary files stay owned."""
import struct
import tempfile
from pathlib import Path
import unittest
from check import check
from schedule import HERE, schedule
from test_long import synthetic


class OfflineTests(unittest.TestCase):
    def test_file_backed_and_stale_endpoint(self):
        good = struct.pack('<131072f', *([1.0]*131072))
        stale = struct.pack('<131072f', *([2.0]*131072))
        with tempfile.TemporaryDirectory(prefix='synthetic-', dir=HERE) as tmp:
            root = Path(tmp)
            (root/'stdout.log').write_text(synthetic(256))
            for op in schedule(256)['operations']:
                if op['api'] == 'snapshot':
                    (root/op['file']).write_bytes(good)
            self.assertEqual(len(check(root,256)),6)
            # Wrong finite branch artifact while independently replayed control stays good.
            (root/'a.f32').write_bytes(stale)
            with self.assertRaisesRegex(ValueError,'logits mismatch'):
                check(root,256)
            (root/'a.f32').write_bytes(good)
            # Even a finite-only capacity endpoint is checked across its entire vector.
            bad = bytearray(good)
            struct.pack_into('<f',bad,524284,float('nan'))
            (root/'capacity.f32').write_bytes(bad)
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                check(root,256)

if __name__ == '__main__':
    unittest.main()
