"""Checker-only contract copies with synthetic vectors; NOT official-token traces."""
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import runner


class ShortCheckerTests(unittest.TestCase):
    def test_copied_contracts_and_faults(self):
        contracts = json.loads((Path(__file__).parent/'contracts.json').read_text())
        def vector(a, b):
            if len(a) != 131072*4 or a != b:
                raise ValueError('synthetic vector mismatch')
            if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f', a)):
                raise ValueError('nonfinite')
        for name, contract in contracts.items():
            with self.subTest(contract=name), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                rows = copy.deepcopy(contract['rows']['routes']+contract['rows']['worker'])
                for row in rows:
                    for key in ('file', 'representative'):
                        if key in row:
                            row[key] = str(work/row[key])
                for filename in contract['files']:
                    (work/filename).write_bytes(struct.pack('<f', 1.0)*131072)
                    rows.extend([dict(type='prefix_hashes', file=str(work/filename),
                                      kind='stub_input_state_not_KV', sha256=['a'*64]*52),
                                 dict(type='resource_point', footprint_point=1)])
                def check(stub=True):
                    (work/'stdout').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    return runner.short_check(work, contract, vector, stub=stub)
                check()
                with self.assertRaisesRegex(ValueError, 'native KV evidence'):
                    check(stub=False)
                hashes = next(r for r in rows if r['type'] == 'prefix_hashes')
                hashes['sha256'].pop()
                with self.assertRaisesRegex(ValueError, 'all52'):
                    check()
                hashes['sha256'].append('a'*64)
                endpoint = next(r for r in rows if r['type'] == 'endpoint')
                endpoint['file'] += '.stale'
                with self.assertRaisesRegex(ValueError, 'contract drift'):
                    check()
                endpoint['file'] = endpoint['file'][:-6]
                duplicate = next((r for r in rows if r['type'] == 'duplicate_verified'), None)
                if duplicate is not None:
                    target = Path(duplicate['file'])
                    target.write_bytes(struct.pack('<f', 2.0)*131072)
                    with self.assertRaisesRegex(ValueError, 'vector mismatch'):
                        check()
                    target.write_bytes(struct.pack('<f', 1.0)*131072)
                target = work/contract['files'][0]
                target.write_bytes(struct.pack('<f', float('nan'))*131072)
                with self.assertRaisesRegex(ValueError, 'nonfinite'):
                    check()
