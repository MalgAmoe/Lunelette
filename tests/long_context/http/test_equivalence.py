"""Original schedule oracle with injected evidence; not C/model correctness."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import driver
from test_driver import Mock


def compare(original, promoted):
    driver.require(original == promoted, 'ordered schedule/evidence transcript differs')


class Tests(unittest.TestCase):
    def test_ordered_transcripts(self):
        original_path = Path(__file__).parent/'original/driver.py'
        pins = json.loads((Path(__file__).parent/'original-sha256.json').read_text())
        self.assertEqual(hashlib.sha256(original_path.read_bytes()).hexdigest(), pins['driver.py'])
        spec = importlib.util.spec_from_file_location('original_driver', original_path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        # Only historical path/pin machinery is replaced. Protocol/schedule,
        # validators, branch selection and journal records remain original.
        original.verify_pins = driver.verify_pins
        original.frozen = driver.frozen
        original.PREP = driver.PREP
        original.HERE = driver.PREP  # original negative.json lookup
        for branch in ('partial', 'eos', 'utf8'):
            for block in (128, 256):
                receipts = []
                for module in (original, driver):
                    with tempfile.TemporaryDirectory() as td:
                        mock = Mock(branch, block)
                        requests = []
                        request = mock.request
                        def capture(method, path, body=None, declared=None):
                            requests.append((method, path, body, declared))
                            return request(method, path, body, declared)
                        mock.request = capture
                        journal = module.Journal(Path(td)/'journal')
                        try:
                            module.Driver(mock, journal, mock, block,
                                          'resident-fp32', 'custom').run()
                        finally:
                            journal.close()
                        receipts.append((requests, journal.path.read_bytes(), mock.ids,
                                         mock.history, mock.trace, mock.points, mock.finish()))
                with self.subTest(branch=branch, block=block):
                    compare(receipts[0], receipts[1])
                    # Prove each transcript component matters to comparison.
                    for index in range(len(receipts[0])):
                        bad = list(receipts[1])
                        bad[index] = None
                        with self.assertRaises(ValueError):
                            compare(receipts[0], tuple(bad))


if __name__ == '__main__':
    unittest.main()
