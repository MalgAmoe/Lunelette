"""Admission tests use tiny receipts, never weights or a compiler."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('long_entry', ROOT/'tools/long_context.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class AdmissionTests(unittest.TestCase):
    def test_entrypoint_symlink_ancestors_and_existing_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root/'real').mkdir()
            (root/'link').symlink_to(root/'real', target_is_directory=True)
            base = [sys.executable, '-B', str(ROOT/'tools/long_context.py')]
            cases = [(['--build', str(root/'link/new')], 'symlink'),
                     (['--build', str(root/'real')], 'fresh directory')]
            for option in ('--out', '--weights', '--tokenizer', '--build'):
                args = dict(build=str(root/'build'), out=str(root/'out'),
                            weights=str(root/'model'), tokenizer=str(root/'tokenizer'))
                args[option[2:]] = str(root/'link/new')
                cases.append(([item for k, v in args.items() for item in ('--'+k, v)], 'symlink'))
            cases.append((['--build', str(root/'build'), '--out', str(root/'real'),
                           '--weights', str(root/'model'), '--tokenizer', str(root/'tokenizer')], 'fresh directory'))
            for args, message in cases:
                with self.subTest(args=args):
                    result = subprocess.run(base+args, capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
            self.assertFalse((root/'real/new').exists())
            self.assertFalse((root/'out').exists())

    def test_tiny_receipt_source_product_and_stale_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            product = root/'binary'
            product.write_bytes(b'tiny')
            receipt = dict(sources=['current'], products=[entry.pin(product)])
            def save():
                (root/'receipt.json').write_text(json.dumps(receipt))
            save()
            with patch.object(entry, 'sources', return_value=['current']):
                self.assertEqual(entry.verify_build(root), receipt)
                product.write_bytes(b'evil')
                with self.assertRaisesRegex(RuntimeError, 'digest drift'):
                    entry.verify_build(root)
                product.write_bytes(b'tiny')
                receipt['sources'] = ['old']
                save()
                with self.assertRaisesRegex(RuntimeError, 'source drift'):
                    entry.verify_build(root)
                receipt['sources'] = ['current']
                receipt['products'][0]['path'] = str(root/'missing')
                save()
                with self.assertRaisesRegex(RuntimeError, 'size drift'):
                    entry.verify_build(root)
                (root/'receipt.json').write_text('{}')
                with self.assertRaises(KeyError):
                    entry.verify_build(root)
