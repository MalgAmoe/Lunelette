import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import build
from campaign_http import compose, CELL
import manifest


class Tests(unittest.TestCase):
    def test_receipts_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            target = root/'build'
            target.mkdir()
            binary = target/'observing-http'
            binary.write_bytes(b'not executed')
            source = root/'source'
            source.write_bytes(b'source')
            sources = [build.pin(source)]
            compiler = build.pin(Path(sys.executable).resolve())
            receipt = dict(schema='http-current-build-v1', sources=sources,
                           compiler=compiler, products=[build.pin(binary)])
            (target/'receipt.json').write_text(json.dumps(receipt))
            weights, tokenizer = root/'weights', root/'tokenizer'
            weights.write_bytes(b'never opened as a model')
            tokenizer.write_bytes(b'never decoded')
            with patch.object(build, 'sources', return_value=sources):
                self.assertEqual(build.verify(target), receipt)
                spec = compose(root/'out', target, Path(sys.executable), weights, tokenizer, 12345, receipt)
                manifest.validate(spec)
                job = spec['jobs'][0]
                self.assertEqual(job['id'], CELL)
                self.assertEqual(job['owners'], 1)
                self.assertEqual(job['prerequisites'], [])
                self.assertEqual(job['timeout'], 1200)
                binary.write_bytes(b'corrupt product')
                with self.assertRaises(manifest.ManifestError):
                    build.verify(target)
                binary.write_bytes(b'not executed')
                extra = target/'unexpected'
                extra.touch()
                with self.assertRaisesRegex(RuntimeError, 'inventory drift'):
                    build.verify(target)
                extra.unlink()
                with patch.object(build, 'sources', return_value=[]):
                    with self.assertRaisesRegex(RuntimeError, 'source drift'):
                        build.verify(target)
                link = root/'alias'
                link.symlink_to(target, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeError, 'symlink'):
                    build.verify(link)
                with self.assertRaises(RuntimeError):
                    build.build(target)
                with self.assertRaisesRegex(RuntimeError, 'source tree'):
                    build.build(build.ROOT/'http-forbidden-build')
                import long_context_http as cli
                with patch.object(cli, 'python_check', return_value=Path(sys.executable)), \
                     patch.object(cli, 'verify', return_value=receipt), \
                     patch.object(sys, 'argv', ['http', '--python', str(sys.executable),
                           '--build', str(target), '--out', str(build.ROOT/'http-forbidden-output'),
                           '--weights', str(weights), '--tokenizer', str(tokenizer), '--port', '12345']), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        cli.main()
                    self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
