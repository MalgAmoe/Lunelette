"""Generator admission tests: stub regex, no package import or Unicode scan."""
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


GENERATOR = Path(__file__).with_name('tokenizer_generate_unicode.py')


class CompileBoundary(Exception):
    """Stop before scanning Unicode or writing output."""


class GeneratorTests(unittest.TestCase):
    def exercise(self, version, existing):
        stub = types.ModuleType('regex')
        if version is not None:
            stub.__version__ = version
        calls = []

        def compile_pattern(pattern):
            calls.append(pattern)
            raise CompileBoundary

        stub.compile = compile_pattern
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'unicode.h'
            if existing:
                output.write_bytes(b'preserve existing output\n')
            with patch.dict(sys.modules, {'regex': stub}), patch.object(
                    sys, 'argv', [str(GENERATOR), str(output)]):
                if version == '2026.9.29':
                    with self.assertRaises(CompileBoundary):
                        runpy.run_path(str(GENERATOR), run_name='__main__')
                    self.assertEqual(calls, [r'\A\p{L}\Z'])
                else:
                    with self.assertRaisesRegex(RuntimeError, 'requires regex==2026.9.29'):
                        runpy.run_path(str(GENERATOR), run_name='__main__')
                    self.assertEqual(calls, [])
            if existing:
                self.assertEqual(output.read_bytes(), b'preserve existing output\n')
            else:
                self.assertFalse(output.exists())

    def test_wrong_version(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                self.exercise('wrong-version', existing)

    def test_missing_version(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                self.exercise(None, existing)

    def test_expected_version_reaches_compile(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                self.exercise('2026.9.29', existing)


if __name__ == '__main__':
    unittest.main()
