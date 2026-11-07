"""Real subprocess census: no model, tokenizer file, or installed changes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build import HERE


class Tests(unittest.TestCase):
    def capture(self, launcher, code, optimize):
        script = ('import sys; sys.path.insert(0, ' + repr(str(HERE)) + '); import build; ' + code +
                  '; from provider import environment; import json; '
                  'print(json.dumps(dict(environment=environment(), paths=sys.path)))')
        return json.loads(subprocess.check_output(
            [str(launcher), '-I', '-B', *optimize, '-c', script], text=True, timeout=30))

    def test_real_reference_routes(self):
        python = Path(sys.executable)
        resolved = python.resolve()
        historical = (
            'import subprocess,json; paths=json.loads(subprocess.check_output('
            '[' + repr(str(python)) + ', "-I", "-B", "-c", '
            '\'import sys,json; print(json.dumps([p for p in sys.path if p.endswith("site-packages")]))\'],'
            'text=True)); '
            '[sys.path.insert(0,p) for p in paths]; '
            'sys.path.insert(0,' + repr(str(HERE)) + ')')
        shared = 'from reference_environment import bootstrap; bootstrap(' + repr(str(python)) + ')'
        for optimize in ([], ['-O']):
            with self.subTest(optimize=optimize):
                old_child = self.capture(resolved, historical, optimize)
                child = self.capture(resolved, shared, optimize)
                replay = self.capture(resolved, shared, optimize)
                direct = self.capture(python, 'pass', optimize)
                self.assertEqual(old_child, child)
                self.assertEqual(child, replay)
                # Exercise the real replay entry point through its environment
                # gate, stopping at the absent journal before any tokenizer use.
                with tempfile.TemporaryDirectory() as directory:
                    work = Path(directory)
                    (work/'reference-environment.json').write_text(
                        json.dumps(old_child['environment']) + '\n')
                    result = subprocess.run(
                        [str(resolved), '-I', '-B', *optimize, str(HERE/'replay.py'),
                         str(work), 'unused', '--reference-python', str(python)],
                        capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('journal.jsonl', result.stderr)
                    self.assertNotIn('versions drift', result.stderr)
                # The previous replay used the direct venv launcher. Keep the
                # observed discrepancy visible without assuming all venvs omit pip.
                extra = set(child['environment']['packages']) - set(direct['environment']['packages'])
                if 'pip' not in direct['environment']['packages']:
                    self.assertIn('pip', extra)
                    self.assertNotEqual(child['environment'], direct['environment'])

    def test_interpreter_drift_rejected(self):
        result = subprocess.run(
            [str(Path(sys.executable).resolve()), '-I', '-B', '-c',
             'import sys; sys.path.insert(0,' + repr(str(HERE)) + '); '
             'from reference_environment import bootstrap; bootstrap("/not/the/interpreter")'],
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('executable drift', result.stderr)
