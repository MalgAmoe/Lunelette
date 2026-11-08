#!/usr/bin/env python3
"""Optional Make/wrapper regression: compile host embedder, never run Metal/model code."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildPaths(unittest.TestCase):
    def test_make_paths_and_publication(self):
        # Relative paths are relative to the Make working directory, not the test.
        with tempfile.TemporaryDirectory(prefix='build-path-', dir=ROOT) as directory:
            base = Path(directory)
            outputs = []
            for absolute in (False, True):
                out = base / ('absolute' if absolute else 'relative')
                build = str(out if absolute else out.relative_to(ROOT))
                target = build + '/metal_shader.h'
                command = ['/usr/bin/make', 'BUILD=' + build, target]
                subprocess.run(command, cwd=ROOT, check=True, timeout=60)
                header = out / 'metal_shader.h'
                expected = subprocess.check_output([
                    str(out / 'embed_shader'), str(ROOT / 'src/metal_kernels.metal')])
                self.assertEqual(header.read_bytes(), expected)
                outputs.append(expected)
                for dependency in ('include/model_facts.h', 'src/metal_kernels.metal'):
                    schedule = subprocess.check_output(command + ['-n', '-W', dependency],
                                                       cwd=ROOT, text=True)
                    self.assertIn('/embed_shader" src/metal_kernels.metal', schedule)
                    if dependency.endswith('model_facts.h'):
                        self.assertIn('tools/embed_shader.c -o', schedule)
                # A helper that writes malformed output then fails must not publish it.
                helper = out / 'embed_shader'
                helper.write_text('#!/bin/sh\nprintf malformed\nexit 1\n')
                helper.chmod(0o755)
                os.utime(helper, (header.stat().st_mtime + 2,) * 2)
                result = subprocess.run(command, cwd=ROOT, timeout=60)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(header.read_bytes(), expected)
                header.unlink()
                result = subprocess.run(command, cwd=ROOT, timeout=60)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(header.exists())
                # The maintained rule intentionally retains failed .tmp evidence.
                self.assertEqual(header.with_suffix('.h.tmp').read_bytes(), b'malformed')
            self.assertEqual(*outputs)

    def test_builders_delegate_before_any_manual_embedding(self):
        import sys
        # Wrapper imports alter interpreter search state; isolate this test even
        # when an external runner discovers it alongside other tooling tests.
        for isolation in (patch.dict(sys.modules),
                          patch.object(sys, 'path', list(sys.path)),
                          patch.object(sys, 'dont_write_bytecode', sys.dont_write_bytecode),
                          patch.dict(os.environ, {'MAKEFLAGS': '--invalid-parent-flag',
                                                  'CC': 'untrusted-parent-cc',
                                                  'CFLAGS': 'untrusted-parent-cflags',
                                                  'CPPFLAGS': 'untrusted-parent-cppflags',
                                                  'LDFLAGS': 'untrusted-parent-ldflags',
                                                  'SDKROOT': 'untrusted-parent-sdk'})):
            isolation.start()
            self.addCleanup(isolation.stop)
        sys.path.insert(0, str(ROOT / 'tools'))
        long = load('build_paths_long', ROOT / 'tools/long_context.py')
        http = load('build_paths_http', ROOT / 'tests/long_context/http/build.py')
        class ReachedMake(Exception):
            pass
        for module in (long, http):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as directory:
                target = Path(directory) / 'fresh'
                calls = []
                def run(argv, **kwargs):
                    calls.append((argv, kwargs))
                    if argv[0] == '/usr/bin/make':
                        self.assertIn('BUILD=' + str(target), argv)
                        self.assertIn('CC=/usr/bin/cc', argv)
                        self.assertEqual(kwargs['timeout'],
                                         30 if argv[-1].endswith('/metal_shader.h') else 180)
                        self.assertEqual(kwargs['cwd'], ROOT)
                        for key in ('MAKEFLAGS', 'CC', 'CFLAGS', 'CPPFLAGS', 'LDFLAGS'):
                            self.assertNotIn(key, kwargs['env'])
                        self.assertEqual(kwargs['env']['SDKROOT'], 'test-query')
                        if module is http:
                            self.assertIn('preexec_fn', kwargs)
                        if str(target / 'metal_runtime.o') in argv:
                            make_calls = [call[0] for call in calls if call[0][0] == '/usr/bin/make']
                            self.assertEqual(make_calls[0][-1], str(target / 'embed_shader'))
                            self.assertEqual(make_calls[1][-1], str(target / 'metal_shader.h'))
                            raise ReachedMake()
                        return subprocess.CompletedProcess(argv, 0)
                    self.assertNotIn('embed_shader', ' '.join(argv))
                    if argv[0] == '/usr/bin/cc':
                        self.assertEqual(argv[1:], ['--version'])
                    text = '/usr/bin/cc' if '--find' in argv else 'test-query'
                    return subprocess.CompletedProcess(argv, 0, text, '')
                with patch.object(module, 'sources', return_value=[]), \
                     patch.object(module, 'pin', return_value={}), \
                     patch.object(module.subprocess, 'run', side_effect=run), \
                     patch.object(module, 'run_build_command', side_effect=run):
                    with self.assertRaises(ReachedMake):
                        module.build(target)
                self.assertFalse((target / 'metal_shader.h').exists())
                self.assertFalse((target / 'receipt.json').exists())
                with self.assertRaises(RuntimeError):
                    module.build(target)
                self.assertEqual(calls[-1][0][0], '/usr/bin/make')


if __name__ == '__main__':
    unittest.main()
