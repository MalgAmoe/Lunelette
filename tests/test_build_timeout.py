#!/usr/bin/env python3
"""Real Make shader calls through both builders; no model or GPU execution."""
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildTimeout(unittest.TestCase):
    def setUp(self):
        for isolation in (patch.dict(sys.modules),
                          patch.object(sys, 'path', list(sys.path)),
                          patch.object(sys, 'dont_write_bytecode', sys.dont_write_bytecode)):
            isolation.start()
            self.addCleanup(isolation.stop)
        sys.path.insert(0, str(ROOT/'tools'))

    def test_exit_before_notification_registration(self):
        import select
        if not hasattr(select, 'kqueue'):
            self.skipTest('Darwin kqueue registration race')
        command = load('timeout_command', ROOT/'tests/long_context/build_command.py')
        real_kqueue = select.kqueue

        class DelayedQueue:
            def __init__(self):
                self.queue = real_kqueue()

            def control(self, changes, *args):
                if changes:
                    time.sleep(.1)  # Allow the child to become an unreaped zombie.
                return self.queue.control(changes, *args)

            def close(self):
                self.queue.close()

        with tempfile.TemporaryFile() as log, patch.object(select, 'kqueue', DelayedQueue):
            for status in (0, 7):
                with self.subTest(status=status):
                    result = command.run_build_command(['/bin/sh', '-c', 'exit '+str(status)],
                                                       timeout=2, stdout=log, stderr=log)
                    self.assertEqual(result.returncode, status)

    def test_signal_mask_restored_if_notification_close_fails(self):
        import select
        if not hasattr(select, 'kqueue'):
            self.skipTest('Darwin notification cleanup')
        command = load('timeout_command', ROOT/'tests/long_context/build_command.py')
        real_kqueue = select.kqueue

        class CloseFailure:
            def __init__(self):
                self.queue = real_kqueue()

            def control(self, *args):
                return self.queue.control(*args)

            def close(self):
                self.queue.close()
                raise OSError('injected notification close failure')

        original = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        try:
            with tempfile.TemporaryFile() as log, patch.object(select, 'kqueue', CloseFailure):
                with self.assertRaisesRegex(OSError, 'injected notification close failure'):
                    command.run_build_command(['/bin/sleep', '.1'], timeout=2,
                                              stdout=log, stderr=log)
            self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), original)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, original)

    def test_shader_lifecycle(self):
        modules = (load('timeout_long', ROOT/'tools/long_context.py'),
                   load('timeout_http', ROOT/'tests/long_context/http/build.py'))
        real_run = subprocess.run
        # An unrelated, finite process in another session must survive cleanup.
        unrelated = subprocess.Popen(['/bin/sleep', '120'], start_new_session=True)
        self.addCleanup(lambda: (unrelated.terminate(), unrelated.wait()))
        for module in modules:
            for old in (False, True):
                for mode in ('success', 'nonzero', 'timeout', 'cancel'):
                    with self.subTest(builder=module.__name__, old=old, mode=mode), \
                         tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as tmp:
                        target = Path(tmp)/'build'
                        marker = target/'late'
                        pidfile = target/'pid'
                        groupfile = target/'group'
                        class Finished(Exception):
                            pass
                        def dispatch(argv, **kwargs):
                            if argv[0] != '/usr/bin/make':
                                text = '/usr/bin/cc' if '--find' in argv else 'query'
                                return subprocess.CompletedProcess(argv, 0, text, '')
                            if argv[-1] == str(target/'embed_shader'):
                                header = target/'metal_shader.h'
                                if old:
                                    header.write_text('old')
                                helper = target/'embed_shader'
                                delay = 2 if mode in ('timeout', 'cancel') else 0
                                # Ignore TERM to exercise forced descendant teardown. Finite
                                # even on broken baseline; the retained fd exposes late writes.
                                helper.write_text('#!/bin/sh\ntrap "" TERM\necho $$ > "'+str(pidfile)+'"\n'
                                                  '/bin/ps -o pgid= -p $$ > "'+str(groupfile)+'"\n'
                                                  'echo started >&2\nsleep '+str(delay)+'\n'
                                                  'echo late\necho done > "'+str(marker)+'"\n'
                                                  'exit '+('7' if mode == 'nonzero' else '0')+'\n')
                                helper.chmod(0o755)
                                os.utime(helper, (time.time()+2,)*2)
                                return subprocess.CompletedProcess(argv, 0)
                            if argv[-1] != str(target/'metal_shader.h'):
                                raise Finished()
                            kwargs['timeout'] = .8 if mode == 'timeout' else 5
                            if mode == 'cancel':
                                signal.setitimer(signal.ITIMER_REAL, .8)
                            try:
                                return real_command(argv, **kwargs)
                            finally:
                                signal.setitimer(signal.ITIMER_REAL, 0)
                        def interrupt(signum, frame):
                            raise KeyboardInterrupt()
                        previous = signal.signal(signal.SIGALRM, interrupt)
                        # Baseline has subprocess.run; fixed builders expose run_build_command.
                        real_command = getattr(module, 'run_build_command', real_run)
                        seam = 'run_build_command' if hasattr(module, 'run_build_command') else None
                        with patch.object(module, 'sources', return_value=[]), \
                             patch.object(module, 'pin', return_value={}), \
                             patch.object(module.subprocess, 'run', side_effect=dispatch):
                            try:
                                with patch.object(module, seam, side_effect=dispatch) if seam else nullcontext():
                                    expected = (KeyboardInterrupt if mode == 'cancel' else
                                                subprocess.TimeoutExpired if mode == 'timeout' else
                                                (subprocess.CalledProcessError if module is modules[0] else RuntimeError)
                                                if mode == 'nonzero' else Finished)
                                    with self.assertRaises(expected):
                                        module.build(target)
                            finally:
                                signal.signal(signal.SIGALRM, previous)
                        header = target/'metal_shader.h'
                        if mode in ('timeout', 'cancel'):
                            # Snapshot at return, then wait beyond the finite broken helper.
                            snapshot = {p.name: p.read_bytes() for p in target.iterdir() if p.is_file()}
                            state = real_run(['/bin/ps', '-o', 'stat=', '-p', pidfile.read_text().strip()],
                                             capture_output=True, text=True).stdout.strip()
                            group = int(groupfile.read_text())
                            census = real_run(['/bin/ps', '-axo', 'pgid=,stat='],
                                              capture_output=True, text=True, check=True).stdout
                            live = [line for line in census.splitlines()
                                    if int(line.split()[0]) == group and
                                    not line.split()[1].startswith('Z')]
                            time.sleep(2.2)
                            self.assertNotEqual(group, os.getpgrp())
                            self.assertEqual(live, [])
                            self.assertFalse(state and not state.startswith('Z'), state)
                            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in target.iterdir() if p.is_file()})
                            self.assertFalse(marker.exists())
                        if mode != 'success':
                            self.assertEqual(header.read_text() if header.exists() else None, 'old' if old else None)
                        else:
                            self.assertEqual(header.read_text(), 'late\n')
                        self.assertIn(b'started', (target/'build.log').read_bytes())
                        self.assertIsNone(unrelated.poll())


if __name__ == '__main__':
    unittest.main()
