import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import supervisor as s


class Tests(unittest.TestCase):
    def config(self, mode='ok'):
        return dict(mode='fake-only', pins=s.source_pins(),
                    jobs=[dict(id='one', fixture=mode, owners=1, files={'result': 64})],
                    artifact_cap=s.MiB, build_cap=0, headroom=s.MiB,
                    process_seconds=.15, total_seconds=5)

    def test_short_writes_and_no_progress(self):
        for failure in (False, 0, None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=s.HERE) as directory:
                pump = s.Streams(Path(directory), dict(stdout=5, stderr=5, diagnostics=5))
                original = pump.files['stdout']
                class ShortWriter:
                    def write(self, data):
                        if failure is not False and pump.counts['stdout'] >= 2:
                            return failure
                        return original.write(data[:2])
                pump.files['stdout'] = ShortWriter()
                try:
                    if failure is False:
                        pump.write('stdout', b'abcdef')
                        self.assertEqual(pump.counts['stdout'], 5)
                        self.assertTrue(pump.truncated)
                        self.assertEqual((Path(directory)/'stdout').read_bytes(), b'abcde')
                    else:
                        with self.assertRaises(s.Stop):
                            pump.write('stdout', b'abcde')
                        self.assertEqual(pump.counts['stdout'], 2)
                        self.assertEqual((Path(directory)/'stdout').read_bytes(), b'ab')
                finally:
                    pump.files['stdout'] = original
                    pump.close()

    def test_stream_long_raw_and_terminal_diagnostics(self):
        diag = b'SUPERVISOR_DIAG ' + json.dumps(dict(finite=1, expected=1,
            nonfinite=0, state_valid=True, memory_error=False,
            classification='pass')).encode()
        ledger = json.dumps({'layers': [{'finite': 8192, 'expected': 8192,
            'name': 'layer-' + str(i)} for i in range(500)]}).encode()
        self.assertGreater(len(ledger), 26000)
        for name in ('stdout', 'stderr'):
            with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
                pump = s.Streams(Path(directory), dict(stdout=65536, stderr=65536, diagnostics=16384))
                try:
                    raw = ledger + b'\n' + diag + b'\n'
                    for offset in range(0, len(raw), 137):
                        chunk = raw[offset:offset+137]
                        pump.write(name, chunk)
                        pump.feed(name, chunk)
                        self.assertLessEqual(len(pump.lines[name]), 8192)
                    self.assertIsNone(pump.fatal)
                    self.assertEqual(pump.classification, 'pass')
                    self.assertEqual((Path(directory)/name).read_bytes(), raw)
                    pump.diagnostic(diag.replace(b'"pass"', b'"known-independent-mismatch"'))
                    self.assertEqual(pump.classification, 'pass')
                    self.assertEqual(pump.fatal, 'duplicate terminal diagnostic')
                finally:
                    pump.close()

    def test_oversized_diagnostic_complete_and_partial(self):
        for complete in (False, True):
            for chunk_size in (137, 16384):
                with self.subTest(complete=complete, chunk_size=chunk_size), tempfile.TemporaryDirectory(dir=s.HERE) as directory:
                    pump = s.Streams(Path(directory), dict(stdout=32768, stderr=32768, diagnostics=16384))
                    try:
                        raw = b'SUPERVISOR_DIAG ' + b'x'*9000 + (b'\n' if complete else b'')
                        for offset in range(0, len(raw), chunk_size):
                            pump.feed('stdout', raw[offset:offset+chunk_size])
                        self.assertEqual(pump.fatal, 'diagnostic line bound exceeded')
                        self.assertIsNone(pump.classification)
                        self.assertLessEqual(len(pump.lines['stdout']), 8192)
                    finally:
                        pump.close()

    def test_zero_coverage_and_manifest_source_pin(self):
        self.assertIn('manifest.py', s.source_pins())
        config = self.config(); config['pins']['manifest.py'] = 'drift'
        with self.assertRaises(s.Stop): s.validate(config)
        with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
            pump = s.Streams(Path(directory), dict(stdout=4096, stderr=4096, diagnostics=4096))
            try:
                pump.diagnostic(b'SUPERVISOR_DIAG ' + json.dumps(dict(finite=0,
                    expected=0, nonfinite=0, state_valid=True, memory_error=False,
                    classification='pass')).encode())
                self.assertIsNotNone(pump.fatal)
            finally:
                pump.close()

    def test_success_no_retry(self):
        with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
            self.assertEqual(s.run_fake_campaign(directory, self.config())['launches'], 1)
            with self.assertRaises(s.Stop):
                s.run_fake_campaign(directory, self.config())
            self.assertEqual(json.loads((Path(directory) / 'one.launch.json').read_text())['state'], 'launch-intent-no-retry')

    def test_failures_and_cleanup(self):
        for mode in ('fail', 'sleep', 'extra', 'symlink', 'leak'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=s.HERE) as directory:
                with self.assertRaises(s.Stop):
                    s.run_fake_campaign(directory, self.config(mode))
                result = json.loads((Path(directory) / 'one.result.json').read_text())
                self.assertNotEqual(result['state'], 'complete')
                with self.assertRaises(s.Stop):
                    s.run_fake_campaign(directory, self.config(mode))

    def test_validation(self):
        cases = []
        for key, value in [('mode', 'real'), ('extra', True), ('process_seconds', 1201), ('total_seconds', 14401), ('artifact_cap', 65*s.MiB)]:
            config = self.config(); config[key] = value; cases.append(config)
        config = self.config(); config['pins']['fake_child.py'] = 'bad'; cases.append(config)
        for name in ('../escape', '/tmp/escape', 'a/b', '..'):
            config = self.config(); config['jobs'][0]['id'] = name; cases.append(config)
        config = self.config(); config['jobs'][0]['files']['extra'] = 1; cases.append(config)
        config = self.config(); config['jobs'] = [dict(config['jobs'][0], id='j'+str(i), owners=3) for i in range(17)]; cases.append(config)
        config = self.config(); config['jobs'] *= 33; cases.append(config)
        for config in cases:
            with self.assertRaises(s.Stop): s.validate(config)
        with self.assertRaises(s.Stop): s.real_launch(execution_authorized=True)

    def test_exclusive_and_symlink(self):
        with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
            root = Path(directory)
            with (root/'ownership').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(s.Stop): s.run_fake_campaign(root, self.config())
            (root/'link').symlink_to(root, target_is_directory=True)
            with self.assertRaises(s.Stop): s.run_fake_campaign(root/'link', self.config())

    def test_disk_and_spawn_failure(self):
        with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
            with patch.object(s.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 0})()):
                with self.assertRaises(s.Stop): s.run_fake_campaign(directory, self.config())
            self.assertFalse((Path(directory)/'one.launch.json').exists())
        with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
            with patch.object(s.subprocess, 'Popen', side_effect=OSError('injected')):
                with self.assertRaises(OSError): s.run_fake_campaign(directory, self.config())
            self.assertTrue((Path(directory)/'one.launch.json').exists())
            with self.assertRaises(s.Stop): s.run_fake_campaign(directory, self.config())

    def test_census(self):
        with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
            root = Path(directory); (root/'result').write_bytes(b'123')
            with self.assertRaises(s.Stop): s.census(root, {'result': 2}, 10)
            with self.assertRaises(s.Stop): s.census(root, {'result': 4}, 2)
            with self.assertRaises(s.Stop): s.census(root, {}, 10)

    def test_source_drift_and_total_stop(self):
        original = s.source_pins
        config = self.config()
        with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
            with patch.object(s, 'source_pins', side_effect=[original(), {'drift': 'yes'}]):
                with self.assertRaises(s.Stop): s.run_fake_campaign(directory, config)
            self.assertFalse((Path(directory)/'one.launch.json').exists())
        config = self.config('sleep'); config['total_seconds'] = .05
        with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
            with self.assertRaises(s.Stop): s.run_fake_campaign(directory, config)


class FutureTests(unittest.TestCase):
    def test_failed_child_validation_before_acceptance(self):
        from test_manifest import specimen, pin
        import subprocess
        for mode in ('badexit', 'missing', 'reject', 'exception', 'leak', 'unsafe'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=s.HERE) as directory:
                base = Path(directory); root = base/'campaign'; root.mkdir()
                entry = base/'tiny.py'
                script = "from pathlib import Path\nimport sys\nprint('failure evidence', flush=True)\n"
                if mode != 'missing':
                    script += "Path('result').write_text('partial64')\n"
                if mode == 'unsafe':
                    script += "Path('unknown').write_text('x')\n"
                if mode == 'leak':
                    script += "import subprocess\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                script += 'sys.exit(%d)\n' % (1 if mode in ('badexit', 'reject', 'exception') else 0)
                entry.write_text(script)
                spec = specimen(2)
                for job in spec['jobs']:
                    job.update(argv=[str(Path(s.sys.executable).resolve()), '-I', '-B', str(entry)],
                               binary=pin(entry), sources=[pin(entry)], cwd=str(root/job['id']))
                seen = []
                def validator(job, work, code):
                    seen.append(code)
                    self.assertFalse(s.alive(children[-1].pid))
                    self.assertIsNotNone(children[-1].returncode)
                    if mode != 'missing':
                        self.assertEqual((work/'result').read_text(), 'partial64')
                    # Metrics are represented by caller-owned data, not new unreserved files.
                    metrics.append({'exit': code, 'partial': mode != 'missing'})
                    if mode == 'exception': raise ValueError('offline metrics failed')
                    return 'reject' if mode == 'reject' else 'pass'
                children = []; metrics = []
                def backend(*a, **kw):
                    child = subprocess.Popen(*a, **kw); children.append(child); return child
                with self.assertRaises((s.Stop, ValueError)):
                    s._run_future_test(root, spec, validator, backend=backend,
                        guard=s._FUTURE_TEST_BACKEND, phase_prepare=lambda *a: None)
                self.assertEqual(len(seen), 0 if mode in ('leak', 'unsafe') else 1)
                self.assertEqual(len(metrics), len(seen))
                self.assertEqual(len(children), 1)
                self.assertFalse((root/(spec['jobs'][1]['id']+'.launch.json')).exists())
                receipt = json.loads((root/(spec['jobs'][0]['id']+'.result.json')).read_text())
                self.assertEqual(receipt['state'], 'stopped')
                if mode == 'badexit': self.assertIn('child exit 1', receipt['reason'])
                if mode == 'missing': self.assertIn('missing required output', receipt['reason'])
                self.assertIn(b'failure evidence', (root/spec['jobs'][0]['id']/'stdout').read_bytes())
                self.assertEqual(json.loads((root/'campaign.result.json').read_text())['state'], 'stopped')

    def test_callback_wall_deadlines(self):
        from test_manifest import specimen, pin
        import subprocess
        import time
        for phase in ('prepare', 'validator', 'pin'):
            for hanging in (False, True):
                with self.subTest(phase=phase, hanging=hanging), tempfile.TemporaryDirectory(dir=s.HERE) as directory:
                    base = Path(directory)
                    root = base/'campaign'; root.mkdir()
                    entry = base/'tiny.py'
                    entry.write_text("from pathlib import Path\nPath('result').write_text('tiny')\n")
                    spec = specimen(2)
                    spec['total_seconds'] = .3
                    for job in spec['jobs']:
                        job.update(argv=[str(Path(s.sys.executable).resolve()), '-I', '-B', str(entry)],
                                   binary=pin(entry), sources=[pin(entry)], cwd=str(root/job['id']))
                    calls = []
                    native = subprocess.Popen
                    def backend(*args, **kwargs):
                        calls.append(args)
                        return native(*args, **kwargs)
                    def block():
                        if hanging:
                            while True:
                                time.sleep(10)
                        time.sleep(10)
                    def prepare(*args):
                        if phase == 'prepare': block()
                    def validator(*args):
                        block()
                        return 'pass'
                    verify = s.manifest_api.verify
                    def slow_pin(*args):
                        block()
                        return verify(*args)
                    started = time.monotonic()
                    with patch.object(s.manifest_api, 'verify', slow_pin if phase == 'pin' else verify):
                        with self.assertRaisesRegex(s.Stop, 'total wall deadline'):
                            s._run_future_test(root, spec, validator, backend=backend,
                                guard=s._FUTURE_TEST_BACKEND, phase_prepare=prepare)
                    self.assertLess(time.monotonic()-started, 2)
                    self.assertEqual(len(calls), 1 if phase == 'validator' else 0)
                    self.assertFalse((root/(spec['jobs'][1]['id']+'.launch.json')).exists())
                    receipt = json.loads((root/'campaign.result.json').read_text())
                    self.assertEqual(receipt['state'], 'stopped')
                    self.assertEqual(receipt['real_gate'], 'CLOSED')
                    if phase == 'validator':
                        self.assertEqual(json.loads((root/(spec['jobs'][0]['id']+'.result.json')).read_text())['state'], 'stopped')

    def test_alarm_ownership_and_restoration(self):
        import signal
        import time
        previous = signal.getsignal(signal.SIGALRM)
        def handler(*args): pass
        signal.signal(signal.SIGALRM, handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, 10, 10)
            with self.assertRaisesRegex(s.Stop, 'existing ITIMER_REAL'):
                with s.wall_deadline(time.monotonic()+.1): self.fail('entered')
            remaining, interval = signal.getitimer(signal.ITIMER_REAL)
            self.assertGreater(remaining, 9)
            self.assertEqual(interval, 10)
            self.assertIs(signal.getsignal(signal.SIGALRM), handler)
            signal.setitimer(signal.ITIMER_REAL, 0)
            for fail in (False, True):
                try:
                    with s.wall_deadline(time.monotonic()+1):
                        if fail: raise ValueError('callback')
                except ValueError:
                    pass
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
                self.assertIs(signal.getsignal(signal.SIGALRM), handler)
            with self.assertRaises(s.Stop):
                with s.wall_deadline(time.monotonic()+.02): time.sleep(1)
            self.assertIs(signal.getsignal(signal.SIGALRM), handler)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def test_pinned_command_and_validator_rejection(self):
        from test_manifest import specimen, pin
        import subprocess
        for reject in (False, True):
            with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
                base = Path(directory)
                root = base/'campaign'; root.mkdir()
                entry = base/'tiny.py'
                entry.write_text("from pathlib import Path\nimport resource\nif resource.getrlimit(resource.RLIMIT_CORE) != (0, 0): raise RuntimeError('core limit')\nif resource.getrlimit(resource.RLIMIT_FSIZE) != (64, 64): raise RuntimeError('file limit')\nif Path('result').read_text() != 'prepared input': raise RuntimeError('prepare missing')\nPath('result').write_text('tiny')\n")
                data = base/'input'; data.write_text('tiny input')
                spec = specimen(2)
                for job in spec['jobs']:
                    job.update(argv=[str(Path(s.sys.executable).resolve()), '-I', '-B', str(entry)],
                               binary=pin(entry), sources=[pin(entry)], inputs=[pin(data)],
                               cwd=str(root/job['id']), env={'ONLY_PINNED':'yes'})
                calls = []; validated = []; children = []
                native = subprocess.Popen
                def spy(argv, **kwargs):
                    job = spec['jobs'][len(calls)]
                    self.assertEqual(argv, job['argv'])
                    self.assertEqual(kwargs['env'], job['env'])
                    self.assertEqual(kwargs['cwd'], job['cwd'])
                    self.assertEqual((Path(kwargs['cwd'])/'result').read_text(), 'prepared input')
                    calls.append(argv)
                    child = native(argv, **kwargs)  # Only temporary tiny script.
                    children.append(child)
                    return child
                def prepare(job, work):
                    self.assertFalse((root/(job['id']+'.launch.json')).exists())
                    (work/'result').write_text('prepared input')
                def validator(job, work, code):
                    self.assertIsNotNone(children[-1].returncode)
                    self.assertEqual(code, 0)
                    self.assertEqual((work/'result').read_text(), 'tiny')
                    validated.append(job['id'])
                    return 'reject' if reject else 'pass'
                with patch.object(s.subprocess, 'Popen', side_effect=AssertionError('unguarded spawn')):
                    if reject:
                        with self.assertRaisesRegex(s.Stop, 'phase validator rejected'):
                            s._run_future_test(root, spec, validator, backend=spy, guard=s._FUTURE_TEST_BACKEND, phase_prepare=prepare)
                        self.assertFalse((root/(spec['jobs'][1]['id']+'.launch.json')).exists())
                    else:
                        result = s._run_future_test(root, spec, validator, backend=spy, guard=s._FUTURE_TEST_BACKEND, phase_prepare=prepare)
                        self.assertEqual(result['launches'], 2)
                self.assertEqual(len(calls), 1 if reject else 2)
                self.assertEqual(len(validated), len(calls))
                with self.assertRaises(s.Stop):
                    s.real_launch(execution_authorized=True)

    def test_prepare_failure_and_census_no_retry(self):
        from test_manifest import specimen
        for unknown in (False, True):
            with self.subTest(unknown=unknown), tempfile.TemporaryDirectory(dir=s.HERE) as directory:
                root = Path(directory)
                spec = specimen()
                spec['jobs'][0]['cwd'] = str(root/spec['jobs'][0]['id'])
                def prepare(job, work):
                    if unknown:
                        (work/'unreserved').write_text('x')
                    else:
                        raise RuntimeError('prepare failed')
                backend = lambda *a, **k: self.fail('must not spawn')
                with patch.object(s.subprocess, 'Popen', side_effect=AssertionError('must not spawn')):
                    with self.assertRaises((RuntimeError, s.Stop)):
                        s._run_future_test(root, spec, lambda *a:'pass', backend=backend,
                                           guard=s._FUTURE_TEST_BACKEND, phase_prepare=prepare)
                self.assertFalse(list(root.glob('*.launch.json')))
                self.assertEqual((root/'ownership').read_text(), 'claimed\n')
                receipt = json.loads((root/'campaign.result.json').read_text())
                self.assertEqual(receipt['state'], 'stopped')
                self.assertIn('unknown output' if unknown else 'prepare failed', receipt['reason'])
                with self.assertRaisesRegex(s.Stop, 'retry forbidden'):
                    s._run_future_test(root, spec, lambda *a:'pass', backend=backend,
                                       guard=s._FUTURE_TEST_BACKEND, phase_prepare=prepare)

    def test_guard_cwd_and_inputs(self):
        from test_manifest import specimen
        with tempfile.TemporaryDirectory(dir=s.HERE) as directory:
            root = Path(directory)
            spec = specimen()
            backend = lambda *a, **k: self.fail('must not spawn')
            with self.assertRaisesRegex(s.Stop, 'mandatory phase prepare'):
                s._run_future_test(root, spec, lambda *a:'pass', backend=backend, guard=s._FUTURE_TEST_BACKEND)
            with self.assertRaisesRegex(s.Stop, 'guard'):
                s._run_future_test(root, spec, lambda *a:'pass', backend=backend, guard=object(), phase_prepare=lambda *a: None)
            with self.assertRaisesRegex(s.Stop, 'cwd'):
                s._run_future_test(root, spec, lambda *a:'pass', backend=backend, guard=s._FUTURE_TEST_BACKEND, phase_prepare=lambda *a: None)
            spec['jobs'][0]['cwd'] = str(root/spec['jobs'][0]['id'])
            spec['jobs'][0]['inputs'] = [dict(path=str(root/'absent'), sha256='0'*64, size=1)]
            with self.assertRaises(s.Stop):
                s._run_future_test(root, spec, lambda *a:'pass', backend=backend, guard=s._FUTURE_TEST_BACKEND, phase_prepare=lambda *a: None)
            self.assertFalse(list(root.glob('*.launch.json')))


if __name__ == '__main__':
    unittest.main()
