"""Model-free tests; optional existing protocol server argument."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from interactive_workload import Adapter, Output, command
from real_workload import Service

SERVER = (sys.argv.pop() if __name__ == '__main__' and len(sys.argv) > 1
          and not sys.argv[-1].startswith('-') else None)


class Fake:
    def __init__(self):
        self.started = threading.Event()
        self.done = threading.Event()

    def request(self, method, path, body=None):
        if path.endswith('/cancel'):
            self.done.set()
        return 200, {'session_id': 1}

    def generate(self, session, identity, budget, on_event=None):
        self.started.set()
        self.done.wait(2)
        on_event({'event': 'text_delta', 'text': 'hi'})
        return {'event': 'cancelled', 'generated_tokens': 1}


class Tests(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(command('{"id":"x","op":"generate","budget":1}')['budget'], 1)
        for raw in ['[]', '{"op":"quit","id":"x","id":"y"}',
                    '{"id":"x","op":"generate","budget":true}',
                    '{"id":"x","op":"generate","budget":257}',
                    '{"id":"x","op":"quit","extra":0}',
                    '{"id":"../x","op":"quit"}',
                    json.dumps({'id': 'x', 'op': 'message', 'content': 'x' * 4097})]:
            with self.assertRaises((ValueError, TypeError)):
                command(raw)

    def test_concurrency_cancel_budget_ids(self):
        fake, events = Fake(), []
        adapter = Adapter(fake, events.append, 2)
        adapter.dispatch({'op': 'create', 'id': 'a'})
        adapter.dispatch({'op': 'generate', 'id': 'b', 'budget': 2})
        self.assertTrue(fake.started.wait(1))
        with self.assertRaisesRegex(ValueError, 'busy'):
            adapter.dispatch({'op': 'reset', 'id': 'c'})
        adapter.dispatch({'op': 'inspect', 'id': 'd'})
        adapter.dispatch({'op': 'cancel', 'id': 'e'})
        adapter.worker.join(2)
        self.assertFalse(adapter.worker.is_alive())
        self.assertIsNone(adapter.fatal)
        self.assertTrue(any(e['event'] == 'text' for e in events))
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            adapter.dispatch({'op': 'continue', 'id': 'f', 'budget': 1})
        with self.assertRaisesRegex(ValueError, 'already used'):
            adapter.dispatch({'op': 'inspect', 'id': 'a'})

    def test_tool_schema_mapping(self):
        from interactive_workload import tool_operation
        for profile, answer in [('calculator', 43), ('investigation', '19 [LARCH-STOCK:P1]'),
                                ('observation', 'pump-A 84 ALERT')]:
            value = command(json.dumps({'id': 't', 'op': 'tools', 'profile': profile,
                'budget': 64, 'max_calls': 1, 'max_generated': 128,
                'wall_seconds': 30, 'answer': answer}))
            mapped = tool_operation(value)
            self.assertEqual(mapped['op'], 'native_' + ('tool' if profile == 'calculator'
                                                      else profile) + '_loop')
            self.assertNotIn('profile', mapped)

    def test_checkpoint_preflight(self):
        fake = Fake()
        fake.info = {'checkpoints': False, 'prefix_reuse': 'legacy'}
        fake.control = Mock(side_effect=RuntimeError('corrupt control'))
        adapter = Adapter(fake, lambda e: None, 2)
        adapter.session = 1
        with self.assertRaisesRegex(ValueError, 'not supported'):
            adapter.dispatch({'id': 'c', 'op': 'checkpoint'})
        fake.info.update(checkpoints=True, prefix_reuse='on')
        with self.assertRaisesRegex(ValueError, 'no published'):
            adapter.dispatch({'id': 'r', 'op': 'restore'})
        fake.control.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'corrupt'):
            adapter.dispatch({'id': 'bad', 'op': 'checkpoint'})
        adapter.dispatch({'id': 'i', 'op': 'inspect'})

    def test_tool_transition_cancel(self):
        fake, events = Fake(), []
        fake.native = True
        fake.info = {'tools': True, 'conversation_protocol': 'native-calculator-v13'}
        fake.absolute_deadline = 12345
        published, release = threading.Event(), threading.Event()
        cancels = []
        original = fake.request
        def request(method, path, body=None):
            if path.endswith('/cancel'):
                cancels.append(path)
                if len(cancels) == 1:
                    return 404, {'error': 'not admitted'}
            return original(method, path, body)
        fake.request = request
        def generate(session, identity, budget, on_event):
            published.set()
            self.assertTrue(release.wait(2))
            on_event({'event': 'started'})
            return {'event': 'cancelled'}
        fake.generate = generate
        def loop(service, session, op, reserve):
            old = service.absolute_deadline
            try:
                service.absolute_deadline = 1
                reserve(op['id'] + '-0')
                service.generate(session, op['id'] + '-0', 1)
                with self.assertRaisesRegex(ValueError, 'cancelled'):
                    service.request('POST', '/messages', {})
                with self.assertRaisesRegex(ValueError, 'cancelled'):
                    service.generate(session, op['id'] + '-1', 1)
                return {'protocol_success': True}
            finally:
                service.absolute_deadline = old
        adapter = Adapter(fake, events.append, 4)
        adapter.session = 1
        with patch('interactive_workload.native_workload_tools.run_loop', loop):
            adapter.dispatch({'id': 't', 'op': 'tools', 'profile': 'calculator',
                'max_generated': 2})
            self.assertTrue(published.wait(1))
            # Cancellation races with admission; callback rechecks latched intent.
            with self.assertRaisesRegex(ValueError, 'admission'):
                adapter.dispatch({'id': 'c', 'op': 'cancel'})
            release.set()
            adapter.worker.join(2)
        self.assertIsNone(adapter.fatal)
        self.assertEqual(cancels, ['/requests/t-0/cancel', '/requests/t-0/cancel'])
        self.assertEqual(fake.absolute_deadline, 12345)
        with self.assertRaisesRegex(ValueError, 'already used'):
            adapter.reserve('t-0')

    def test_tool_internal_id_collision(self):
        fake = Fake()
        fake.native = True
        fake.info = {'tools': True, 'conversation_protocol': 'native-calculator-v13'}
        fake.generate = Mock(side_effect=AssertionError('must not generate'))
        adapter = Adapter(fake, lambda e: None, 4)
        adapter.session = 1
        adapter.reserve('t-0')
        def loop(service, session, op, reserve):
            reserve('t-0')
            service.generate(session, 't-0', 1)
        with patch('interactive_workload.native_workload_tools.run_loop', loop):
            adapter.dispatch({'id': 't', 'op': 'tools', 'profile': 'calculator',
                              'max_generated': 2})
            adapter.worker.join(2)
        self.assertIsInstance(adapter.fatal, ValueError)
        fake.generate.assert_not_called()

    def test_tool_cancel_between_requests(self):
        fake = Fake()
        fake.native = True
        fake.info = {'tools': True, 'conversation_protocol': 'native-calculator-v13'}
        between, release = threading.Event(), threading.Event()
        calls = []
        def generate(session, identity, count, on_event):
            calls.append(identity)
            return {'event': 'completed'}
        fake.generate = generate
        def loop(service, session, op, reserve):
            reserve('t-0')
            service.generate(session, 't-0', 1)
            between.set()
            self.assertTrue(release.wait(2))
            with self.assertRaisesRegex(ValueError, 'cancelled'):
                service.generate(session, 't-1', 1)
            return {'protocol_success': True}
        events = []
        adapter = Adapter(fake, events.append, 4)
        adapter.session = 1
        with patch('interactive_workload.native_workload_tools.run_loop', loop):
            adapter.dispatch({'id': 't', 'op': 'tools', 'profile': 'calculator',
                              'max_generated': 2})
            self.assertTrue(between.wait(1))
            adapter.dispatch({'id': 'c', 'op': 'cancel'})
            self.assertEqual(events[-1]['result'], {'cancel_pending': True})
            release.set()
            adapter.worker.join(2)
        self.assertIsNone(adapter.fatal)
        self.assertEqual(calls, ['t-0'])
        with self.assertRaisesRegex(ValueError, 'no active'):
            adapter.dispatch({'id': 'c2', 'op': 'cancel'})

    def test_config_popen_failure_closes_logs(self):
        opened = []
        original = Path.open
        def opening(path, *args, **kwargs):
            if path.name == 'config.json':
                raise OSError('config failure')
            stream = original(path, *args, **kwargs)
            opened.append(stream)
            return stream
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, 'open', opening), patch.object(
                    Service, 'record', side_effect=OSError('log failure')):
                with self.assertRaisesRegex(OSError, 'config failure'):
                    Service('unused', Path(directory) / 'config', evidence_config={})
            self.assertTrue(all(stream.closed for stream in opened))
            opened.clear()
            with patch.object(Path, 'open', opening), patch(
                    'real_workload.subprocess.Popen', side_effect=OSError('spawn failure')), \
                    patch.object(Service, 'record', side_effect=OSError('log failure')):
                with self.assertRaisesRegex(OSError, 'spawn failure'):
                    Service('unused', Path(directory) / 'spawn')
            self.assertTrue(all(stream.closed for stream in opened))

    def test_record_failure_no_wire(self):
        import io
        service = Service.__new__(Service)
        service.log, service.log_bytes = io.StringIO(), 1
        connection = Mock()
        service.connect = lambda: connection
        with self.assertRaisesRegex(RuntimeError, 'limit'):
            service.request('POST', '/mutate', {})
        connection.request.assert_not_called()
        connection.close.assert_called_once()

    def test_startup_io_failure_cleanup(self):
        processes = []
        real_popen = subprocess.Popen
        def popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        with tempfile.TemporaryDirectory() as directory:
            child = Path(directory) / 'child'
            child.write_text('#!/usr/bin/env python3\nimport sys,time\nsys.stdout.write("partial")\nsys.stdout.flush()\ntime.sleep(10)\n')
            child.chmod(0o700)
            with patch('real_workload.subprocess.Popen', popen), patch.object(
                    Service, 'record', side_effect=OSError('record failed')), patch.object(
                    Service, 'write_diagnostic', side_effect=OSError('disk failed')):
                with self.assertRaises(Exception):
                    Service(child, Path(directory) / 'out', log_bytes=16384,
                            absolute_deadline=time.monotonic() + .1)
            self.assertIsNotNone(processes[0].poll())
            self.assertTrue(processes[0].stderr.closed)

    def test_long_message_schema(self):
        raw = json.dumps({'id': 'long', 'op': 'message', 'content': '\x01' * 32768})
        self.assertLess(len(raw.encode()), 262144)
        self.assertEqual(len(command(raw, 32768)['content']), 32768)
        with self.assertRaises(ValueError):
            command(raw)
        with self.assertRaises(ValueError):
            command(json.dumps({'id': 'x', 'op': 'message', 'content': 'x' * 32769}), 32768)

    def test_terminal_immediate_mutation(self):
        fake = Fake()
        fake.done.set()
        observed = threading.Event()
        release = threading.Event()
        def emit(event):
            if event.get('id') == 'g' and event['event'] == 'final':
                observed.set()
                release.wait(1)
        adapter = Adapter(fake, emit, 2)
        adapter.dispatch({'op': 'create', 'id': 's'})
        adapter.dispatch({'op': 'generate', 'id': 'g', 'budget': 1})
        self.assertTrue(observed.wait(1))
        # Keep final publication in progress until dispatch has begun its handoff.
        timer = threading.Timer(.05, release.set)
        timer.start()
        adapter.dispatch({'op': 'reset', 'id': 'r'})
        timer.join()
        self.assertFalse(adapter.worker.is_alive())

    def test_cancel_admission_is_recoverable(self):
        fake = Fake()
        original = fake.request
        fake.request = lambda method, path, body=None: ((404, {'error': 'not admitted'})
            if path.endswith('/cancel') else original(method, path, body))
        adapter = Adapter(fake, lambda event: None, 2)
        adapter.dispatch({'op': 'create', 'id': 's'})
        adapter.dispatch({'op': 'generate', 'id': 'g', 'budget': 1})
        self.assertTrue(fake.started.wait(1))
        with self.assertRaisesRegex(ValueError, 'admission'):
            adapter.dispatch({'op': 'cancel', 'id': 'c'})
        self.assertTrue(adapter.cancelled.is_set())
        adapter.dispatch({'op': 'inspect', 'id': 'i'})
        fake.done.set()
        adapter.worker.join(2)
        self.assertIsNone(adapter.fatal)

    def test_context_flag_schema(self):
        script = str(Path(__file__).resolve().parents[1] / 'tools/interactive_workload.py')
        for flags in (['--native-context', '257'], ['--native-chunk', '128']):
            result = subprocess.run([sys.executable, script, '--server', 'unused',
                '--output', 'unused', '--test-only'] + flags, capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_legacy_record(self):
        import io
        service = Service.__new__(Service)
        service.log = io.StringIO()
        service.record('legacy')
        self.assertEqual(json.loads(service.log.getvalue())['kind'], 'legacy')

    def test_startup_failure_and_logcap_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            child = Path(directory) / 'child'
            child.write_text('#!/usr/bin/env python3\nimport sys\nsys.stdout.write("x" * 10000)\nsys.stdout.flush()\n')
            child.chmod(0o700)
            for cap in (128, 16384):
                output = Path(directory) / str(cap)
                with self.assertRaises(Exception):
                    Service(child, output, log_bytes=cap)
                self.assertLessEqual((output / 'server.stderr').stat().st_size, cap)
                self.assertIn('shutdown', (output / 'client.ndjson').read_text()) if cap > 128 else None

    def test_output_refusal(self):
        with tempfile.TemporaryFile() as stream:
            with self.assertRaises(ValueError):
                Output(stream.fileno(), time.monotonic() + 1)
        r, w = os.pipe()
        try:
            output = Output(w, time.monotonic() + 2)
            while True:
                try:
                    os.write(w, b'x' * 4096)
                except BlockingIOError:
                    break
            with self.assertRaises(TimeoutError):
                output.emit({'event': 'test'})
            os.close(r)
            r = None
            with self.assertRaises(BrokenPipeError):
                output.emit({'event': 'test'})
        finally:
            if r is not None:
                os.close(r)
            os.close(w)

    @unittest.skipUnless(SERVER, 'supply existing no-weight protocol server')
    def test_close_record_io_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            service = Service(SERVER, Path(directory) / 'service')
            with patch.object(service, 'record', side_effect=OSError('disk failed')):
                self.assertFalse(service.close())
            self.assertIsNotNone(service.process.poll())
            self.assertTrue(service.log.closed)
            self.assertTrue(service.stderr.closed)
            self.assertTrue(service.process.stderr.closed)

    @unittest.skipUnless(SERVER, 'supply existing no-weight protocol server')
    def test_callback_mutation_and_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            for raises in (False, True):
                service = Service(SERVER, Path(directory) / str(raises), log_bytes=32768)
                try:
                    _, created = service.request('POST', '/sessions', {})
                    session = created['session_id']
                    service.request('POST', f'/sessions/{session}/messages',
                                    {'role': 'user', 'content': 'hello'})
                    def callback(event):
                        event.clear()
                        if raises:
                            raise LookupError('callback exploded')
                    if raises:
                        with self.assertRaisesRegex(LookupError, 'callback exploded'):
                            service.generate(session, 'cb', 2, on_event=callback)
                        self.assertIn('callback exploded', (service.output / 'client.ndjson').read_text())
                    else:
                        result = service.generate(session, 'cb', 2, on_event=callback)
                        self.assertEqual(result['event'], 'completed')
                finally:
                    service.close()

    @unittest.skipUnless(SERVER, 'supply existing no-weight protocol server')
    def test_callback_logging_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            service = Service(SERVER, Path(directory) / 'service', log_bytes=32768,
                              absolute_deadline=time.monotonic() + 5)
            try:
                writers = [threading.Thread(target=lambda: [service.record('concurrent', n=i)
                    for i in range(20)]) for _ in range(4)]
                for writer in writers:
                    writer.start()
                for writer in writers:
                    writer.join()
                status, created = service.request('POST', '/sessions', {})
                self.assertEqual(status, 200)
                session = created['session_id']
                service.request('POST', f'/sessions/{session}/messages',
                                {'role': 'user', 'content': 'hello'})
                events = []
                result = service.generate(session, 'callback', 2, on_event=events.append)
                self.assertEqual(events[0]['event'], 'started')
                self.assertEqual(events[-1]['event'], result['event'])
                with self.assertRaisesRegex(RuntimeError, 'limit'):
                    for i in range(100):
                        service.record('cap', data='x' * 1024)
                self.assertLessEqual((service.output / 'client.ndjson').stat().st_size, 32768)
                self.assertIn('limit', service.stderr_error)
                for line in (service.output / 'client.ndjson').read_text().splitlines():
                    self.assertIsInstance(json.loads(line), dict)
            finally:
                service.close()

    @unittest.skipUnless(SERVER, 'supply existing no-weight protocol server')
    def test_idle_deadline_broken_stdout(self):
        script = str(Path(__file__).resolve().parents[1] / 'tools/interactive_workload.py')
        with tempfile.TemporaryDirectory() as directory:
            for name in ('idle', 'broken', 'signal'):
                path = Path(directory) / name
                process = subprocess.Popen([sys.executable, script, '--server', SERVER,
                    '--output', str(path), '--test-only', '--seconds', '1'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    if name == 'broken':
                        process.stdout.close()
                    else:
                        self.assertEqual(json.loads(process.stdout.readline())['event'], 'ready')
                        if name == 'signal':
                            process.terminate()
                    self.assertNotEqual(process.wait(timeout=10), 0)
                    self.assertIn('shutdown', (path / 'client.ndjson').read_text())
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    for stream in (process.stdin, process.stdout, process.stderr):
                        stream.close()

    @unittest.skipUnless(SERVER, 'supply existing no-weight protocol server')
    def test_live_cancel_and_eof(self):
        script = str(Path(__file__).resolve().parents[1] / 'tools/interactive_workload.py')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'live'
            process = subprocess.Popen([sys.executable, script, '--server', SERVER,
                '--output', str(path), '--test-only', '--seconds', '5'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertEqual(json.loads(process.stdout.readline())['event'], 'ready')
                def send(value):
                    process.stdin.write((json.dumps(value) + '\n').encode())
                    process.stdin.flush()
                send({'id': 'a', 'op': 'create'})
                self.assertTrue(json.loads(process.stdout.readline())['ok'])
                send({'id': 'b', 'op': 'message', 'content': 'hello'})
                self.assertTrue(json.loads(process.stdout.readline())['ok'])
                send({'id': 'c', 'op': 'generate', 'budget': 100})
                # Wait for actual stream activity, not an assumed scheduling delay.
                self.assertEqual(json.loads(process.stdout.readline())['event'], 'text')
                send({'id': 'd', 'op': 'inspect'})
                send({'id': 'e', 'op': 'reset'})
                send({'id': 'f', 'op': 'cancel'})
                events = []
                while not any(e.get('id') == 'c' and e['event'] == 'final' for e in events) or not any(e.get('id') == 'f' for e in events):
                    events.append(json.loads(process.stdout.readline()))
                self.assertTrue(any(e.get('id') == 'd' and e.get('ok') for e in events))
                self.assertTrue(any(e.get('id') == 'e' and not e.get('ok') for e in events))
                process.stdin.close()
                self.assertEqual(process.wait(timeout=10), 0, process.stderr.read())
                self.assertIn('shutdown', (path / 'client.ndjson').read_text())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                for stream in (process.stdout, process.stderr):
                    stream.close()

    @unittest.skipUnless(SERVER, 'supply existing no-weight protocol server')
    def test_process_eof_bounds_exclusive_cleanup(self):
        script = str(Path(__file__).resolve().parents[1] / 'tools/interactive_workload.py')
        with tempfile.TemporaryDirectory() as directory:
            for name, data, expected in [
                ('eof', b'', 0),
                ('turn', b'{"id":"a","op":"create"}\n'
                    b'{"id":"cp","op":"checkpoint"}\n'
                    b'{"id":"rs","op":"restore"}\n'
                    b'{"id":"b","op":"inspect"}\n', 0),
                ('long', b'x' * 262145, 1),
                ('partial', b'{', 1),
                ('commands', b'{"id":"create","op":"create"}\n' + b''.join(
                    (json.dumps({'id': 'x' + str(i), 'op': 'inspect'}) + '\n').encode()
                    for i in range(256)), 1),
                ('active-quit', b'{"id":"a","op":"create"}\n'
                    b'{"id":"b","op":"message","content":"hello"}\n'
                    b'{"id":"c","op":"generate","budget":256}\n'
                    b'{"id":"q","op":"quit"}\n', 0),
                ('active-eof', b'{"id":"a","op":"create"}\n'
                    b'{"id":"b","op":"message","content":"hello"}\n'
                    b'{"id":"c","op":"generate","budget":256}\n', 0)]:
                path = Path(directory) / name
                args = [sys.executable, '-B'] + (['-O'] if sys.flags.optimize else []) + [script,
                    '--server', SERVER, '--output', str(path), '--test-only', '--seconds', '3']
                result = subprocess.run(args, input=data, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, expected, result.stderr)
                if name == 'turn':
                    events = [json.loads(line) for line in result.stdout.splitlines()]
                    for identity in ('cp', 'rs'):
                        self.assertTrue(any(e.get('id') == identity and e.get('event') == 'final'
                                            and e.get('ok') is False for e in events))
                    self.assertTrue(any(e.get('id') == 'b' and e.get('ok') for e in events))
                self.assertTrue((path / 'config.json').exists())
                log = (path / 'client.ndjson').read_text()
                self.assertIn('shutdown', log)
                again = subprocess.run(args, input=b'', capture_output=True, timeout=10)
                self.assertNotEqual(again.returncode, 0)
                self.assertEqual((path / 'client.ndjson').read_text(), log)


if __name__ == '__main__':
    unittest.main()
