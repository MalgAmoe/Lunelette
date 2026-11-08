"""Model-free asynchronous control contracts, including adversarial wire metadata."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from workload_scenarios import control

WORK = ('prefill_attempted prefill_committed append_attempted append_committed '
        'pending_attempted pending_committed prefill_calls append_calls decode_calls '
        'decode_attempted decode_committed recompute_attempted recompute_committed '
        'logit_restores').split()


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ControlledService:
    """Wire-shaped fixtures: running admission, polled terminal, authoritative ledger."""
    native = True
    request_seconds = 2

    def __init__(self, policy='on', operation='checkpoint', checkpoint='1'):
        self.clock = Clock()
        self.absolute_deadline = None
        self.info = dict(prefix_reuse=policy, context_capacity=256, checkpoints=True)
        self.calls = []
        self.delay = 0
        self.http = {}
        self.before = dict(session_id=7, state='ready', logical_tokens=2,
                           committed_tokens=2, pending_tokens=0, remaining=254,
                           token_ids=[10, 11])
        self.after = copy.deepcopy(self.before)
        if operation == 'restore' and policy == 'off':
            self.after['committed_tokens'] = 0
        self.admission = dict(self.before, id='request', operation=operation,
                              status='running', state='control')
        self.terminal = dict(self.admission, status='completed', state='ready',
                             checkpoint_id=checkpoint, session_valid=True, recovery='none',
                             generated_tokens=0, token_ids=[], stop_reason='none',
                             cache_restored=operation == 'restore' and policy == 'on',
                             metrics={'work': dict.fromkeys(WORK, 0)})
        if operation == 'restore' and policy == 'on':
            self.terminal['metrics']['work']['logit_restores'] = 1
        for key in ('state', 'logical_tokens', 'committed_tokens', 'pending_tokens'):
            self.terminal['before_' + key] = self.before[key]
            self.terminal[key] = self.after[key]
        self.polls = 0
        self.running_polls = 1

    def bounded_timeout(self, seconds):
        if self.absolute_deadline is not None:
            remaining = self.absolute_deadline - self.clock.now
            if remaining <= 0:
                raise TimeoutError('deadline')
            return min(seconds, remaining)
        return seconds

    def request(self, method, path, body=None):
        self.bounded_timeout(1)
        self.clock.now += self.delay
        self.bounded_timeout(1)
        self.calls.append((method, path, copy.deepcopy(body)))
        status = self.http.get((method, path))
        if status is not None:
            return status, {'error': 'injected'}
        if method == 'POST':
            return 202, copy.deepcopy(self.admission)
        if path.startswith('/requests/'):
            self.polls += 1
            return 200, copy.deepcopy(self.admission if self.polls <= self.running_polls else self.terminal)
        ledger = self.before if not any(c[0] == 'POST' for c in self.calls) else self.after
        return 200, copy.deepcopy(ledger)


class Controls(unittest.TestCase):
    def invoke(self, service, saved=None):
        with patch('workload_scenarios.time.monotonic', service.clock.monotonic), patch('workload_scenarios.time.sleep', service.clock.sleep):
            return control(service, 7, service.admission['operation'], 'request', saved)

    def saved(self):
        return dict(checkpoint_id='1', ledger=copy.deepcopy(ControlledService().before))

    def test_save_restore_policies_and_checkpoint_bounds(self):
        for policy in ('on', 'off'):
            for checkpoint in ('1', '9' * 19, '18446744073709551615'):
                with self.subTest(policy=policy, checkpoint=checkpoint):
                    save = ControlledService(policy, checkpoint=checkpoint)
                    result = self.invoke(save)
                    self.assertEqual(result['checkpoint_id'], checkpoint)
                    self.assertEqual(save.polls, 2)
                    self.assertEqual(save.calls[-1][:2], ('GET', '/sessions/7'))
                    restore = ControlledService(policy, 'restore', checkpoint)
                    restored = self.invoke(restore, result)
                    self.assertEqual(restored['ledger']['token_ids'], [10, 11])
                    self.assertIs(restored['terminal']['cache_restored'], policy == 'on')
                    self.assertEqual(restore.calls[1][2], {'id': 'request', 'checkpoint_id': checkpoint})
                    self.assertIsNone(restore.absolute_deadline)

    def test_wrong_admission_and_terminal(self):
        cases = {'id': 'wrong', 'operation': 'reset', 'session_id': 8,
                 'status': 'completed', 'state': 'ready'}
        for key, bad in cases.items():
            service = ControlledService()
            service.admission[key] = bad
            # invoke operation must remain the requested checkpoint, not injected metadata.
            with self.subTest(admission=key), patch('workload_scenarios.time.monotonic', service.clock.monotonic), patch('workload_scenarios.time.sleep', service.clock.sleep), self.assertRaises(RuntimeError):
                control(service, 7, 'checkpoint', 'request')
            self.assertEqual(sum(c[0] == 'POST' for c in service.calls), 1)
        cases = {'id': 'wrong', 'operation': 'reset', 'session_id': 8, 'state': 'control',
                 'before_state': 'wrong', 'before_logical_tokens': 3,
                 'before_committed_tokens': True, 'before_pending_tokens': False,
                 'logical_tokens': True, 'committed_tokens': 3, 'pending_tokens': False,
                 'generated_tokens': False, 'token_ids': [1], 'session_valid': 1,
                 'recovery': 'reset', 'stop_reason': 'length'}
        for key, bad in cases.items():
            service = ControlledService()
            service.terminal[key] = bad
            with self.subTest(terminal=key), self.assertRaises(RuntimeError):
                self.invoke(service)

    def test_checkpoint_ids_and_ledgers(self):
        for bad in ('0', '', '01', '-1', '18446744073709551616', '1' * 64, 1, True):
            service = ControlledService(checkpoint=bad)
            with self.subTest(checkpoint=bad), self.assertRaises(RuntimeError):
                self.invoke(service)
        for key, bad in {'session_id': 8, 'state': 'control', 'remaining': 255,
                         'token_ids': [10, True], 'logical_tokens': True,
                         'committed_tokens': False, 'pending_tokens': False}.items():
            service = ControlledService()
            service.after[key] = bad
            with self.subTest(ledger=key), self.assertRaises(RuntimeError):
                self.invoke(service)

    def test_restore_work_all_fields_and_identity(self):
        for policy in ('on', 'off'):
            for key in WORK:
                service = ControlledService(policy, 'restore')
                service.terminal['metrics']['work'][key] += 1
                with self.subTest(policy=policy, work=key), self.assertRaises(RuntimeError):
                    self.invoke(service, self.saved())
            for key, bad in (('cache_restored', 1 if policy == 'on' else 0),
                             ('checkpoint_id', '2'), ('session_valid', False)):
                service = ControlledService(policy, 'restore')
                service.terminal[key] = bad
                with self.subTest(policy=policy, field=key), self.assertRaises(RuntimeError):
                    self.invoke(service, self.saved())

    def test_http_and_failed_terminals_never_retry_post(self):
        for method, path in (('GET', '/sessions/7'), ('POST', '/sessions/7/checkpoint'),
                             ('GET', '/requests/request')):
            for status in (409, 500):
                service = ControlledService()
                service.http[method, path] = status
                with self.subTest(method=method, status=status), self.assertRaises(RuntimeError):
                    self.invoke(service)
                self.assertLessEqual(sum(c[0] == 'POST' for c in service.calls), 1)
        for status in ('cancelled', 'error', 'unknown'):
            service = ControlledService()
            service.terminal['status'] = status
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                self.invoke(service)
            self.assertEqual(sum(c[0] == 'POST' for c in service.calls), 1)

    def test_deadline_includes_initial_get_and_restores_global(self):
        for global_deadline in (None, 100.5, 1000):
            service = ControlledService()
            service.absolute_deadline = global_deadline
            service.delay = 3
            with self.subTest(global_deadline=global_deadline), self.assertRaises(TimeoutError):
                self.invoke(service)
            self.assertEqual(service.absolute_deadline, global_deadline)
            self.assertFalse(any(c[0] == 'POST' for c in service.calls))
        service = ControlledService()
        service.absolute_deadline = 100.02
        service.running_polls = 100
        with self.assertRaises(TimeoutError):
            self.invoke(service)
        self.assertEqual(service.absolute_deadline, 100.02)
        self.assertEqual(sum(c[0] == 'POST' for c in service.calls), 1)

    def test_interrupt_restores_deadline(self):
        service = ControlledService()
        service.absolute_deadline = 1000
        original = service.request
        def interrupted(method, path, body=None):
            if path.startswith('/requests/'):
                raise KeyboardInterrupt()
            return original(method, path, body)
        service.request = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(service)
        self.assertEqual(service.absolute_deadline, 1000)


class TransportBounds(unittest.TestCase):
    def test_failure_classification(self):
        from types import SimpleNamespace
        from real_workload import run
        from workload_scenarios import DeadlineExceeded
        scenario = dict(version=2, mode='free_greedy', prefix_reuse='on', repetitions=2,
                        wall_seconds=120, operations=[{'op': 'create'}, {'op': 'close'}])
        for error, kind in ((TimeoutError('inactivity'), 'runtime'),
                            (DeadlineExceeded('application budget'), 'limit')):
            service = SimpleNamespace(native=True, info=dict(prefix_reuse='on'),
                                      record=lambda *a, **k: None)
            with patch('real_workload.run_once', side_effect=error) as execute:
                report = run(service, scenario)
            self.assertEqual(report['failure_kind'], kind)
            self.assertIsNone(report['task_success'])
            self.assertEqual(execute.call_count, 1)

    def test_trickling_body_obeys_application_deadline(self):
        import socket
        import threading
        import time
        from workload_client import Service
        from workload_scenarios import DeadlineExceeded
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(1)
        stop = threading.Event()
        errors = []

        def serve():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(1)
                    connection.recv(4096)
                    body = b'{"value":"' + b'x' * 128 + b'"}'
                    connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: ' +
                                       str(len(body)).encode() + b'\r\n\r\n')
                    for byte in body:
                        if stop.is_set():
                            break
                        connection.sendall(bytes([byte]))
                        stop.wait(.01)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                errors.append(error)

        service = Service.__new__(Service)
        service.port = listener.getsockname()[1]
        service.record = lambda *a, **k: None
        service.absolute_deadline = time.monotonic() + .12
        worker = threading.Thread(target=serve)
        worker.start()
        try:
            with self.assertRaises(DeadlineExceeded):
                service.request('GET', '/probe')
        finally:
            stop.set()
            worker.join(2)
            listener.close()
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])

    def test_final_body_read_may_close_socket(self):
        from workload_client import Service
        class Connection:
            status = 200
            closed = False
            def request(self, *args):
                pass
            def getresponse(self):
                return self
            def settimeout(self, seconds):
                if self.closed:
                    raise OSError('closed socket')
            def isclosed(self):
                return self.closed
            def read1(self, size):
                self.closed = True
                return b'{}'
            def close(self):
                self.closed = True
        connection = Connection()
        connection.sock = connection
        service = Service.__new__(Service)
        service.connect = lambda: connection
        service.record = lambda *a, **k: None
        self.assertEqual(service.request('GET', '/probe'), (200, {}))

    def test_inactivity_is_not_application_exhaustion_and_body_is_bounded(self):
        import time
        from workload_client import Service
        from workload_scenarios import DeadlineExceeded, ProtocolError

        class Connection:
            status = 200
            def __init__(self, failure):
                self.failure = failure
                self.closed = False
            def request(self, *args):
                pass
            def getresponse(self):
                return self
            def read1(self, size):
                if self.failure:
                    raise TimeoutError('inactivity')
                return b'x' * size
            def close(self):
                self.closed = True

        for failure in (True, False):
            service = Service.__new__(Service)
            service.absolute_deadline = time.monotonic() + 120
            service.record = lambda *a, **k: None
            connection = Connection(failure)
            service.connect = lambda: connection
            with self.assertRaises(TimeoutError if failure else ProtocolError) as caught:
                service.request('GET', '/probe')
            self.assertNotIsInstance(caught.exception, DeadlineExceeded)
            self.assertTrue(connection.closed)


if __name__ == '__main__':
    unittest.main()
