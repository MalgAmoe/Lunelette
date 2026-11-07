"""Real loopback regression: bounded reads and deadlines, never a model."""
import importlib.util
from pathlib import Path
import socket
import threading
import unittest

from driver import SocketTransport


def exchange(transport, path, body=None, declared=None, payload=b'{}'):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    listener.settimeout(2)
    result, errors = {}, []
    def serve():
        try:
            with listener.accept()[0] as peer:
                peer.settimeout(.15)
                raw = b''
                while b'\r\n\r\n' not in raw:
                    chunk = peer.recv(1024)
                    if not chunk or len(raw) + len(chunk) > 4096:
                        raise RuntimeError('header cap/EOF')
                    raw += chunk
                header, content = raw.split(b'\r\n\r\n', 1)
                length = int(next(line.split(b':', 1)[1] for line in header.split(b'\r\n')
                                  if line.lower().startswith(b'content-length:')))
                result.update(header=header, length=length)
                try:
                    while len(content) < length:
                        chunk = peer.recv(min(4096, length-len(content)))
                        if not chunk:
                            break
                        content += chunk
                except socket.timeout:
                    result['timeout'] = True
                result['content'] = content
                peer.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: '+str(len(payload)).encode()+
                             b'\r\nConnection: close\r\n\r\n'+payload)
                try:
                    result['client_closed'] = peer.recv(1) == b''
                except ConnectionResetError:
                    result['client_closed'] = True
        except BaseException as error:
            errors.append(error)
        finally:
            listener.close()
    thread = threading.Thread(target=serve)
    thread.start()
    failure = None
    try:
        transport(listener.getsockname()[1]).request('POST' if body is not None else 'GET',
                                                   path, body, declared)
    except BaseException as error:
        failure = error
    finally:
        thread.join(3)
    if thread.is_alive():
        raise RuntimeError('socket regression server did not stop')
    if errors:
        raise errors[0]
    return result, failure


class Tests(unittest.TestCase):
    def test_bodyless_get(self):
        spec = importlib.util.spec_from_file_location('original_socket',
                       Path(__file__).parent/'original/driver.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        for path in ('/info', '/sessions/1', '/requests/probe'):
            with self.subTest(path=path):
                old, error = exchange(original.SocketTransport, path)
                self.assertIsNone(error)
                self.assertEqual(old['length'], 2)
                self.assertTrue(old.get('timeout'))
                new, error = exchange(SocketTransport, path)
                self.assertIsNone(error)
                self.assertEqual(new['length'], 0)
                self.assertNotIn('timeout', new)
                self.assertEqual(new['content'], b'')
                self.assertTrue(new['client_closed'])
                self.assertTrue(old['client_closed'])

    def test_explicit_malformed_and_caps(self):
        result, error = exchange(SocketTransport, '/sessions/1/messages', b'', 262145)
        self.assertIsNone(error)
        self.assertEqual(result['length'], 262145)
        self.assertTrue(result['timeout'])
        self.assertTrue(result['client_closed'])
        result, error = exchange(SocketTransport, '/info', payload=b'x'*262145)
        self.assertTrue(result['client_closed'])
        self.assertIsInstance(error, ValueError)
        self.assertIn('response cap', str(error))
        result, error = exchange(SocketTransport, '/info', payload=b'not-json')
        self.assertTrue(result['client_closed'])
        self.assertIsInstance(error, ValueError)
        # A fresh socket after failures demonstrates bounded server cleanup.
        _, error = exchange(SocketTransport, '/info')
        self.assertIsNone(error)


if __name__ == '__main__':
    unittest.main()
