from contextlib import ExitStack
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import os

import child


class Tests(unittest.TestCase):
    def test_owned_startup_and_failures(self):
        for mode in ('ready', 'exit', 'late-exit', 'occupied', 'overflow'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td, ExitStack() as stack:
                root = Path(td)
                listener = stack.enter_context(socket.socket())
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
                if mode != 'occupied':
                    listener.close()
                script = root/'server'
                script.write_text('#!'+sys.executable+'\nimport socket,sys,json,time\n'+
                    ('sys.exit(9)\n' if mode == 'exit' else
                     'sys.stderr.write("x"*600000); sys.stderr.flush(); time.sleep(5)\n' if mode == 'overflow' else
                     's=socket.socket(); s.bind(("127.0.0.1", '+str(port)+')); s.listen(1)\n'
                     'print(json.dumps(dict(event="listening",port='+str(port)+')),file=sys.stderr,flush=True)\n'
                     + ('time.sleep(.1)\n' if mode == 'late-exit' else 'time.sleep(20)\n')))
                script.chmod(0o700)
                previous = Path.cwd()
                stack.callback(os.chdir, previous)
                os.chdir(root)
                class Evidence:
                    def __init__(self, *args):
                        pass
                class Driver:
                    def __init__(self, *args):
                        pass
                    def run(self):
                        if mode == 'late-exit':
                            time.sleep(.3)
                stack.enter_context(patch('driver.verify_pins'))
                stack.enter_context(patch('build.verify'))
                stack.enter_context(patch('provider.Evidence', Evidence))
                stack.enter_context(patch('driver.Driver', Driver))
                args = SimpleNamespace(server=str(script), model='unused', tokenizer='unused',
                                       port=port, block=256, strategy='compact-fp8-arithmetic', mode='mps')
                if mode == 'ready':
                    child.child(args)
                else:
                    with self.assertRaises((ValueError, RuntimeError)):
                        child.child(args)
                self.assertLessEqual((root/'startup-stderr.log').stat().st_size, 524288)


if __name__ == '__main__':
    unittest.main()
