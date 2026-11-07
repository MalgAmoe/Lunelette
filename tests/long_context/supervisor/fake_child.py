"""Tiny fixed backend, never executes manifest argv or reads its inputs."""
import json
import os
import signal
import sys
import time

mode = sys.argv[1]
print('fake stdout before '+mode, flush=True)
print('fake stderr before '+mode, file=sys.stderr, flush=True)
d = dict(finite=4, expected=4, nonfinite=0, state_valid=True,
         memory_error=False, classification='pass')
if mode == 'nonfinite': d['nonfinite'] = 1
if mode == 'finitecount': d['finite'] = 3
if mode == 'state': d['state_valid'] = False
if mode == 'memory': d['memory_error'] = True
if mode == 'known': d['classification'] = 'known-independent-mismatch'
print('SUPERVISOR_DIAG '+json.dumps(d), file=sys.stderr, flush=True)
if mode in ('nonfinite','finitecount','state','memory','sleep'):
    time.sleep(30)
elif mode in ('fail','partial'):
    os.write(1, b'partial stdout without newline')
    os.write(2, b'partial stderr without newline')
    sys.exit(7)
elif mode == 'flood':
    # finite synthetic overflow, not an unbounded disk writer
    os.write(1, b'x'*8192)
    time.sleep(30)
elif mode == 'extra':
    with open('unexpected', 'xb') as stream: stream.write(b'x')
elif mode == 'symlink':
    os.symlink('result', 'unexpected')
elif mode == 'leak':
    if os.fork() == 0:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        time.sleep(30)
        os._exit(0)
elif mode in ('ok','known'):
    with open('result', 'xb') as stream: stream.write(b'fake-only\n')
else:
    sys.exit(99)
