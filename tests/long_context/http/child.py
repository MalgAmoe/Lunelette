"""Serial D supervisor. Execution authorized; no model validation claimed."""
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import signal
import subprocess
import time
import threading
def reserve(extra):
    total = sum(p.stat().st_size for p in Path.cwd().rglob('*') if p.is_file())
    if total + extra > 64 * 1024 * 1024:
        raise ValueError('HTTP aggregate artifact cap')



def child(args):
    from driver import Driver, Journal, SocketTransport, verify_pins
    from provider import Evidence
    verify_pins()
    from build import verify
    verify(Path(args.server).parent)
    # Persistent official-tokenizer memory must exist BEFORE the server
    # performs its unchanged global-availability admission check.
    transport = SocketTransport(args.port)
    evidence = Evidence(args.tokenizer, "observation.jsonl", transport)
    # Inherits supervisor-owned session/group; never setsid the server.
    argv = [args.server, '--model', args.model, '--tokenizer', args.tokenizer,
            '--context', '4352', '--prefill-chunk', str(args.block), '--prefix-reuse', 'on',
            '--strategy', args.strategy, '--prefill', args.mode, '--port', str(args.port)]
    with ExitStack() as files:
        stderr = files.enter_context(Path('startup-stderr.log').open('xb'))
        stdout = files.enter_context(Path('server-stdout.log').open('xb'))
        journal = Journal(Path.cwd()/'journal.jsonl')
        files.callback(journal.close)
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        failures = []
        def capture(pipe, target):
            size = 0
            try:
                while True:
                    raw = pipe.read1(16384)
                    if not raw:
                        break
                    if size + len(raw) > 524288:
                        raise ValueError('server output 524288-byte cap')
                    reserve(len(raw))
                    target.write(raw)
                    target.flush()
                    size += len(raw)
            except BaseException as error:
                failures.append(str(error))
                try:
                    p.terminate()
                except ProcessLookupError:
                    pass
            finally:
                pipe.close()
        threads = []
        try:
            threads = [threading.Thread(target=capture, args=pair) for pair in
                       ((p.stdout, stdout), (p.stderr, stderr))]
            for thread in threads:
                thread.start()
            deadline = time.monotonic() + 120
            while True:
                if p.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('startup failed/deadline')
                if sum(f.stat().st_size for f in (Path('startup-stderr.log'), Path('server-stdout.log'))) > 1048576:
                    raise ValueError('startup output cap')
                # Never connect merely because an unrelated server owns this port.
                # The listening record must come from this owned child's stderr first.
                listening = False
                for line in Path('startup-stderr.log').read_text(errors='replace').splitlines():
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict) and record.get('event') == 'listening' and record.get('port') == args.port:
                        listening = True
                if listening:
                    with socket.create_connection(('127.0.0.1', args.port), .1):
                        break
                time.sleep(.05)
            raw = Path('startup-stderr.log').read_bytes()
            journal.record('startup_guard_observations', {
                'status': 'silent' if not raw else 'captured_unclassified',
                'artifact': 'startup-stderr.log', 'allocator_admission_metrics': 'unavailable'})
            Driver(transport, journal, evidence, args.block, args.strategy, args.mode).run()
        finally:
            unexpected_exit = p.poll()
            try:
                p.terminate()
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=2)
            for thread in threads:
                if thread.ident is not None:
                    thread.join(timeout=3)
                if thread.is_alive():
                    raise RuntimeError('server capture not reaped')
            p.stdout.close()
            p.stderr.close()
            if failures:
                raise RuntimeError('; '.join(failures))
            if unexpected_exit is not None or p.returncode not in (0, -signal.SIGTERM):
                raise RuntimeError('unexpected server exit/forced cleanup: ' + str(p.returncode))
