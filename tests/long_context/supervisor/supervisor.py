"""POSIX serial supervisor with separate real and model-free admission.

Not a sandbox: trusted reviewed writers only. Polling is not a disk quota.
"""
from contextlib import contextmanager
import threading
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import selectors
import resource
import manifest as manifest_api
import stat
import subprocess
import sys
import time

GiB = 1024 ** 3
MiB = 1024 ** 2
HERE = Path(__file__).resolve().parent


class Stop(RuntimeError):
    pass


@contextmanager
def wall_deadline(deadline):
    """Interrupt trusted main-thread Python callbacks and interruptible syscalls.

    Refuse, never borrow, an existing real timer. Not a sandbox for callbacks
    that suppress BaseException, change signals, or run noninterruptible C code.
    """
    if threading.current_thread() is not threading.main_thread():
        raise Stop('wall deadline requires main thread')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise Stop('existing ITIMER_REAL refused')
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise Stop('total wall deadline')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise Stop('total wall deadline')
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        yield
        if time.monotonic() >= deadline:
            raise Stop('total wall deadline')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def real_launch(root=None, manifest=None, phase_validator=None, *, phase_prepare=None, **unknown):
    """Launch an authorized strict manifest; callbacks are trusted reviewed code.

    No retries, builds or implicit environment. All pins (including inputs) are
    verified under the campaign deadline before any child can start. Accepted
    known-independent-mismatch remains a discrepancy, never a numerical pass.
    """
    if unknown or root is None or not callable(phase_validator) or not callable(phase_prepare):
        raise Stop('strict manifest and mandatory phase prepare/validator required')
    m = json.loads(json.dumps(manifest))
    try:
        manifest_api.validate(m)
    except manifest_api.ManifestError as exc:
        raise Stop(str(exc)) from exc
    root = safe_root(root)
    for job in m['jobs']:
        if job['cwd'] != str(root / job['id']):
            raise Stop('pinned cwd must be fresh job directory')
    config = dict(mode='real', pins=source_pins(),
        jobs=[dict(id=j['id'], owners=j['owners']) for j in m['jobs']],
        **{k:m[k] for k in ('artifact_cap','build_cap','headroom','process_seconds','total_seconds')})
    return _run(root, config, m, backend=subprocess.Popen,
                phase_validator=phase_validator, phase_prepare=phase_prepare,
                guard=_REAL_BACKEND)


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Stop('source pin is not a regular file')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def source_pins():
    return {name: digest(HERE / name) for name in ('supervisor.py', 'fake_child.py', 'manifest.py')}


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise Stop('integer outside envelope')
    return value


def seconds(value, ceiling):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= ceiling:
        raise Stop('invalid time budget')
    return value


def leaf(name):
    if type(name) is not str or not name or len(name) > 64 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise Stop('unsafe job name')
    return name


def validate(config):
    if type(config) is not dict or set(config) != {'mode', 'pins', 'jobs', 'artifact_cap', 'build_cap', 'headroom', 'process_seconds', 'total_seconds'}:
        raise Stop('unknown/missing configuration')
    if config['mode'] != 'fake-only' or config['pins'] != source_pins():
        raise Stop('closed mode or source drift')
    # Production envelopes are 2 GiB artifacts / 1 GiB builds. This interface
    # deliberately admits at most 64 MiB combined, including metadata allowance.
    a = integer(config['artifact_cap'], 16384, 64 * MiB)
    b = integer(config['build_cap'], 0, 64 * MiB)
    if a + b > 64 * MiB or a > 2 * GiB or b > GiB:
        raise Stop('storage envelope')
    integer(config['headroom'], MiB, GiB)
    seconds(config['process_seconds'], 1200)
    seconds(config['total_seconds'], 14400)
    jobs = config['jobs']
    if type(jobs) is not list or not 1 <= len(jobs) <= 32:
        raise Stop('32 process ceiling')
    names = set()
    owners = 0
    for job in jobs:
        if type(job) is not dict or set(job) != {'id', 'fixture', 'owners', 'files'}:
            raise Stop('unknown job configuration')
        name = leaf(job['id'])
        if name in names:
            raise Stop('duplicate job')
        names.add(name)
        if job['fixture'] not in ('ok', 'fail', 'sleep', 'extra', 'symlink', 'leak'):
            raise Stop('not a bundled fake child')
        owners += integer(job['owners'], 1, 3)
        if job['files'] != {'result': 64}:
            raise Stop('unapproved output/writer bound')
    if owners > 48:
        raise Stop('48 sequential-owner ceiling')
    # Metadata worst-case allowance, plus all declared writer reservations.
    if 32768 + len(jobs) * (16384 + 3*4096 + 64) > a:
        raise Stop('insufficient artifact reservation')
    return config


def durable(path, value, limit=4096):
    data = (json.dumps(value, sort_keys=True) + '\n').encode()
    if len(data) > limit:
        raise Stop('receipt too large')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cleanup(process):
    for sig, grace in ((signal.SIGTERM, .3), (signal.SIGKILL, 2)):
        process.poll()  # Reap before signalling; Darwin may report EPERM for zombies.
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            # Permission is not proof of absence. Reap and verify below, then
            # escalate; a persistent group is always a campaign failure.
            process.poll()
        until = time.monotonic() + grace
        while time.monotonic() < until:
            process.poll()
            if not alive(process.pid):
                process.wait()
                return
            time.sleep(.01)
    process.wait(timeout=2)
    if alive(process.pid):
        raise Stop('group remains (possibly unreaped descendant): campaign stopped')


def safe_root(root):
    root = Path(os.path.abspath(root))
    for part in (root, *root.parents):
        if part.is_symlink():
            raise Stop('symlink campaign path')
    if not root.is_dir():
        raise Stop('campaign root must already exist')
    return root


def census(root, allowed, cap):
    total = 0
    for entry in root.rglob('*'):
        rel = entry.relative_to(root).as_posix()
        st = entry.lstat()
        if rel not in allowed:
            raise Stop('unknown output: ' + rel)
        limit = allowed[rel]
        if limit is None:
            if not stat.S_ISDIR(st.st_mode):
                raise Stop('expected directory')
        elif not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > limit:
            raise Stop('symlink/hardlink/type/per-file limit: ' + rel)
        else:
            total += st.st_size
    if total > cap:
        raise Stop('aggregate artifact cap')
    return total


def run_fake_campaign(root, config):
    return _run(root, validate(json.loads(json.dumps(config))))


def run_manifest_fake(root, manifest, scenarios=None):
    """Same lifecycle, exact real specs retained, only bundled fake child launched."""
    m = json.loads(json.dumps(manifest))
    try:
        manifest_api.preflight(m)
    except manifest_api.ManifestError as exc:
        raise Stop(str(exc)) from exc
    # No build or real input access in this model-free execution route.
    if m['artifact_cap'] + m['build_cap'] > 64*MiB:
        raise Stop('model-free combined 64MiB cap')
    scenarios = dict(scenarios or {})
    if set(scenarios) - {j['id'] for j in m['jobs']}:
        raise Stop('unknown scenario job')
    modes = {'ok','fail','sleep','extra','symlink','leak','nonfinite','finitecount','state','memory','partial','flood','known'}
    if any(v not in modes for v in scenarios.values()):
        raise Stop('unapproved fake scenario')
    for j in m['jobs']:
        bounds = {w['name']:w['max_bytes'] for w in j['writers']}
        if any(name.startswith('result/') for name in bounds):
            raise Stop('result path reserved by tiny backend')
        if 'result' in bounds and bounds['result'] < 10:
            raise Stop('bundled result writer requires ten byte reservation')
    config = dict(mode='fake-only', pins=source_pins(), jobs=[dict(id=j['id'], fixture=scenarios.get(j['id'], 'ok'), owners=j['owners'], files={'result':64}) for j in m['jobs']], artifact_cap=m['artifact_cap'], build_cap=m['build_cap'], headroom=m['headroom'], process_seconds=m['process_seconds'], total_seconds=m['total_seconds'])
    return _run(root, config, m)


# Identity guard, deliberately not an authorization token or public enable switch.
_FUTURE_TEST_BACKEND = object()
_REAL_BACKEND = object()


def _run_future_test(root, manifest, phase_validator, *, backend, guard, phase_prepare=None):
    """Prepared command path, ONLY for injected model-free test backends.

    Main must supply phase_prepare(job, freshwork) and phase A/B/C/D validators
    in a separately authorized future
    integration. Neither this sentinel nor this helper authorizes real execution.
    """
    if guard is not _FUTURE_TEST_BACKEND or not callable(backend) or backend is subprocess.Popen:
        raise Stop('private test backend guard')
    if not callable(phase_validator):
        raise Stop('mandatory phase validator')
    if not callable(phase_prepare):
        raise Stop('mandatory phase prepare')
    m = json.loads(json.dumps(manifest))
    try:
        manifest_api.validate(m)  # No input access before the guarded admission.
    except manifest_api.ManifestError as exc:
        raise Stop(str(exc)) from exc
    if m['artifact_cap'] + m['build_cap'] > 64*MiB:
        raise Stop('combined 64MiB cap')
    root = safe_root(root)
    for j in m['jobs']:
        if j['cwd'] != str(root / j['id']):
            raise Stop('pinned cwd must be fresh job directory')
    config = dict(mode='future-test', pins=source_pins(),
        jobs=[dict(id=j['id'], owners=j['owners']) for j in m['jobs']],
        **{k:m[k] for k in ('artifact_cap','build_cap','headroom','process_seconds','total_seconds')})
    return _run(root, config, m, backend=backend, phase_validator=phase_validator,
                guard=guard, phase_prepare=phase_prepare)


def _run(root, config, manifest=None, *, backend=None, phase_validator=None, guard=None, phase_prepare=None):
    """Run strict dict jobs serially in an existing empty dedicated directory.

    Any existing campaign is refused, even completed: no resume/retry path.
    Receipts are durable intent BEFORE Popen, then immutable result afterward.
    owners is reserved metadata, not evidence of real model owner construction.
    """
    future = backend is not None
    real = future and guard is _REAL_BACKEND and backend is subprocess.Popen
    if future and (not (real or (guard is _FUTURE_TEST_BACKEND and backend is not subprocess.Popen))
                   or not callable(phase_validator) or not callable(phase_prepare)):
        raise Stop('private test backend guard / validator')
    backend_name = 'real-command' if real else ('future-command-test' if future else 'tiny-fake')
    gate = 'AUTHORIZED' if real else 'CLOSED'
    # Snapshot caller data so mutation cannot change an admitted job mid-flight.
    config = json.loads(json.dumps(config))
    root = safe_root(root)
    fd = os.open(root / 'ownership', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    previous_term = signal.getsignal(signal.SIGTERM)
    campaign_started = False
    campaign_outcome = {'state':'stopped', 'reason':'interrupted'}
    def interrupted(signum, frame):
        raise Stop('supervisor interrupted by SIGTERM')
    try:
        signal.signal(signal.SIGTERM, interrupted)  # Main-thread API only.
        if os.fstat(fd).st_nlink != 1:
            raise Stop('hardlinked ownership file')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Stop('campaign exclusively owned') from exc
        if set(p.name for p in root.iterdir()) != {'ownership'} or os.fstat(fd).st_size:
            raise Stop('resume/retry forbidden; preserve prior receipts')
        os.write(fd, b'claimed\n')
        os.fsync(fd)
        start = time.monotonic()
        deadline = start + config['total_seconds']
        metadata_limit = manifest_api.META-4096 if manifest else 28672
        allowed = {'ownership': 64, 'campaign.json': metadata_limit-64, 'campaign.result.json':4096}
        # Campaign configuration is immutable and includes exact source pins.
        durable(root / 'campaign.json', {'manifest': manifest, 'supervisor_pins':config['pins']} if manifest else config, metadata_limit-64)
        campaign_started = True
        if real:
            # Campaign-wide admission: a later bad input must prevent job one.
            try:
                with wall_deadline(deadline):
                    for spec in manifest['jobs']:
                        for pin in manifest_api.all_pins(spec):
                            manifest_api.verify(pin)
            except manifest_api.ManifestError as exc:
                raise Stop(str(exc)) from exc
        completed = {}
        for job in config['jobs']:
            if time.monotonic() >= deadline:
                raise Stop('total wall deadline')
            with wall_deadline(deadline):
                if config['pins'] != source_pins():
                    raise Stop('source drift')
            census(root, allowed, config['artifact_cap'])
            # Full logical reservation remains available before each launch;
            # this is free-space preflight, NOT a physical filesystem reservation.
            if shutil.disk_usage(root).free < config['artifact_cap'] + config['build_cap'] + config['headroom']:
                raise Stop('reservation/headroom preflight')
            name = job['id']
            spec = next(j for j in manifest['jobs'] if j['id'] == name) if manifest else None
            if spec:
                if any(completed.get(p) not in ('pass','known-independent-mismatch') for p in spec['prerequisites']):
                    raise Stop('prerequisite not accepted')
                try:
                    if future:
                        manifest_api.validate(manifest)
                        with wall_deadline(deadline):
                            for pin in manifest_api.all_pins(spec):
                                manifest_api.verify(pin)
                    else:
                        with wall_deadline(deadline):
                            manifest_api.preflight(dict(manifest, jobs=manifest['jobs']))
                except manifest_api.ManifestError as exc:
                    raise Stop(str(exc)) from exc
            bounds = {w['name']: w['max_bytes'] for w in spec['writers']} if spec else {'stdout':4096,'stderr':4096,'diagnostics':4096,'result':64}
            if not future:
                bounds.setdefault('result',64)  # Internal tiny-backend output; charged to JOB_META.
            for suffix in ('launch', 'result'):
                allowed[f'{name}.{suffix}.json'] = 4096
            allowed[name] = None
            for output, bound in bounds.items():
                allowed[f'{name}/{output}'] = bound
                for parent in Path(output).parents:
                    if parent != Path('.'):
                        allowed[f'{name}/{parent.as_posix()}'] = None
            work = root / name
            work.mkdir()
            # Only directories are prepared: no placeholder output evidence.
            for output in bounds:
                (work/output).parent.mkdir(parents=True, exist_ok=True)
            if future:
                with wall_deadline(deadline):
                    phase_prepare(json.loads(json.dumps(spec)), work)
                    if real:
                        try:
                            for pin in manifest_api.all_pins(spec):
                                manifest_api.verify(pin)
                        except manifest_api.ManifestError as exc:
                            raise Stop(str(exc)) from exc
            census(root, allowed, config['artifact_cap'])
            if time.monotonic() >= deadline:
                raise Stop('total deadline during prelaunch checks')
            durable(root / f'{name}.launch.json', {'state': 'launch-intent-no-retry', 'job': name, 'pins': config['pins'], 'manifest_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()})
            process = None
            outcome = {'state': 'stopped', 'reason': 'launch failed'}
            pump = None
            try:
                pump = Streams(work, bounds)
                if time.monotonic() >= deadline:
                    raise Stop('total deadline before spawn')
                stop_at = min(deadline, time.monotonic() + (spec['timeout'] if spec else config['process_seconds']))
                argv = spec['argv'] if future else [sys.executable, '-I', '-B', str(HERE / 'fake_child.py'), job['fixture']]
                direct_max = max((w['max_bytes'] for w in spec['writers']
                                  if w['writer'] == 'reviewed-direct'), default=0) if future else 0
                def child_limits():
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                    # Metal compilation uses OS-managed files outside our evidence
                    # writers. A process-wide output-sized cap breaks compilation.
                    # Real jobs retain bounded writers, census, space/time checks.
                    if direct_max and not real:
                        resource.setrlimit(resource.RLIMIT_FSIZE, (direct_max, direct_max))
                process = (backend if future else subprocess.Popen)(argv, cwd=spec['cwd'] if future else work,
                                           env=spec['env'] if future else {'PATH': '/usr/bin:/bin'}, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           start_new_session=True, close_fds=True, preexec_fn=child_limits)
                pump.attach(process)
                while True:
                    pump.poll()
                    census(root, allowed, config['artifact_cap'])
                    if shutil.disk_usage(root).free < config['headroom']:
                        raise Stop('free-space headroom stop')
                    with wall_deadline(stop_at):
                        if config['pins'] != source_pins():
                            raise Stop('source drift during job')
                    if spec:
                        try:
                            with wall_deadline(stop_at):
                                for p in spec['sources'] + [spec['binary']] + ([spec['interpreter']] if spec['interpreter'] else []):
                                    manifest_api.verify(p)
                        except manifest_api.ManifestError as exc:
                            raise Stop(str(exc)) from exc
                    if time.monotonic() >= stop_at:
                        raise Stop('process/total deadline')
                    code = process.poll()
                    if code is not None:
                        pump.drain()
                        # poll() has reaped the child. Never inspect offline results
                        # until all pipes and the owned process group are gone and
                        # the output census is safe, even on a failed child exit.
                        if alive(process.pid):
                            raise Stop('successful parent leaked descendants')
                        census(root, allowed, config['artifact_cap'])
                        classification = None if future else pump.classification
                        if future:
                            with wall_deadline(deadline):
                                classification = phase_validator(json.loads(json.dumps(spec)), work, code)
                            census(root, allowed, config['artifact_cap'])
                        # Validators collect failure metrics; they cannot waive
                        # the exit-code or required-output contracts.
                        if code not in (spec['accepted_returncodes'] if spec else [0]):
                            raise Stop('child exit ' + str(code))
                        if spec and any(w['required'] and not (work/w['name']).is_file() for w in spec['writers']):
                            raise Stop('missing required output')
                        if 'result' in bounds and not (work / 'result').is_file():
                            raise Stop('missing required result')
                        if future:
                            if classification not in ('pass', 'known-independent-mismatch') or classification != spec['accept']:
                                raise Stop('phase validator rejected: '+str(classification))
                        elif spec and classification != spec['accept']:
                            raise Stop('unexpected result classification: '+str(classification))
                        outcome = {'state': 'complete', 'returncode': code, 'classification': classification, 'backend':backend_name, 'real_outputs_verified':real}
                        completed[name] = classification
                        break
                    time.sleep(.01)
            except BaseException as exc:
                outcome = {'state': 'stopped', 'reason': type(exc).__name__ + ': ' + str(exc)[:512]}
                raise
            finally:
                try:
                    if process is not None:
                        cleanup(process)
                except BaseException as exc:
                    outcome = {'state': 'cleanup-failed', 'reason': str(exc)[:512]}
                    raise
                finally:
                    if pump is not None:
                        try:
                            pump.drain(check=False)
                        finally:
                            pump.close()
                        outcome['streams'] = pump.counts
                        outcome['streams_truncated'] = pump.truncated
                    if process is not None:
                        outcome['returncode'] = process.returncode
                    durable(root / f'{name}.result.json', outcome)
            census(root, allowed, config['artifact_cap'])
        campaign_outcome = {'state': 'complete', 'launches': len(config['jobs']), 'owners_reserved': sum(j['owners'] for j in config['jobs']), 'backend':backend_name, 'real_gate':gate}
        return campaign_outcome
    except BaseException as exc:
        campaign_outcome = {'state':'stopped', 'reason':type(exc).__name__+': '+str(exc)[:512], 'backend':backend_name, 'real_gate':gate}
        raise
    finally:
        try:
            if campaign_started:
                durable(root / 'campaign.result.json', campaign_outcome)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            os.close(fd)


class Streams:
    """Only parent writes raw files; pipe read size and retained lines are bounded."""
    def __init__(self, work, bounds):
        self.bounds = bounds
        self.files = {}
        self.counts = {}
        self.truncated = False
        self.selector = selectors.DefaultSelector()
        self.lines = {'stdout': b'', 'stderr': b''}
        self.classification = None
        self.diagnostic_seen = False
        self.skipping = {'stdout': False, 'stderr': False}
        self.fatal = None
        try:
            for name in ('stdout','stderr','diagnostics'):
                self.files[name] = open(work/name, 'xb', buffering=0)
                self.counts[name] = 0
        except BaseException:
            self.close()
            raise

    def attach(self, process):
        for name in ('stdout','stderr'):
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, name)

    def write(self, name, data):
        room = self.bounds[name] - self.counts[name]
        kept = memoryview(data)[:room]
        offset = 0
        while offset < len(kept):
            written = self.files[name].write(kept[offset:])
            if type(written) is not int or written <= 0 or written > len(kept)-offset:
                self.fatal = 'bounded writer made no valid progress: '+name
                raise Stop(self.fatal)
            offset += written
            self.counts[name] += written
        if len(data) > room:
            self.truncated = True
            self.fatal = 'bounded writer exhausted: '+name

    def diagnostic(self, line):
        if not line.startswith(b'SUPERVISOR_DIAG '):
            return
        if len(line) > 8192:
            self.fatal = 'diagnostic line bound exceeded'
            return
        if self.diagnostic_seen:
            self.fatal = 'duplicate terminal diagnostic'
            return
        self.diagnostic_seen = True
        self.write('diagnostics', line+b'\n')
        try:
            d = json.loads(line[len(b'SUPERVISOR_DIAG '):])
            if set(d) != {'finite','expected','nonfinite','state_valid','memory_error','classification'}:
                raise ValueError('schema')
            if any(type(d[k]) is not int or d[k] < 0 for k in ('finite','expected','nonfinite')):
                raise ValueError('counts')
            if type(d['state_valid']) is not bool or type(d['memory_error']) is not bool:
                raise ValueError('flags')
            if not d['expected'] or d['nonfinite'] or d['finite'] != d['expected'] or not d['state_valid'] or d['memory_error']:
                self.fatal = 'fatal numerical/state/memory diagnostic'
                return
            if d['classification'] not in ('pass','known-independent-mismatch'):
                raise ValueError('classification')
            self.classification = d['classification']
        except (ValueError, TypeError, KeyError):
            self.fatal = 'invalid diagnostic'

    def poll(self, check=True):
        for key, _ in self.selector.select(0):
            data = os.read(key.fileobj.fileno(), 4096)
            name = key.data
            if not data:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
                if self.lines[name]:
                    self.diagnostic(self.lines[name])
                    self.lines[name] = b''
                continue
            self.write(name, data)
            self.feed(name, data)
        if check and self.fatal:
            raise Stop(self.fatal)

    def feed(self, name, data):
        """Retain only a possible diagnostic prefix/line, never arbitrary raw lines."""
        prefix = b'SUPERVISOR_DIAG '
        pieces = data.split(b'\n')
        for i, piece in enumerate(pieces):
            complete = i < len(pieces)-1
            if not self.skipping[name]:
                buf = self.lines[name] + piece
                if not (prefix.startswith(buf) or buf.startswith(prefix)):
                    self.skipping[name] = True
                    self.lines[name] = b''
                elif len(buf) > 8192:
                    self.fatal = 'diagnostic line bound exceeded'
                    self.skipping[name] = True
                    self.lines[name] = b''
                else:
                    self.lines[name] = buf
            if complete:
                if not self.skipping[name]:
                    self.diagnostic(self.lines[name])
                self.lines[name] = b''
                self.skipping[name] = False

    def drain(self, check=True):
        # Called after exit/group cleanup: finite pipe contents, bounded read buffers.
        until = time.monotonic()+.2
        while self.selector.get_map() and time.monotonic() < until:
            self.poll(check=check)
            time.sleep(.001)
        if check and self.selector.get_map():
            raise Stop('unclosed output pipe')

    def close(self):
        for key in list(self.selector.get_map().values()):
            self.selector.unregister(key.fileobj)
            key.fileobj.close()
        self.selector.close()
        for stream in self.files.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
