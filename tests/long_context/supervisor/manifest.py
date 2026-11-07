"""Manifest contract. Validation never executes a command or opens model inputs."""
import hashlib
import json
import math
from pathlib import Path

MiB = 1024**2
GiB = 1024**3
META = 1024*1024
JOB_META = 16384


class ManifestError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise ManifestError(message)


def natural(x, ceiling):
    require(type(x) is int and 0 <= x <= ceiling, 'invalid byte/count bound')
    return x


def pin(p):
    require(type(p) is dict and set(p) == {'path', 'sha256', 'size'}, 'pin schema')
    require(type(p['path']) is str and Path(p['path']).is_absolute(), 'absolute pin path required')
    require(type(p['sha256']) is str and len(p['sha256']) == 64 and all(c in '0123456789abcdef' for c in p['sha256']), 'sha256 required')
    natural(p['size'], 2**63-1)


def verify(p):
    pin(p)
    path = Path(p['path'])
    require(not any(part.is_symlink() for part in (path, *path.parents)), 'symlink pin')
    require(path.is_file() and path.stat().st_size == p['size'], 'pin size drift: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            h.update(chunk)
    require(h.hexdigest() == p['sha256'], 'pin digest drift: '+str(path))


def all_pins(job):
    return job['sources'] + job['inputs'] + [job['binary']] + ([job['interpreter']] if job['interpreter'] else [])


def validate(m):
    require(type(m) is dict and set(m) == {'schema', 'jobs', 'artifact_cap', 'build_cap', 'build_reservations', 'headroom', 'total_seconds', 'process_seconds'}, 'manifest schema')
    require(m['schema'] == 'supervisor-manifest-v1', 'manifest version')
    for key, cap in [('artifact_cap', 2*GiB), ('build_cap', GiB), ('headroom', GiB)]:
        natural(m[key], cap)
    require(m['headroom'] >= MiB, 'headroom minimum')
    for key, ceiling in [('total_seconds', 14400), ('process_seconds', 1200)]:
        require(type(m[key]) in (int, float) and math.isfinite(m[key]) and 0 < m[key] <= ceiling, 'time envelope')
    require(type(m['build_reservations']) is list, 'build reservation list')
    build = 0
    seen_build = set()
    for r in m['build_reservations']:
        require(type(r) is dict and set(r) == {'id', 'max_bytes', 'writer_contract'}, 'build schema')
        require(type(r['id']) is str and r['id'] not in seen_build, 'duplicate build')
        seen_build.add(r['id'])
        require(type(r['writer_contract']) is str and bool(r['writer_contract']), 'build writer proof reference')
        build += natural(r['max_bytes'], GiB)
    require(build <= m['build_cap'], 'build reservations exceed 1GiB/configured cap')
    require(type(m['jobs']) is list and 1 <= len(m['jobs']) <= 32, '32 launch ceiling')
    seen = set(); owners = 0; artifact = META
    for j in m['jobs']:
        require(type(j) is dict and set(j) == {'id','argv','cwd','env','sources','binary','interpreter','inputs','prerequisites','accept','accepted_returncodes','owners','timeout','writers'}, 'job schema')
        name = j['id']
        require(type(name) is str and 0 < len(name) <= 64 and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name) and name not in seen, 'job id')
        require(type(j['argv']) is list and bool(j['argv']) and all(type(x) is str and '\x00' not in x for x in j['argv']), 'exact argv')
        require(type(j['cwd']) is str and Path(j['cwd']).is_absolute(), 'absolute cwd')
        require(type(j['env']) is dict and all(type(k) is str and type(v) is str and k and '=' not in k and '\x00' not in k+v for k,v in j['env'].items()), 'exact env')
        require(type(j['sources']) is list and bool(j['sources']) and type(j['inputs']) is list, 'source/input pins')
        require(j['interpreter'] is None or type(j['interpreter']) is dict and bool(j['interpreter']), 'nullable interpreter pin')
        for p in all_pins(j): pin(p)
        executable = j['interpreter'] or j['binary']
        require(j['argv'][0] == executable['path'], 'argv executable pin')
        if j['interpreter']:
            require(j['binary']['path'] in j['argv'][1:], 'interpreted entry pin absent from argv')
        require(type(j['prerequisites']) is list and all(type(p) is str for p in j['prerequisites']) and len(set(j['prerequisites'])) == len(j['prerequisites']) and all(p in seen for p in j['prerequisites']), 'prerequisites must precede job')
        require(j['accept'] in ('pass', 'known-independent-mismatch'), 'explicit result classification required')
        require(type(j['accepted_returncodes']) is list and bool(j['accepted_returncodes']) and all(type(code) is int and 0 <= code <= 255 for code in j['accepted_returncodes']), 'accepted exit codes')
        require(j['accept'] != 'pass' or j['accepted_returncodes'] == [0], 'pass requires zero exit')
        owners += natural(j['owners'], 48)
        require(j['owners'] >= 1, 'owner reservation')
        require(type(j['timeout']) in (int,float) and math.isfinite(j['timeout']) and 0 < j['timeout'] <= m['process_seconds'], 'job timeout')
        require(type(j['writers']) is list and bool(j['writers']), 'writer reservations')
        names = set()
        for w in j['writers']:
            require(type(w) is dict and set(w) == {'name','max_bytes','writer','contract','required'}, 'writer schema')
            require(type(w['name']) is str and 0 < len(w['name']) <= 240 and w['name'] not in names and not Path(w['name']).is_absolute() and all(part not in ('','.', '..') and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in part) for part in w['name'].split('/')), 'safe unique reserved writer path')
            require(not any(w['name'].startswith(n+'/') or n.startswith(w['name']+'/') for n in names), 'file/directory reservation conflict')
            names.add(w['name'])
            natural(w['max_bytes'], 2*GiB)
            require(w['writer'] in ('supervisor-pipe', 'reviewed-direct') and type(w['contract']) is str and bool(w['contract']) and type(w['required']) is bool, 'writer bound contract')
            if w['name'] in ('stdout','stderr','diagnostics'):
                require(w['writer'] == 'supervisor-pipe', 'raw streams must use bounded supervisor writers')
            artifact += w['max_bytes']
        require({'stdout','stderr','diagnostics'} <= names, 'raw streams and diagnostic writer required')
        artifact += JOB_META
        seen.add(name)
    require(owners <= 48, '48 owner ceiling')
    require(artifact <= m['artifact_cap'], 'artifact reservations exceed 2GiB/configured cap')
    require(len(json.dumps(m).encode()) <= META-8192, 'manifest metadata reservation exceeded')
    return {'artifact_reserved': artifact, 'build_reserved': build, 'launches': len(seen), 'owners': owners, 'real_gate': 'CLOSED'}


def preflight(m, *, verify_inputs=False):
    """Default skips input content (may be weights); does not claim verified inputs."""
    result = validate(m)
    for j in m['jobs']:
        require(Path(j['cwd']).is_dir(), 'missing cwd')
        for p in j['sources'] + [j['binary']] + ([j['interpreter']] if j['interpreter'] else []): verify(p)
        if verify_inputs:
            for p in j['inputs']: verify(p)
    return dict(result, inputs_verified=verify_inputs)
