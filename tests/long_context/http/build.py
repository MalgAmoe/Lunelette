"""Current-source observer build; no checkpoint is opened by this module."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'tools'))
from long_context import safe_path, pin
from build_command import run_build_command
sys.path.insert(0, str(HERE.parent/'supervisor'))
import manifest


def sources():
    names = json.loads((HERE.parent/'engine-sources.json').read_text())
    paths = {ROOT/name for name in names}
    paths.update(p for p in HERE.parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update((HERE.parent/'engine-sources.json',
                  ROOT/'tools/long_context_http.py', ROOT/'tools/long_context.py',
                  ROOT/'docs/long-context-http-testing.md', ROOT/'docs/long-context-testing.md'))
    paths.update(p for p in (HERE.parent/'supervisor').glob('*.py'))
    return [pin(p) for p in sorted(paths)]


def verify(target):
    target = safe_path(target)
    receipt_path = safe_path(target/'receipt.json')
    if receipt_path.stat().st_size > 262144:
        raise RuntimeError('HTTP receipt 256KiB cap')
    for path in target.iterdir():
        safe_path(path)
        if not path.is_file():
            raise RuntimeError('HTTP unexpected build directory/entry')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != 'http-current-build-v1':
        raise RuntimeError('HTTP receipt schema')
    products = receipt.get('products', [])
    expected = {str(p) for p in target.iterdir() if p.is_file() and p.name != 'receipt.json'}
    if {p['path'] for p in products} != expected or str(target/'observing-http') not in expected:
        raise RuntimeError('HTTP product inventory drift')
    if receipt['sources'] != sources():
        raise RuntimeError('HTTP build source drift; fresh build required')
    for record in receipt['products']:
        if safe_path(record['path']).parent != target:
            raise RuntimeError('HTTP product outside build')
        manifest.verify(record)
    manifest.verify(receipt['compiler'])
    return receipt


def build(target):
    target = safe_path(target, fresh=True)
    if target == ROOT or ROOT in target.parents or target in ROOT.parents:
        raise RuntimeError('HTTP build must be outside the source tree')
    target.mkdir(parents=True)
    before = sources()
    env = dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', LANG='C', LC_ALL='C',
               HOME=os.environ['HOME'], TMPDIR=tempfile.gettempdir())
    commands = []
    # Compiler output is bounded by a timeout and a regular-file size limit;
    # build artifacts are not model-run artifacts and no launch owner is added.
    import resource
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2*1024*1024, 2*1024*1024))
    log = (target/'build.log').open('xb')
    def census():
        paths = list(target.iterdir())
        if len(paths) > 32 or any(not p.is_file() or p.is_symlink() for p in paths):
            raise RuntimeError('HTTP fixed build inventory bound')
        if sum(p.stat().st_size for p in paths) > 64*1024*1024:
            raise RuntimeError('HTTP 64MiB build cap')
    def run(argv, expected=0, *, timeout=180):
        commands.append(argv)
        log.write((json.dumps(argv)+'\n').encode())
        log.flush()
        result = run_build_command(argv, cwd=ROOT, env=env, stdout=log, stderr=log,
                                timeout=timeout, preexec_fn=limits)
        census()
        if result.returncode != expected:
            raise RuntimeError('HTTP build/refusal command failed: '+str(argv))
    def query(argv):
        result = subprocess.run(argv, env=env, check=True, capture_output=True,
                                text=True, timeout=30)
        return result.stdout.strip()
    try:
        cc = query(['/usr/bin/xcrun', '--find', 'clang'])
        sdk = query(['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-path'])
        env['SDKROOT'] = sdk
        compiler = pin(cc)
        version = query([cc, '--version'])
        sdk_version = query(['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-version'])
        developer_directory = query(['/usr/bin/xcode-select', '-p'])
        make_shader = ['/usr/bin/make', 'BUILD='+str(target), 'CC='+cc]
        run([*make_shader, str(target/'embed_shader')])
        run([*make_shader, str(target/'metal_shader.h')], timeout=30)
        objects = [target/(name+'.o') for name in
                   ('metal_weights', 'metal_yarn', 'metal_runtime', 'metal_encode', 'metal_mps')]
        run(['/usr/bin/make', 'BUILD='+str(target), 'CC='+cc,
             'CFLAGS=-std=c11 -O2 -Wall -Wextra -Wpedantic', 'CPPFLAGS=-Iinclude',
             'LDFLAGS=', 'METAL_FLAGS=-fobjc-arc -fno-objc-exceptions -ffp-contract=off -mmacosx-version-min=15.0',
             *map(str, objects)])
        flags = [cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Wpedantic',
                 '-Wno-deprecated-declarations', '-ffp-contract=off', '-mmacosx-version-min=15.0',
                 '-DSERVICE_NATIVE', '-DSERVICE_COMMIT="http-current-source"', '-DSERVICE_DIRTY="true"',
                 *['-I'+str(ROOT/n) for n in ('include', 'src', 'service')]]
        units = [HERE/'observe.c'] + [ROOT/n for n in
                 ('service/session.c', 'service/tool_grammar.c', 'service/framing.c', 'src/tokenizer.c',
                  'src/tokenizer_json.c', 'src/generation.c', 'src/model_facts.c', 'src/safetensors.c')]
        for unit in units:
            obj = target/('host-'+unit.stem+'.o')
            run([*flags, '-MD', '-MF', str(obj.with_suffix('.d')), '-c', str(unit), '-o', str(obj)])
            objects.append(obj)
        binary = target/'observing-http'
        run([cc, *map(str, objects), '-pthread', '-lm', '-framework', 'Foundation',
             '-framework', 'Metal', '-framework', 'MetalPerformanceShaders', '-o', str(binary)])
        # Refuse all these before creating observation.jsonl or touching inputs.
        base = ['--model', 'absent', '--tokenizer', 'absent', '--context', '4352',
                '--prefill-chunk', '256', '--prefix-reuse', 'on', '--strategy',
                'compact-fp8-arithmetic', '--prefill', 'mps', '--port', '12345']
        refusals = [[], base[:-2], base[:-2]+['--context', '4352']]
        for index, value in ((0, '--unknown'), (1, ''), (3, '-'), (5, '256'),
                             (7, '128'), (9, 'off'), (11, 'resident-fp32'),
                             (13, 'custom'), (15, '0'), (15, '65536'), (15, '12x')):
            invalid = list(base)
            invalid[index] = value
            refusals.append(invalid)
        for argv in refusals:
            # cwd is a fresh build, never repository root, for exclusive trace check.
            commands.append([str(binary), *argv])
            log.flush()
            offset = log.tell()
            result = run_build_command([str(binary), *argv], cwd=target, env=env,
                                    stdout=log, stderr=log, timeout=5, preexec_fn=limits)
            refusal_output = (target/'build.log').read_bytes()[offset:]
            if result.returncode != 2 or b'missingargs:' not in refusal_output or (target/'observation.jsonl').exists():
                raise RuntimeError('pre-model argument refusal failed')
        if before != sources():
            raise RuntimeError('source changed during HTTP build')
    finally:
        log.close()
    receipt = dict(schema='http-current-build-v1', sources=before, commands=commands,
                   compiler=compiler, compiler_version=version, sdk=sdk, sdk_version=sdk_version,
                   developer_directory=developer_directory, environment=env,
                   model_execution='NOT_RUN', products=[pin(p) for p in sorted(target.iterdir()) if p.is_file()])
    with (target/'receipt.json').open('x') as stream:
        encoded = json.dumps(receipt, indent=2)+'\n'
        if len(encoded.encode()) > 262144:
            raise RuntimeError('HTTP receipt 256KiB cap')
        stream.write(encoded)
    census()
    return receipt
