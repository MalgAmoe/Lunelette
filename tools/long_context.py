#!/usr/bin/env python3
"""Optional, explicitly invoked real-model regression tooling; no downloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shlex

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / 'tests/long_context'
sys.path.insert(0, str(HELPERS))
import runner
import campaign
from build_command import run_build_command


def safe_path(path, *, fresh=False):
    # Do not resolve: that would erase evidence of a symlink ancestor.
    path = Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError('symlink or parent traversal: '+str(path))
    if fresh and path.exists():
        raise RuntimeError('fresh directory required: '+str(path))
    return path


def pin(path):
    path = safe_path(path)
    return dict(path=str(path), size=path.stat().st_size, sha256=runner.sha(path))


def sources():
    names = json.loads((HELPERS/'engine-sources.json').read_text())
    paths = {ROOT/name for name in names}
    paths.update(p for p in HELPERS.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update((ROOT/'tools/long_context.py', ROOT/'docs/development/long-context.md'))
    return [pin(p) for p in sorted(paths)]


def build(out):
    out = safe_path(out, fresh=True)
    out.mkdir(parents=True, exist_ok=False)
    before = sources()
    # Allowlist, rather than trying to enumerate every compiler/Make injection.
    env = dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', LANG='C', LC_ALL='C',
               HOME=os.environ['HOME'], TMPDIR=tempfile.gettempdir())
    log = out/'build.log'
    def capture(cmd, *, required=True):
        result = subprocess.run(cmd, env=env, cwd=ROOT, check=False,
                                timeout=30, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        with log.open('a') as stream:
            stream.write(shlex.join(cmd)+'\n'+result.stdout+'\nexit='+str(result.returncode)+'\n')
        if required:
            result.check_returncode()
        return result.stdout.strip()
    cc = capture(['/usr/bin/xcrun', '--find', 'clang'])
    sdk = capture(['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-path'])
    env['SDKROOT'] = sdk
    compiler = pin(cc)
    version = capture([cc, '--version'])
    capture(['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-version'])
    capture(['/usr/bin/xcode-select', '-p'])
    capture(['/usr/bin/xcodebuild', '-version'], required=False)
    with log.open('a') as stream:
        stream.write(json.dumps(dict(compiler=compiler, environment=env), sort_keys=True)+'\n')
    commands = []
    def run(cmd, *, timeout=180):
        commands.append(cmd)
        with log.open('a') as stream:
            stream.write(shlex.join(cmd)+'\n')
            stream.flush()
            run_build_command(cmd, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
    objects = [out/(n+'.o') for n in ('metal_weights','metal_yarn','metal_runtime','metal_encode','metal_mps')]
    make_shader = ['/usr/bin/make', 'BUILD='+str(out), 'CC='+cc]
    run([*make_shader, str(out/'embed_shader')])
    run([*make_shader, str(out/'metal_shader.h')], timeout=30)
    run(['/usr/bin/make', 'BUILD='+str(out), 'CC='+cc,
         'CFLAGS=-std=c11 -O2 -Wall -Wextra -Wpedantic', 'CPPFLAGS=-Iinclude',
         'LDFLAGS=', 'METAL_FLAGS=-fobjc-arc -fno-objc-exceptions -ffp-contract=off -mmacosx-version-min=15.0',
         *map(str, objects)])
    sys.path.insert(0, str(HELPERS/'long'))
    from schedule import header
    header(out/'b_schedule.h')
    flags = [cc,'-ffp-contract=off','-std=c11','-O2','-DNDEBUG','-Wall','-Wextra','-Wpedantic',
             '-Wno-deprecated-declarations',
             '-DSERVICE_COMMIT="long-context"','-DSERVICE_DIRTY="true"',
             '-I'+str(ROOT), '-I'+str(ROOT/'include'), '-I'+str(ROOT/'src'),
             '-I'+str(ROOT/'service'), '-I'+str(out)]
    common = [str(HELPERS/'short/short.c'), *[str(ROOT/p) for p in
              ('service/framing.c','service/tool_grammar.c','src/tokenizer.c','src/tokenizer_json.c','src/generation.c')]]
    run(flags+['-DCONTROLLER_STUB',*common,'-o',str(out/'short-stub'),'-lm','-pthread'])
    libs = ['-lm','-pthread','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders']
    tail = [*map(str,objects), str(ROOT/'src/model_facts.c'),str(ROOT/'src/safetensors.c')]
    run(flags+['-x','objective-c',*common,'-x','none',*tail,'-o',str(out/'short-linked'),*libs])
    run(flags+['-fobjc-arc','-mmacosx-version-min=15.0',str(HELPERS/'long/runtime_probe.m'),*tail,'-o',str(out/'runtime-probe'),*libs])
    if before != sources():
        raise RuntimeError('sources changed during build')
    receipt = dict(sources=before, commands=commands, compiler=compiler, compiler_version=version, environment=env, sdk=sdk,
                   model_execution='NEVER RUN', products=[pin(p) for p in out.iterdir() if p.is_file()])
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


def self_test():
    with tempfile.TemporaryDirectory(prefix='lunelette-long-',
                                     dir=Path(tempfile.gettempdir()).resolve()) as tmp:
        out = Path(tmp)/'build'
        build(out)
        receipt = json.loads((out/'receipt.json').read_text())
        fixture = out/'tokenizer-fixture'
        subprocess.run([receipt['compiler']['path'],
                        '-std=c11','-O2','-I'+str(ROOT),'-I'+str(ROOT/'include'),
                        str(HELPERS/'tokenizer_fixture.c'),str(ROOT/'src/tokenizer.c'),
                        str(ROOT/'src/tokenizer_json.c'),'-o',str(fixture)],env=receipt['environment'],check=True,timeout=60)
        subprocess.run([str(fixture)],cwd=out,check=True,timeout=30)
        # Forced branch IDs must decode to valid UTF-8, unlike opaque fixture ranks.
        import base64
        tokenizer = out/'test_tokenizer_fixture.tmp'
        data = json.loads(tokenizer.read_text())
        for token in (9000,9001):
            data['vocab'][token-1000]['token_bytes'] = base64.b64encode(
                ('fixture_output_'+str(token)).encode()).decode()
        tokenizer.write_text(json.dumps(data))
        subprocess.run([sys.executable,'-O' if sys.flags.optimize else '-B','-B',str(HELPERS/'short/check.py'),str(out),
                        str(out/'test_tokenizer_fixture.tmp')],check=True,timeout=180)
        env = dict(os.environ, LONG_CONTEXT_BUILD=str(out), PYTHONDONTWRITEBYTECODE='1')
        for sub in ('long','supervisor','.'):
            subprocess.run([sys.executable,'-O' if sys.flags.optimize else '-B','-B','-m','unittest','discover','-s',str(HELPERS/sub)],
                           env=env,check=True,timeout=180)


def verify_build(target):
    target = safe_path(target)
    receipt = json.loads(safe_path(target/'receipt.json').read_text())
    if receipt['sources'] != sources():
        raise RuntimeError('build source drift; use a fresh build directory')
    sys.path.insert(0, str(HELPERS/'supervisor'))
    import manifest
    for record in receipt['products']:
        if safe_path(record['path']).parent != target:
            raise RuntimeError('product outside build')
        manifest.verify(record)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--build',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--weights',type=Path)
    p.add_argument('--tokenizer',type=Path)
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.build:
        p.error('--build required')
    target = safe_path(args.build)
    if not args.out:
        build(target)
        return
    if not args.weights or not args.tokenizer:
        p.error('run requires --weights --tokenizer --out --build')
    sys.path.insert(0,str(HELPERS/'supervisor'))
    import manifest
    import supervisor
    root = safe_path(args.out, fresh=True)
    weights = safe_path(args.weights)
    tokenizer = safe_path(args.tokenizer)
    receipt = verify_build(target)
    spec = campaign.compose(root,target,pin(weights),pin(tokenizer),receipt['sources'],pin)
    manifest.validate(spec)
    root.mkdir(parents=True,exist_ok=False)
    callbacks = runner.Callbacks(supervisor)
    print(json.dumps(supervisor.real_launch(root,spec,callbacks.validate,phase_prepare=callbacks.prepare)))


if __name__ == '__main__':
    main()
