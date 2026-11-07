#!/usr/bin/env python3
"""Optional bounded HTTP repeat: explicit paths only; never installs dependencies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'tests/long_context/http'
sys.path.insert(0, str(HERE))
from build import build, verify, safe_path, pin


def python_check(path):
    # Virtualenv launchers are normally symlinks. Permit only this explicit,
    # read-only interpreter input; build/output/model paths still reject symlinks.
    path = Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in path.parents):
        raise RuntimeError('unsafe --python ancestor/traversal')
    if not path.is_file():
        raise RuntimeError('--python must name an existing interpreter')
    script = ('import importlib.metadata as m; import sys; '
              'v=m.version("mistral_common"); '
              'sys.exit(0 if v=="1.13.1" else "requires mistral_common==1.13.1; found "+v)')
    result = subprocess.run([str(path), '-I', '-B', '-c', script], capture_output=True,
                            text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('explicit --python requires official mistral_common==1.13.1; '
                           'no installation attempted: '+result.stderr[-2048:])
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--build', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--weights', type=Path)
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--port', type=int)
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    python = python_check(args.python)
    if args.self_test:
        if args.out or args.weights or args.build or args.child:
            p.error('--self-test cannot build or run a model')
        for optimize in ([], ['-O']):
            subprocess.run([str(python), '-B', *optimize, '-m', 'unittest', 'discover',
                            '-s', str(HERE), '-v'], check=True, timeout=180,
                           env={'PATH':'/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE':'1'})
            if args.tokenizer:
                subprocess.run([str(python), '-I', '-B', *optimize, str(HERE/'check_tokenizer.py'),
                                '--tokenizer', str(safe_path(args.tokenizer))], check=True, timeout=60)
        return
    if not args.build:
        p.error('--build required')
    target = safe_path(args.build)
    if not args.out and not args.child:
        if args.weights or args.tokenizer or args.port:
            p.error('build-only mode does not take model inputs')
        build(target)
        return
    if not args.weights or not args.tokenizer or args.port is None or not 0 < args.port < 65536:
        p.error('run requires explicit --build --out --weights --tokenizer --port --python')
    receipt = verify(target)
    if args.child:
        from reference_environment import bootstrap
        bootstrap(python)
        if not args.out or Path.cwd() != safe_path(args.out):
            p.error('child cwd must be explicit owned --out')
        from provider import environment
        encoded = json.dumps(environment(), sort_keys=True)+'\n'
        if len(encoded.encode()) > 16384:
            raise RuntimeError('reference environment receipt cap')
        with Path('reference-environment.json').open('x') as stream:
            stream.write(encoded)
        from child import child
        from types import SimpleNamespace
        child(SimpleNamespace(server=str(target/'observing-http'), model=str(safe_path(args.weights)),
                              tokenizer=str(safe_path(args.tokenizer)), port=args.port,
                              block=256, strategy='compact-fp8-arithmetic', mode='mps'))
        return
    from campaign_http import compose, Callbacks
    import supervisor
    import manifest
    root = safe_path(args.out, fresh=True)
    # Refuse source/output/build collisions before creating the campaign.
    if root == target or root in target.parents or target in root.parents or root == ROOT or ROOT in root.parents:
        p.error('--out must be outside source and build trees')
    spec = compose(root, target, python, safe_path(args.weights), safe_path(args.tokenizer),
                   args.port, receipt)
    manifest.validate(spec)
    root.mkdir(parents=True)
    callbacks = Callbacks(target, args.tokenizer)
    print(json.dumps(supervisor.real_launch(root, spec, callbacks.validate,
                                           phase_prepare=callbacks.prepare)))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print('HTTP repeat refused: '+str(error), file=sys.stderr)
        raise SystemExit(2)
