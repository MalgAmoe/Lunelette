"""One D-256 HTTP cell, using only the existing process-group supervisor."""
import json
from pathlib import Path
import subprocess

from build import ROOT, HERE, pin, verify
from driver import require

CELL = 'D-256-mps-compact-fp8-arithmetic'


def compose(root, target, python, weights, tokenizer, port, receipt):
    entry = ROOT/'tools/long_context_http.py'
    def writer(name, cap, required=True, pipe=False):
        return dict(name=name, max_bytes=cap, required=required,
                    writer='supervisor-pipe' if pipe else 'reviewed-direct',
                    contract='bounded exclusive HTTP writer; docs/long-context-http-testing.md')
    interpreter = pin(python.resolve())
    sources = receipt['sources'] + receipt['products'] + [pin(target/'receipt.json')]
    # Preserve venv identity without asking the supervisor to admit symlink pins.
    cfg = python.parent.parent/'pyvenv.cfg'
    if cfg.is_file():
        sources.append(pin(cfg))
    job = dict(id=CELL, argv=[interpreter['path'], '-I', '-B', str(entry),
               '--child', '--python', str(python), '--build', str(target),
               '--out', str(root/CELL), '--weights', str(weights),
               '--tokenizer', str(tokenizer), '--port', str(port)],
               cwd=str(root/CELL), env={'PATH':'/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE':'1',
                                      'PYTHONNOUSERSITE':'1'},
               sources=sources, binary=pin(entry), interpreter=interpreter,
               inputs=[pin(weights), pin(tokenizer)], prerequisites=[], accept='pass',
               accepted_returncodes=[0], owners=1, timeout=1200,
               writers=[writer(n, 2*1024*1024 if n != 'diagnostics' else 16384, pipe=True)
                        for n in ('stdout', 'stderr', 'diagnostics')] +
                       [writer('reference-environment.json', 16384), writer('startup-stderr.log', 524288), writer('server-stdout.log', 524288),
                        writer('journal.jsonl', 2*1024*1024), writer('observation.jsonl', 4*1024*1024),
                        writer('failure-logits.f32', 524288, False), writer('validation.json', 16384)])
    return dict(schema='supervisor-manifest-v1', jobs=[job], artifact_cap=32*1024*1024,
                build_cap=0, build_reservations=[], headroom=1024**3,
                total_seconds=1200, process_seconds=1200)


class Callbacks:
    def __init__(self, target, tokenizer):
        self.target, self.tokenizer = target, tokenizer

    def prepare(self, job, work):
        verify(self.target)

    def validate(self, job, work, code):
        # The same explicit interpreter/venv performs independent official decode
        # offline. No model or HTTP request is made by the validator.
        argv = job['argv']
        python = argv[argv.index('--python')+1]
        require(str(Path(python).resolve()) == job['interpreter']['path'], 'interpreter symlink drift')
        result = subprocess.run([job['interpreter']['path'], '-I', '-B', str(HERE/'replay.py'),
                                 str(work), str(self.tokenizer), '--reference-python', python],
                                capture_output=True, timeout=120, text=True)
        value = dict(schema='http-validation-v1', classification='pass' if code == 0 and result.returncode == 0 else 'rejected',
                     child_exit=code, replay_exit=result.returncode,
                     detail=(result.stdout+result.stderr)[-8192:],
                     scope='schedule/state/work smoke; not numerical or task-quality acceptance')
        with (work/'validation.json').open('x') as stream:
            json.dump(value, stream, allow_nan=False)
            stream.write('\n')
        return value['classification']
