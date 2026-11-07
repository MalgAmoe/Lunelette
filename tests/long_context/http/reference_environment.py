"""Recover the explicit reference venv on the supervisor's resolved interpreter."""
import json
from pathlib import Path
import subprocess
import sys

from build import HERE, safe_path


def bootstrap(python):
    """Preserve the historical child path order, including base distributions.

    Call before importing the provider. The launcher must already be the resolved
    pinned interpreter; invoking the venv launcher directly is not equivalent.
    """
    python = Path(python)
    if python.resolve() != Path(sys.executable).resolve():
        raise RuntimeError('explicit --python executable drift from supervisor-pinned interpreter')
    paths = json.loads(subprocess.check_output(
        [str(python), '-I', '-B', '-c',
         'import sys,json; print(json.dumps([p for p in sys.path if p.endswith("site-packages")]))'],
        text=True, timeout=30))
    for path in paths:
        sys.path.insert(0, str(safe_path(path)))
    sys.path.insert(0, str(HERE))
