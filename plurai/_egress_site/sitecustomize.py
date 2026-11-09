# ruff: noqa: INP001 — a path entry for Python subprocesses, deliberately not a package.
"""Arms ``plurai.egress_guard`` in a Python subprocess of the agent child, then defers to any other ``sitecustomize``.

Put on the subprocess's ``PYTHONPATH`` by :func:`plurai.egress_guard.arm_from_env`. The guard module is
loaded BY PATH, so a subprocess in a venv without the SDK is covered too.
"""

import contextlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
# Never break the subprocess, and never write to its stdio (its spawner's); the parent counts it not covered.
with contextlib.suppress(Exception):
    _spec = importlib.util.spec_from_file_location('_plurai_egress_guard', _here.parent / 'egress_guard.py')
    if _spec is not None and _spec.loader is not None:
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        _module.arm_from_env('python-subprocess')

# Ours shadows the next sitecustomize on the path; load that one as well.
sys.path[:] = [p for p in sys.path if os.path.realpath(p or '.') != str(_here)]
_mine = sys.modules.pop('sitecustomize', None)
try:
    importlib.import_module('sitecustomize')
except ImportError:
    if _mine is not None:
        sys.modules['sitecustomize'] = _mine
