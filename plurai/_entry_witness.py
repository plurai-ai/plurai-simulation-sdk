"""The entry witness: did the bench's DECLARED agent function start during a driven turn?

A ``module:Qual.name`` ref is resolved from ``sys.modules`` ONLY (importing an agent module from the
child could boot a server) and armed with ``sys.monitoring`` PY_START on that one code object, so
nothing else in the process pays for it. A start counts only while :meth:`EntryWitness.turn` is
open — a PROCESS-GLOBAL flag, not a ContextVar, because an agent may run its entry on a worker
thread that never sees the turn's context. A start during ``reset`` / ``seed_state`` does not count.

Three states, cumulative since Init: ``witnessed`` (a start inside a driven turn), ``not_witnessed``
(armed before at least one driven turn and never started in one) and ``not_determinable`` (a guard
refused to observe, with its reason). A guard can never yield ``witnessed``.

Known ceilings, both honest in direction or documented: the flag is process-global, so a background
thread that ``seed_state`` or an import started and that calls the entry on its own is counted while
a turn is open; and a module the harness first imports INSIDE a turn arms only from the next one, so
a pipeline pass (one per child) over a lazily imported agent reads ``not_determinable``.
"""

from __future__ import annotations

import inspect
import sys
import sysconfig
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

WITNESSED = 'witnessed'
NOT_WITNESSED = 'not_witnessed'
NOT_DETERMINABLE = 'not_determinable'

_TOOL_NAME = 'plurai-entry-witness'
#: Tried in order; DEBUGGER (0) and COVERAGE (1) last, since those are the ones other tools hold.
_TOOL_IDS = (3, 4, 2, 5, 1, 0)


class EntryWitness:
    """Arm, observe and report one declared entry function for one child process."""

    def __init__(self, ref: str, harness_module: types.ModuleType) -> None:
        self.ref = ref
        self._harness_module = harness_module
        self._harness_dir = Path(getattr(harness_module, '__file__', '') or '.').resolve().parent
        self._code: types.CodeType | None = None
        self._tool: int | None = None
        self._refusal: str | None = None
        self._unresolved: str | None = None
        self._in_turn = False
        self._armed_turns = 0
        self._hit = False

    def _resolve(self) -> None:
        """Try to arm; leaves either a code object, a permanent refusal, or a retryable miss."""
        if self._code is not None or self._refusal is not None:
            return
        monitoring = getattr(sys, 'monitoring', None)
        if monitoring is None:
            self._refusal = 'this interpreter has no sys.monitoring, so the entry cannot be observed'
            return
        module_name, sep, qualname = self.ref.partition(':')
        if not sep or not module_name or not qualname:
            self._refusal = f'`{self.ref}` is not a `module:Qual.name` ref'
            return
        module = self._harness_module if module_name == self._harness_module.__name__ else sys.modules.get(module_name)
        if module is None:
            self._unresolved = f'the harness never imported `{module_name}` before a turn started'
            return
        target: Any = module
        try:
            for part in qualname.split('.'):
                target = inspect.getattr_static(target, part)
            # Follows ``functools.wraps`` AND a static/classmethod's ``__wrapped__`` to the body.
            target = inspect.unwrap(target)
        except (AttributeError, ValueError) as exc:
            self._refusal = f'`{self.ref}` does not resolve: {exc}'
            return
        if not isinstance(target, types.FunctionType):
            self._refusal = f'`{self.ref}` names a `{type(target).__name__}`, not a function the turn runs'
            return
        code = target.__code__
        if code.co_qualname != qualname:
            # A monkeypatch, a mock, a decorator without ``wraps`` or a factory closure: the name
            # resolves, but to code that is not the declared body (and may be shared by others).
            self._refusal = f'`{self.ref}` resolves to `{code.co_qualname}`, not the declared function'
            return
        if not Path(code.co_filename).is_file():
            # ``mock.patch(autospec=True)`` of a module-level function, or a harness ``exec``: the
            # declared name, compiled from no file, so no path guard below can place it.
            self._refusal = f'`{self.ref}` resolves to code with no source file ({code.co_filename}), not the declared function'
            return
        if not module.__name__ == module_name == target.__module__:
            # A same-name body from another module patched in, or the name aliased in sys.modules.
            self._refusal = f'`{self.ref}` resolves to a body defined in `{target.__module__}`; name its defining module'
            return
        filename = Path(code.co_filename).resolve()
        if filename == self._harness_dir or self._harness_dir in filename.parents:
            self._refusal = f'`{self.ref}` names a harness symbol ({filename.name}), which the harness itself runs'
            return
        libraries = {Path(p).resolve() for key, p in sysconfig.get_paths().items() if key in ('stdlib', 'platstdlib', 'purelib', 'platlib')}
        if any(lib == filename or lib in filename.parents for lib in libraries):
            self._refusal = f'`{self.ref}` names an installed library symbol, which the harness may call on every turn'
            return
        for tool in _TOOL_IDS:
            try:
                monitoring.use_tool_id(tool, _TOOL_NAME)
            except ValueError:
                continue
            self._tool = tool
            break
        else:
            self._refusal = 'no free sys.monitoring tool id (a debugger or coverage tool holds them all)'
            return
        monitoring.register_callback(self._tool, monitoring.events.PY_START, self._on_start)
        monitoring.set_local_events(self._tool, code, monitoring.events.PY_START)
        self._code = code

    def _on_start(self, _code: types.CodeType, _offset: int) -> Any:
        if not self._in_turn:
            return None
        self._hit = True
        return sys.monitoring.DISABLE

    @contextmanager
    def turn(self) -> Iterator[None]:
        """Open one driven turn (or pipeline pass): resolve if not yet armed; starts inside it count."""
        self._resolve()
        if self._code is not None:
            self._armed_turns += 1
        self._in_turn = True
        try:
            yield
        finally:
            self._in_turn = False

    def report(self) -> dict[str, str]:
        """The cumulative ``{state, reason}`` since Init."""
        if self._hit:
            return {'state': WITNESSED, 'reason': f'`{self.ref}` started during a driven turn'}
        if self._refusal is not None:
            return {'state': NOT_DETERMINABLE, 'reason': self._refusal}
        if self._code is None or self._armed_turns == 0:
            return {'state': NOT_DETERMINABLE, 'reason': self._unresolved or f'`{self.ref}` was not armed before a driven turn'}
        return {'state': NOT_WITNESSED, 'reason': f'`{self.ref}` never started during {self._armed_turns} driven turn(s)'}

    def close(self) -> None:
        """Clear the local events and free the tool id. Idempotent."""
        if self._tool is None:
            return
        monitoring = sys.monitoring
        if self._code is not None:
            monitoring.set_local_events(self._tool, self._code, 0)
        monitoring.register_callback(self._tool, monitoring.events.PY_START, None)
        monitoring.free_tool_id(self._tool)
        self._tool = None
