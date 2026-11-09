"""The mocking environment's leak detector, armed in the Python agent child before the harness loads.

``egress.production_hosts`` names the customer's OWN production systems that the mock stands in
for. The preflight (:mod:`plurai.egress`) runs in the RUNNER and can only say whether the machine
reaches one; the harness substitutes its clients inside ``plurai.child``, so that is where a
substitution that missed is caught. One ``sys.addaudithook``, process-wide (so a call nested in a
tool, a library, a thread or an event loop is covered alike):

* BLOCKS the block set — ``production_hosts`` minus ``egress.allow_hosts`` (the operator's consent
  to run a surface live), the mock, the Plurai backend and loopback — and the addresses those
  hosts resolved to, with a ``PermissionError`` the agent sees as a tool error;
* RECORDS every other host the process resolves, with the calling line: the model provider,
  public services, a live tool's own calls. Nothing is blocked that cannot be attributed to a
  mocked production system;
* RECORDS subprocess spawns, and never blocks them. A Python subprocess arms the same guard
  through the ``sitecustomize`` in ``_egress_site/`` (put on its ``PYTHONPATH``); a non-Python
  one is recorded and not covered.

The runner hands the block set over in :data:`GUARD_ENV`; rows are appended as JSONL to the file
it names and folded into ``run_summary.egress.runtime``. Stdlib-only, so a subprocess can load it
by path without the SDK installed.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

#: ``{"deny": [host], "env": [VAR], "allow": [host], "log": <jsonl path>}``, set by the runner.
GUARD_ENV = 'PLURAI_EGRESS_GUARD'

#: What ``guarded`` covers and what it does not, stated wherever the verdict is.
GUARD_SCOPE = (
    'the agent process (in-process) and each Python subprocess that reports arming (python-subprocess, via '
    'sitecustomize); not covered: a Python subprocess started with its own environment or -I/-E/-S, non-Python '
    'subprocesses (recorded, not blocked), a Python subprocess that cannot write the record (stays armed, counted not '
    'covered), an event loop made directly from uvloop, C-level sockets (grpc core, libpq, libcurl) and ctypes'
)

#: Where a guard runs. Only ``in-process`` (``plurai.child``) owns its stderr: a subprocess's stdio belongs to whoever
#: spawned it, often the agent's own tool, so a guard there reports a failure on no stdio at all.
Scope = Literal['in-process', 'python-subprocess']

#: The fix-it text a block carries, in the record and on the report line.
BLOCK_REMEDY = (
    'if this surface was approved to run LIVE, add its host to `egress.allow_hosts` (scripts/record_egress.py) '
    'to consent to live use; otherwise route it through the mock'
)

#: The directory whose ``sitecustomize.py`` arms a Python subprocess.
SITE_DIR = str(Path(__file__).parent / '_egress_site')

_RESOLVE = frozenset({'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyname_ex'})
_CONNECT = frozenset({'socket.connect', 'socket.sendto'})
_SPAWN = frozenset({'subprocess.Popen', 'os.posix_spawn', 'os.exec', 'os.spawn', 'os.system'})
_EVENTS = _RESOLVE | _CONNECT | _SPAWN

#: Frames under these roots are library code, not the line that asked for the host.
_LIBRARY_ROOTS = tuple(
    {os.path.realpath(p) for key in ('stdlib', 'platstdlib', 'purelib', 'platlib') if (p := sysconfig.get_paths().get(key))}
    | {os.path.realpath(Path(__file__).parent)}
)


def _is_local(host: str) -> bool:
    if host in ('', 'localhost'):
        return True
    try:
        address = ipaddress.ip_address(host.split('%', 1)[0])
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _text(value: Any) -> str:
    raw = value.decode('ascii', 'replace') if isinstance(value, bytes) else str(value or '')
    return raw.lower().rstrip('.')


def _addresses(host: str) -> set[str]:
    try:
        return {str(info[4][0]) for info in socket.getaddrinfo(host, None)}
    except OSError:
        return set()


def _host_of(value: str) -> str:
    """The host of an env value: a URL, ``host:port`` or a bare host."""
    value = value.strip()
    if '://' in value:
        return _text(urlsplit(value).hostname)
    return _text(value.split('/', 1)[0].rsplit(':', 1)[0] if value.count(':') <= 1 else value)


class _Guard:
    def __init__(self, spec: dict[str, Any], scope: Scope) -> None:
        self.scope = scope
        self.log = str(spec['log'])
        self.allow = frozenset(_text(h) for h in spec.get('allow') or ())
        self.deny: frozenset[str] = frozenset()
        self.deny_ips: frozenset[str] = frozenset()
        self.unresolved: list[str] = []
        self.names: set[str] = set()
        self._shared: set[str] = set()  # every address ever answered for a name that is not declared
        self._seen: set[tuple[str, str, str | None]] = set()
        self._busy = threading.local()
        self._cwd = os.path.realpath(Path.cwd())
        self.add_hosts(str(h) for h in spec.get('deny') or ())

    def add_hosts(self, hosts: Any) -> None:
        """Join hosts to the block set. Resolved while the hook is idle, so arming is never itself refused;
        loopback addresses are left out, so a declared name mapped to 127.0.0.1 cannot take the local mock down."""
        new = {_text(h) for h in hosts if h} - self.deny
        resolved = {h: _addresses(h) for h in new}
        self.unresolved += sorted(h for h, ips in resolved.items() if not ips)
        self.deny |= new
        self.deny_ips |= {ip for ips in resolved.values() for ip in ips if not _is_local(ip)}

    def add_env_hosts(self, names: Any) -> None:
        """Resolve ``{env: VAR}`` production hosts from THIS process's environment, once the harness built it."""
        for name in names:
            raw = os.environ.get(str(name))
            host = _host_of(raw) if raw else ''
            blocked = bool(host) and not _is_local(host) and host not in self.allow
            if blocked:
                self._busy.on = True
                try:
                    self.add_hosts([host])
                finally:
                    self._busy.on = False
            self._record('env_host', env=str(name), host=host or None, blocked=blocked)

    def after_harness_loaded(self) -> None:
        """In ``plurai.child``: resolve the ``env`` hosts, and hand the grown block set to Python subprocesses."""
        spec = json.loads(os.environ[GUARD_ENV])
        self.add_env_hosts(spec.get('env') or ())
        os.environ[GUARD_ENV] = json.dumps({**spec, 'deny': sorted(self.deny)})

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        if event == 'import' and args and args[0] == 'uvloop':
            # A loop made directly from uvloop resolves and connects in libuv, with no audit event.
            self._record('not_covered', command='uvloop', why='uvloop loaded: an event loop made directly from it is not covered')
            return
        if event not in _EVENTS or getattr(self._busy, 'on', False):
            return
        self._busy.on = True
        try:
            self._handle(event, args)
        finally:
            self._busy.on = False

    def _handle(self, event: str, args: tuple[Any, ...]) -> None:
        if event in _SPAWN:
            self._record_spawn(event, args)
            return
        if event in _RESOLVE:
            host = _text(args[0])
            if self._denied(host):
                self._block(host)
            if not _is_local(host):
                if host not in self.names:
                    self._shared |= _addresses(host)  # its first answer, before the connect that follows
                self.names.add(host)
                self._record('observed', host=host)
            return
        address = args[1]
        if isinstance(address, tuple) and address:
            ip = _text(address[0])
            if self._denied(ip):
                self._block(ip)
            if not _is_local(ip) and ip not in self._shared:  # a raw-address connect never passes through name resolution; a name's answer folds into it
                self._record('observed', host=ip)

    def _denied(self, target: str) -> bool:
        """A declared production host, or an address that belongs only to declared hosts."""
        return target in self.deny or (target in self.deny_ips and self._exclusive(target))

    def _exclusive(self, ip: str) -> bool:
        """The address belongs ONLY to declared hosts: it was never answered for an allowed host (the mock,
        the backend, a consented host) or any other name this process reached. Frontends share addresses and
        answers rotate per lookup, so every answer seen is kept, never only the latest."""
        for name in (self.allow | self.names) - self.deny:
            self._shared |= _addresses(name)
        return ip not in self._shared

    def _record_spawn(self, event: str, args: tuple[Any, ...]) -> None:
        """One row per spawn site. A Python one says whether it CAN arm; only its own `armed` row counts it covered."""
        # Where argv sits: Popen (executable, args, cwd, env), posix_spawn/exec (path, argv, env), spawn (mode, path, args, env).
        first = {'subprocess.Popen': 1, 'os.posix_spawn': 1, 'os.exec': 1, 'os.spawn': 2}.get(event, 0)
        raw = args[first] if first < len(args) else None
        argv = [os.fsdecode(a) for a in raw] if isinstance(raw, (list, tuple)) else str(raw or '').split()
        command = Path(argv[0]).name if argv else '?'
        if not command.startswith('python'):
            self._record('spawned', command=command)
            return
        env = {'subprocess.Popen': 3, 'os.posix_spawn': 2, 'os.exec': 2, 'os.spawn': 3}.get(event)
        env_map = args[env] if env is not None and env < len(args) else None
        reason = None
        if env_map is not None and (GUARD_ENV not in env_map or SITE_DIR not in str(env_map.get('PYTHONPATH', ''))):
            reason = 'started with its own environment'
        for token in argv[1:]:
            if token in ('-c', '-m') or not token.startswith('-') or token.startswith('--'):
                break
            if set(token[1:]) & {'I', 'E', 'S'}:
                reason = f'started with {token}'
        self._record('spawned', command=command, python=True, cannot_arm=reason)

    def _block(self, host: str) -> None:
        self._record('blocked', host=host, remedy=BLOCK_REMEDY)
        raise PermissionError(f'[plurai-egress] refused: {host} is a declared production host the mock stands in for. {BLOCK_REMEDY}.')

    def _callsite(self) -> str | None:
        """The first frame outside the stdlib, site-packages and this SDK (``<string>`` is ``python -c`` code);
        else the first library frame, by file name only."""
        frame = sys._getframe(1)
        first: str | None = None
        while frame is not None:
            raw = frame.f_code.co_filename
            if raw != __file__:
                path = os.path.realpath(raw)
                if not raw.startswith('<frozen') and not path.startswith(_LIBRARY_ROOTS):
                    return f'{os.path.relpath(path, self._cwd) if path.startswith(self._cwd + os.sep) else raw}:{frame.f_lineno}'
                first = first or f'{Path(raw).name}:{frame.f_lineno}'
            frame = frame.f_back
        return first

    def _record(self, kind: str, **fields: Any) -> None:
        subject = str(fields.get('host') or fields.get('command') or fields.get('env') or '') + str(fields.get('cannot_arm') or '')
        if (kind, subject, None) in self._seen:  # one row per observed host, whatever calls it next
            return
        callsite = None if kind in ('armed', 'env_host') else self._callsite()
        key = (kind, subject, None if kind == 'observed' else callsite)
        if key in self._seen:
            return
        self._seen.add(key)
        line = json.dumps({'kind': kind, **fields, 'callsite': callsite, 'scope': self.scope, 'pid': os.getpid()}) + '\n'
        try:
            fd = os.open(self.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line.encode('utf-8'))
            finally:
                os.close(fd)
        except OSError as exc:
            if self.scope == 'in-process':  # a subprocess stays armed and silent: the parent counts it not covered
                print(f'[plurai-egress] WARNING: could not record {kind} to {self.log} ({exc}).', file=sys.stderr)


def arm_from_env(scope: Scope = 'in-process') -> _Guard | None:
    """Arm the guard when the runner asked for it; at most once per process.

    ``in-process`` (``plurai.child``) resolves its ``env`` hosts AFTER the harness loads
    (:meth:`_Guard.after_harness_loaded`).
    ``python-subprocess`` inherits an environment that is already built, so it resolves them now.
    A guard that cannot arm or record says so on stderr only ``in-process``; in a subprocess it stays silent,
    and the run continues.
    """
    raw = os.environ.get(GUARD_ENV)
    if not raw or getattr(sys, '_plurai_egress_guard', None) is not None:
        return None
    try:
        spec = json.loads(raw)
        guard = _Guard(spec, scope)
    except Exception as exc:  # noqa: BLE001 — the runner reports no armed process
        if scope == 'in-process':
            print(f'[plurai-egress] WARNING: the egress guard did NOT arm ({type(exc).__name__}: {exc}).', file=sys.stderr)
        return None
    sys.addaudithook(guard)
    sys._plurai_egress_guard = guard  # type: ignore[attr-defined]
    _keep_the_default_loop(guard)
    guard._record('armed', deny=sorted(guard.deny), unresolved=guard.unresolved)
    if 'uvloop' in sys.modules:  # imported before arming: the audit event has already gone by
        guard('import', ('uvloop',))
    if scope != 'in-process':
        guard.add_env_hosts(spec.get('env') or ())
    # Every Python subprocess from here on runs the arming ``sitecustomize`` first.
    rest = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p and p != SITE_DIR]
    os.environ['PYTHONPATH'] = os.pathsep.join([SITE_DIR, *rest])
    return guard


def armed_in_process() -> bool | None:
    """``True`` when this process armed the guard as the agent child; the child's Init reply carries it."""
    guard = getattr(sys, '_plurai_egress_guard', None)
    return True if guard is not None and guard.scope == 'in-process' else None


def _keep_the_default_loop(guard: _Guard) -> None:
    """A uvloop event-loop POLICY is not installed while the guard is armed.

    libuv resolves and connects without audit events, so every later ``asyncio.run`` would bypass the
    guard; a local mock's uvicorn installs that policy process-wide. The default loop is kept and the
    refusal recorded. A loop made directly from uvloop is outside this and stated in :data:`GUARD_SCOPE`.
    """
    import asyncio  # noqa: PLC0415 — only when arming

    install = asyncio.set_event_loop_policy

    def _set_event_loop_policy(policy: Any) -> None:
        if policy is not None and type(policy).__module__.split('.', 1)[0] == 'uvloop':
            guard._record('uvloop_policy_refused', command='uvloop')
            return
        install(policy)

    asyncio.set_event_loop_policy = _set_event_loop_policy
