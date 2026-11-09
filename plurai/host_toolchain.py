"""Whether this host gives the agent what its own files declare it needs, checked before a run.

``bench_manifest.json`` → ``harness.host_requirements`` names each program or module the agent's own
files say it needs on this host (the scan writes it, citing the line). This module resolves each one
to ONE typed fact, where the agent would use it:

* ``present`` — found, and its declared ``probe`` exited 0 (a module: it imports);
* ``missing`` — not on this host;
* ``fails_sandboxed`` — found, and its probe exited non-zero inside the Bash sandbox the agent's CLI
  runs it in, without a grant;
* ``broken`` — found, and it does not work where the agent runs it outside any sandbox: the probe
  exited non-zero, or a module raised on import;
* ``unverified`` — found, and no probe is declared: nothing shows it runs where the agent runs it. A
  module is never one — its import IS its probe (``import x`` in the agent's interpreter, ``require('x')``
  by ``node`` from a fresh folder) — and an interpreter declared with no probe of its own is ``present``
  once a declared module imported through it;
* ``not_checked`` — the lookup itself could not run (the agent's shell may not run programs here, say);
  ``detail`` says why.

Only ``missing`` / ``broken`` / ``fails_sandboxed`` are measured gaps (:data:`GAP_STATUSES`). Each row
carries its ``remedy`` from one table (:data:`REMEDY`), keyed by the status: it says where the program failed and who acts, never why it failed.

Each non-present row is one gate finding (:func:`host_findings`): ``missing``, ``broken`` and
``fails_sandboxed`` are measured gaps and HOLD the run until an overrule lifts them; ``unverified``
and ``not_checked`` measured nothing, so they ship with their finding. A probe that never reached a
verdict (a timeout, a session that died) is ``not_checked``, never ``broken``.

WHERE the agent runs it decides where the probe runs. An agent whose programs run through the Claude
Code CLI's own ``Bash`` (``surfaces.host_effects`` lists ``Bash``) runs them under
:func:`plurai.host_effects.confine`, so the probe runs there too: the agent's own CLI, confined by
the same call, is driven by a stand-in model endpoint on loopback that asks for exactly one Bash
call per probe. Nothing reaches a model. Any other agent starts the program itself, so the probe
runs as a plain process in the agent's interpreter environment. Either way the verdict is the exit
status of the run, never its wording. A probe element ``{agent}/<rel>`` is the agent's own file,
passed to the runner as an absolute path.

The run gate (``plurai-run-simulation``) and the plugin's run-gate both call this one resolver, in
the agent's interpreter: ``python -m plurai.host_toolchain --workdir <plurai_sim> --json``. Its record
carries what is missing as a structured ``install`` plan (:func:`install_plan`, with its block head) — built only
from the install lines the agent's own install docs give (``install`` + ``install_evidence``) — for ONE copy-paste
install block (:func:`render_install_block`); nothing here runs it.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

from plurai import exit_codes
from plurai._platform import utf8_child_text, utf8_stdio
from plurai.gates.apply import make_finding
from plurai.gates.catalogue import HOST_REQUIREMENT_GAP, HOST_REQUIREMENT_NOT_CHECKED, HOST_REQUIREMENT_UNVERIFIED
from plurai.gates.contract import GateFinding
from plurai.host_effects import (
    _CLI_KEYS,
    _CLI_PROVIDER_SWITCHES,
    BENCH_MANIFEST_FILENAME,
    UNSANDBOXED_GRANT_KEY,
    HostRequirement,
    confine,
    declared_host_requirements,
)
from plurai.user_path import expand_user_path

Status = Literal['present', 'missing', 'fails_sandboxed', 'broken', 'unverified', 'not_checked']

#: The folder beside ``bench_manifest.json`` whose files a probe's working folder starts with (a
#: fixture the declared probe converts, say). Optional.
PROBE_INPUTS_DIRNAME = 'host_probes'

#: Seconds one probe may run. A cold start of a document converter is slow; a hang is a failure.
PROBE_TIMEOUT_SECONDS = 120

#: Where the probes ran: through the agent's own confined CLI Bash, or as the agent's own process.
Confinement = Literal['agent_cli_bash', 'agent_process']

#: A probe element naming the agent's own file: ``{agent}/<path under the agent's folder>``.
AGENT_PATH_PREFIX = '{agent}/'

#: What the operator does about each gap. Where it failed, never why: nothing on the run can tell
#: that a program failed because of the sandbox.
REMEDY: dict[str, str] = {
    'missing': 'install it on this machine, then re-run',
    'fails_sandboxed': (
        "it is installed, and its probe failed inside the Bash sandbox the agent's CLI runs it in, without a grant; "
        "nothing here tells why — a missing dependency fails there too. Fix this host's other gaps first, then re-run. "
        f'Only if it still fails there can {UNSANDBOXED_GRANT_KEY} in {BENCH_MANIFEST_FILENAME} let it out: a granted '
        'program runs with full read, write and network access to this machine, and a command leaves the sandbox only '
        'when every program in it is granted'
    ),
    'broken': 'repair it on this machine, then re-run',
    'unverified': "declare a probe on the agent's documented route to it — re-run /simulation-scan-agent",
    'not_checked': 'the check could not run here — fix what the detail names, then re-run',
}

#: The statuses that are a MEASURED gap (they hold the run); the rest of the non-present ones measured nothing.
GAP_STATUSES: frozenset[str] = frozenset({'missing', 'broken', 'fails_sandboxed'})

#: How a gap reads in the run's one host stamp, and whether it may claim the agent took a fallback path.
_STAMP: dict[str, tuple[str, bool]] = {
    'missing': ('is missing', True),
    'broken': ('is broken', True),
    'not_checked': ('is not checked', True),
    'fails_sandboxed': ('fails inside the Bash sandbox', False),
    'unverified': ('is not proven usable', False),
}

#: The environment a stand-in CLI session gets on top of the confined options' own: every provider
#: switch off and every sign-in key blank, so the CLI's only route is the loopback stand-in.
_STAND_IN_ENV_OFF = {**dict.fromkeys(_CLI_PROVIDER_SWITCHES, '0'), **dict.fromkeys(_CLI_KEYS, '')}


@dataclass(frozen=True)
class RequirementStatus:
    """One declared requirement, resolved on this host where the agent would use it."""

    name: str
    kind: str
    status: Status
    detail: str = ''
    #: What the operator does about it ('' for ``present``): :data:`REMEDY`, plus the route's own clause.
    remedy: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {'name': self.name, 'kind': self.kind, 'status': self.status, 'detail': self.detail, 'remedy': self.remedy}


@dataclass(frozen=True)
class HostToolchain:
    """Every declared requirement's status, and where the probes ran."""

    requirements: tuple[RequirementStatus, ...] = ()
    confinement: Confinement = 'agent_process'
    #: What is missing, structured for the install block (:func:`install_plan`), and the block's head.
    install_plan: Mapping[str, Any] = field(default_factory=dict)
    install_head: str = ''

    @property
    def install_block(self) -> str:
        """The ONE copy-paste install block (:func:`render_install_block`); '' when nothing is missing."""
        return render_install_block(self.install_plan, self.install_head) if self.install_plan else ''

    def gaps(self) -> tuple[RequirementStatus, ...]:
        """The requirements this host does not give the agent, or that could not be checked."""
        return tuple(r for r in self.requirements if r.status != 'present')

    def as_dict(self) -> dict[str, Any]:
        install = {'head': self.install_head, **self.install_plan} if self.install_plan else {}
        return {'confinement': self.confinement, 'requirements': [r.as_dict() for r in self.requirements], 'install': install}


@dataclass(frozen=True)
class ProbeRun:
    """One probe's outcome: ``exit_ok`` is ``None`` when it never ran (``detail`` says why).
    ``sandboxed``: it ran inside the Bash sandbox, as the runner's own confinement record says."""

    exit_ok: bool | None
    detail: str = ''
    sandboxed: bool = False


#: ``(argvs, workdir, timeout) -> one ProbeRun per argv``: how a set of probes is run. Injectable for tests.
ProbeRunner = Callable[[Sequence[Sequence[str]], Path, float], list[ProbeRun]]


def _head(text: str, *, redact: Sequence[str] = (), limit: int = 300) -> str:
    """The first ``limit`` characters of ``text`` on one line, with the given paths named generically."""
    for path in sorted({p for p in redact if p}, key=len, reverse=True):
        text = text.replace(path, '<probe dir>' if 'plurai-host-probe' in path else '<home>')
    return ' '.join(text.split())[:limit]


@contextlib.contextmanager
def _probe_folder(workdir: Path) -> Iterator[Path]:
    """A fresh working folder for the probes, starting with a copy of ``<workdir>/host_probes/``."""
    with tempfile.TemporaryDirectory(prefix='plurai-host-probe-') as tmp:
        folder = Path(tmp) / 'run'
        inputs = workdir / PROBE_INPUTS_DIRNAME
        if inputs.is_dir():
            shutil.copytree(inputs, folder)
        else:
            folder.mkdir()
        yield folder


def run_as_agent_process(argvs: Sequence[Sequence[str]], workdir: Path, timeout: float) -> list[ProbeRun]:
    """Each probe as a plain process in this interpreter's environment, as an agent that starts it itself would."""
    runs: list[ProbeRun] = []
    with _probe_folder(workdir) as folder:
        redact = (str(folder.parent), str(Path.home()))
        for argv in argvs:
            try:
                done = subprocess.run(list(argv), cwd=folder, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603
            except FileNotFoundError:
                runs.append(ProbeRun(False, f'{argv[0]} could not be started (not found)'))
                continue
            except subprocess.TimeoutExpired:
                # Never reached a verdict: not "exited non-zero".
                runs.append(ProbeRun(None, f'timed out after {timeout:.0f}s'))
                continue
            except OSError as exc:
                runs.append(ProbeRun(False, f'{type(exc).__name__}: {exc}'))
                continue
            detail = '' if done.returncode == 0 else f'exit {done.returncode}: ' + _head(done.stderr or done.stdout, redact=redact)
            runs.append(ProbeRun(done.returncode == 0, detail.rstrip(': ')))
    return runs


class _StandInModel:
    """A loopback Messages-API endpoint that asks the CLI for exactly one ``Bash`` call per probe.

    Each streamed reply is the next probe as a ``tool_use`` while one is left, then a plain end of
    turn; a request that offers no ``Bash`` tool (a side call of the CLI's own) gets the plain end.
    It records each probe's ``tool_result`` by its call id: ``is_error`` is the CLI's own flag for a
    command that exited non-zero.
    """

    def __init__(self, commands: Sequence[str]) -> None:
        self.commands = list(commands)
        self.results: dict[int, dict[str, Any]] = {}
        self.asked = 0
        self.offered_bash = False
        self._lock = threading.Lock()
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def do_HEAD(self) -> None:
                self.send_response(200)
                self.end_headers()

            def do_GET(self) -> None:
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get('content-length') or 0)) or b'{}')
                except ValueError:
                    body = {}
                if 'count_tokens' in self.path:
                    self._send_json({'input_tokens': 1})
                    return
                blocks, stop = model.reply(body if isinstance(body, dict) else {})
                if body.get('stream'):
                    self._send_stream(body, blocks, stop)
                else:
                    self._send_json(
                        {
                            'id': 'msg_probe',
                            'type': 'message',
                            'role': 'assistant',
                            'model': body.get('model'),
                            'content': blocks,
                            'stop_reason': stop,
                            'stop_sequence': None,
                            'usage': {'input_tokens': 1, 'output_tokens': 1},
                        }
                    )

            def _send_json(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('content-type', 'application/json')
                self.send_header('content-length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_stream(self, body: dict[str, Any], blocks: list[dict[str, Any]], stop: str) -> None:
                self.send_response(200)
                self.send_header('content-type', 'text/event-stream')
                self.end_headers()

                def event(name: str, data: dict[str, Any]) -> None:
                    self.wfile.write(f'event: {name}\ndata: {json.dumps(data)}\n\n'.encode())

                event(
                    'message_start',
                    {
                        'type': 'message_start',
                        'message': {
                            'id': 'msg_probe',
                            'type': 'message',
                            'role': 'assistant',
                            'model': body.get('model'),
                            'content': [],
                            'stop_reason': None,
                            'stop_sequence': None,
                            'usage': {'input_tokens': 1, 'output_tokens': 1},
                        },
                    },
                )
                for index, block in enumerate(blocks):
                    if block['type'] == 'text':
                        event('content_block_start', {'type': 'content_block_start', 'index': index, 'content_block': {'type': 'text', 'text': ''}})
                        event('content_block_delta', {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': block['text']}})
                    else:
                        start = {'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': {}}
                        event('content_block_start', {'type': 'content_block_start', 'index': index, 'content_block': start})
                        delta = {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}
                        event('content_block_delta', {'type': 'content_block_delta', 'index': index, 'delta': delta})
                    event('content_block_stop', {'type': 'content_block_stop', 'index': index})
                event('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': stop, 'stop_sequence': None}, 'usage': {'output_tokens': 1}})
                event('message_stop', {'type': 'message_stop'})
                self.wfile.flush()

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)

    @staticmethod
    def call_id(index: int) -> str:
        return f'toolu_plurai_host_probe_{index}'

    def reply(self, body: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
        with self._lock:
            for message in body.get('messages') or ():
                content = message.get('content') if isinstance(message, dict) else None
                for block in content if isinstance(content, list) else ():
                    if isinstance(block, dict) and block.get('type') == 'tool_result':
                        for index in range(len(self.commands)):
                            if block.get('tool_use_id') == self.call_id(index):
                                self.results.setdefault(index, block)
            offers_bash = any(isinstance(tool, dict) and tool.get('name') == 'Bash' for tool in body.get('tools') or ())
            self.offered_bash = self.offered_bash or offers_bash
            if offers_bash and self.asked < len(self.commands) and self.asked <= len(self.results):
                index = self.asked
                self.asked += 1
                return [
                    {
                        'type': 'tool_use',
                        'id': self.call_id(index),
                        'name': 'Bash',
                        'input': {'command': self.commands[index], 'description': 'host requirement probe'},
                    }
                ], 'tool_use'
            return [{'type': 'text', 'text': 'done'}], 'end_turn'

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.server.server_address[1]}'


def _tool_result_text(block: Mapping[str, Any]) -> str:
    content = block.get('content')
    if isinstance(content, list):
        return ' '.join(str(part.get('text') or '') for part in content if isinstance(part, dict))
    return str(content or '')


def run_in_agent_cli_bash(argvs: Sequence[Sequence[str]], workdir: Path, timeout: float) -> list[ProbeRun]:
    """Each probe as one ``Bash`` call of the agent's own CLI, confined by :func:`plurai.host_effects.confine`.

    ONE CLI session serves every probe. The command is the argv alone (``shlex.join``), so a program
    the operator let out of the sandbox leaves it exactly as the agent's own bare call would. A probe
    the session never ran (no CLI, a CLI that failed, no ``Bash`` on offer) is ``exit_ok=None``.
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except ImportError as exc:
        return [ProbeRun(None, f"claude_agent_sdk is not importable in the agent's interpreter ({exc})")] * len(argvs)
    model = _StandInModel([shlex.join(list(argv)) for argv in argvs])
    thread = threading.Thread(target=model.server.serve_forever, daemon=True)
    thread.start()
    session_error = ''
    try:
        with _probe_folder(workdir) as folder:
            env = {
                **_STAND_IN_ENV_OFF,
                'ANTHROPIC_BASE_URL': model.url,
                'ANTHROPIC_API_KEY': 'plurai-host-probe',
                'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
                # no CLAUDE_CONFIG_DIR of its own: confine refuses one (it would move the settings it
                # confines with), and a keyed CLI gets a config dir of the scenario's own from confine,
                # so the probe session's transcript and state never reach the operator's ~/.claude
            }
            # the CLI's own stderr (its sign-in notices) is not this command's output
            options = ClaudeAgentOptions(env=env, allowed_tools=['Bash'], max_turns=len(argvs) + 2, setting_sources=[], stderr=lambda _line: None)
            record = confine(options, run_dir=folder, repo=workdir.parent, workdir=workdir, trail=[])
            bash = str((record.get('denied') or {}).get('Bash') or '')
            if bash.startswith('disallowed'):
                return [ProbeRun(None, f"the agent's Bash does not run here ({bash})")] * len(argvs)
            # a command leaves the sandbox when Bash itself is granted live, or when its one program is granted
            let_out = set((record.get('unsandboxed_grant') or {}).get('names') or ())
            bash_live = 'Bash' in (record.get('granted') or ())

            async def drive() -> None:
                async for _message in query(prompt='Run the host requirement probes.', options=options):
                    pass

            try:
                asyncio.run(asyncio.wait_for(drive(), timeout=timeout * len(argvs) + 60))
            except Exception as exc:  # noqa: BLE001 — a session that died is reported per probe, below
                session_error = f'{type(exc).__name__}: {exc}'
            redact = (str(folder.parent), str(Path.home()))
            runs = []
            for index in range(len(argvs)):
                result = model.results.get(index)
                if result is None:
                    why = session_error or ('the CLI offered no Bash tool' if not model.offered_bash else 'the CLI never ran it')
                    runs.append(ProbeRun(None, f"the agent's CLI did not run the probe ({_head(why, redact=redact)})"))
                elif result.get('is_error'):
                    sandboxed = not bash_live and argvs[index][0] not in let_out
                    runs.append(ProbeRun(False, _head(_tool_result_text(result), redact=redact) or 'exited non-zero', sandboxed=sandboxed))
                else:
                    runs.append(ProbeRun(True))
            return runs
    finally:
        model.server.shutdown()
        model.server.server_close()


def _runs_in_agent_cli_bash(workdir: Path) -> bool:
    """Whether the agent runs host programs through its CLI's own ``Bash``: the scan lists ``Bash``
    among ``surfaces.host_effects`` (the CLI built-ins the agent has, which :func:`confine` confines)."""
    try:
        manifest = json.loads((workdir / BENCH_MANIFEST_FILENAME).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    surfaces = manifest.get('surfaces') if isinstance(manifest, dict) else None
    effects = surfaces.get('host_effects') if isinstance(surfaces, dict) else None
    return isinstance(effects, list) and 'Bash' in effects


def _status(requirement: HostRequirement, status: Status, detail: str) -> RequirementStatus:
    """A gap, with its :data:`REMEDY`. A route probe (headed by another program) that failed sandboxed adds
    that a grant of the program does not free the route: the command's program is the route's head."""
    remedy = REMEDY.get(status, '')
    if status == 'fails_sandboxed' and requirement.probe and requirement.probe[0] != requirement.name:
        remedy += f"; the agent's route (`{requirement.probe[0]} …`) stays sandboxed even with a grant of {requirement.name}"
    return RequirementStatus(requirement.name, requirement.kind, status, detail, remedy)


def _expand_agent_paths(probe: Sequence[str], agent: Path) -> list[str]:
    """The probe as the runner gets it: each ``{agent}/<rel>`` element is that file of the agent, absolute."""
    return [str(agent / arg[len(AGENT_PATH_PREFIX) :]) if arg.startswith(AGENT_PATH_PREFIX) else arg for arg in probe]


def _import_status(requirement: HostRequirement) -> RequirementStatus:
    """A python module: absent from the interpreter is ``missing``; importing it raising is ``broken``."""
    try:
        found = importlib.util.find_spec(requirement.name) is not None
    except (ImportError, ValueError) as exc:
        return _status(requirement, 'broken', f'{type(exc).__name__}: {exc}')
    if not found:
        return _status(requirement, 'missing', "not importable by the agent's interpreter")
    # In a fresh interpreter: an import that raises must not take this process's state with it.
    code = 'import importlib, sys; importlib.import_module(sys.argv[1])'
    try:
        done = subprocess.run([sys.executable, '-c', code, requirement.name], capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS, check=False)  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _status(requirement, 'not_checked', f'the import could not be run: {type(exc).__name__}: {exc}')
    if done.returncode != 0:
        last = (done.stderr.strip().splitlines() or [f'exit {done.returncode}'])[-1]
        return _status(requirement, 'broken', f'importing it raised: {_head(last, redact=(str(Path.home()),))}')
    return RequirementStatus(requirement.name, requirement.kind, 'present')


def _node_status(requirement: HostRequirement) -> RequirementStatus:
    """A node module, from a fresh folder as the agent runs node: unresolvable is ``missing``; ``require`` raising is ``broken``."""
    node = shutil.which('node')
    if node is None:
        # its import probe's own program is absent: the agent cannot require it either — a measured gap
        return _status(requirement, 'missing', _NO_NODE_DETAIL)
    with tempfile.TemporaryDirectory(prefix='plurai-host-probe-') as cwd:
        steps = (
            ('require.resolve(process.argv[1])', 'missing', 'node cannot resolve it from a fresh folder'),
            ('require(process.argv[1])', 'broken', 'require() raised'),
        )
        for script, failed, meaning in steps:
            try:
                done = subprocess.run(  # noqa: S603
                    [node, '-e', script, requirement.name],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=PROBE_TIMEOUT_SECONDS,
                    check=False,
                    **utf8_child_text(),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return _status(requirement, 'not_checked', f'node could not be run: {type(exc).__name__}: {exc}')
            if done.returncode != 0:
                return _status(requirement, failed, f'{meaning} (exit {done.returncode})')  # type: ignore[arg-type]
    return RequirementStatus(requirement.name, requirement.kind, 'present')


def _interpreter(kind: str) -> str | None:
    """The program a module kind's import probe runs (:func:`_import_status`, :func:`_node_status`)."""
    return {'python_module': sys.executable, 'node_module': shutil.which('node')}.get(kind)


def _same_program(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    try:
        return Path(a).samefile(b)
    except OSError:
        return False


def _proven_by_imports(declared: Sequence[HostRequirement], statuses: dict[int, RequirementStatus]) -> None:
    """An executable with no probe of its own that is the interpreter a declared module imported through, where the
    agent runs it, is ``present``: that import ran it. Never the reverse — a module that failed proves nothing of it."""
    for index, requirement in enumerate(declared):
        if statuses[index].status != 'unverified':
            continue
        found = shutil.which(requirement.name)
        imported = [m.name for j, m in enumerate(declared) if statuses[j].status == 'present' and _same_program(found, _interpreter(m.kind))]
        if imported:
            detail = f'not probed itself; {", ".join(imported)} imported through it'
            statuses[index] = RequirementStatus(requirement.name, requirement.kind, 'present', detail)


#: The block's first line; ``{agent}`` is the agent's folder, where the docs' relative lines run.
INSTALL_BLOCK_HEAD = "# install on this machine, then re-run — run each line in {agent} (the agent's folder), as the agent's own install docs give it:"

#: The detail of a node module found missing because ``node`` itself is not on PATH (:func:`_node_status`).
_NO_NODE_DETAIL = 'node, which its import probe runs, is not on PATH'


def install_plan(requirements: Sequence[Mapping[str, Any]], declared: Sequence[HostRequirement]) -> dict[str, Any]:
    """What this host is missing, structured for the ONE install block (:func:`render_install_block`), from the
    ``missing`` rows of ``requirements``: ``lines`` — each install command the agent's own install docs give, with
    its citation and the names it installs (a command once, however many cite it); ``uncited`` — the names that
    declare none (the docs give none, or the scan predates the field); ``interpreters`` — ``node``, when a node
    module is missing because node is not on PATH, with the modules that need it. Never a command of this module's
    own. The plugin's run-gate reads this from the resolver's JSON and drops the names whose gap it overruled."""
    cited = {(r.name, r.kind): r for r in declared if r.install and r.install_evidence}
    lines: dict[str, dict[str, Any]] = {}
    uncited: list[str] = []
    needs_node: list[str] = []
    for row in requirements:
        if row.get('status') != 'missing':
            continue
        name = str(row.get('name'))
        if row.get('kind') == 'node_module' and row.get('detail') == _NO_NODE_DETAIL:
            needs_node.append(name)
        if (req := cited.get((name, str(row.get('kind'))))) is None:
            uncited.append(name)
        else:
            lines.setdefault(req.install, {'command': req.install, 'evidence': req.install_evidence, 'names': []})['names'].append(name)
    interpreters = [{'name': 'node', 'for': needs_node}] if needs_node else []
    return {'interpreters': interpreters, 'lines': list(lines.values()), 'uncited': uncited}


def render_install_block(plan: Mapping[str, Any], head: str) -> str:
    """The ONE copy-paste block ('' when nothing is missing), every line but the commands a comment so it pastes whole:

    * ``head`` (:data:`INSTALL_BLOCK_HEAD`, naming the agent's folder);
    * ``# node is not on PATH: <names> need it — install it first`` per interpreter, unless the block names it already;
    * ``<command>  # <path:line>: <names>`` per cited install command, verbatim;
    * ``# <names>: no install line declared``.

    The plugin's run-gate renders the same plan with the same four shapes (``bench_manifest._render_install_block``):
    change both together. The operator installs; nothing here runs anything."""
    lines = [line for line in plan.get('lines') or () if line.get('names')]
    uncited = list(plan.get('uncited') or ())
    shown = {name for line in lines for name in line['names']} | set(uncited)
    interpreters = [i for i in plan.get('interpreters') or () if i.get('for') and i.get('name') not in shown]
    if not (lines or uncited or interpreters):
        return ''
    out = [head]
    out += [f'# {i["name"]} is not on PATH: {", ".join(i["for"])} need it — install it first' for i in interpreters]
    out += [f'{line["command"]}  # {line["evidence"]}: {", ".join(line["names"])}' for line in lines]
    if uncited:
        out.append(f'# {", ".join(uncited)}: no install line declared')
    return '\n'.join(out)


def install_block(requirements: Sequence[Mapping[str, Any]], declared: Sequence[HostRequirement], agent: Path) -> str:
    """:func:`render_install_block` of :func:`install_plan`, headed by ``agent``, the folder its relative lines run in."""
    return render_install_block(install_plan(requirements, declared), INSTALL_BLOCK_HEAD.format(agent=agent))


def check_host_toolchain(workdir: str | Path, *, probe_runner: ProbeRunner | None = None, timeout: float = PROBE_TIMEOUT_SECONDS) -> HostToolchain:
    """Resolve every requirement ``<workdir>/bench_manifest.json`` declares, on this host, where the agent uses it.

    Meant for the AGENT's interpreter, with its ``PATH`` headed by that interpreter (the child's own
    rule, which :func:`main` applies), so a module and a program resolve as they will for the agent.
    ``probe_runner`` defaults to where the bench says the agent runs its programs (see the module docstring).
    """
    root = Path(workdir)
    declared = declared_host_requirements(root)
    in_cli = _runs_in_agent_cli_bash(root)
    confinement: Confinement = 'agent_cli_bash' if in_cli else 'agent_process'
    if declared is None:
        return HostToolchain(unreadable_requirements(root), confinement)
    runner = probe_runner or (run_in_agent_cli_bash if in_cli else run_as_agent_process)
    statuses: dict[int, RequirementStatus] = {}
    to_probe: list[int] = []
    for index, requirement in enumerate(declared):
        if requirement.kind == 'python_module':
            statuses[index] = _import_status(requirement)
        elif requirement.kind == 'node_module':
            statuses[index] = _node_status(requirement)
        elif shutil.which(requirement.name) is None:
            statuses[index] = _status(requirement, 'missing', 'not on PATH')
        elif (head := requirement.probe[:1]) and head[0] != requirement.name and not head[0].startswith(AGENT_PATH_PREFIX) and shutil.which(head[0]) is None:
            # the documented route's own program: absent, the route cannot start, whatever the requirement itself is
            statuses[index] = _status(requirement, 'missing', f'{head[0]}, which its documented route runs, is not on PATH')
        elif requirement.probe:
            to_probe.append(index)
        else:
            statuses[index] = _status(requirement, 'unverified', 'no probe declared: nothing shows it runs where the agent runs it')
    if to_probe:
        agent = root.resolve().parent
        runs = runner([_expand_agent_paths(declared[i].probe, agent) for i in to_probe], root, timeout)
        for index, run in zip(to_probe, runs, strict=True):
            requirement = declared[index]
            if run.exit_ok is None:
                statuses[index] = _status(requirement, 'not_checked', run.detail)
            elif run.exit_ok:
                statuses[index] = RequirementStatus(requirement.name, requirement.kind, 'present')
            else:
                failed = 'fails_sandboxed' if run.sandboxed else 'broken'
                statuses[index] = _status(requirement, failed, f'`{shlex.join(requirement.probe)}` failed: {run.detail}')
    if confinement == 'agent_process':
        # an import runs where this process runs; under the agent's confined CLI Bash it proves nothing of the Bash route
        _proven_by_imports(declared, statuses)
    rows = tuple(statuses[i] for i in range(len(declared)))
    head = INSTALL_BLOCK_HEAD.format(agent=root.resolve().parent)
    return HostToolchain(rows, confinement, install_plan([r.as_dict() for r in rows], declared), head)


def unreadable_requirements(workdir: Path) -> tuple[RequirementStatus, ...]:
    """The declaration cannot be read: every entry that names itself is ``not_checked``, and when none does, one row says so."""
    reason = f'{BENCH_MANIFEST_FILENAME} harness.host_requirements could not be read (a wrong shape, an unknown kind, or a probe with no probe_evidence)'
    try:
        entries = json.loads((workdir / BENCH_MANIFEST_FILENAME).read_text(encoding='utf-8'))['harness']['host_requirements']
        names = [(str(e['name']), str(e.get('kind') or '?')) for e in entries if isinstance(e, dict) and isinstance(e.get('name'), str) and e['name'].strip()]
    except (OSError, ValueError, KeyError, TypeError):
        names = []
    remedy = REMEDY['not_checked']
    return tuple(RequirementStatus(name, kind, 'not_checked', reason, remedy) for name, kind in names) or (
        RequirementStatus('host_requirements', 'declaration', 'not_checked', reason, remedy),
    )


def gap_phrase(requirements: Sequence[Mapping[str, Any]]) -> str:
    """``conv fails inside the Bash sandbox and lib is missing`` — the gaps, in one phrase, for a stamp."""
    parts = [f'{r.get("name")} {_predicate(r)}' for r in requirements if r.get('status') != 'present']
    if len(parts) < 2:
        return ''.join(parts)
    return ', '.join(parts[:-1]) + ' and ' + parts[-1]


def _predicate(requirement: Mapping[str, Any]) -> str:
    status = str(requirement.get('status'))
    return _STAMP[status][0] if status in _STAMP else f'is {status.replace("_", " ")}'


def _remedy(requirement: Mapping[str, Any]) -> str:
    """The row's own remedy; a row from a resolver that wrote none gets its status's."""
    return str(requirement.get('remedy') or REMEDY.get(str(requirement.get('status')), ''))


def host_stamp(toolchain: Mapping[str, Any]) -> str:
    """The ONE headline stamp a run over a host with gaps carries ('' when there is none): a host fact."""
    gaps = [r for r in toolchain.get('requirements') or [] if r.get('status') != 'present']
    if not gaps:
        return ''
    if any(_STAMP.get(str(r.get('status')), ('', True))[1] for r in gaps):
        return f"measured on a host where {gap_phrase(gaps)} — graded rows include the agent's fallback path"
    return f'measured on a host where {gap_phrase(gaps)} — every row is graded'


def host_findings(toolchain: Mapping[str, Any], declared: Sequence[HostRequirement]) -> tuple[GateFinding, ...]:
    """One gate finding per requirement this host does not give the agent, keyed by its status.

    Evidence is the requirement's typed facts (name, kind, status, where the probe ran, the declared
    probe argv); the probe's own output and paths ride ``observed``, so an overrule keeps binding
    across runs over the same host.
    """
    probes = {r.name: list(r.probe) for r in declared}
    out: list[GateFinding] = []
    for row in toolchain.get('requirements') or []:
        status = str(row.get('status'))
        if status == 'present':
            continue
        name = str(row.get('name'))
        fields: dict[str, Any] = {
            'subject': name,
            'evidence': {
                'name': name,
                'kind': str(row.get('kind')),
                'status': status,
                'confinement': toolchain.get('confinement'),
                'probe': probes.get(name, []),
            },
            'observed': {'detail': str(row.get('detail') or '')},
            'remedy': f'{name} ({row.get("kind")}) {status.replace("_", " ")}: {_remedy(row)}',
            'draft_ref': None,
        }
        if status in GAP_STATUSES:
            out.append(make_finding(HOST_REQUIREMENT_GAP, **fields))
        elif status == 'unverified':
            out.append(make_finding(HOST_REQUIREMENT_UNVERIFIED, **fields))
        else:
            out.append(make_finding(HOST_REQUIREMENT_NOT_CHECKED, **fields))
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    what = 'Resolve the host requirements bench_manifest.json declares, where the agent uses them.'
    parser = exit_codes.UsageParser(prog='python -m plurai.host_toolchain', description=what)
    parser.add_argument('--workdir', required=True, type=expand_user_path, help='The plurai_sim folder holding bench_manifest.json.')
    parser.add_argument('--json', action='store_true', help='Print the resolved toolchain as JSON on stdout.')
    parser.add_argument('--timeout', type=float, default=PROBE_TIMEOUT_SECONDS, help='Seconds each probe may run.')
    args = parser.parse_args(argv)
    from plurai.child import _head_path_with_running_interpreter

    _head_path_with_running_interpreter()  # the agent's shell finds this interpreter first; so does this lookup
    toolchain = check_host_toolchain(args.workdir, timeout=args.timeout)
    if args.json:
        print(json.dumps(toolchain.as_dict()))
    else:
        for row in toolchain.requirements:
            print(f'{row.name} ({row.kind}): {row.status}' + (f' — {row.detail}' if row.detail else ''))
        if toolchain.install_block:
            print(toolchain.install_block)
    # Non-zero only for a MEASURED gap; a reader that needs the rest reads the JSON.
    return exit_codes.REQUIREMENT_GAP if any(r.status in GAP_STATUSES for r in toolchain.requirements) else exit_codes.OK


if __name__ == '__main__':
    sys.exit(main())
