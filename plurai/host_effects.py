"""Which of the agent's host-effect tools the operator let act on this machine.

Testing is a minimal intervention in the user's environment. A tool that writes to the host or
reaches an external service (the scan marks it ``consent: required`` on ``agent_topology.json``)
runs for real only when the operator granted it at the scan's approval question. The grant is
recorded in ``plurai_sim/.host_effects_consent``, in the same grammar as ``.probe_consent``:

* ``granted: Bash, Write`` — asked, allowed, for exactly these names;
* ``declined`` — asked, refused;
* ``unasked`` — no ask happened (headless, or no listed tool).

Only ``granted`` grants anything. Absent, unreadable or any other content is the empty grant, so a
headless run, a fresh clone and a garbled file all deny. The harness applies the deny to the
agent's own options, reports what it applied on ``agent_state[HOST_EFFECTS_KEY]``, and the run
continues; ``run_summary.json`` carries the roll-up under the same key.

What the agent under test READS is a different matter from consent: its tool results become the
trail, and the trail leaves the machine (``/user-turn``, the judges' LLM, the artifacts, the
upload). So its reads are scoped whatever was granted — :func:`host_read_scope`, applied by the
harness (sandbox + hook + shell). Confinement is the only control: the child sends what the
harness returns as it is — but for the values of the credentials the agent's CLI signs in with,
which it sends as their names (:func:`redact_cli_credentials`).
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import functools
import importlib.metadata
import json
import logging
import os
import re
import shlex
import shutil
import signal
import site
import stat
import sys
import sysconfig
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from urllib.parse import quote, unquote, urlparse
from pathlib import Path
from typing import Any, Literal, cast

from plurai._platform import is_windows
from plurai._protocol import TRAIL_REFUSAL_KEY, TRAIL_STATUS_DENIED, HarnessFault
from plurai.credentials import MIN_TOKEN_LENGTH
from plurai.orchestrator.local_paths import LOCAL_PATH_MARKER

#: The sidecar's name, beside ``.probe_consent`` in the bench's ``plurai_sim/`` directory.
CONSENT_FILENAME = '.host_effects_consent'

#: The ``agent_state`` key a harness reports its applied host-effects record on, per turn:
#: ``{'granted': [tool, ...], 'denied': {tool: how}, 'cwd': str, 'shell_home': str, 'auth': route,
#: 'cli_config': ..., 'read_surfaces': {...}, 'reads_observed': {...}}`` (:func:`confine`).
HOST_EFFECTS_KEY = 'host_effects'

#: The record's ``trail`` value when :func:`confine` records every tool call into the harness's
#: trail: the typed declaration that the turn's ``tool_trail`` is a recording, not a list a harness
#: happened to return (:func:`trail_recorded`).
TRAIL_RECORDED = 'recorded'


def trail_recorded(agent_state: Any) -> bool:
    """Whether one turn's ``agent_state`` declares its trail lane: its host-effects record says
    :func:`confine` recorded the trail, and the turn returned ``tool_trail``. A trail key alone is
    not a recording — a harness can return a list nothing ever appended to."""
    from plurai.observability.tool_call_recovery import AGENT_STATE_TOOL_TRAIL_KEY  # noqa: PLC0415 — the package loads every sink; the harness imports this module

    record = agent_state.get(HOST_EFFECTS_KEY) if isinstance(agent_state, dict) else None
    return isinstance(record, dict) and record.get('trail') == TRAIL_RECORDED and AGENT_STATE_TOOL_TRAIL_KEY in agent_state


#: The Claude Code built-ins :func:`confine` confines IN PLACE or switches off by name when ungranted:
#: its per-tool dispatch keys (file edits scoped to the scenario, ``Bash`` sandboxed, the rest
#: disallowed). Not the consent set: every built-in off :data:`READ_ONLY_BUILTINS` needs a grant.
#: The plugin keeps a parity-pinned copy.
HOST_EFFECT_TOOLS = (
    'Bash',
    'Write',
    'Edit',
    'NotebookEdit',
    'WebFetch',
    'WebSearch',
    'RemoteTrigger',
    'PushNotification',
    'ScheduleWakeup',
    'CronCreate',
    'CronDelete',
)
_WRITE_TOOLS = {'Write', 'Edit', 'NotebookEdit'}

#: Per runtime, the built-ins known to only read, to act only on the agent's own background tasks
#: (which run under the same confinement), or to run under the same confinement (a sub-agent runs
#: under the same options): the ONLY built-ins that run without a grant. A built-in the CLI adds
#: later is not here, so it is denied until granted. The plugin keeps a parity-pinned copy.
READ_ONLY_BUILTINS: Mapping[str, tuple[str, ...]] = {
    'claude_agent_sdk': (
        'Read',
        'Glob',
        'Grep',
        'LS',
        'NotebookRead',
        'ToolSearch',
        'Skill',
        'TodoWrite',
        'Task',
        'Agent',
        'Workflow',
        'ListAgents',
        'CronList',
        'TaskOutput',
        'BashOutput',
        'TaskStop',
        'KillShell',
        'ListMcpResourcesTool',
        'ReadMcpResourceTool',
    ),
}

#: The CLI names a tool it serves from an MCP server with this prefix; such a tool is not a built-in.
_MCP_PREFIX = 'mcp__'
_UNGRANTED = 'denied: a built-in off the read-only list, and plurai_sim/.host_effects_consent does not grant it'

#: The values the CLI reads a provider switch as on (its own env-flag rule: trimmed, case-folded).
_CLI_FLAG_ON = frozenset({'1', 'true', 'yes', 'on'})

#: The provider switches the CLI signs in through, each read as a flag (:data:`_CLI_FLAG_ON`).
_CLI_PROVIDER_SWITCHES = (
    'CLAUDE_CODE_USE_BEDROCK',
    'CLAUDE_CODE_USE_FOUNDRY',
    'CLAUDE_CODE_USE_ANTHROPIC_AWS',
    'CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD',
    'CLAUDE_CODE_USE_MANTLE',
    'CLAUDE_CODE_USE_VERTEX',
)

#: The Anthropic credentials the CLI signs in with when no provider switch is on.
_CLI_KEYS = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN')

#: The environment routes the agent's CLI signs in from, in the CLI's order: a provider switch
#: outranks a key. With none, it signs in with the operator's own Claude login (:data:`EXISTING_LOGIN`).
CLI_CREDENTIAL_ENV = (*_CLI_PROVIDER_SWITCHES, *_CLI_KEYS)

#: The route ``record['auth']`` names when the environment declares none: the operator's existing login.
EXISTING_LOGIN = 'existing_login'

#: The route ``record['auth']`` names when the environment declares none and the operator's user settings name
#: an ``apiKeyHelper``: the CLI takes its key from that command before the existing login.
API_KEY_HELPER = 'apiKeyHelper'

#: The keys of the CLI's settings schema that sign it in (``_KEY_SETTINGS``: an Anthropic key) or refresh its
#: provider credentials. The CLI runs with a config dir of the scenario's own, so :func:`confine` carries these
#: from the operator's user settings into it (:func:`_carry_operator_sign_in`).
_KEY_SETTINGS = ('apiKeyHelper',)
_SIGN_IN_SETTINGS = (*_KEY_SETTINGS, 'awsCredentialExport', 'awsAuthRefresh', 'gcpAuthRefresh')

#: ``record['cli_config']``: where the agent's CLI keeps its config, session transcripts and
#: ``tool-results/``. ``scenario``: a folder of the scenario's own, on every route (:func:`_cli_config_env`).
#: ``operator``: the operator's own config dir.
CLI_CONFIG_SCENARIO = 'scenario'
CLI_CONFIG_OPERATOR = 'operator'

#: The variables a backend bearer token is read from. The value is a credential of the Plurai backend, never of the
#: agent: the runner passes the resolved token on as ``PLURAI_TOKEN`` alone (:mod:`plurai.child_env`), and no
#: output of the SDK carries it (:func:`redact_cli_credentials`).
BACKEND_TOKEN_ENV = ('PLURAI_TOKEN', 'NETWORK__API_KEY')

#: Every variable whose VALUE is a credential the SDK redacts from what it writes: the keys and provider
#: credentials the agent's CLI may hold (its own tables read them), and the backend token
#: (:data:`BACKEND_TOKEN_ENV`). The child sends none of these values out of the process
#: (:func:`redact_cli_credentials`).
CLI_SECRET_ENV = (
    *_CLI_KEYS,
    'AWS_BEARER_TOKEN_BEDROCK',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_SESSION_TOKEN',
    'ANTHROPIC_AWS_API_KEY',
    'ANTHROPIC_FOUNDRY_API_KEY',
    'ANTHROPIC_FOUNDRY_AUTH_TOKEN',
    'AZURE_CLIENT_SECRET',
    *BACKEND_TOKEN_ENV,
)


#: Who owns each ``StopFailure`` error type (the CLI's own enum), with the one step that clears it.
#: ``provider_infra``: the provider's infrastructure or the sign-in's, never the agent's (:data:`INFRA_STOP_CLASSES`).
#: ``agent``: the model's own reply hit its output budget, which production shows the user the same way.
#: Any other type is the agent's visible outcome and is graded as its reply (:data:`AGENT_VISIBLE_STOPS`).
_STOP_OWNERS: Mapping[str, tuple[Literal['provider_infra', 'agent'], str | None]] = {
    'authentication_failed': ('provider_infra', 'sign in to Claude (run `claude`), then re-run'),
    'oauth_org_not_allowed': ('provider_infra', 'sign in to Claude with an account your organisation allows, then re-run'),
    'account_on_hold': ('provider_infra', 'resolve the hold on your Claude account, then re-run'),
    'verification_required': ('provider_infra', "complete your organisation's verification in Claude, then re-run"),
    'cloud_credential_error': ('provider_infra', 'refresh your cloud provider credentials, then re-run'),
    'billing_error': ('provider_infra', "check your Claude plan's usage limits, then re-run"),
    'rate_limit': ('provider_infra', "re-run once the provider's rate limit clears"),
    'overloaded': ('provider_infra', 're-run once the provider recovers'),
    'server_error': ('provider_infra', 're-run once the provider recovers'),
    'model_not_found': ('provider_infra', 'set a model your account can use, then re-run'),
    'max_output_tokens': ('agent', None),
}


#: The framework-neutral class of each ``provider_infra`` stop above, and whether repeating the call can change
#: it (``terminal``). ``server_error`` is every 5xx AND a dropped connection (CLI 2.1.281 types ``ECONNRESET`` so);
#: ``overloaded`` a 529; ``rate_limit`` a 429. These are typed on the side of a turn that returned
#: (``agent_state['provider_turn_fault']``); nothing the agent said is removed but the CLI's own banner.
INFRA_STOP_CLASSES: Mapping[str, tuple[str, bool]] = {
    'authentication_failed': ('infra_auth', True),
    'oauth_org_not_allowed': ('infra_auth', True),
    'verification_required': ('infra_auth', True),
    'cloud_credential_error': ('infra_credentials', True),
    'account_on_hold': ('infra_account', True),
    'billing_error': ('infra_billing', True),
    'model_not_found': ('infra_model_not_found', True),
    'rate_limit': ('infra_rate_limit', False),
    'overloaded': ('infra_overloaded', False),
    'server_error': ('infra_server_error', False),
}

#: The CLI's other ``StopFailure`` types: what the end user of the agent sees, graded as the agent's reply.
#: ``invalid_request`` is a safety-classifier refusal, a context overflow, an oversized/protected PDF or image,
#: an effort/advisor setting; ``unknown`` a bare 4xx; ``max_output_tokens`` the model's own output budget. (A
#: gateway sign-in or provider-credential failure also arrives as ``unknown``: its typed kind is on the CLI's
#: stream message, never on this hook, so it is not keyed here.)
AGENT_VISIBLE_STOPS: frozenset[str] = frozenset({'invalid_request', 'unknown', 'max_output_tokens'})

#: The harness-fault source every ``StopFailure`` note carries.
STOP_FAILURE_SOURCE = 'claude_code.StopFailure'


async def _note_stop_failure(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:  # noqa: ARG001
    """A claude_agent_sdk ``StopFailure`` hook: the CLI ended the turn on an error, and fires this before
    its result. A provider or sign-in error is noted for the child as a harness fault, which types the
    turn from it — so a harness that never sees the CLI's result (it drives the agent's own ``send()``)
    cannot return the CLI's error text as the agent's reply. Every stop is also noted apart, in order,
    with its query's ``prompt_id`` and the CLI's own banner: the turn's last end types a raise the turn
    makes on it (the agent's own stop, its failed trial), and a turn that returned on infrastructure."""
    from plurai._mocking.turn_context import note_cli_stop, note_harness_fault  # noqa: PLC0415 — the mock layer imports this module

    error = str(input_data.get('error') or '')
    owner, remedy = _STOP_OWNERS.get(error, (None, None))
    banner = str(input_data.get('last_assistant_message') or '')
    # a later end in the turn (a retry's) supersedes it
    note_cli_stop(error, agent_owned=owner == 'agent', last_assistant_message=banner, prompt_id=str(input_data.get('prompt_id') or ''))
    if owner == 'provider_infra':
        note_harness_fault(STOP_FAILURE_SOURCE, f"the agent's Claude Code CLI ended its turn on {error} — {remedy}", kind='provider_infra')
    return {}


async def _note_stop(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:  # noqa: ARG001
    """A claude_agent_sdk ``Stop`` hook: the CLI ended a query cleanly. Noted so an earlier query's
    ``StopFailure`` in the same turn (an agent that retried) is not how the turn ended. Never blocks."""
    from plurai._mocking.turn_context import note_cli_stop  # noqa: PLC0415 — the mock layer imports this module

    note_cli_stop('', agent_owned=False, ok=True, prompt_id=str(input_data.get('prompt_id') or ''))
    return {}


async def _note_query_started(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:  # noqa: ARG001
    """A claude_agent_sdk ``UserPromptSubmit`` hook: the CLI started a query, whose end the child waits
    for by its ``prompt_id`` (the CLI fires ``StopFailure`` without awaiting it, and no end hook at all
    for a query its step cap ended). Never blocks or edits the prompt."""
    from plurai._mocking.turn_context import note_cli_query_started  # noqa: PLC0415 — the mock layer imports this module

    note_cli_query_started(str(input_data.get('prompt_id') or ''), transcript_path=str(input_data.get('transcript_path') or ''))
    return {}


def _hook_matcher(hook: Any) -> Any:
    """A claude_agent_sdk ``HookMatcher`` on every tool."""
    from claude_agent_sdk import HookMatcher  # noqa: PLC0415 — the agent's own dependency, never plurai's

    return HookMatcher(hooks=[hook])


def cli_hooks(scope: HostReadScope, *, trail: list[dict[str, Any]] | None, on_read_denied: Callable[[], None] | None = None) -> dict[str, list[Any]]:
    """The hooks every ``claude_agent_sdk`` session plurai builds or confines carries: ``scope``'s read gate
    on the read tools (denies appended to ``trail``; ``None``: no trail kept; each deny also reported to
    ``on_read_denied``), and the three that tell the child how each CLI query ended — started
    (``UserPromptSubmit``), ended cleanly (``Stop``), ended on an error (``StopFailure``). Pass as
    ``ClaudeAgentOptions(hooks=...)``."""
    from claude_agent_sdk import HookMatcher  # noqa: PLC0415 — the agent's own dependency, never plurai's

    read_gate: Any = scope.pre_tool_use_hook(trail=trail, on_deny=on_read_denied)  # claude_agent_sdk HookCallbacks
    started: Any = _note_query_started
    stop: Any = _note_stop
    stop_failure: Any = _note_stop_failure
    return {
        'PreToolUse': [HookMatcher(matcher='Read|Glob|Grep|NotebookRead', hooks=[read_gate])],
        'UserPromptSubmit': [HookMatcher(hooks=[started])],
        'Stop': [HookMatcher(hooks=[stop])],
        'StopFailure': [HookMatcher(hooks=[stop_failure])],
    }


def _routes(name: str, value: str | None) -> bool:
    """Whether ``value`` routes the CLI through ``name``: a switch the CLI reads as on, a key that is not blank."""
    text = (value or '').strip()
    return text.lower() in _CLI_FLAG_ON if name in _CLI_PROVIDER_SWITCHES else bool(text)


def _cli_auth_route(cli_env: Mapping[str, str], settings: Mapping[str, Any]) -> str:
    """The route the agent's CLI signs in with under ``cli_env`` and the user ``settings`` it loads, by
    NAME, in the CLI's order: a provider switch the CLI reads as on, else a key that is not blank, else
    an ``apiKeyHelper``, else the operator's existing login."""
    helper = settings.get('apiKeyHelper')
    fallback = API_KEY_HELPER if isinstance(helper, str) and helper.strip() else EXISTING_LOGIN
    return next((name for name in CLI_CREDENTIAL_ENV if _routes(name, cli_env.get(name))), fallback)


#: The shortest value redacted as a credential: none is shorter, and replacing a shorter one would rewrite ordinary text.
#: The backend token is at least this long (:data:`plurai.credentials.MIN_TOKEN_LENGTH`), so none escapes redaction by being short.
_MIN_SECRET_LEN = MIN_TOKEN_LENGTH

#: Credential values a harness handed the agent's CLI (``options.env``), beyond this process's environment.
_CLI_SECRETS_HANDED: dict[str, str] = {}


def _cli_secret_values(env: Mapping[str, str]) -> dict[str, str]:
    """``{value: variable}`` for every :data:`CLI_SECRET_ENV` credential ``env`` holds."""
    return {value: name for name in CLI_SECRET_ENV if len(value := (env.get(name) or '').strip()) >= _MIN_SECRET_LEN}


def redact_value(value: Any) -> Any:
    """``value`` with :func:`redact_cli_credentials` applied to every string in it: mappings, sequences,
    sets, dataclasses and pydantic models are rebuilt around the redacted strings, anything else is
    returned as it is. What a third-party client (an observability sink) is handed goes through this."""
    return _redact_walk(value, credential_redactor())


def _redact_walk(value: Any, redact: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {_redact_walk(key, redact): _redact_walk(item, redact) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_redact_walk(item, redact) for item in value]
        return items if isinstance(value, list) else type(value)(items)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name: _redact_walk(getattr(value, f.name), redact) for f in dataclasses.fields(value) if f.init})
    model_copy = getattr(value, 'model_copy', None)
    model_fields = getattr(type(value), 'model_fields', None)
    if callable(model_copy) and model_fields:
        return model_copy(update={name: _redact_walk(getattr(value, name), redact) for name in model_fields})
    return value


def register_backend_token(token: str | None) -> None:
    """Add the backend token a run resolved (from the environment or the saved login) to the values
    :func:`redact_cli_credentials` replaces, as ``<PLURAI_TOKEN redacted>``. A token held in the
    environment under :data:`BACKEND_TOKEN_ENV` is redacted without this."""
    if token and len(token := token.strip()) >= _MIN_SECRET_LEN:
        _CLI_SECRETS_HANDED[token] = BACKEND_TOKEN_ENV[0]


def redact_cli_credentials(text: str) -> str:
    """``text`` with the value of every credential the SDK may hold — the agent's CLI credentials in this
    process's environment or handed to :meth:`HostReadScope.agent_shell_env`, and the backend token
    (:data:`BACKEND_TOKEN_ENV`, :func:`register_backend_token`) — written as ``<VARIABLE redacted>``, in its
    plain, its JSON-escaped and its URL-encoded spelling. The one redaction every output of the SDK goes through."""
    return credential_redactor()(text)


def credential_redactor() -> Callable[[str], str]:
    """:func:`redact_cli_credentials` for the credentials held right now, with the secret list built once: the function to
    hand a walk over many strings, so the environment is read once per walk and not once per string."""
    values = {**_cli_secret_values(os.environ), **_CLI_SECRETS_HANDED}
    replacements = [
        (spelling, f'<{values[value]} redacted>')
        for value in sorted(values, key=len, reverse=True)
        for spelling in dict.fromkeys((value, json.dumps(value)[1:-1], quote(value, safe='')))
    ]

    def redact(text: str) -> str:
        for spelling, marker in replacements:
            text = text.replace(spelling, marker)
        return text

    return redact


class RedactingStream:
    """A text stream that writes whole lines through :func:`plurai.host_effects.redact_cli_credentials`
    (a value split across two writes is still one line), and is its stream in every other respect
    (``encoding``, ``fileno``, ``isatty``, ``buffer``…).

    A partial line is held as its chunks and joined once, when its line end (or a ``flush``) comes: a
    long line written in many small pieces (``json.dump``, ``print(end='')``) costs its length, not its
    square. A ``\r`` ends a line too — a progress bar redraws its line, no credential value holds one —
    so such a writer holds one chunk, and its line reaches the log as the plain stream's did.
    ``flush`` sends a held partial line on, redacted. Each call takes the held chunks before it writes:
    the wrapped write lets another thread run, and what that thread writes meanwhile is held for the
    next line, not dropped or sent twice. Not an ``io.TextIOBase``: an ``isinstance`` check on
    ``sys.stdout`` sees a plain object.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._pending: list[str] = []

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f'write() argument must be str, not {type(text).__name__}')
        end = max(text.rfind('\n'), text.rfind('\r')) + 1  # past the last line end: every whole line goes now
        if not end:
            if text:
                self._pending.append(text)
            return len(text)
        pending, self._pending = self._pending, ([text[end:]] if end < len(text) else [])
        self._stream.write(redact_cli_credentials(''.join((*pending, text[:end]))))
        return len(text)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        pending, self._pending = self._pending, []
        if pending:
            self._stream.write(redact_cli_credentials(''.join(pending)))
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        if name == '_stream':  # before ``__init__`` ran (copy, unpickle): no stream to delegate to, and no recursion
            raise AttributeError(name)
        return getattr(self._stream, name)


@contextlib.contextmanager
def redacted_process_output() -> Iterator[None]:
    """Run the body with this process's ``stdout`` / ``stderr`` and every log record passing
    :func:`redact_cli_credentials`, so a credential an error body, a traceback or a log call carries
    reaches no terminal, log file or captured stream."""
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = RedactingStream(out), RedactingStream(err)
    factory = logging.getLogRecordFactory()

    def _redacting(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = factory(*args, **kwargs)
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a malformed log call is the logging module's to report, not a reason to fail the caller
            message = str(record.msg)
        record.msg, record.args = redact_cli_credentials(message), None
        return record

    logging.setLogRecordFactory(_redacting)
    try:
        yield
    finally:
        logging.setLogRecordFactory(factory)
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        sys.stdout, sys.stderr = out, err


#: The keys :func:`observed_writes` counts under, in its order; a run's summary sums the same keys.
#: ``cli_own`` is never filled from ``events`` — the CLI's own session-transcript / tool-results
#: writes are its own runtime bookkeeping, not a tool call, so
#: :func:`observed_writes` structurally cannot see them. A caller that CAN check for them (
#: :func:`cli_own_write_landed`) folds the result in afterward; a caller that never checks leaves it
#: at 0, which is exactly why ``complete`` (``run_simulation.py``) requires the key be present, not
#: merely zero — an unread channel is not a clean one.
WRITE_LANDINGS: tuple[str, ...] = ('scenario_cwd', 'scenario_shell_home', 'cli_own', 'outside')


def observed_writes(record: dict[str, Any], events: list[Any]) -> dict[str, int]:
    """Where the agent's successful file writes landed, from its tool events — not from the deny:
    the scenario folder (``record['cwd']``), its shell's own home (``record['shell_home']``) or outside both."""
    roots = {'scenario_cwd': record.get('cwd'), 'scenario_shell_home': record.get('shell_home')}
    out = dict.fromkeys(WRITE_LANDINGS, 0)
    for event in events:
        args = event.get('args') if isinstance(event, dict) else None
        target = (args.get('file_path') or args.get('notebook_path')) if isinstance(args, dict) else None
        # only a refused or denied write is no landing: one whose outcome never settled may have landed
        denied = TRAIL_STATUS_DENIED in (event.get('status'), event.get('status_token')) or event.get('_denied') is True
        if not isinstance(target, str) or event.get('tool') not in _WRITE_TOOLS or event.get('status') == 'error' or denied:
            continue
        real = os.path.realpath(target)
        out[next((k for k, r in roots.items() if r and _under(real, os.path.realpath(r))), 'outside')] += 1
    return out


def _operator_cli_config() -> Path:
    """The operator's own Claude config folder: this process's ``CLAUDE_CONFIG_DIR``, else ``~/.claude``."""
    configured = os.environ.get('CLAUDE_CONFIG_DIR')
    return Path(configured) if configured else Path.home() / '.claude'


def cli_project_dir(cwd: str | Path, *, config_dir: str | Path | None = None) -> Path:
    """Where the CLI's OWN session transcript / ``tool-results/`` for a scenario run under ``cwd``
    lands: ``<config_dir>/projects/<cwd, every character outside [a-zA-Z0-9] folded to '-'>``.
    ``config_dir`` defaults to the operator's own config folder (:func:`_operator_cli_config`); pass
    a scenario's own ``CLAUDE_CONFIG_DIR`` (:func:`_cli_config_env`) to read that one instead.

    The CLI folds its own working directory, the PHYSICAL path (``getcwd``), so ``cwd`` is resolved
    first: a ``tempfile.mkdtemp`` folder under macOS's ``/var`` is ``/private/var`` to the CLI.
    The fold matches the CLI's own slug function (CLI v2.1.281:
    ``k(e){return e.replace(/[^a-zA-Z0-9]/g,"-")}``, reached from a brand-new cwd via ``gT(e)=k(e)``
    since a never-before-seen path has no cached override): an underscore folds too
    (``my_agent`` → ``my-agent``), and every scenario cwd holds one (``plurai_scenario_``). One CLI
    behavior this does NOT reproduce: ``gT`` hashes and truncates names over 200 characters
    (``Le(e)``, algorithm not recovered) — a scenario path folding to over 200 chars is a known, named gap, not a
    silently wrong guess; :func:`cli_own_write_landed` still only ever under-reports from it, never
    claims a false landing."""
    base = Path(config_dir) if config_dir is not None else _operator_cli_config()
    slug = re.sub(r'[^a-zA-Z0-9]', '-', os.path.realpath(cwd))
    return base / 'projects' / slug


def cli_own_write_landed(cwd: str | Path, *, config_dir: str | Path | None = None) -> bool:
    """Whether the CLI's own project dir for ``cwd`` (:func:`cli_project_dir`) exists and holds
    anything.

    No before/after snapshot needed: every scenario's ``cwd`` is a fresh ``tempfile.mkdtemp`` folder
    the harness template makes once and never reuses, so a project dir existing for it AT ALL — checked once, after the
    scenario ends — can only have been created by THIS scenario's CLI process; nothing earlier could
    have named a path that did not exist yet.
    """
    try:
        return any(cli_project_dir(cwd, config_dir=config_dir).iterdir())
    except (OSError, RuntimeError):  # RuntimeError: Path.home() cannot resolve — never treated as a leak
        return False


def host_effects_granted(workdir: str | Path) -> frozenset[str]:
    """The tool names ``<workdir>/.host_effects_consent`` grants. Anything but ``granted: …`` is ∅."""
    try:
        text = (Path(workdir) / CONSENT_FILENAME).read_text(encoding='utf-8').strip()
    except (OSError, UnicodeDecodeError):
        return frozenset()
    head, sep, names = text.partition(':')
    if not sep or head.strip() != 'granted':
        return frozenset()
    return frozenset(name.strip() for name in names.split(',') if name.strip())


#: The scan's manifest beside ``simulate.py``. Its schema is the plugin's (``bench_manifest.schema.json``);
#: this module reads one key of it, ``harness.host_requirements``.
BENCH_MANIFEST_FILENAME = 'bench_manifest.json'

#: The manifest schema's closed ``kind`` enum for a declared host requirement.
HOST_REQUIREMENT_KINDS: tuple[str, ...] = ('executable', 'python_module', 'node_module')


@dataclass(frozen=True)
class HostRequirement:
    """One ``harness.host_requirements`` entry: what the agent's own files say it needs on this host.

    ``probe``: an argv that runs the invocation ``probe_evidence`` (the ``<path>:<line>`` of the agent's
    files that documents it) shows (``kind: executable`` only; empty when none is declared) — headed by the
    program itself, or by the agent's own route to it (a wrapper script). An element ``{agent}/<rel>``
    names the agent's own file. It is run, never parsed: :mod:`plurai.host_toolchain` runs it where the
    agent would, and its exit status is the verdict. The plugin's ``validate`` holds it to the cited line.

    ``install``: the command the agent's own install docs give for it, verbatim, and ``install_evidence``
    the ``<path>:<line>`` of that line (any kind; both empty when the docs give none). Never written by
    the scan itself: the plugin's ``validate`` holds it to the cited line, and the run gate prints only a
    cited one (:func:`plurai.host_toolchain.install_block`). Nothing runs it, so a malformed one (not a string)
    reads as none — it never makes the declaration unreadable.
    """

    name: str
    kind: str
    probe: tuple[str, ...] = ()
    probe_evidence: str = ''
    install: str = ''
    install_evidence: str = ''


def declared_host_requirements(workdir: str | Path) -> tuple[HostRequirement, ...] | None:
    """The host requirements ``<workdir>/bench_manifest.json`` declares.

    ``()`` when the file or the key is absent: nothing is declared. ``None`` when the declaration is
    there and cannot be read (unparseable, a wrong shape, a ``kind`` outside
    :data:`HOST_REQUIREMENT_KINDS`, a ``probe`` that is not an argv, one on a module kind, one with no
    ``probe_evidence``, or an ``{agent}/`` element that climbs out of the agent): nobody can say what this
    host gives the agent, and the caller says so.
    """
    path = Path(workdir) / BENCH_MANIFEST_FILENAME
    if not path.exists():
        return ()
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    harness = manifest.get('harness')
    if harness is None:
        return ()
    if not isinstance(harness, dict):
        return None
    entries = harness.get('host_requirements')
    if entries is None:
        return ()
    if not isinstance(entries, list):
        return None
    declared: list[HostRequirement] = []
    for entry in entries:
        name = entry.get('name') if isinstance(entry, dict) else None
        kind = entry.get('kind') if isinstance(entry, dict) else None
        probe = entry.get('probe', []) if isinstance(entry, dict) else None
        cited = entry.get('probe_evidence', '') if isinstance(entry, dict) else None
        install = entry.get('install', '') if isinstance(entry, dict) else None
        install_cited = entry.get('install_evidence', '') if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.strip() or not isinstance(kind, str) or kind not in HOST_REQUIREMENT_KINDS:
            return None
        if not isinstance(probe, list) or not all(isinstance(arg, str) for arg in probe) or not isinstance(cited, str):
            return None
        if not isinstance(install, str) or not isinstance(install_cited, str):
            # print-only: a malformed install line is no install line, never an unreadable declaration
            install = install_cited = ''
        if probe and (kind != 'executable' or not cited.strip() or any('..' in arg.split('/') for arg in probe if arg.startswith('{agent}/'))):
            return None
        declared.append(
            HostRequirement(
                name=name.strip(),
                kind=kind,
                probe=tuple(probe),
                probe_evidence=cited.strip(),
                install=install.strip(),
                install_evidence=install_cited.strip(),
            )
        )
    return tuple(declared)


#: Where the operator's answer lives in ``bench_manifest.json``, as the remedy names it.
UNSANDBOXED_GRANT_KEY = 'consent.unsandboxed'

#: A program the Bash sandbox may leave out: one bare executable name. The CLI reads a trailing
#: ``:*`` as a prefix and an unescaped ``*`` as a wildcard (``*`` alone would unsandbox every
#: command), and it matches a command's own words, so a path or a space never names the program.
_BARE_EXECUTABLE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._+-]*')

#: Programs never granted: a shell, or a program whose job is to run another command. A grant of one
#: would let every command out of the sandbox. An interpreter (python, node, …) may be granted; the
#: scan says that frees every script it runs. The plugin keeps a parity-pinned copy.
UNGRANTABLE_PROGRAMS = (
    'sh',
    'bash',
    'zsh',
    'dash',
    'ksh',
    'mksh',
    'ash',
    'fish',
    'csh',
    'tcsh',
    'pwsh',
    'busybox',
    'env',
    'xargs',
    'sudo',
    'doas',
    'su',
    'nohup',
    'nice',
    'timeout',
    'time',
    'stdbuf',
    'setsid',
    'chroot',
    'script',
    'command',
    'builtin',
    'exec',
    'eval',
    'osascript',
    'open',
    'genv',
    'gxargs',
    'gtimeout',
    'gnohup',
    'gnice',
    'gstdbuf',
    'gtime',
)


@dataclass(frozen=True)
class UnsandboxedGrant:
    """The declared programs the operator let run outside the Bash sandbox: ``bench_manifest.json`` →
    ``consent.unsandboxed`` (``state``, ``names``), asked at the scan beside the host-effects consent.

    A granted program runs outside the Bash sandbox, with full read, write and network access to the
    host, and with ANY arguments — including the ones that make it run other commands
    (``<program> -e "<anything>"``, ``find -exec``, a converter's own macros). A Bash command leaves the
    sandbox only when every command in it is a granted program: ``cd dir && <program> …`` and a
    script that starts the program itself stay sandboxed.

    ``state``: ``granted``, ``declined`` or ``unasked`` as the operator answered (no manifest, or one
    without the key, is ``unasked``); ``unreadable`` when the manifest or the block cannot be read,
    which lets nothing out. ``names``: what the grant lets out — empty unless ``granted``.
    """

    state: Literal['granted', 'declined', 'unasked', 'unreadable']
    names: tuple[str, ...] = ()


def unsandboxed_grant(workdir: str | Path) -> UnsandboxedGrant:
    """The :class:`UnsandboxedGrant` ``<workdir>/bench_manifest.json`` records.

    A granted name that is not a bare program name (a path, a space, ``*``, ``x:*``) or that
    ``harness.host_requirements`` does not declare as ``kind: executable`` is a :class:`HarnessFault`
    naming the fix: the grant never reaches further than the programs the agent declares.
    """
    path = Path(workdir) / BENCH_MANIFEST_FILENAME
    if not path.exists():
        return UnsandboxedGrant('unasked')
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return UnsandboxedGrant('unreadable')
    consent = manifest.get('consent') if isinstance(manifest, dict) else None
    if not isinstance(manifest, dict) or not isinstance(consent, dict | None):
        return UnsandboxedGrant('unreadable')
    block = (consent or {}).get('unsandboxed')
    if block is None:
        return UnsandboxedGrant('unasked')
    state, names = (block.get('state'), block.get('names', [])) if isinstance(block, dict) else (None, None)
    if state not in ('granted', 'declined', 'unasked') or not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return UnsandboxedGrant('unreadable')
    if state != 'granted':
        return UnsandboxedGrant(state)
    declared = declared_host_requirements(workdir)
    if declared is None:
        return UnsandboxedGrant('unreadable')
    executables = {r.name for r in declared if r.kind == 'executable'}
    for name in names:
        if not _BARE_EXECUTABLE.fullmatch(name):
            raise HarnessFault(
                f'{BENCH_MANIFEST_FILENAME} {UNSANDBOXED_GRANT_KEY} grants {name!r}, which is not a bare program name — '
                'name the program itself (no path, no space, no `*` or `:*`), or remove it'
            )
        if name.casefold() in {program.casefold() for program in UNGRANTABLE_PROGRAMS}:  # a case-insensitive volume runs BASH as bash
            raise HarnessFault(
                f'{BENCH_MANIFEST_FILENAME} {UNSANDBOXED_GRANT_KEY} grants {name!r}, a shell or a program that runs '
                'other commands: granting it would let every command out of the Bash sandbox — remove it'
            )
        if name not in executables:
            raise HarnessFault(
                f'{BENCH_MANIFEST_FILENAME} {UNSANDBOXED_GRANT_KEY} grants {name!r}, which harness.host_requirements does not declare '
                'as kind: executable — re-run /simulation-scan-agent to declare it, or remove it from the grant'
            )
    return UnsandboxedGrant('granted', tuple(sorted(set(names))))


#: The first folder of a home path that is kept by name: the OS-standard user folders (macOS, XDG).
#: Any other name is somebody's own and is reported as a segment count.
_STANDARD_FOLDERS = frozenset({'Desktop', 'Documents', 'Downloads', 'Library', 'Movies', 'Music', 'Pictures', 'Public', 'Templates', 'Videos'})

#: Every file a zsh, bash or sh may read at startup from the home folder.
_SHELL_STARTUP_FILES = ('.zshenv', '.zprofile', '.zshrc', '.zlogin', '.zlogout', '.bashrc', '.bash_profile', '.bash_login', '.bash_logout', '.profile')

#: Why the read-tool hook refused a call: the tool error the agent sees, and the trail's result.
_READ_DENIED = 'outside the scenario folder and the agent repo'

#: What the agent's shell wrapper says when it refuses a command naming a home startup file.
_STARTUP_REFUSED = "the operator's shell startup files are outside the scenario"

#: A variable name the agent's shell wrapper can read and set.
_SHELL_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

#: The shell wrapper's filter: ``_plurai_keep <value> <denied>`` sets ``_plurai_kept`` to the
#: ``:``-separated entries of ``value`` not in ``denied``, in their order, empty entries included.
_KEEP_ENTRIES = (
    '_plurai_keep() {\n'
    '  _plurai_kept= _plurai_any= _plurai_rest=$1:\n'
    '  while [ -n "$_plurai_rest" ]; do\n'
    '    _plurai_entry=${_plurai_rest%%:*}; _plurai_rest=${_plurai_rest#*:}\n'
    '    case :$2: in *:"$_plurai_entry":*) [ -n "$_plurai_entry" ] && continue;; esac\n'
    '    _plurai_kept=${_plurai_any:+$_plurai_kept:}$_plurai_entry; _plurai_any=1\n'
    '  done\n'
    '}\n'
)


def _on_path(program: str, path: str) -> str | None:
    """The first folder of ``path`` holding an executable ``program``."""
    return next((d for d in path.split(os.pathsep) if d and os.access(os.path.join(d, program), os.X_OK)), None)


def _env_shebang(program: Path) -> str | None:
    """The interpreter a ``#!/usr/bin/env [-S] <name>`` script asks ``PATH`` for; ``None`` for any other file."""
    try:
        with program.open('rb') as f:
            words = f.readline(512).decode('utf-8', 'replace').removeprefix('#!').split()
    except OSError:
        return None
    if not words or Path(words[0]).name != 'env':
        return None
    return next((w for w in words[1:] if not w.startswith('-') and '=' not in w and '/' not in w), None)


#: Where a read tool names what it reads: a file, a notebook, a search root, a glob.
_READ_TARGET_KEYS = ('file_path', 'notebook_path', 'path')

#: A path token inside free text: absolute, ``~``, ``~user/``, ``$HOME`` or ``${HOME}``. Delimited by
#: whitespace, quotes, shell punctuation and ``:`` (PATH-like lists), never by ``/`` or ``.``; never
#: ending before a ``<`` (a rewrite's own ``<N segments>``, so a second scrub changes nothing).
_PATH_TOKEN = re.compile(
    r"""(?<![\w.~$/-])(?:~|~[A-Za-z_][\w.-]*(?=/)|\$\{HOME\}|\$HOME)(?![\w{])(?:/[^\s'"`;|&<>(){}\[\],:]*)?(?!<)"""
    r"""|(?<![\w.~$/-])/[^\s'"`;|&<>(){}\[\],:]+(?!<)"""
)

#: A line that is ONE path (a ``find``/``which``/``mdfind`` result line): taken whole, spaces
#: included, so a file name with a space in it is not cut in two.
_PATH_LINE = re.compile(r"""(?P<indent>[ \t]*)(?P<path>(?:~|~[A-Za-z_][\w.-]*|\$\{HOME\}|\$HOME)?/[^\t:<>'"`]*[^\s:<>'"`])(?P<tail>\s*)""")


#: An ``ls -l`` line up to its group column: mode, link count, owner, group. The owner and group
#: name who owns each file, which a data value never does; the rest of the line is the listing's.
_LISTING_OWNER_GROUP = r'^(?P<mode>[ \t]*[-dlcbps][-rwxsStT]{9}[@+.]?[ \t]+\d+[ \t]+)(?P<owner>\S+)(?P<gap>[ \t]+)(?P<group>\S+)(?=[ \t])'


def _segments(n: int) -> str:
    return f'<{n} segment{"" if n == 1 else "s"}>'


@dataclass(frozen=True)
class HostReadScope:
    """What the agent under test may read: the scenario folder, its repo and its interpreter.

    A path is OUT of scope when it lies inside a ``deny`` root and inside no ``allow`` root — the
    CLI sandbox's own rule (``allowRead`` takes precedence over ``denyRead``), shared by all three
    consumers so the sandbox, the read-tool hook and :func:`scrub_parent_record` cannot disagree.
    """

    home: Path
    deny: tuple[Path, ...]
    allow: tuple[Path, ...]
    #: Import roots of the interpreter NOT allowed back because each is the home, an ancestor of it
    #: or one of its user-content folders (a crafted ``.pth``, a symlink to the home).
    refused_imports: int = 0
    #: The on-disk identities of the roots, and of every ancestor of a deny root: a path is matched
    #: by what it IS, so a case variant (``/users/…`` on a case-insensitive volume) or a symlink
    #: into a root cannot pass for a different path.
    _deny_ids: frozenset[tuple[int, int]] = field(init=False, repr=False, compare=False)
    _allow_ids: frozenset[tuple[int, int]] = field(init=False, repr=False, compare=False)
    _deny_ancestor_ids: frozenset[tuple[int, int]] = field(init=False, repr=False, compare=False)
    #: Every spelling of the home (:func:`_both_forms`), computed once: the home does not move during a scrub.
    _home_forms: tuple[str, ...] = field(init=False, repr=False, compare=False)
    #: The HOME :meth:`agent_shell_env` gave the agent's shell (``None`` before it is made).
    _shell_home: Path | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_deny_ids', _identities(self.deny))
        object.__setattr__(self, '_allow_ids', _identities(self.allow))
        object.__setattr__(self, '_deny_ancestor_ids', _identities(a for d in self.deny for a in _ancestors(str(d))))
        object.__setattr__(self, '_home_forms', tuple(str(h) for h in _both_forms([self.home])))

    def _allow_back(self, root: Path) -> None:
        """Allow ``root`` (a folder of this scenario's own under a denied parent) back, in every consumer."""
        object.__setattr__(self, 'allow', tuple(dict.fromkeys((*self.allow, *_both_forms([root])))))
        object.__setattr__(self, '_allow_ids', _identities(self.allow))

    def _home_prefix(self, p: str) -> str | None:
        """The spelling of the home ``p`` lies under, or ``None`` when it lies under none."""
        return next((h for h in self._home_forms if _under(p, h)), None)

    def _expand(self, token: str) -> str:
        for lead in ('${HOME}', '$HOME', '~'):
            if token == lead or token.startswith(lead + '/'):
                return str(self.home) + token[len(lead) :]
        if token.startswith('~') and '/' in token:  # ~user/…: that user's home
            user, rest = token[1:].split('/', 1)
            return f'{self.home if user == self.home.name else self.home.parent / user}/{rest}'
        return token

    def out_of_scope(self, path: str | Path) -> bool:
        p = os.path.normpath(self._expand(str(path)))
        return any(_under(p, d) for d in self.deny) and not any(_under(p, a) for a in self.allow)

    def _on_disk_in(self, path: str, roots: tuple[Path, ...], ids: frozenset[tuple[int, int]]) -> bool:
        return any(_under(path, r) for r in roots) or any(_identity(a) in ids for a in _ancestors(path))

    def _read_denied(self, target: str) -> bool:
        """What a read of ``target`` would actually open: its realpath, matched by identity."""
        real = os.path.realpath(target)
        if self._on_disk_in(real, self.allow, self._allow_ids):
            return False
        covers = any(_under(str(d), real) for d in self.deny) or _identity(real) in self._deny_ancestor_ids
        return covers or self._on_disk_in(real, self.deny, self._deny_ids)

    def _inside_deny(self, target: str) -> bool:
        """Whether ``target`` (its realpath, matched by identity) lies inside a deny root and no allow
        root. Unlike :meth:`_read_denied`, a folder that only HOLDS a deny root (``/``) is not inside
        one: listing it reads none of the denied content."""
        real = os.path.realpath(target)
        return self._on_disk_in(real, self.deny, self._deny_ids) and not self._on_disk_in(real, self.allow, self._allow_ids)

    def sandbox_filesystem(self) -> dict[str, list[str]]:
        """The CLI's ``sandbox.filesystem`` read keys: the harness passes them verbatim."""
        return {'denyRead': [str(p) for p in self.deny], 'allowRead': [str(p) for p in self.allow]}

    def pre_tool_use_hook(
        self, *, trail: list[dict[str, Any]] | None, on_deny: Callable[[], None] | None = None
    ) -> Callable[[dict[str, Any], str | None, Any], Awaitable[dict[str, Any]]]:
        """A claude_agent_sdk ``PreToolUse`` hook for Read/Glob/Grep/NotebookRead. Hooks run in every
        permission mode, ``bypassPermissions`` included, and before the allow list.

        A denied call fires no PostToolUse or PostToolUseFailure, so the trail recorders never
        see it: the deny is appended to ``trail`` here, as a ``denied`` call — a decision, not a failed
        dispatch (``None``: no trail kept) — and reported to ``on_deny``.
        """

        async def hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:  # noqa: ARG001
            tool_input = input_data.get('tool_input') or {}
            cwd = str(input_data.get('cwd') or Path.cwd())
            # the read tools run in the CLI, whose ``~`` is the operator's home
            targets = [str(tool_input[k]) for k in _READ_TARGET_KEYS if isinstance(tool_input.get(k), str) and tool_input[k]]
            pattern = tool_input.get('pattern')
            if input_data.get('tool_name') == 'Glob' and isinstance(pattern, str):  # judged by the folder it searches
                base = self._expand(str(tool_input.get('path') or cwd))
                targets += [os.path.join(base, self._expand(_glob_root(p))) for p in _brace_alternatives(pattern)]
            for target in targets:
                expanded = self._expand(target)
                absolute = expanded if Path(expanded).is_absolute() else os.path.join(cwd, expanded)
                if self._read_denied(absolute):
                    if trail is not None:
                        call_id = input_data.get('tool_use_id') or tool_use_id
                        trail.append(
                            {
                                'tool': input_data.get('tool_name'),
                                'args': tool_input,
                                'call_id': call_id,
                                'result': _READ_DENIED,
                                'status': TRAIL_STATUS_DENIED,
                                TRAIL_REFUSAL_KEY: {'by': 'confinement', 'reason': _READ_DENIED},
                            }
                        )
                    if on_deny is not None:
                        on_deny()
                    return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny', 'permissionDecisionReason': _READ_DENIED}}
            return {}

        return hook

    def _rewrite(self, token: str) -> str:
        p = os.path.normpath(self._expand(token))
        home = self._home_prefix(p)
        if home is not None:
            rest = [s for s in p[len(home) :].split('/') if s]
            if not rest:
                return '~'
            first = rest[0] if rest[0] in _STANDARD_FOLDERS else _segments(1)
            return '/'.join(['~', first, *([_segments(len(rest) - 1)] if len(rest) > 1 else [])])
        root = max((str(d) for d in self.deny if _under(p, str(d))), key=len)
        rest = [s for s in p[len(root) :].split('/') if s]
        return root if not rest else f'{root.rstrip("/")}/{_segments(len(rest))}'

    def _redact(self, token: str, counter: list[int]) -> str:
        if self.out_of_scope(token):
            new = self._rewrite(token)
            if len(counter) > 1:  # :meth:`home_named`: the home as spelled, not a ``~`` the shell reads as its own
                counter[1] += self._home_prefix(os.path.normpath(token)) is not None
        else:  # readable, so kept whole — but what ships never names the home
            p = os.path.normpath(self._expand(token))
            home = self._home_prefix(p)
            new = token if home is None else '~' + p[len(home) :]
        counter[0] += new != token
        return new

    def _drop_username(self, text: str, counter: list[int]) -> str:
        """The username → ``<user>`` where the home spells it: after its own users folder
        (``…/Users/<user>/``, ``home/<user>``), as ``~<user>``, in the home's encoded prefix
        (``-Users-<user>-Documents``), and in the owner and group columns of an ``ls -l`` line.
        Never in free text or a data value: "Ann" in a reply, ``/contacts/ann`` in a record or
        ``/dev/null`` is not the operator."""
        user = self.home.name
        if not user:
            return text
        name, parent, end = re.escape(user), re.escape(self.home.parent.name), r'(?![\w-]|\.\w)'  # a whole segment: not ann.txt, ann-v2
        forms = [rf'(?<![\w.~$/-])(?P<tilde>~){name}{end}', _LISTING_OWNER_GROUP]
        if parent:  # a home with a users folder above it; ``/root`` has none, and its token rule covers it
            forms += [rf'(?<=/)(?P<users>{parent}/){name}{end}', rf'(?<=-{parent}-){name}(?![\w.])']
        pattern = re.compile('|'.join(forms), re.MULTILINE)

        def rewrite(m: re.Match[str]) -> str:
            if m['mode'] is None:
                counter[0] += 1
                if m.groupdict().get('users') and len(counter) > 1:  # a relative spelling the tokenizer never takes whole (``../../Users/<user>/x``)
                    counter[1] += 1
                return (m.groupdict().get('users') or m['tilde'] or '') + '<user>'
            owner, group = ('<user>' if m[c] == user else m[c] for c in ('owner', 'group'))
            counter[0] += (m['owner'] == user) + (m['group'] == user)
            return m['mode'] + owner + m['gap'] + group

        return pattern.sub(rewrite, text)

    def _scrub_text(self, text: str, counter: list[int]) -> str:
        """A path spelled with JSON-escaped slashes or percent-encoded is the same path: when its
        decoded form names the home, the decoded, scrubbed form is what ships."""
        for marker, decode in _ENCODINGS:
            if marker in text.casefold():
                decoded = decode(text)
                found = [0] * len(counter)
                scrubbed = self._scrub_plain(decoded, found)
                if found[0]:
                    counter[:] = [c + f for c, f in zip(counter, found, strict=True)]
                    text = scrubbed
        return self._scrub_plain(text, counter)

    def _scrub_plain(self, text: str, counter: list[int]) -> str:
        out = []
        for line in text.splitlines(keepends=True):
            whole = _PATH_LINE.fullmatch(line)
            if whole and self.out_of_scope(whole['path']):
                out.append(whole['indent'] + self._redact(whole['path'], counter) + whole['tail'])
            else:
                out.append(_PATH_TOKEN.sub(lambda m: self._redact(m.group(0), counter), line))
        return self._drop_username(''.join(out), counter)

    def _walk(self, obj: Any, counter: list[int]) -> Any:
        if isinstance(obj, str):
            return self._scrub_text(obj, counter)
        if isinstance(obj, list | tuple):
            return [self._walk(v, counter) for v in obj]
        if not isinstance(obj, dict):
            return obj
        out = {}
        for k, v in obj.items():  # keys too: a du- or stat-style result is keyed by path
            key = self._scrub_text(k, counter) if isinstance(k, str) else k
            out[key] = self._walk(v, counter)
        return out

    def scrub(self, obj: Any) -> tuple[Any, int]:
        """``obj`` (JSON-shaped) with every out-of-scope host path rewritten, and how many were.

        * A path token into the deny region becomes ``~/<standard folder>/<N segments>`` (or
          ``<deny root>/<N segments>``): the username and the deeper names are dropped. Keys are
          scrubbed like values, and a JSON-escaped or percent-encoded spelling like the plain one.
        * Paths outside the deny region, and inside the scenario folder or the repo, are kept —
          with a home prefix written ``~``, and the username as a path segment written ``<user>``:
          allowed to read is not the same as fit to ship.
        """
        counter = [0]
        return self._walk(obj, counter), counter[0]

    def home_named(self, obj: Any) -> int:
        """How many path tokens in ``obj`` (JSON-shaped) spell the operator's home literally and lie
        out of scope: what a record that leaves this machine discloses of the home. Tokenized as
        :meth:`scrub` tokenizes (JSON-escaped and percent-encoded spellings included); a path the scope
        allows (a repo under the home), one under no spelling of the home (a mock route ``/users/42``)
        and a ``~`` or ``$HOME`` (the agent shell's own home) are not counted. A relative spelling that names the
        home after its users folder (``../../Users/<user>/…``) counts: whether it is in scope cannot be told
        without its base."""
        counter = [0, 0]
        self._walk(obj, counter)
        return counter[1]

    def _denies_entry(self, entry: str) -> bool:
        """Whether the sandbox refuses ``entry`` (a PATH-list entry, or a whole variable value): out of
        scope as spelled and as what it IS on disk (a venv spelled ``/tmp/…`` is its ``/private/tmp/…``)."""
        return bool(entry) and self.out_of_scope(entry) and self._inside_deny(self._expand(entry))

    def _shell_env_drops(self, cli_env: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
        """Per variable of ``cli_env`` whose value is a list naming one or more paths (``os.pathsep``
        separated: ``PATH``, ``NODE_PATH``, a lone ``VIRTUAL_ENV``), the path entries the scope denies;
        each entry is judged alone, so a relative one (``.``, ``node_modules/.bin``) is kept and stops
        nothing. A denied entry is dead in the sandbox and only names what lies there. A value holding
        ``://`` is a URL (``file://…``, ``sqlite:////…``), not a path list: it is left whole."""
        drops = {}
        for name, value in cli_env.items():
            if '://' in value:
                continue
            entries = [e for e in value.split(os.pathsep) if _PATH_LINE.fullmatch(e)]
            if not _SHELL_NAME.fullmatch(name):
                continue
            if denied := tuple(dict.fromkeys(e for e in entries if self._denies_entry(e))):
                drops[name] = denied
        return drops

    def agent_shell_env(self, *, cli_env: Mapping[str, str], sandbox_grant: Sequence[str] | None = None) -> dict[str, str]:
        """The CLI environment under which the agent's shell keeps no state in, and reads none of the
        startup files of, the operator's home.

        The CLI itself keeps the operator's ``HOME``: its sign-in (the macOS keychain is found through
        it) stays the operator's own, while its config and its transcripts go to a config dir of the
        scenario's own (:func:`confine`, :func:`_cli_config_env`); its auto-memory is the
        scenario's (``CLAUDE_CODE_REMOTE_MEMORY_DIR``, which leaves the sign-in untouched). The
        agent's SHELL runs with a home of the scenario's own — ``HOME`` and the XDG folders under it — in a folder no other
        scenario can read (:func:`_scenario_root`), so what it derives from ``$HOME`` lands there.
        The credential values in ``cli_env`` (the environment the CLI starts with) are noted, so the
        child never sends them out (:func:`redact_cli_credentials`).

        The CLI snapshots the user's shell by running it as a login shell that sources the home rc
        file, and runs every command in that shell; zsh reads ``$ZDOTDIR/.zshenv`` on every start
        and bash ``$BASH_ENV``. ``CLAUDE_CODE_SHELL`` points it at a wrapper that starts the same
        shell with no user startup files (``zsh -f`` / ``bash --noprofile --norc``), refuses a
        command that names one of them (the snapshot is then skipped), and hands the agent's own
        nested shells an empty ``ZDOTDIR``, no ``BASH_ENV``/``ENV`` and the scenario's home itself,
        whatever environment the CLI passes it — and an empty global git config of its own. It all
        goes when this process ends — one process per scenario.

        ``sandbox_grant``: the programs the operator let out of the Bash sandbox, when the shell runs
        inside it (:func:`confine`); ``None`` when it does not. Inside it, what the shell inherits names
        nothing the scope denies (:meth:`_shell_env_drops`) — a denied entry is dead there and only
        names what lies there: each denied entry of a path-list variable is dropped, in place, and a
        variable left with none is unset; an entry the CLI adds at run time is none the scope judged,
        and stands. A granted program found only in a dropped folder is linked from a folder of the
        scenario's own, first on ``PATH``. The CLI process keeps its own environment: its npm and its
        sign-in need it.
        """
        _CLI_SECRETS_HANDED.update(_cli_secret_values(cli_env))
        real = _operator_shell()
        flavour = 'zsh' if 'zsh' in Path(real).name else 'bash'
        root = _scenario_root('sh-')
        self._allow_back(root)
        (root / 'zdotdir').mkdir()
        (root / 'gitconfig').touch()
        shell_home = root / 'home'
        shell_home.mkdir()
        memory = shell_home / '.claude'  # the CLI's auto-memory: <memory>/projects/<project>/memory
        object.__setattr__(self, '_shell_home', shell_home)
        startup = '|'.join('*' + shlex.quote(str(self.home / name)) + '*' for name in _SHELL_STARTUP_FILES)
        flags = '-f' if flavour == 'zsh' else '--noprofile --norc'
        home_vars = {'HOME': shell_home, **{name: shell_home / sub for name, sub in _XDG_HOMES}}
        drops = self._shell_env_drops(cli_env) if sandbox_grant is not None else {}
        links = {}
        dropped = drops.get('PATH', ())
        for program in sandbox_grant or ():  # where the operator's own PATH finds it, when that folder is dropped
            found = _on_path(program, cli_env.get('PATH', ''))
            if found is not None and found in dropped:
                links[program] = os.path.join(found, program)
                # its ``#!/usr/bin/env <interpreter>`` (an npm-global or nvm CLI's ``node``), when only a dropped folder holds it
                interpreter = _env_shebang(Path(links[program]))
                if interpreter and (where := _on_path(interpreter, cli_env.get('PATH', ''))) in dropped:
                    links.setdefault(interpreter, os.path.join(where, interpreter))
        if links:
            (root / 'bin').mkdir()
            for program, target in links.items():
                (root / 'bin' / program).symlink_to(target)
        env_filter = ''.join(
            f'if [ -n "${{{name}+x}}" ]; then _plurai_keep "${name}" {shlex.quote(os.pathsep.join(denied))}; '
            f'if [ -n "$_plurai_kept" ]; then {name}=$_plurai_kept; export {name}; else unset {name}; fi; fi\n'
            for name, denied in drops.items()
        )
        if links:
            env_filter += f'PATH={shlex.quote(str(root / "bin"))}${{PATH:+:$PATH}}; export PATH\n'
        wrapper = root / flavour
        wrapper.write_text(
            '#!/bin/sh\n'
            "# The agent's shell: it starts from none of the operator's startup files (plurai.host_read_scope).\n"
            'for arg do\n'
            f'  case $arg in {startup}) echo "plurai: {_STARTUP_REFUSED}" >&2; exit 1;; esac\n'
            'done\n'
            'unset BASH_ENV ENV\n'
            + (_KEEP_ENTRIES + env_filter + 'unset _plurai_kept _plurai_rest _plurai_entry _plurai_any\n' if env_filter else '')
            + f'ZDOTDIR={shlex.quote(str(root / "zdotdir"))}; export ZDOTDIR\n'
            # git dies on an unreadable ~/.gitconfig, and that file carries the operator's identity
            # and credential helpers: git gets an empty global config of its own.
            f'GIT_CONFIG_GLOBAL={shlex.quote(str(root / "gitconfig"))}; export GIT_CONFIG_GLOBAL\n'
            + ''.join(f'{name}={shlex.quote(str(path))}; export {name}\n' for name, path in home_vars.items())
            + f'exec {shlex.quote(real)} {flags} "$@"\n',
            encoding='utf-8',
        )
        wrapper.chmod(0o755)
        return {'CLAUDE_CODE_SHELL': str(wrapper), 'CLAUDE_CODE_REMOTE_MEMORY_DIR': str(memory)}


#: The keys :func:`observed_reads` counts a call's paths under; a run's summary sums the same keys.
#: A call is judged by the paths it NAMES (its command, a read tool's target) and by what came back
#: for them — never by a path its output merely mentions (an environment dump, a module search list,
#: the CLI's own bookkeeping keys), which is not a read.
#:
#: * ``in_scope``: a named path the scope allows.
#: * ``refused``: an out-of-scope path the output shows was not read — the OS's refusal of it — or a
#:   read-tool call the gate denied.
#: * ``unseen``: an out-of-scope path named with no decision either way: its refusal or its content,
#:   if any, is hidden (``2>/dev/null``, a pipe through ``grep``), or the output cannot be told apart
#:   from what the call's other paths returned. The run reads ``unobserved``, never ``scoped``.
#: * ``outside``: an out-of-scope path whose content came back — a listing naming what lies inside
#:   it, or output from a call whose only source is that path. The run reads ``escaped``.
READ_LANDINGS: tuple[str, ...] = ('in_scope', 'refused', 'unseen', 'outside')

#: The built-in tools that read the host for the agent: the gated read tools, and the shell tools
#: (a command, and the output of a background one).
_READ_TOOLS = ('Read', 'Glob', 'Grep', 'NotebookRead')
_SHELL_TOOLS = ('Bash', 'BashOutput', 'TaskOutput')
READ_CHANNEL_TOOLS: tuple[str, ...] = (*_SHELL_TOOLS, *_READ_TOOLS)

#: How a call is told a path was not read, in any shell's casing: EPERM (the macOS sandbox), EACCES,
#: ENOENT (how the Linux sandbox hides a denied path), the macOS sandbox's authorization denial, and
#: the agent's own shell wrapper refusing a home startup file (:data:`_STARTUP_REFUSED`).
_NOT_READ = re.compile(
    rf'operation not permitted|permission denied|no such file or directory|authorization denied|{re.escape(_STARTUP_REFUSED)}', re.IGNORECASE
)

#: What makes a shell command more than one simple command, whose output then has more than one
#: producer: a separator or pipe (``;``, ``|``, ``&``, a line break), a command or process substitution
#: (backticks, ``$(``, ``<(``, ``>(``) and the test operator (``[``, ``[[``, ``test``), whose verdict
#: another command turns into output.
_COMPOUND = re.compile(r'[;|&`\n]|\$\(|[<>]\(|(?<!\S)(?:\[\[?|test)(?!\S)')

#: The root folder named as a word of its own (``find / …``, ``du -sh /``): no path token spells it.
_ROOT_WORD = re.compile(r'(?<!\S)/(?![^\s;|&)])')

#: Paths a command names that are no read source: the device files a redirect writes to.
_NOT_A_SOURCE = '/dev'

#: The most distinct paths one call is judged by (each costs a realpath and a stat per ancestor): a
#: call past it (a ``find /`` listing) is left unjudged, so the run reads unobserved, not scoped.
_MAX_JUDGED_PATHS = 5000


def _shell_paths(scope: HostReadScope, text: str, shell_home: str) -> set[str]:
    """The absolute paths ``text`` names, as the agent's shell reads them: its ``~`` and ``$HOME``
    are its own home (``shell_home``), ``~user/`` that user's."""
    found = {'/'} if _ROOT_WORD.search(text) else set()
    for line in text.splitlines():
        whole = _PATH_LINE.fullmatch(line)
        for token in [whole['path']] if whole else [m.group(0) for m in _PATH_TOKEN.finditer(line)]:
            lead = next((lead for lead in ('${HOME}', '$HOME', '~') if token == lead or token.startswith(lead + '/')), None)
            found.add(os.path.normpath(shell_home + token[len(lead) :] if lead else scope._expand(token)))
    return found


def _shell_output(response: Any) -> tuple[bool, list[str], list[str]]:
    """A shell call's own output: whether it ran to success, its stdout lines, and every output line.

    A success is the CLI's result object, of which only ``stdout`` and ``stderr`` are the command's:
    its other keys (where it persisted a large output, its hints) are the CLI's own bookkeeping. A
    failure is the CLI's error text, the command's output folded into it.
    """
    if isinstance(response, dict):
        stdout, stderr = (str(response.get(k) or '') for k in ('stdout', 'stderr'))
        return True, stdout.splitlines(), [*stdout.splitlines(), *stderr.splitlines()]
    text = response if isinstance(response, str) else ''
    return False, [], text.splitlines()


def _read_inside_denied(scope: HostReadScope, path: str) -> bool:
    """Whether learning ``path`` took reading a folder out of scope: it and its folder are both out of scope."""
    return scope._inside_deny(path) and scope._inside_deny(str(Path(path).parent))


def observed_reads(scope: HostReadScope, *, tool: Any, tool_input: Any, response: Any, shell_home: str) -> dict[str, int]:
    """Where one built-in call's reads landed, by :data:`READ_LANDINGS`, judged by ``scope`` — the rule
    the sandbox and the read-tool gate enforce.

    A read tool that ran passed the gate: ``in_scope``, unless a file list it returned names a path
    out of scope in a folder out of scope (``outside``). A shell call is judged by the absolute paths its
    command names. For each, in this order:

    1. ``outside`` when a success's stdout lists a path inside it that is out of scope in a folder out
       of scope, on a line not saying it was not read: only a read of that folder yields it (a denied
       folder's NAME, seen from its readable parent, is not such a read, nor is an allowed path);
    2. ``in_scope`` when the scope allows the path;
    3. ``refused`` when an output line naming it (or a path inside it) says it was not read;
    4. ``outside`` when a success printed output that is no refusal, no line names it, and the
       command is one simple command naming no other source: that output can only be its content
       (in a compound command — ``cat P 2>/dev/null || echo none`` — another command may have
       printed it);
    5. else ``unseen``.

    A relative path, or one the command builds at run time, is not judged: the sandbox still holds
    it, the count does not see it. The output is the command's own text, so a command that prints a
    refusal it was not given reads ``refused``: the observer backs the sandbox, it does not replace it.
    A call naming more than :data:`_MAX_JUDGED_PATHS` paths raises
    ``OverflowError``: it is not judged.
    """
    out = dict.fromkeys(READ_LANDINGS, 0)
    if tool in _READ_TOOLS:
        listed = response.get('filenames') if isinstance(response, dict) else None
        files = [f for f in listed if isinstance(f, str) and Path(f).is_absolute()] if isinstance(listed, list) else []
        if len(files) > _MAX_JUDGED_PATHS:
            raise OverflowError(f'{len(files)} paths in one call')
        out['outside' if any(_read_inside_denied(scope, f) for f in files) else 'in_scope'] = 1
        return out
    command = tool_input.get('command') if isinstance(tool_input, dict) else None
    if tool not in _SHELL_TOOLS or not isinstance(command, str):
        return out
    named = _shell_paths(scope, command, shell_home)
    ok, stdout, lines = _shell_output(response)
    parsed = [(line, _shell_paths(scope, line, shell_home)) for line in lines]
    if len(named) + sum(len(paths) for _line, paths in parsed) > _MAX_JUDGED_PATHS:
        raise OverflowError(f'{len(named)} paths in one call')
    denied = {p for p in named if scope._inside_deny(p)}
    sources = [p for p in named - denied if not _under(p, _NOT_A_SOURCE)]
    sole_source = not sources and not _COMPOUND.search(command)  # one simple command, naming nothing else
    listed = {q for line, paths in parsed[: len(stdout)] if ok and not _NOT_READ.search(line) for q in paths}
    read_from_denied = {q for q in listed if _read_inside_denied(scope, q)}
    for path in named:
        mentioned = [line for line, paths in parsed if any(_under(q, path) or _under(path, q) for q in paths)]
        if any(q != path and _under(q, path) for q in read_from_denied):
            out['outside'] += 1
        elif path not in denied:
            out['in_scope'] += 1
        elif any(_NOT_READ.search(line) for line in mentioned):
            out['refused'] += 1
        elif ok and not mentioned and sole_source and any(line.strip() and not _NOT_READ.search(line) for line in stdout):
            out['outside'] += 1
        else:
            out['unseen'] += 1
    return out


def _read_observer(scope: HostReadScope, shell_home: str, counts: dict[str, int]) -> tuple[Callable[..., Awaitable[dict[str, Any]]], Callable[[], None]]:
    """A ``PostToolUse``/``PostToolUseFailure`` hook that folds each read-channel call into ``counts``
    (:func:`observed_reads`, plus ``calls``), and the callback the read gate reports a deny to. A call
    it could not judge is left out of ``calls``, so the run cannot read as observed."""

    async def hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:  # noqa: ARG001
        tool = input_data.get('tool_name')
        if tool in READ_CHANNEL_TOOLS:
            try:
                response = input_data.get('tool_response', input_data.get('error'))
                seen = observed_reads(scope, tool=tool, tool_input=input_data.get('tool_input'), response=response, shell_home=shell_home)
            except Exception:  # noqa: BLE001 — an unjudged call stays out of ``calls``: the run then reads unobserved
                return {}
            for key, n in seen.items():
                counts[key] += n
            counts['calls'] += 1
        return {}

    def refused() -> None:
        counts['refused'] += 1
        counts['calls'] += 1

    return hook, refused


def _operator_credential_store(cli_env: Mapping[str, str]) -> str:
    """``CLAUDE_SECURESTORAGE_CONFIG_DIR`` naming the store the operator's own CLI keeps its login in:
    their own value when set, else their relocated ``CLAUDE_CONFIG_DIR``, else EMPTY — the CLI's
    default store (the unsuffixed keychain item, ``~/.claude/.credentials.json``). A non-empty
    ``~/.claude`` there is a different store: the CLI suffixes the keychain item by a hash of the path."""
    if 'CLAUDE_SECURESTORAGE_CONFIG_DIR' in cli_env:
        return cli_env['CLAUDE_SECURESTORAGE_CONFIG_DIR']
    return cli_env.get('CLAUDE_CONFIG_DIR') or ''


def declared_tool_name(name: Any, declared: Iterable[str]) -> Any:
    """The key the config declares for a tool the CLI names by its MCP wire name (``mcp__<server>__<tool>``):
    the one declared key that is ``<tool>`` or ``<server>__<tool>``. A name that is not a wire name (no server
    part), is itself declared, or that matches no declared key or more than one, is kept as it is."""
    if not isinstance(name, str) or not name.startswith(_MCP_PREFIX) or name in declared:
        return name
    rest = name[len(_MCP_PREFIX) :]
    server, sep, tool = rest.partition('__')
    if not (server and sep and tool):
        return name
    matches = [k for k in declared if rest == k or rest.endswith(f'__{k}')]
    return matches[0] if len(matches) == 1 else name


def _config_documents(workdir: str | Path) -> Iterator[dict[str, Any]]:
    """Every config document beside the harness in ``workdir``, at both rungs in order. A config that exists and
    cannot be read raises its ``WorkspaceConfigError``."""
    from plurai.ingest.workspace_config import bench_config_roots, discover_config_documents  # noqa: PLC0415

    for root in bench_config_roots(Path(workdir) / 'simulate.py'):
        for _path, document in discover_config_documents(root):
            yield document


@contextlib.contextmanager
def _config_readable(unknown: str) -> Iterator[None]:
    """A config beside the harness that exists and cannot be read is a :class:`HarnessFault`: ``unknown`` is unknown."""
    from plurai.ingest.workspace_config import WorkspaceConfigError  # noqa: PLC0415

    try:
        yield
    except WorkspaceConfigError as error:
        raise HarnessFault(f'the config beside the harness cannot be read, so {unknown} is unknown: {error}') from error


def read_tool_keys(workdir: str | Path) -> frozenset[str]:
    """The ``tools:`` keys of the config beside the harness in ``workdir``. A config that exists and
    cannot be read raises its ``WorkspaceConfigError``: a reader that is not :func:`confine` decides
    what that means, and no harness fault is recorded on its behalf."""
    keys: set[str] = set()
    for document in _config_documents(workdir):
        tools = document.get('tools')
        keys |= {str(k) for k in tools if k} if isinstance(tools, dict) else set()
    return frozenset(keys)


def declared_tool_keys(workdir: str | Path) -> frozenset[str]:
    """:func:`read_tool_keys`; a config that exists and cannot be read is a :class:`HarnessFault`."""
    with _config_readable('which tools the agent declares'):
        return read_tool_keys(workdir)


#: :func:`confine`'s record key for the tools the session does not have, ``{name: owner}``: ``harness`` when the
#: harness's own options left it out (not in its ``tools`` list, or in its ``disallowed_tools``), ``confinement``
#: when :func:`confine` switched it off. The CLI answers a call to one before any hook fires.
SWITCHED_OFF_KEY = 'switched_off'
_TOOL_USE_ERROR_TAG = re.compile(r'</?tool_use_error>')
#: An errored tool result, as the CLI writes it (compact) or as any JSON writer does.
_IS_ERROR_TRUE = re.compile(r'"is_error"\s*:\s*true')


def switched_off_tools(known: Iterable[str], *, offered: Sequence[str] | None, harness_off: Iterable[str], confine_off: Iterable[str]) -> dict[str, str]:
    """``{name: owner}`` for each tool the session does not have (:data:`SWITCHED_OFF_KEY`). ``offered`` is the
    harness's own ``tools`` list (``None`` = every built-in is offered), read against the ``known`` built-in names;
    the harness owns what it left out, :func:`confine` only what it switched off itself."""
    out: dict[str, str] = {}
    if offered is not None:
        offered_names = {str(t) for t in offered}
        out.update({name: 'harness' for name in known if name not in offered_names})
    out.update({str(name): 'harness' for name in harness_off if name})
    for name in confine_off:
        if name:
            out.setdefault(str(name), 'confinement')
    return dict(sorted(out.items()))


def _jsonl_dicts(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Each line that is a JSON object; any other line (a partial or still-written row) is skipped."""
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def cli_transcript_rows(path: str | Path, prompt_ids: Iterable[str] = (), *, offsets: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """The rows of the CLI's session transcript (one JSON object per line) that :func:`reconcile_trail` reads:
    none when the part read holds no errored tool result, else every row from the first one of the queries
    ``prompt_ids`` (every row when none is named). Unparseable lines are skipped, an unreadable file read as empty.

    With ``offsets`` (``{path: byte offset}``, one per session), only what was appended since the last read of
    ``path`` is read, and its offset moves past the last complete line read. The file is append-only and each
    turn's queries are new, so no row before them is needed again; a file shorter than its offset is read whole."""
    key = str(path)
    start = offsets.get(key, 0) if offsets is not None else 0
    try:
        with open(path, 'rb') as fh:
            if fh.seek(0, os.SEEK_END) < start:
                start = 0
            fh.seek(start)
            data = fh.read()
    except OSError:
        return []
    if offsets is not None:
        offsets[key] = start + data.rfind(b'\n') + 1
    text = data.decode('utf-8', errors='replace')
    if not _IS_ERROR_TRUE.search(text):
        return []
    wanted = [p for p in prompt_ids if p]
    started = not wanted
    rows: list[dict[str, Any]] = []
    # lazy: once the query's first row is found, every later line is parsed
    for row in _jsonl_dicts(line for line in text.splitlines() if started or any(p in line for p in wanted)):
        if not started:
            if row.get('promptId') not in wanted:
                continue
            started = True
        rows.append(row)
    return rows


def _tool_result_text(block: Mapping[str, Any]) -> str:
    content = block.get('content')
    if isinstance(content, list):
        content = '\n'.join(str(b.get('text') or '') for b in content if isinstance(b, dict) and b.get('type') == 'text')
    return _TOOL_USE_ERROR_TAG.sub('', str(content or '')).strip()


def reconcile_trail(trail: list[Any], rows: Sequence[Mapping[str, Any]], prompt_ids: Iterable[str], *, record: Any, declared: Iterable[str] = ()) -> list[Any]:
    """``trail`` with a row for each tool call the CLI answered with an error in the queries ``prompt_ids``
    (rows of its session transcript) whose id the trail lacks, each placed where the transcript ran it.

    The CLI answers a call to a tool the session does not have, and a call whose input it rejects,
    before any hook fires, so the recorders never see it. A call to a tool ``record`` (:func:`confine`'s
    host-effects record) names switched off (:data:`SWITCHED_OFF_KEY`) is a ``denied`` row, refused ``by``
    the owner it names, whatever the CLI's wording; any other errored call is an ``error`` row, a tool the
    model made up included. MCP wire names are recorded by their ``declared`` key (:func:`declared_tool_name`).
    A call is placed by the first row that ran it: the CLI's later request blobs repeat earlier calls."""
    wanted = {p for p in prompt_ids if p}
    uses: dict[str, tuple[int, str, Any]] = {}
    errored: dict[str, str] = {}
    for i, row in enumerate(rows):
        message = row.get('message')
        content = message.get('content') if isinstance(message, Mapping) else None
        for block in content if isinstance(content, list) else ():
            if not isinstance(block, Mapping):
                continue
            if block.get('type') == 'tool_use' and isinstance(block.get('id'), str):
                uses.setdefault(block['id'], (i, str(block.get('name') or ''), block.get('input')))
            elif block.get('type') == 'tool_result' and block.get('is_error') is True and isinstance(block.get('tool_use_id'), str):
                if row.get('promptId') in wanted:
                    errored.setdefault(block['tool_use_id'], _tool_result_text(block))
    known = {entry.get('call_id') for entry in trail if isinstance(entry, Mapping)}
    off = record.get(SWITCHED_OFF_KEY) if isinstance(record, Mapping) else None
    switched_off: Mapping[str, Any] = off if isinstance(off, Mapping) else {}
    missing: list[tuple[int, dict[str, Any]]] = []
    for call_id, text in errored.items():
        if call_id in known or call_id not in uses:
            continue
        at, name, args = uses[call_id]
        entry: dict[str, Any] = {'tool': declared_tool_name(name, declared), 'args': args if isinstance(args, dict) else {}, 'call_id': call_id, 'result': text}
        if (by := switched_off.get(name)) in ('confinement', 'harness'):
            entry.update({'status': TRAIL_STATUS_DENIED, TRAIL_REFUSAL_KEY: {'by': by, 'reason': text}})
        else:
            entry['status'] = 'error'
        missing.append((at, entry))
    if not missing:
        return trail
    pending = sorted(missing, key=lambda item: item[0])
    out: list[Any] = []
    for entry in trail:
        ran = uses.get(entry.get('call_id')) if isinstance(entry, Mapping) else None
        ran_at = ran[0] if ran is not None else None
        while pending and ran_at is not None and pending[0][0] < ran_at:
            out.append(pending.pop(0)[1])
        out.append(entry)
    return [*out, *(entry for _at, entry in pending)]


def _trail_recorder(trail: list[dict[str, Any]], status: str, declared: frozenset[str] = frozenset()) -> Callable[..., Awaitable[dict[str, Any]]]:
    """A ``PostToolUse`` (``status`` ``ok``) or ``PostToolUseFailure`` (``error``) hook that appends
    each call to ``trail``, an MCP tool by its ``declared`` key (:func:`declared_tool_name`). A
    failure carries ``error``, not ``tool_response``."""

    async def hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:  # noqa: ARG001
        trail.append(
            {
                'tool': declared_tool_name(input_data.get('tool_name'), declared),
                'args': input_data.get('tool_input'),
                'call_id': input_data.get('tool_use_id') or tool_use_id,
                'result': input_data.get('tool_response', input_data.get('error')),
                'status': status,
            }
        )
        return {}

    return hook


def _cli_config_env(route: str, cli_env: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    """``record['cli_config']`` for a CLI signing in by ``route``, and the environment that puts it there.

    Every route gets a config dir of the scenario's own (:func:`_scenario_root`): the CLI's
    settings, session transcripts and ``tool-results/`` land there, not in the operator's home.
    ``CLAUDE_SECURESTORAGE_CONFIG_DIR`` says where its credential store is: a declared route (a
    provider switch, a key or a token) signs in without the operator's, so the store follows the
    scenario's folder; any other route (the existing login, an ``apiKeyHelper``) keeps the operator's own
    store (:func:`_operator_credential_store`).
    """
    config = str(_scenario_root('cfg-'))
    store = config if route in CLI_CREDENTIAL_ENV else _operator_credential_store(cli_env)
    return CLI_CONFIG_SCENARIO, {'CLAUDE_CONFIG_DIR': config, 'CLAUDE_SECURESTORAGE_CONFIG_DIR': store}


def _operator_login_key(cli_env: Mapping[str, str]) -> str | None:
    """The API key ``/login`` saved into the operator's global config (``primaryApiKey`` in
    ``<CLAUDE_CONFIG_DIR or ~>/.claude.json``), which the CLI reads after its keychain item. One the
    CLI cannot parse holds no key for it either."""
    path = Path(cli_env.get('CLAUDE_CONFIG_DIR') or Path.home()) / '.claude.json'
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    key = config.get('primaryApiKey') if isinstance(config, dict) else None
    return key if isinstance(key, str) and key.strip() else None


def _carry_operator_sign_in(settings: Mapping[str, Any], cli_env: Mapping[str, str], config: Path, route: str) -> None:
    """The operator's sign-in the CLI keeps in its config, carried into the scenario's config dir
    ``config``, where the CLI now reads it: the user ``settings`` that sign it in or refresh its
    provider credentials (:data:`_SIGN_IN_SETTINGS`), and the API key ``/login`` saved in the global
    config (:func:`_operator_login_key`). Nothing else of either file moves; the key is redacted like
    any credential the CLI holds.

    A route the environment declares (``route``, :data:`CLI_CREDENTIAL_ENV`) is the one the run
    signs in with: the operator's own Anthropic keys (:data:`_KEY_SETTINGS`, the ``/login`` key)
    stay behind, since the CLI would send a key of theirs before a declared token."""
    declared = route in CLI_CREDENTIAL_ENV
    names = [name for name in _SIGN_IN_SETTINGS if not (declared and name in _KEY_SETTINGS)]
    carried = {name: settings[name] for name in names if isinstance(settings.get(name), str)}
    key = None if declared else _operator_login_key(cli_env)
    for name, document in (('settings.json', carried), ('.claude.json', {'primaryApiKey': key} if key else {})):
        if document:
            (config / name).write_text(json.dumps(document), encoding='utf-8')
            (config / name).chmod(0o600)
    if key and len(key) >= _MIN_SECRET_LEN:
        _CLI_SECRETS_HANDED[key] = 'primaryApiKey'


#: Encodings a path can be spelled in inside a string, and how to read it back.
_ENCODINGS: tuple[tuple[str, Callable[[str], str]], ...] = (('\\/', lambda s: s.replace('\\/', '/')), ('%2f', unquote))

#: macOS: the data volume's own spelling of every path on it (a firmlink, not a symlink).
_DATA_VOLUME = '/System/Volumes/Data'


#: macOS volumes are case-insensitive by default: ``/users/x`` IS ``/Users/x`` there.
_CASE_FOLDING = sys.platform == 'darwin'


def _under(path: str | Path, root: str | Path) -> bool:
    path, root = str(path), str(root)
    if _CASE_FOLDING:
        path, root = path.casefold(), root.casefold()
    return path == root or path.startswith(root.rstrip('/') + '/')


def _identity(path: str | Path) -> tuple[int, int] | None:
    try:
        st = Path(path).stat()
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino)


def _identities(paths: Any) -> frozenset[tuple[int, int]]:
    return frozenset(i for i in map(_identity, paths) if i is not None)


def _ancestors(path: str) -> Iterator[str]:
    """``path`` and each parent up to ``/``."""
    current = os.path.normpath(path)
    while True:
        yield current
        parent = str(Path(current).parent)
        if parent == current:
            return
        current = parent


def _glob_root(pattern: str) -> str:
    """The folder a glob searches: its text up to the last ``/`` before the first wildcard, so
    ``/U*/*/Downloads/*`` searches ``/``, not a path named ``/U``."""
    wild = re.search(r'[*?\[{]', pattern)
    return pattern if wild is None else pattern[: pattern.rfind('/', 0, wild.start()) + 1]


def _brace_alternatives(pattern: str, limit: int = 64) -> list[str]:
    """``a{b,c}d`` → ``abd``, ``acd``: a brace can hold whole roots (``{/x,/y}/*``). Past ``limit``
    alternatives, or an unbalanced brace, the pattern is judged from ``/``."""
    out, todo = [], [pattern]
    while todo:
        p = todo.pop()
        start = p.find('{')
        if start < 0:
            out.append(p)
            continue
        depth, parts, last = 0, [], start + 1
        for i in range(start, len(p)):
            depth += {'{': 1, '}': -1}.get(p[i], 0)
            if depth == 1 and p[i] == ',':
                parts.append(p[last:i])
                last = i + 1
            elif depth == 0:
                todo += [p[:start] + alt + p[i + 1 :] for alt in [*parts, p[last:i]]]
                break
        else:
            return ['/']
        if len(out) + len(todo) > limit:
            return ['/']
    return out


#: The XDG base folders the agent's shell gets under its own home, so a tool that honours them writes there.
_XDG_HOMES = (('XDG_CONFIG_HOME', '.config'), ('XDG_CACHE_HOME', '.cache'), ('XDG_DATA_HOME', '.local/share'), ('XDG_STATE_HOME', '.local/state'))

#: The file in a scenario root naming the process it belongs to, so a later run can sweep a dead one's.
_OWNER_FILE = '.plurai-owner-pid'


#: Where the per-user scenario-roots parent is made (a module value so a test can point it elsewhere).
_SCENARIO_ROOTS_DIR = Path('/tmp')  # noqa: S108


class CliAgentUnsupportedOnWindows(HarnessFault):  # noqa: N818 - a HarnessFault, named for what it refuses
    """A CLI-driven agent (``claude_agent_sdk`` confined per scenario) on native Windows, which has no port yet."""


def _scenario_roots_parent(*, is_win: bool | None = None) -> Path:
    # Every confined scenario root comes through here; its per-user /tmp parent, sandbox and signals are POSIX.
    if is_windows() if is_win is None else is_win:
        raise CliAgentUnsupportedOnWindows('CLI-driven agents are not supported on native Windows yet; run this agent under WSL or on macOS/Linux')
    return _SCENARIO_ROOTS_DIR / f'plurai-{os.getuid()}'


#: What the permissive profile opens for writing in the shared ``/tmp`` so LibreOffice (soffice) can start. It places
#: its single-instance pipe in ``/tmp`` (``/tmp/OSL_PIPE_<uid>_SingleOfficeIPC_<hash of its profile>``) after an
#: ``access(/tmp, W_OK)`` check, and with either refused it exits 1 without a word and converts nothing. Two entries
#: per spelling of ``/tmp`` (it is ``/private/tmp`` on macOS): the pipe names of that program and this user, and the
#: folder ITSELF, as a pattern whose last character is a one-character class (``/tm[p]``: the CLI reads a plain path
#: as a whole subtree, a pattern as exactly what it matches). ``/tmp`` stays closed to every other write, and no root
#: the read scope denies (the CLI's and every scenario's own folders) lies under a pipe name.
_SOFFICE_IPC_WRITES = [
    entry
    for root in ('/tmp', '/private/tmp')  # noqa: S108
    for entry in (f'{root[:-1]}[{root[-1]}]', f'{root}/OSL_PIPE_{os.getuid() if hasattr(os, "getuid") else 0}_*')
]


def _scenario_roots() -> Path:
    """``/tmp/plurai-<uid>``: the one parent of every scenario's shell home and CLI temp root. The read
    scope denies it, and allows back only this scenario's own, so concurrent scenarios and trials
    cannot read each other's. /tmp, not $TMPDIR: under a Claude Code session that is the shared root
    the sandbox denies — and short, because the CLI's temp root must stay under ~30 bytes."""
    parent = _scenario_roots_parent()
    try:
        parent.mkdir(mode=0o700, exist_ok=True)
    except FileExistsError:
        pass  # a file, or a link to one, holds the name: refused below
    st = parent.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise HarnessFault(f'{parent} is a link or belongs to another user — remove it, then re-run')
    if stat.S_IMODE(st.st_mode) != 0o700:  # made earlier, or by an older run, with a looser mode
        parent.chmod(0o700)
    return parent


#: Seconds after which a scenario root with no owner file is a leftover, not one being made: a root
#: is written its owner file right after ``mkdtemp``, so only a root re-created after its removal lacks one.
_ORPHAN_ROOT_SECONDS = 60


@functools.lru_cache(maxsize=1)
def _sweep_dead_scenario_roots() -> None:
    """Remove the scenario roots of processes that are gone (a SIGKILL skips ``atexit``), and the
    roots left with no owner file (re-created after their removal). Once per process."""
    for root in _scenario_roots().iterdir():
        try:
            pid = int((root / _OWNER_FILE).read_text(encoding='utf-8'))
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(root, ignore_errors=True)
        except FileNotFoundError:
            with contextlib.suppress(OSError):
                if time.time() - root.lstat().st_mtime > _ORPHAN_ROOT_SECONDS:
                    shutil.rmtree(root, ignore_errors=True)
        except (OSError, ValueError):
            continue  # not ours to judge: a live process of another user, or an unreadable owner file


#: Seconds the agent's CLI is given to exit on SIGTERM (it flushes its session transcript) before SIGKILL.
_CLI_EXIT_SECONDS = 5.0


def _end_cli_processes(timeout: float = _CLI_EXIT_SECONDS) -> None:
    """At the scenario's end, before its roots are removed: end the agent's CLI processes that
    claude_agent_sdk still tracks, and wait for them. A harness keeps its CLI session for the
    scenario's life and never disconnects it; left to claude_agent_sdk's own exit hook (a SIGTERM
    after this process's removals), the CLI flushes its session transcript into a config dir that
    was already removed, and re-creates it. What the scenario records of that transcript, it
    records before this process ends."""
    transport = sys.modules.get('claude_agent_sdk._internal.transport.subprocess_cli')
    live = [p.pid for p in list(getattr(transport, '_ACTIVE_CHILDREN', ())) if getattr(p, 'returncode', 0) is None]
    live = [pid for pid in live if not _reaped(pid)]  # one reaped off the loop keeps ``returncode`` None: its pid may be another's now
    for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, timeout)):
        for pid in live:
            with contextlib.suppress(OSError):
                os.kill(pid, sig)
        deadline = time.monotonic() + wait
        while live and time.monotonic() < deadline:
            live = [pid for pid in live if not _reaped(pid)]
            if live:
                time.sleep(0.05)
        if not live:
            return


async def end_cli_processes_in_loop(timeout: float = 2.0) -> None:
    """:func:`_end_cli_processes` for a caller that still owns the running loop: end the agent's CLI
    processes SIGTERM first (SIGKILL on timeout), then close their pipe transports while the loop is
    alive; closed at interpreter exit, on a closed loop, they raise ``Event loop is closed``. Never raises."""
    import asyncio

    transport = sys.modules.get('claude_agent_sdk._internal.transport.subprocess_cli')
    procs: list[Any] = list(getattr(transport, '_ACTIVE_CHILDREN', ()))
    for p in procs:
        with contextlib.suppress(Exception):
            if getattr(p, 'returncode', 0) is None:
                p.terminate()
                try:
                    await asyncio.wait_for(p.wait(), timeout)
                except TimeoutError:
                    p.kill()
                    await asyncio.wait_for(p.wait(), timeout)
    for p in procs:
        with contextlib.suppress(Exception):
            await p.aclose()


def _reaped(pid: int) -> bool:
    """Whether child ``pid`` has exited (reaped here, or by whoever else waits on it)."""
    try:
        return os.waitpid(pid, os.WNOHANG)[0] == pid
    except ChildProcessError:
        return True


def _scenario_root(prefix: str) -> Path:
    """A fresh 0700 folder of this scenario's own under :func:`_scenario_roots`, gone when this process ends."""
    _sweep_dead_scenario_roots()
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=_scenario_roots()))
    (root / _OWNER_FILE).write_text(str(os.getpid()), encoding='utf-8')
    atexit.register(shutil.rmtree, root, True)
    return root


def _operator_shell() -> str:
    """The shell the CLI would pick: ``$SHELL`` when it is a zsh or a bash, else the first found."""
    declared = os.environ.get('SHELL', '')
    if any(f in Path(declared).name for f in ('zsh', 'bash')) and os.access(declared, os.X_OK):
        return declared
    found = next((p for p in ('/bin/zsh', '/usr/bin/zsh', '/bin/bash', '/usr/bin/bash') if os.access(p, os.X_OK)), None)
    return found or shutil.which('bash') or '/bin/sh'


def _both_forms(paths: list[Path]) -> tuple[Path, ...]:
    """Each path as given, as resolved (``/tmp`` is ``/private/tmp`` on macOS) and, on macOS, as the
    data volume spells it — deduplicated."""
    out: dict[str, None] = {}
    for p in paths:
        absolute = Path(os.path.normpath(Path(p).absolute()))
        for form in (str(absolute), str(absolute.resolve())):
            out.setdefault(form)
            if sys.platform == 'darwin' and Path(_DATA_VOLUME + form).exists() and not form.startswith(_DATA_VOLUME):
                out.setdefault(_DATA_VOLUME + form)
    return tuple(Path(p) for p in out)


#: Folders under the home that hold the operator's own content: never an import root, whatever names one.
_USER_CONTENT = ('Desktop', 'Documents', 'Downloads', 'Pictures', 'Movies', 'Music', 'Library/Mobile Documents')


def _site_dirs() -> tuple[str, ...]:
    """This interpreter's own site-packages folders, its user site included when enabled."""
    dirs = list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        dirs.append(site.getusersitepackages())
    return tuple(d for d in dirs if Path(d).is_dir())


@functools.lru_cache(maxsize=4)
def _dist_targets(site_dirs: tuple[str, ...]) -> tuple[str, ...]:
    """What the distributions INSTALLED in ``site_dirs`` import from: an editable install's source
    tree (``direct_url.json``) and each folder a ``.pth`` file listed in their ``RECORD`` names. A
    ``.pth`` no distribution installed, and ``PYTHONPATH``, are not asked."""
    out = []
    for dist in importlib.metadata.distributions(path=list(site_dirs)):
        try:
            info = json.loads(dist.read_text('direct_url.json') or '{}')
        except (OSError, ValueError):
            info = {}
        url = urlparse(str(info.get('url') or ''))
        if (info.get('dir_info') or {}).get('editable') and url.scheme == 'file':
            out.append(unquote(url.path))
        for record in dist.files or ():
            if record.suffix != '.pth':
                continue
            pth = Path(str(dist.locate_file(record)))
            try:
                lines = pth.read_text(encoding='utf-8').splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            out += [os.path.join(pth.parent, ln.strip()) for ln in lines if ln.strip() and not ln.startswith(('#', 'import ', 'import\t'))]
    return tuple(out)


def _import_roots(home: Path) -> tuple[list[Path], int]:
    """Where THIS interpreter imports from — it is the agent's interpreter (the child heads PATH
    with it): its prefixes, its site-packages and what the distributions installed there import
    from, each by its realpath — and how many candidates were refused for being the home, an
    ancestor of it, or one of its user-content folders."""
    candidates = [sys.prefix, sys.base_prefix, *_site_dirs(), *_dist_targets(_site_dirs())]
    protected = [str(h / sub) for h in _both_forms([home]) for sub in ('', *_USER_CONTENT)]
    roots, refused = [], 0
    for real in dict.fromkeys(os.path.realpath(c) for c in candidates):  # judged by what it IS
        if not Path(real).is_dir():
            continue
        if any(_under(f, real) for f in protected):  # it is one of them, or holds one
            refused += 1
        else:
            roots.append(Path(real))
    return roots, refused


def scrub_parent_record(obj: Any, *, repo: str | Path) -> Any:
    """A record the PARENT writes about the run (the egress preflight, the SDK identity) as it may
    leave this machine: :meth:`HostReadScope.scrub`, with ``repo`` read as allowed — so its paths
    ship whole from ``~``, and no path names the operator."""
    return host_read_scope(run_dir=None, repo=repo, harness=None).scrub(obj)[0]


#: A path character in free text: not whitespace or a control character, a quote, shell punctuation or ``:`` (which starts a
#: ``:line`` suffix).
_PATH_CHAR = r"""[^\s\x00-\x1f\x7f'"`;|&<>(){}\[\],:]"""

#: The rest of a path after its root.
_PATH_RUN = re.compile(_PATH_CHAR + '*')

#: Words the service reads as an infrastructure outage in an untyped error (a status code, a timeout, a capacity
#: phrase). A project-relative path that spells one ships as the stand-in, so the rule never creates one.
_OUTAGE_WORDS = re.compile(
    r'\b[45]\d\d\b|resource[ _-]?exhausted|timed?[ _-]?out|connection|unavailable|overloaded|rate[ _-]?limit|quota|exhaust|capacity|throttl'
    r'|per[ _-]?(?:minute|second)',
    re.IGNORECASE,
)

_NOT_A_PATH_END = '.'

#: How many passes make a text stable: a project-relative result can itself hold a spelling of a root.
_MAX_PASSES = 4

#: A ``file:`` URL prefix (``file:``, ``file://``, ``file://localhost``) at the end of the text before a root.
_FILE_PREFIX = re.compile(r'(?<![\w+.-])file:(?://(?:localhost)?)?$', re.IGNORECASE)

#: A part of a package's path after its ``site-packages`` folder: word characters, dots, hyphens and plus signs.
_PACKAGE_SEGMENT = re.compile(r'[\w.+-]+')

#: Spellings of a slash inside a string: JSON-escaped, ``/``-escaped and percent-encoded (a run of ``%XX``, so a
#: multi-byte character decodes whole). The text is decoded up to :data:`_MAX_PASSES` times, to find a root, never to
#: rewrite the text.
_ENCODED = re.compile(r'(?:%[0-9A-Fa-f]{2})+|\\/|\\u002f', re.IGNORECASE)


def _decoded_view(text: str) -> tuple[str, list[int] | None, list[int] | None]:
    """``text`` with its escaped and percent-encoded spellings decoded, and for every character of the result where it
    starts and ends in ``text`` (``None, None`` when nothing decodes: the view is ``text`` itself). A decoded character
    maps to the whole escape it came from, so a span of the view maps back to a span of ``text`` that covers it."""
    starts: list[int] | None = None
    ends: list[int] | None = None
    for _ in range(_MAX_PASSES):
        if _ENCODED.search(text) is None:
            break
        if starts is None or ends is None:
            starts, ends = list(range(len(text))), list(range(1, len(text) + 1))
        out: list[str] = []
        new_starts: list[int] = []
        new_ends: list[int] = []
        pos = 0
        for m in _ENCODED.finditer(text):
            out.append(text[pos : m.start()])
            new_starts += starts[pos : m.start()]
            new_ends += ends[pos : m.start()]
            chunk = m.group()
            decoded = bytes.fromhex(chunk.replace('%', '')).decode('utf-8', errors='replace') if chunk[0] == '%' else '/'
            out.append(decoded)
            new_starts += [starts[m.start()]] * len(decoded)
            new_ends += [ends[m.end() - 1]] * len(decoded)
            pos = m.end()
        out.append(text[pos:])
        new_starts += starts[pos:]
        new_ends += ends[pos:]
        text, starts, ends = ''.join(out), new_starts, new_ends
    return text, starts, ends


@functools.lru_cache(maxsize=1)
def _interpreter_roots() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The folders this interpreter lives in (its prefixes) and the ones it imports packages from (``site-packages``), read
    once: the interpreter does not move during a run."""
    paths = sysconfig.get_paths()
    sites = dict.fromkeys(p for p in (*_site_dirs(), paths.get('purelib'), paths.get('platlib')) if p)
    return tuple(p for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix) if p), tuple(sites)


def _alternation(words: Iterable[str]) -> str:
    """A regex group that matches any of ``words``, longest first, then in text order, so the pattern is the same for the
    same set whatever order the set iterates in."""
    return '(?:' + '|'.join(re.escape(w) for w in sorted(words, key=lambda w: (-len(w), w))) + ')'


#: Stands in for the ``user:password@`` part of a URL. It holds no digit and no character a URL's userinfo may hold.
USERINFO_STAND_IN = '[plurai: credentials not sent]'

#: The userinfo of a ``scheme://`` URL in a text as written: everything after ``://`` up to the last ``@`` that comes before
#: the first ``/``, ``?``, ``#``, whitespace or quote. An encoded character (``%2F``, ``%40``) is part of it, so it is read in
#: the text as written, never in a decoded view. Fail closed: ``https://api.example.com:443,admin@corp.example`` (a URL, a comma
#: and an address) reads as userinfo too.
_USERINFO = re.compile(r"""(?<=://)[^\s/?#'"`]+@""")


def _root_form(root: str) -> str:
    """``root`` (a path with ``/`` separators) as a regex that also reads a drive letter, a rooted backslash and mixed or
    doubled separators between its folders: ``/Users/jane`` matches ``\\Users\\jane`` and ``C:/Users//jane``. It starts at one
    separator, so a long run of separators costs one step per character."""
    return r'(?:(?<!\w)[A-Za-z]:)?[\\/]' + r'[\\/]+'.join(re.escape(segment) for segment in root.strip('/').split('/'))


@dataclass(frozen=True)
class _HomeSpellings:
    """The home as the pattern needs it: its spellings as a path and the user names (``~<name>``)."""

    forms: frozenset[str] = frozenset()
    names: frozenset[str] = frozenset()

    @property
    def dashed(self) -> frozenset[str]:
        """The home with its slashes written as hyphens (``-Users-jane``), for a home of two folders or more."""
        return frozenset(h.replace('/', '-') for h in self.forms if h.count('/') >= 2)

    @property
    def tails(self) -> frozenset[str]:
        """The last two folders of the home (``home/jane``): a relative spelling of it."""
        return frozenset('/'.join(h.strip('/').split('/')[-2:]) for h in self.forms if h.count('/') >= 2)


@functools.lru_cache(maxsize=32)
def _roots_pattern(posix: frozenset[str], windows: frozenset[str], home: _HomeSpellings, *, fold: bool) -> re.Pattern[str]:
    """The one pattern that finds a spelling of a known root in a text, as the named group that says which kind:

    * ``windows`` — a root with a drive, either separator, any case, and an extended-length prefix;
    * ``deep`` — a root of two folders or more (``/Users/jane``), found anywhere, glued to a word included;
    * ``shallow`` — a root of one folder (``/root``, ``/app``), which is also an ordinary segment of a relative path or a
      URL (``https://pypi.org/simple/root/``): found only where it does not follow a word character, ``.``, ``~`` or ``-``;
    * ``tail`` — the last two folders of the home with no root before them (``home/jane/x``), a spelling of the home;
    * ``tilde`` — the home as ``~/``, ``~<user>``, ``$HOME`` or ``${HOME}``, each not followed by a word character (``$HOMEBREW``
      and ``~janet`` are no spelling of the home, ``~jane.`` and ``$HOME,`` are), and a bare ``~`` names nothing;
    * ``dashed`` — the home with its slashes written as hyphens, as a tool names a folder after it.

    Every alternation is a group of its own, so a boundary in front of it guards every alternative. A posix root also
    reads a drive letter, a rooted backslash and mixed separators, and ignores case where the file system does
    (``fold``). A root ends at a path separator or at a character that is no part of a path token, so ``/Users/jane`` is
    not found in ``/Users/janet``.
    """
    end = rf'(?=[\\/]|(?!{_PATH_CHAR}))'
    case = '(?i:{})' if fold else '(?:{})'

    def posix_forms(roots: Iterable[str]) -> str:
        return case.format('|'.join(_root_form(r) for r in sorted(roots, key=lambda r: (-len(r), r))))

    parts: list[str] = []
    if windows:
        forms = [r'[\\/]+'.join(re.escape(segment) for segment in w.split('/') if segment) for w in sorted(windows, key=lambda w: (-len(w), w))]
        parts.append(rf'(?P<windows>(?<![\w.~$/\\-])(?:\\\\[?.]\\)?(?i:{"|".join(forms)}){end})')
    if deep := [r for r in posix if r.count('/') > 1]:
        parts.append(rf'(?P<deep>{posix_forms(deep)}){end}')
    if shallow := [r for r in posix if r.count('/') == 1]:
        parts.append(rf'(?P<shallow>(?<![\w.~-]){posix_forms(shallow)}){end}')
    if tails := home.tails:
        forms = '|'.join(r'[\\/]+'.join(re.escape(segment) for segment in t.split('/')) for t in sorted(tails, key=lambda t: (-len(t), t)))
        parts.append(rf'(?P<tail>(?<![\w.~$/\\-]){case.format(forms)}){end}')
    if home.forms:
        names = f'~{_alternation(home.names)}(?!\\w)|' if home.names else ''
        parts.append(rf'(?P<tilde>(?<![\w.~$/\\-])(?:~(?=[\\/])|{names}\$\{{HOME\}}(?!\w)|\$HOME(?!\w)))')
    if dashed := home.dashed:
        parts.append(rf'(?P<dashed>(?<![\w-]){_alternation(dashed)})(?=-|(?![\w.]))')
    return re.compile('|'.join(parts))


class _Shippable:
    """One pass of :func:`shippable_text`: the folders of this machine, read once.

    The rule knows the folders the operator owns (:meth:`_read_known_roots`): the home, the project, the working folder, the
    temporary folder and the folders of the running interpreter. It finds a spelling of one of them in a text and replaces
    the path from there to its end; every other byte of the text ships as it was.

    Nothing here raises: a folder that cannot be read (a deleted working folder, no home) is left out, and a text
    whose rewrite fails ships as :data:`~plurai.orchestrator.local_paths.LOCAL_PATH_MARKER` whole.
    """

    def __init__(self, project: str | Path | None) -> None:
        try:
            home = Path(os.path.normpath(Path.home()))
        except (RuntimeError, OSError, KeyError):
            home = Path('/')
        self.scope = HostReadScope(home=home, deny=(), allow=())
        self.changed = 0
        self.project_forms: tuple[str, ...] = ()
        self.project_windows = ''
        home_real = _realpath(home)
        self.user_names = frozenset({home.name, Path(home_real).name} - {''}) if home.parent != home else frozenset()
        project_path = _absolute(project) if project is not None else None
        if project_path is not None and not self._holds_a_home(project_path, home_real):
            self.project_forms = tuple(dict.fromkeys((*_forms(project_path), _realpath(project_path))))
            self.project_windows = _windows_form(str(project))
        self._read_known_roots(home)

    def _read_known_roots(self, home: Path) -> None:
        """Read the folders of this machine, once, and build the pattern that finds them."""
        try:
            cwd: Path | None = Path(os.path.normpath(Path.cwd()))
        except OSError:
            cwd = None
        prefixes, sites = _interpreter_roots()
        folders = [*([cwd] if cwd else []), Path(tempfile.gettempdir()), *(Path(p) for p in (*prefixes, *sites))]
        spellings = {form for folder in folders for form in _forms(folder)} | set(self.project_forms)
        home_forms = frozenset(self.scope._home_forms) - {'/'} if self.user_names else frozenset()  # noqa: SLF001
        self.site_dirs: tuple[str, ...] = tuple(dict.fromkeys(_windows_form(f) or f for site in sites for f in _forms(Path(site))))
        windows = {_windows_form(f) for f in spellings if re.match(r'[A-Za-z]:', f) and _windows_form(f)} | ({self.project_windows} - {''})
        posix = {f for f in spellings | home_forms if f not in ('', '/') and not (re.match(r'[A-Za-z]:', f) and _windows_form(f))}
        self.pattern = _roots_pattern(frozenset(posix), frozenset(windows), _HomeSpellings(home_forms, self.user_names), fold=_CASE_FOLDING)

    def _holds_a_home(self, project: Path, home_real: str) -> bool:
        """Whether a path relative to ``project`` could spell the user: it is the home folder, a folder above it
        (``/Users``, ``/home``, ``/``) or a users folder, so ``Users/<user>/…`` would be a relative path."""
        home = self.scope.home
        return any(_under(h, project) for h in (home, home_real)) or project.parent == project or project.as_posix() in ('/Users', '/home')

    def names_the_user(self, shown: str) -> bool:
        """Whether a relative path has a segment that is the user name or the home folder's name."""
        return any(segment in self.user_names for segment in re.split(r'[\\/]', shown))

    def relative(self, path: str) -> str | None:
        """``path`` (absolute, normalised) relative to the project, or ``None`` when it lies outside it."""
        if not self.project_forms:
            return None
        root = next((r for r in self.project_forms if _under(path, r)), None)
        return None if root is None else path[len(root) :].strip('/') or '.'

    def stand_in(self, path: str, *, text: bool = False) -> str:
        """``path`` as a record may show it: relative to the project when it lies under it, else the stand-in. In a free
        text (``text``) a path outside the project that lies in a ``site-packages`` folder of this interpreter keeps the part
        from the package on (:meth:`package_frame`)."""
        shown = self.relative(path)
        if shown is None:
            return self.package_frame(path) if text else LOCAL_PATH_MARKER
        return LOCAL_PATH_MARKER if _OUTAGE_WORDS.search(shown) or self.names_the_user(shown) else shown

    def package_frame(self, path: str) -> str:
        """The stand-in followed by the part of ``path`` after a ``site-packages`` folder this interpreter imports from
        (``<stand-in>langgraph/pregel/main.py``): the package's own file names hold nothing of the machine and tell which
        library a traceback frame is in. The stand-in alone when the path is in no such folder, a segment of the part
        is empty, ``.``, ``..`` or no package file name, or the part spells an outage word or the user name. The result
        holds no absolute path, so shipping it again changes nothing."""
        slashed = path.replace('\\', '/')
        for folder in self.site_dirs:
            if _under(slashed, folder) and slashed != folder:
                rest = slashed[len(folder.rstrip('/')) + 1 :]
                segments = rest.split('/')
                named = all(_PACKAGE_SEGMENT.fullmatch(s) and s not in ('.', '..') for s in segments)
                if named and not (_OUTAGE_WORDS.search(rest) or self.names_the_user(rest)):
                    return LOCAL_PATH_MARKER + rest
                break
        return LOCAL_PATH_MARKER

    def posix(self, token: str) -> str:
        body = token.rstrip(_NOT_A_PATH_END)
        end = token[len(body) :]
        if body[:1] not in ('/', '~', '$'):  # a drive letter or a rooted backslash spells the same path
            body = re.sub(r'^[A-Za-z]:', '', body).replace('\\', '/')
        return self.stand_in(os.path.normpath(self.scope._expand(body)), text=True) + end  # noqa: SLF001

    def windows(self, token: str, *, text: bool = False) -> str:
        body = token.rstrip(_NOT_A_PATH_END)
        end = token[len(body) :]
        slashed = _windows_form(body)
        root = self.project_windows
        if root and (slashed.casefold() == root.casefold() or slashed.casefold().startswith(root.casefold() + '/')):
            shown = slashed[len(root) :].strip('/') or '.'
            return (LOCAL_PATH_MARKER if _OUTAGE_WORDS.search(shown) or self.names_the_user(shown) else shown) + end
        return (self.package_frame(slashed) if text else LOCAL_PATH_MARKER) + end

    @staticmethod
    def _token_end(view: str, end: int, starts: list[int] | None, ends: list[int] | None) -> int:
        """Where the path token that has read up to ``view[end]`` ends: at the first whitespace or shell punctuation. A
        space that came from an escape (``%20``) is part of the token, so a ``file:`` or percent-encoded path, which
        carries its spaces encoded, is whole."""
        while True:
            end = _PATH_RUN.match(view, end).end()  # type: ignore[union-attr]  # ``*`` always matches
            if starts is None or ends is None or end >= len(view) or not view[end].isspace() or ends[end] - starts[end] < 2:
                return end
            end += 1

    def once(self, text: str) -> str:
        """``text`` with every path under a known root replaced, from the root to the end of the path: relative to the
        project when it lies under it, else the stand-in. The path ends at the first whitespace, or at the closing quote when
        a quote opened before it on the same line. A root is found in the text as written and in its decoded view
        (JSON-escaped, ``\\u002f`` and percent-encoded slashes), and a span found in the view is replaced in the text it
        came from, so every other byte ships as it was. The ``user:password@`` of a URL is replaced in the text as written."""
        text = _USERINFO.sub(f'{USERINFO_STAND_IN}@', text)
        view, starts, ends = _decoded_view(text)
        edits: list[tuple[int, int, str]] = []
        for m in _USERINFO.finditer(view):  # the userinfo of a URL spelled with an escaped separator (``https%3A%2F%2Fu%3Ap%40h``)
            edits.append((m.start(), m.end(), f'{USERINFO_STAND_IN}@'))
        spans = [(start, end) for start, end, _ in edits]
        quotes = {q: [0, -1] for q in '\'"`'}  # per quote character: how many came before the match on this line, and the last
        pos = line_end = scanned = ui = 0
        while (m := self.pattern.search(view, pos)) is not None:
            kind = m.lastgroup
            start = m.start()
            while ui < len(spans) and spans[ui][1] <= start:
                ui += 1
            if ui < len(spans) and spans[ui][0] < m.end():  # inside a userinfo
                pos = spans[ui][1]
                continue
            if m.end() > line_end:  # a new line: its end and its quotes are looked up once
                line_end = view.find('\n', m.end())
                line_end = len(view) if line_end < 0 else line_end
                scanned = view.rfind('\n', 0, start) + 1
                quotes = {q: [0, -1] for q in quotes}
            for q, state in quotes.items():
                i = view.find(q, scanned, start)
                while i != -1:
                    state[:] = [state[0] ^ 1, i]
                    i = view.find(q, i + 1, start)
            scanned = start
            file_url = _FILE_PREFIX.search(view, max(0, start - 17), start)
            inside = max((state[1], q) for q, state in quotes.items() if state[0])[1] if any(state[0] for state in quotes.values()) else None
            closing = view.find(inside, m.end(), line_end) if inside else -1
            end = closing if closing >= 0 else self._token_end(view, m.end(), starts, ends)
            token = view[start:end]
            if file_url or kind in ('dashed', 'tail'):
                replaced = LOCAL_PATH_MARKER
            elif kind == 'windows':
                replaced = self.windows(token, text=True)
            else:
                replaced = self.posix(token)
            edits.append((file_url.start() if file_url else start, end, replaced))
            pos = max(end, m.end())
        if not edits:
            return text
        out: list[str] = []
        at = 0
        for start, end, replaced in sorted(edits, key=lambda edit: edit[:2]):
            first = start if starts is None else starts[start]
            last = end if ends is None else ends[end - 1]
            first = max(first, at)
            out += [text[at:first], replaced]
            at = max(last, first)
        out.append(text[at:])
        return ''.join(out)

    def text(self, text: str) -> str:
        try:
            shipped = text
            for _ in range(_MAX_PASSES):
                again = self.once(shipped)
                if again == shipped:
                    break
                shipped = again
            else:
                shipped = LOCAL_PATH_MARKER
        except Exception:  # noqa: BLE001 - a rule that cannot read a text never ships it raw
            shipped = LOCAL_PATH_MARKER
        self.changed += shipped != text
        return shipped

    def key(self, key: str, taken: dict[Any, Any]) -> str:
        """``key`` shipped, numbered (``<key> #2``) when another key of the same object already ships as that text."""
        shipped = self.text(key)
        if shipped not in taken:
            return shipped
        n = 2
        while f'{shipped} #{n}' in taken:
            n += 1
        return f'{shipped} #{n}'

    def walk(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, os.PathLike):
            return self.text(os.fspath(obj)) if isinstance(os.fspath(obj), str) else obj
        if isinstance(obj, list | tuple):
            return [self.walk(v) for v in obj]
        if isinstance(obj, dict):
            out: dict[Any, Any] = {}
            for k, v in obj.items():
                out[self.key(k, out) if isinstance(k, str) else k] = self.walk(v)
            return out
        return obj


def _realpath(path: str | Path) -> str:
    """``os.path.realpath``, or the path as given when it cannot be resolved (an embedded NUL)."""
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return str(path)


def _absolute(path: str | Path) -> Path | None:
    try:
        return Path(os.path.normpath(Path(path).absolute()))
    except (OSError, ValueError):
        return None


def _forms(path: Path) -> tuple[str, ...]:
    try:
        return tuple(str(p) for p in _both_forms([path]))
    except (OSError, ValueError, RuntimeError):
        return (str(path),)


def _windows_form(path: str) -> str:
    """``path`` with ``/`` separators when it is a Windows drive, UNC, extended-length or rooted path, else ``''``."""
    return path.replace('\\', '/').rstrip('/') if re.match(r'[A-Za-z]:[\\/]|\\', path) else ''


def shippable_text(obj: Any, *, project: str | Path | None) -> Any:
    """``obj`` (JSON-shaped) as a record the SDK writes about a run may leave this machine.

    The rule for texts the SDK writes (an error, a summary, a log line), always on. It knows the folders this machine
    owns, its known roots: the home, the project, the working folder, the temporary folder and the prefixes and
    ``site-packages`` folders of the running interpreter, each as written and resolved. It finds a spelling of one in a
    text (the root as a path, as ``~``, ``~<user>``, ``$HOME`` or ``${HOME}``, as a ``file:`` URL, with a drive and either
    separator on Windows, JSON-escaped, ``\\u002f``-escaped or percent-encoded) and replaces the path from there to its
    end:

    * a path under ``project`` is written relative to it (``project`` itself: ``.``; ``None``: no project). A project
      that is the home folder or holds it (``/Users``, ``/home``, ``/``) is no project, the home is still a root, and a
      project-relative result with a segment that is the user name is the stand-in;
    * a path under any other known root is
      :data:`~plurai.orchestrator.local_paths.LOCAL_PATH_MARKER`, and a ``:line`` suffix after it stays. A path
      ends at the first whitespace or shell punctuation, or at the closing quote when a quote opened before it on the same
      line (``File "/home/jane/My Project/a.py"``). Residual, accepted: an unquoted path with a space ships the words after
      the space (``/home/jane/My Project/a.py`` becomes ``<stand-in> Project/a.py``); the root and the part before the first
      space never ship. A path in a ``site-packages`` folder of this interpreter is the stand-in followed by the part after that folder
      (``<stand-in>langgraph/pregel/main.py``) when that part is a plain package path (no empty, ``.`` or ``..``
      segment, no space, no outage word, no user name);
    * every other byte of a text ships as it was. That includes any path that is under no known root (``/opt/data/x.csv``,
      another user's home), URLs and routes (``/api/orders/42``), template advice (``<out>/<scenario>/transcript.json``)
      and relative paths. Residual, accepted: such a path names a place of this machine and is sent as written; the rule
      does not guess from the shape of a text what is a path;
    * a spelling of a root is exact, not guessed: a drive letter, a rooted backslash and mixed or doubled separators
      read as one (``\\Users\\jane``, ``C:/Users/jane``), case is ignored where the file system ignores it, the last two
      folders of the home with nothing before them (``home/jane/x``) are the home. A relative path (``../x``, ``x/y``) is
      never read from the working folder and ships as written;
    * the ``user:password@`` of any ``scheme://`` URL, read in the text as written (an encoded ``%2F`` or ``%40`` is part of
      it), is replaced by ``[plurai: credentials not sent]@``; the host and the rest are kept. The userinfo is everything
      after ``://`` up to the last ``@`` before the first ``/``, ``?``, ``#``, whitespace or quote, so a URL, a comma and an
      address (``https://api.example.com:443,admin@corp.example``) also read as userinfo: accepted, to stay closed for
      credentials. This is a rule about credentials, not about paths;
    * a root is found anywhere in a text, a URL, a query or after a placeholder included. A root of one folder (``/root``,
      ``/app``) is found only where it does not follow a word character, ``.``, ``~`` or ``-``, because it is also a
      segment of a relative path or a URL (``src/root/main.py``, ``https://pypi.org/simple/root/``). A root of two folders
      or more (``/home/jane``) is found wherever it appears, as the tail of a longer path or in a URL path included: the
      asymmetry is deliberate.

    The roots are read once, when the rule is built (a few path resolutions); matching a text reads no file. A decoded copy
    of a text is read only to find a root, and the replacement is made in the text as written. Applying the rule twice gives
    the same text. Dict keys are rewritten like values; two keys that ship as one text are numbered (``<key> #2``), so no
    value is lost. Nothing raises: a text the rule cannot read ships as the stand-in. :meth:`HostReadScope.scrub` stays
    for what the agent reads and for the opt-in anonymisation.
    """
    return _Shippable(project).walk(obj)


def shippable_roots(roots: Sequence[str], *, project: str | Path | None) -> list[str]:
    """The source roots a scan names, as a record the SDK writes about a run may leave this machine.

    The roots are read as whole values, not as free text, with the project rule of :func:`shippable_text`: a root
    under ``project`` is written relative to it (``.`` for the project itself), and every other absolute, ``~``-led,
    ``$HOME``-led, Windows or ``file:`` root is :data:`~plurai.orchestrator.local_paths.LOCAL_PATH_MARKER`. With no
    project, or a project that is the home folder or holds it, every such root is the stand-in. A relative root is
    written as given unless a segment of it is the user name. Applying the rule twice gives the same list, so a roots
    list that is already shipped comes back unchanged.
    """
    shipper = _Shippable(project)

    def shipped(root: str) -> str:
        if root.lower().startswith('file:'):
            return LOCAL_PATH_MARKER
        if _windows_form(root):
            return shipper.windows(root)
        if _is_local_root(root):
            return shipper.stand_in(os.path.normpath(shipper.scope._expand(root)))  # noqa: SLF001
        return LOCAL_PATH_MARKER if shipper.names_the_user(root) else root

    return [shipped(root) for root in roots]


def _is_local_root(root: str) -> bool:
    return root.startswith(('/', '~', '$HOME', '${HOME}')) or bool(_windows_form(root))


#: Where a scan contract records the roots its citations were resolved against.
_CONTRACT_ROOTS_PATH = ('report', 'sections', 'fact_citations', 'roots')

#: Where a scan contract records the render surface it found (``{"verbs": [...], "source": <config path>}``).
_CONTRACT_RENDER_ACTIONS_PATH = ('render_actions',)

#: The members of the render surface that name the scanning machine: the config file the verbs were read from and the
#: reason a declaration could not be read. The verbs are the agent's own vocabulary (``/refund`` is a verb, not a path)
#: and the service compares them by name, so they and every other member ship as written.
_RENDER_ACTIONS_LOCAL_MEMBERS = ('source', 'unreadable')


def _contract_member(contract: Any, path: tuple[str, ...]) -> Any:
    node = contract
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return node


def contract_roots(contract: Any) -> Any:
    """The citation roots of ``contract``, a scan contract as a document, or ``None`` when it holds none at :data:`_CONTRACT_ROOTS_PATH`."""
    return _contract_member(contract, _CONTRACT_ROOTS_PATH)


def shippable_contract(contract: dict[str, Any], *, typed_from: Path | None = None) -> dict[str, Any]:
    """A copy of ``contract``, a scan contract as a document, with the parts that name the scanning machine shipped.

    The project is ``typed_from``, the folder the scan ran in, which is also where a relative root is read from.
    Without it (a contract read back later, where that folder is not known) there is no project and a relative root is
    kept as written. The citation roots go through :func:`shippable_roots`, and the render surface's config ``source``
    (a path) and the reason it could not be read (a text) through :func:`shippable_roots` and :func:`shippable_text`,
    both with that project. The verbs and every other member of the surface, and every other section, are the
    contract's own; only the dicts on the way to the rewritten members are copied.
    """
    roots = contract_roots(contract)
    shipped = dict(contract)
    if isinstance(roots, list) and all(isinstance(root, str) for root in roots):
        given = [_read_from(root, typed_from) for root in roots]
        shipped = _with_member(shipped, _CONTRACT_ROOTS_PATH, shippable_roots(given, project=typed_from))
    if isinstance(surface := shipped.get(_CONTRACT_RENDER_ACTIONS_PATH[0]), dict):
        shipper = _Shippable(typed_from)
        shipped[_CONTRACT_RENDER_ACTIONS_PATH[0]] = {
            **surface,
            **{k: _render_member(k, surface[k], typed_from, shipper) for k in _RENDER_ACTIONS_LOCAL_MEMBERS if k in surface},
        }
    return shipped


def _render_member(key: str, value: Any, project: Path | None, shipper: _Shippable) -> Any:
    """The render surface's ``source`` is a path, read as a whole value like a root; ``unreadable`` is a text."""
    if key == 'source' and isinstance(value, str):
        return shippable_roots([value], project=project)[0]
    return shipper.walk(value)


def _read_from(root: str, folder: Path | None) -> str:
    """``root`` as a path read from ``folder`` when it is relative; a local, Windows or ``file:`` root is kept as it is."""
    if folder is None or _is_local_root(root) or root.lower().startswith('file:'):
        return root
    return str(folder / root)


def _with_member(contract: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """A copy of ``contract`` with the member at ``path`` replaced by ``value``; only the dicts on the way to it are copied."""
    key, *rest = path
    return {**contract, key: _with_member(contract[key], tuple(rest), value) if rest else value}


def shippable_contract_roots(text: str) -> str:
    """``text``, a scan contract as the scan wrote it, with :func:`shippable_contract` applied to its bytes.

    Only the bytes of the citation roots and of the render surface change: the document is not re-serialised, so
    every other section is the file's own, a leading byte order mark included. A JSON value that holds neither is
    returned as it came. Raises :class:`ValueError` for a text that is not JSON: the rule cannot read it, so it must
    not be sent as it is.
    """
    bom = '\ufeff' if text.startswith('\ufeff') else ''
    body = text[len(bom) :]
    document = json.loads(body)  # ValueError (JSONDecodeError) for a text that is not JSON
    if not isinstance(document, dict):
        return text
    shipped = shippable_contract(document)
    spans: list[tuple[int, int, Any]] = []
    for path in (_CONTRACT_ROOTS_PATH, _CONTRACT_RENDER_ACTIONS_PATH):
        before, after = _contract_member(document, path), _contract_member(shipped, path)
        if before == after:
            continue
        span = (0, len(body))
        for key in path:
            found = _member_span(body, span[0], key)
            if found is None:
                raise ValueError(f'the scan contract member {".".join(path)} could not be located in the text')
            span = found
        spans.append((*span, after))
    for start, end, value in sorted(spans, reverse=True):
        body = body[:start] + json.dumps(value, ensure_ascii=False) + body[end:]
    return bom + body


def _member_span(text: str, start: int, key: str) -> tuple[int, int] | None:
    """The ``(start, end)`` of the value of ``key`` in the JSON object that opens at ``text[start]``; the last one when repeated."""
    decoder = json.JSONDecoder()

    def skip(at: int) -> int:
        while text[at] in ' \t\r\n':
            at += 1
        return at

    found: tuple[int, int] | None = None
    at = skip(text.index('{', start) + 1)
    while text[at] != '}':
        name, at = decoder.raw_decode(text, at)
        at = skip(skip(at) + 1)  # past the colon
        _, end = decoder.raw_decode(text, at)
        if name == key:
            found = (at, end)
        at = skip(end)
        if text[at] == ',':
            at = skip(at + 1)
    return found


def harness_project(simulate_path: str | Path) -> Path:
    """The root above the harness folder (``<project>/plurai_sim/simulate.py``): the project a text about a run writes its paths relative to."""
    return Path(os.path.normpath(Path(simulate_path).absolute())).parent.parent


def shipped_error[E: BaseException](make: Callable[[str], E], message: str, *, project: str | Path | None) -> E:
    """The exception ``make(message)`` builds, with ``message`` as :func:`shippable_text` writes it.

    The exception is what the failed-thread request, ``error.json`` and the transcript's failure metadata read, so
    they all hold the one shipped text. The operator's terminal reads :func:`local_text_of`, which keeps what
    ``make`` renders from ``message`` as written.
    """
    shipped_message = shippable_text(message, project=project)
    if shipped_message == message:
        return make(message)
    shipped = make(shipped_message)
    local = make(message)
    if str(shipped) != str(local):
        set_local_text(shipped, str(local) or repr(local))
    return shipped


def set_local_text(exc: BaseException, text: str) -> None:
    """Record on ``exc`` the text :func:`local_text_of` returns, for an exception whose own text is the shipped one."""
    exc.local_text = text  # type: ignore[attr-defined]


def shippable_error_record(record: dict[str, Any], *, project: str | Path | None = None) -> dict[str, Any]:
    """``record``, a scenario's ``error.json``, with every text through :func:`shippable_text` (the input is not changed).

    The re-drive's ``original_error`` ends with the ``cause`` when it did before: the backend tells a persistent runtime
    fault from a one-off by that suffix, and the two texts are rewritten apart, so the shipped cause is joined back.
    """
    shipped = shippable_text(record, project=project)
    redrive, cause = record.get('redrive'), record.get('cause')
    original = redrive.get('original_error') if isinstance(redrive, dict) else None
    if isinstance(cause, str) and isinstance(original, str) and original.endswith(cause):
        head = shippable_text(original.removesuffix(cause), project=project)
        shipped['redrive'] = {**shipped['redrive'], 'original_error': head + shipped['cause']}
    return shipped


def local_text_of(exc: BaseException) -> str:
    """What the operator's terminal prints for ``exc``: the text as written when :func:`shipped_error` rewrote it."""
    return getattr(exc, 'local_text', None) or str(exc) or repr(exc)


def _repo_entries(repo: Path, harness: str, *, root: str) -> list[Path]:
    """``repo``'s entries but its VCS dir and the harness folder (``harness``: its realpath; a folder
    above it is opened, not allowed whole): the committed rubric is never the agent's to read.

    An entry is the repo's only when it resolves inside ``root`` (the repo's realpath): a link out of
    it is left to the scope's realpath judgement, and a link to a folder holding the harness is
    neither opened nor allowed."""
    out = []
    for entry in sorted(repo.iterdir()) if repo.is_dir() else ():
        real = os.path.realpath(entry)
        if entry.name == '.git' or _under(real, harness) or not _under(real, root):
            continue
        if not _under(harness, real):
            out.append(entry)
        elif not entry.is_symlink():
            out += _repo_entries(entry, harness, root=root)
    return out


def _vcs_dirs(harness: str) -> list[Path]:
    """The git dirs that may hold ``harness`` (its realpath) committed: every ``.git`` beside one of
    its folders up to ``/`` (the repo's own, and an enclosing monorepo's), and for a ``.git`` file (a
    worktree or a submodule) the git dir it names, and that one's common dir."""
    out: list[Path] = []
    for folder in _ancestors(str(Path(harness).parent)):
        dot = Path(folder) / '.git'
        if not os.path.lexists(dot):
            continue
        out.append(dot)
        try:
            pointer = dot.read_text(encoding='utf-8').strip() if dot.is_file() else ''
        except (OSError, ValueError):
            continue
        if pointer.startswith('gitdir:'):
            gitdir = Path(folder) / pointer.removeprefix('gitdir:').strip()  # an absolute one replaces folder
            out.append(gitdir)
            try:
                common = (gitdir / 'commondir').read_text(encoding='utf-8').strip()
            except (OSError, ValueError):
                continue
            out.append(gitdir / common)
    return [Path(os.path.normpath(p)) for p in out if os.path.lexists(p)]


#: ``PLURAI_SANDBOX=strict`` selects the closed Bash sandbox (no network, no extra mach services, nothing
#: writable outside the scenario's own folders, no toolchain reads under the home). Any other value, or none:
#: the permissive profile, which keeps only what protects the RUN: the harness and the expected answers, other
#: scenarios, and the operator's credentials, none of which the agent's shell can read or reach.
SANDBOX_MODE_ENV = 'PLURAI_SANDBOX'
SANDBOX_PROFILE_STRICT = 'strict'
SANDBOX_PROFILE_PERMISSIVE = 'permissive'


def sandbox_profile() -> str:
    return SANDBOX_PROFILE_STRICT if os.environ.get(SANDBOX_MODE_ENV, '').strip().lower() == SANDBOX_PROFILE_STRICT else SANDBOX_PROFILE_PERMISSIVE


#: Toolchain-manager folders under the home the agent's shell may read (the interpreters and CLIs it runs
#: from PATH, the fonts programs load). Credential folders (``.ssh``, ``.aws``, ``.config``, ``.docker``,
#: ``.kube``, ``.gnupg``, ``.netrc``, ``.cargo/credentials*``, ``.claude*``, ``Library/Keychains``) are never listed.
_TOOLCHAIN_HOME_READ = (
    '.local/bin',
    '.local/share/uv',
    '.local/share/mise',
    '.pyenv',
    '.nvm',
    '.volta',
    '.fnm',
    '.asdf',
    '.cargo/bin',
    '.rustup',
    '.bun',
    '.deno',
    '.sdkman',
    '.rbenv',
    'Library/Fonts',
    'Library/pnpm',
)

#: Mach services the agent's shell may look up: every Apple service (the runtime of a macOS app: launch
#: services, fonts, notifications, disk arbitration, configd, ...).
_OPEN_MACH_LOOKUP = ('com.apple.*',)


def _toolchain_reads(home: Path, withheld: Sequence[Path]) -> list[Path]:
    """The :data:`_TOOLCHAIN_HOME_READ` folders that exist, but any that holds or lies under a ``withheld`` path."""
    held = _both_forms(list(withheld))
    out = []
    for sub in _TOOLCHAIN_HOME_READ:
        folder = home / sub
        if folder.is_dir() and not any(_under(f, h) or _under(h, f) for f in _both_forms([folder]) for h in held):
            out.append(folder)
    return out


#: ``harness`` left out: a harness generated for an older SDK. Refused by name, never defaulted.
_HARNESS_UNNAMED: Any = object()


def host_read_scope(*, run_dir: str | Path | None, repo: str | Path, harness: str | Path | None = _HARNESS_UNNAMED) -> HostReadScope:
    """The read scope of the agent under test for one scenario.

    ``deny``: the operator's home, the other users' homes (its parent, never ``/``), the removable
    volumes, the CLI's shared per-uid temp root and the per-user temp root when it is not the shared
    ``/tmp`` (each holds the operator's other sessions); and, by name, the ``harness`` folder and the
    git dirs that may hold it committed (:func:`_vcs_dirs`): a repo in ``/srv``, ``/workspace``, a
    container's ``/app`` or Linux's shared ``/tmp`` lies under none of the roots above, and leaving
    them out of ``allow`` alone denies nothing.
    ``allow``: ``run_dir`` (the scenario folder; ``None`` when the harness declared none), ``repo``
    (but its ``.git`` and the ``harness`` folder; whole when ``harness`` is ``None``), and the
    toolchain folders of the home that exist (:data:`_TOOLCHAIN_HOME_READ`, so a home-installed
    interpreter or node stays on the shell's PATH; none under ``PLURAI_SANDBOX=strict``), and the
    interpreter: ``sys.prefix``, ``sys.base_prefix``, its site-packages and what the
    distributions installed there import from — never ``sys.path`` as such, whose entries a launch
    folder, ``PYTHONPATH`` or a stray ``.pth`` can name. An import root that holds the harness (an
    editable install of the agent's project) is opened around it the way the repo is.
    """
    if harness is _HARNESS_UNNAMED:
        raise HarnessFault('this simulate.py was generated for an older plurai SDK — update the Plurai simulation plugin, then re-run /simulation-scan-agent')
    home = Path.home()
    deny = [home]
    if home.parent != Path(home.anchor):
        deny.append(home.parent)
    deny += [Path('/Volumes')] if sys.platform == 'darwin' else [Path('/media'), Path('/mnt')]
    if hasattr(os, 'getuid'):
        deny.append(Path('/tmp') / f'claude-{os.getuid()}')  # noqa: S108 — a root this names to DENY, not a temp file
        deny.append(_scenario_roots_parent())  # every scenario's own root: each allows back its own
    per_user_tmp = tempfile.gettempdir()  # a per-user temp root holds the operator's other scenarios and sessions
    if os.path.realpath(per_user_tmp) not in {os.path.realpath(p) for p in ('/tmp', '/var/tmp')}:  # noqa: S108 — the shared roots, never denied
        deny.append(Path(per_user_tmp))
    deny_forms = _both_forms(deny)
    roots, refused = _import_roots(home)
    imports = [p for p in roots if any(_under(f, d) for f in _both_forms([p]) for d in deny_forms)]
    if harness is None:
        repo_allow = [Path(repo)]
    else:
        held = os.path.realpath(harness)
        repo_path = Path(os.path.normpath(Path(repo).absolute()))
        repo_allow = _repo_entries(repo_path, held, root=os.path.realpath(repo_path))
        withheld = _both_forms([Path(held), *_vcs_dirs(held)])
        imports = [q for p in imports for q in (_repo_entries(p, held, root=os.path.realpath(p)) if _under(held, p) else [p])]
        imports = [q for q in imports if not any(_under(f, w) for f in _both_forms([q]) for w in withheld)]  # allow wins: never reopen them
        deny_forms = _both_forms([*deny, *withheld])
    toolchain = [] if sandbox_profile() == SANDBOX_PROFILE_STRICT else _toolchain_reads(home, [] if harness is None else [Path(held), *_vcs_dirs(held)])
    allow = [Path(p) for p in (run_dir,) if p is not None] + repo_allow + imports + toolchain
    return HostReadScope(home=Path(os.path.normpath(home)), deny=deny_forms, allow=_both_forms(allow), refused_imports=refused)


def _sandbox_available() -> bool:
    if sys.platform == 'darwin':
        return shutil.which('sandbox-exec') is not None
    return sys.platform.startswith('linux') and bool(shutil.which('bwrap') and shutil.which('socat'))


#: The placeholder a plugin's MCP config names its own folder by; the CLI fills it only for a server it loads from the plugin.
_PLUGIN_ROOT = '${CLAUDE_PLUGIN_ROOT}'


def _plugin_mcp_servers(options: Any) -> dict[str, Any]:
    """The MCP servers the agent's own ``plugins`` declare, keyed as the CLI keys a plugin's server
    (``plugin:<plugin>:<server>``, so their tools keep their names). Strict MCP config drops them from
    the plugin; ``confine`` passes them on as declared servers. One it cannot pass on is refused."""
    found: dict[str, Any] = {}
    for plugin in getattr(options, 'plugins', None) or ():
        if not (isinstance(plugin, dict) and plugin.get('path')):
            raise HarnessFault(
                f"the agent declares a plugin plurai cannot read ({plugin!r}), so its MCP servers would not load — declare them in the agent's mcp_servers"
            )
        root = Path(str(plugin['path']))
        try:
            manifest, declared = (
                json.loads(f.read_text(encoding='utf-8')) if f.is_file() else {} for f in (root / '.claude-plugin' / 'plugin.json', root / '.mcp.json')
            )
        except (OSError, ValueError) as error:
            raise HarnessFault(
                f"the agent's plugin {root} has an unreadable plugin.json or .mcp.json ({error}), so its MCP servers would not load — fix that file"
            ) from error
        name, inline = str(manifest.get('name') or root.name), manifest.get('mcpServers', {})
        if not isinstance(inline, dict) or not isinstance(declared, dict):
            raise HarnessFault(
                f"the agent's plugin {name} declares its MCP servers in a form plurai does not read, so they would not load — declare them in its .mcp.json"
            )
        for server, spec in {**declared.get('mcpServers', declared), **inline}.items():
            carried = json.loads(json.dumps(spec).replace(json.dumps(_PLUGIN_ROOT)[1:-1], json.dumps(str(root))[1:-1]))
            if '${CLAUDE_PLUGIN_' in json.dumps(carried):
                raise HarnessFault(
                    f"the agent's plugin MCP server {name}:{server} names a placeholder only the CLI fills, so it would not load — "
                    "declare it in the agent's mcp_servers"
                )
            found[f'plugin:{name}:{server}'] = carried
    if found and not isinstance(getattr(options, 'mcp_servers', None), dict):
        raise HarnessFault(
            f"the agent's plugin MCP servers {', '.join(sorted(found))} would not load: its mcp_servers is a config file plurai does not "
            'edit — declare them there'
        )
    return found


def _any_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _empty(value: Any) -> bool:
    return value in ([], {})


def _only(expected: bool) -> Callable[[Any], bool]:
    return lambda value: value is expected


def _deny_entries(value: Any) -> bool:
    """Credential entries that only withhold (``mode: deny``); a ``mask`` entry injects at the proxy."""
    return isinstance(value, list) and all(isinstance(v, dict) and v.get('mode') == 'deny' for v in value)


#: The ``sandbox`` keys a settings source may carry, each with the values that can only tighten the
#: sandbox (or that :func:`confine` sets itself). A key outside this table — a future CLI key
#: included — or a value outside its rule (``allowUnixSockets: [docker.sock]``, ``allowAppleEvents:
#: true``, ``ignoreViolations``, ``failIfUnavailable: false``, …) is refused, never merged.
#: ``excludedCommands`` is a list here; which names it may hold is the grant's (:func:`_sandbox_findings`).
_SANDBOX_TIGHTENING: Mapping[str, Any] = {
    'enabled': _any_bool,
    'allowUnsandboxedCommands': _any_bool,
    'autoAllowBashIfSandboxed': _any_bool,
    'failIfUnavailable': _only(True),
    'excludedCommands': _strings,
    'filesystem': {'denyRead': _strings, 'denyWrite': _strings, 'allowRead': _empty, 'allowWrite': _empty, 'disabled': _only(False)},
    'credentials': {'files': _deny_entries, 'envVars': _deny_entries},
    'network': {
        'strictAllowlist': _only(True),
        'deniedDomains': _strings,
        'allowManagedDomainsOnly': _any_bool,
        'allowedDomains': _empty,
        'allowUnixSockets': _empty,
        'allowAllUnixSockets': _only(False),
        'allowLocalBinding': _only(False),
        'allowMachLookup': _empty,
    },
    'ignoreViolations': {'file': _empty, 'network': _empty},
    'enableWeakerNestedSandbox': _only(False),
    'enableWeakerNetworkIsolation': _only(False),
    'allowAppleEvents': _only(False),
}


def _anything(value: Any) -> bool:  # noqa: ARG001 — the rule ignores the value by design
    """Rule for a sandbox key the agent's own settings may set to any value.

    Used only by the permissive profile below (``network`` and ``ignoreViolations.network``): the agent
    under test may open its own network access, since that cannot reach the harness or the operator's
    credentials. ``PLURAI_SANDBOX=strict`` uses :data:`_SANDBOX_TIGHTENING` instead, where every value
    must pass a tightening-only rule (``_empty``, ``_only(...)``).
    """
    return True


#: The permissive profile's rules (the default; :data:`_SANDBOX_TIGHTENING` is ``PLURAI_SANDBOX=strict``):
#: what the agent's own settings may open is everything that leaves the RUN intact — network, mach and
#: socket access, Apple events, programs it runs outside the sandbox. Still refused: what would let the
#: shell read the harness or the operator's credentials, or write outside its own folders (``filesystem``,
#: ``credentials``, a file ``ignoreViolations``, a sandbox that may fail open, a weaker nested sandbox).
_SANDBOX_INTEGRITY: Mapping[str, Any] = {
    **_SANDBOX_TIGHTENING,
    'network': dict.fromkeys(cast(Mapping[str, Any], _SANDBOX_TIGHTENING['network']), _anything),
    'ignoreViolations': {'file': _empty, 'network': _anything},
    'enableWeakerNetworkIsolation': _any_bool,
    'allowAppleEvents': _any_bool,
}


def _sandbox_rules() -> Mapping[str, Any]:
    return _SANDBOX_TIGHTENING if sandbox_profile() == SANDBOX_PROFILE_STRICT else _SANDBOX_INTEGRITY


def _loosening_keys(block: Any, rules: Mapping[str, Any], path: str) -> list[str]:
    """The key paths in ``block`` that :data:`_SANDBOX_TIGHTENING` does not admit with their value."""
    if not isinstance(block, dict):
        return [path]
    found = []
    for key, value in block.items():
        rule = rules.get(key)
        where = f'{path}.{key}'
        if isinstance(rule, Mapping):
            found += _loosening_keys(value, rule, where)
        elif rule is None or not rule(value):
            found.append(where)
    return found


def _setting_sources(options: Any) -> tuple[str, ...]:
    """The settings sources the agent's CLI loads: the ones ``setting_sources`` names, none named: every source."""
    return ('user', 'project', 'local') if options.setting_sources is None else tuple(options.setting_sources)


def _read_settings(where: str, text: str | None, path: Path | None, *, unread: str) -> dict[str, Any] | None:
    """One settings document (``text``, else the file at ``path``; a missing file is ``None``). One that
    exists and cannot be read is a :class:`HarnessFault` naming ``where`` and what that leaves ``unread``."""
    if path is not None and not path.is_file():
        return None
    try:
        settings = json.loads(text if text is not None else cast(Path, path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise HarnessFault(f'{where} cannot be read ({type(error).__name__}), so {unread} — fix it, then re-run') from error
    return settings if isinstance(settings, dict) else None


def _operator_user_settings(options: Any) -> dict[str, Any]:
    """The operator's own user settings, when the agent's CLI loads that tier (else ``{}``). The CLI runs
    with a config dir of the scenario's own (:func:`_cli_config_env`), where that file is not, so
    :func:`confine` hands its ``env`` block to the CLI itself and carries its sign-in keys over
    (:func:`_carry_operator_sign_in`): a sign-in route kept only there still signs it in."""
    if 'user' not in _setting_sources(options):
        return {}
    where = "the operator's user settings (settings.json in the Claude config folder)"
    return _read_settings(where, None, _operator_cli_config() / 'settings.json', unread='the sign-in route it names is unknown') or {}


def _settings_env(settings: Mapping[str, Any]) -> dict[str, str]:
    """A settings document's ``env`` block, its string values."""
    env = settings.get('env')
    return {str(name): value for name, value in env.items() if isinstance(value, str)} if isinstance(env, dict) else {}


def _settings_documents(options: Any, run_dir: Path, config: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every settings document the agent's CLI will load beside the sandbox ``confine`` applies: the
    agent's own ``settings`` (the flag tier), and the settings files of the sources ``setting_sources``
    names (none named: every source), read from where the CLI reads them for this scenario — the
    Claude config folder ``config`` it runs with, and the scenario folder's copied ``.claude/``."""
    flag = (options.settings or '').strip() if isinstance(options.settings, str) else ''
    documents: list[tuple[str, str | None, Path | None]] = []
    if flag:
        inline = flag.startswith('{') and flag.endswith('}')
        documents.append(("the agent's own settings", flag if inline else None, None if inline else Path(flag)))
    files = {
        'user': ('the user settings (settings.json in the Claude config folder)', config / 'settings.json'),
        'project': ('the project settings (.claude/settings.json)', run_dir / '.claude' / 'settings.json'),
        'local': ('the local settings (.claude/settings.local.json)', run_dir / '.claude' / 'settings.local.json'),
    }
    documents += [(files[source][0], None, files[source][1]) for source in _setting_sources(options) if source in files]
    unread = 'its sandbox settings cannot be checked'
    read = [(where, _read_settings(f"{where} the agent's CLI loads", text, path, unread=unread)) for where, text, path in documents]
    return [(where, settings) for where, settings in read if settings is not None]


#: The ``extra_args`` flags an agent may pass its CLI: output and debugging only. Any other flag may
#: replace the settings ``confine`` applies (the CLI keeps the last ``--settings``) or move where the
#: CLI reads them (``--setting-sources``, ``--project-config-root``), so it is refused.
_EXTRA_ARGS_ADMITTED = frozenset({'replay-user-messages', 'debug-to-stderr', 'debug', 'verbose'})

#: A permission rule that adds a host to the Bash sandbox's allowed domains.
_DOMAIN_RULE = re.compile(r'WebFetch\(domain:')


def _channel_findings(options: Any, documents: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """What reaches the CLI around the ``sandbox`` block and would undo it: an ``extra_args`` flag
    outside :data:`_EXTRA_ARGS_ADMITTED`, a ``CLAUDE_CONFIG_DIR`` of the agent's own (the settings it
    names are not the ones checked), and a ``WebFetch(domain:…)`` allow rule, which the CLI adds to
    the sandbox's allowed domains."""
    findings = []
    if flags := sorted(set(options.extra_args or {}) - _EXTRA_ARGS_ADMITTED):
        findings.append(
            f"the agent's own extra_args pass {', '.join('--' + f for f in flags)}, which can replace or move "
            "the settings plurai confines the agent's CLI with — remove it"
        )
    if 'CLAUDE_CONFIG_DIR' in (options.env or {}):
        findings.append("the agent's own env sets CLAUDE_CONFIG_DIR, which moves the settings plurai confines the agent's CLI with — remove it")
    rules = [("the agent's own allowed_tools", list(options.allowed_tools or ()))]
    for where, settings in documents:
        permissions = settings.get('permissions')
        allow = permissions.get('allow') if isinstance(permissions, dict) else None
        rules.append((where, allow if isinstance(allow, list) else []))
    for where, allow in rules:
        if domains := [str(rule) for rule in allow if isinstance(rule, str) and _DOMAIN_RULE.match(rule)]:
            findings.append(f'{where} allows {", ".join(domains)}, which opens those hosts to the sandboxed Bash — remove it')
    return findings


def _sandbox_findings(where: str, block: Any, grant: UnsandboxedGrant) -> list[str]:
    """What one ``sandbox`` block would loosen beyond the profile's rules (:func:`_sandbox_rules`), and,
    under ``PLURAI_SANDBOX=strict``, programs it excludes from the sandbox that the grant does not
    (the permissive profile runs the agent's own exclusions as its settings say)."""
    findings = []
    if loosening := _loosening_keys(block, _sandbox_rules(), 'sandbox'):
        findings.append(f'{where} sets {", ".join(loosening)}, which loosens the Bash sandbox — remove it')
    excluded = block.get('excludedCommands') if isinstance(block, dict) else None
    if sandbox_profile() == SANDBOX_PROFILE_STRICT and _strings(excluded) and (ungranted := sorted(set(excluded) - set(grant.names))):
        findings.append(
            f'{where} excludes {", ".join(ungranted)} from the Bash sandbox, which {BENCH_MANIFEST_FILENAME} '
            f'{UNSANDBOXED_GRANT_KEY} does not grant — grant it there (a declared kind: executable), or remove it'
        )
    return findings


def _declared_native_tools(workdir: Path) -> tuple[str, ...]:
    """``agent.native_tools`` from the config beside the harness (the scan records the agent's live
    built-ins there). A config that exists and cannot be read is a :class:`HarnessFault`."""
    names: list[str] = []
    with _config_readable('which built-ins the agent has'):
        for document in _config_documents(workdir):
            agent = document.get('agent')
            declared = agent.get('native_tools') if isinstance(agent, dict) else None
            names += [str(n) for n in declared or () if n] if isinstance(declared, list) else []
    return tuple(dict.fromkeys(names))


def _builtin_gate(allowed: frozenset[str], *, trail: list[dict[str, Any]] | None) -> Callable[[dict[str, Any], str | None, Any], Awaitable[dict[str, Any]]]:
    """A ``PreToolUse`` hook on every tool: a built-in not in ``allowed`` (the read-only list, the
    grants, and the ones confined in place) is denied, so one the CLI adds later, or one the scan
    never recorded, cannot run ungranted. MCP tools pass: the agent declares those servers."""

    async def hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:  # noqa: ARG001
        name = str(input_data.get('tool_name') or '')
        if not name or name.startswith(_MCP_PREFIX) or name in allowed:
            return {}
        if trail is not None:  # a denied call fires no PostToolUse, so the trail is told here
            call_id = input_data.get('tool_use_id') or tool_use_id
            trail.append(
                {
                    'tool': name,
                    'args': input_data.get('tool_input') or {},
                    'call_id': call_id,
                    'result': _UNGRANTED,
                    'status': TRAIL_STATUS_DENIED,
                    TRAIL_REFUSAL_KEY: {'by': 'confinement', 'reason': _UNGRANTED},
                }
            )
        return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny', 'permissionDecisionReason': _UNGRANTED}}

    return hook


def confine(options: Any, *, run_dir: Path, repo: Path, workdir: Path, trail: list) -> dict[str, Any]:
    """Confine one scenario's ``claude_agent_sdk`` options before connect, and return the record the
    harness reports on ``agent_state[HOST_EFFECTS_KEY]``. MCP servers the agent does not declare in
    ``mcp_servers`` or in its own ``plugins`` never load.

    The record states per read surface what was applied (``read_surfaces``: ``confined`` or
    ``not_confined``) and what was seen: ``reads_observed`` counts, per :data:`READ_LANDINGS`, the
    paths of every built-in read-channel call as it happens (the record is updated in place, so the
    turn that reports it last carries them all), with ``calls`` the calls judged. A run reads as
    scoped only when every surface is confined and every call was judged with nothing undecided
    (``run_simulation``): the scope was enforced and nothing observed contradicts it — not that
    every read was seen, since a path a command builds at run time is held by the sandbox alone.

    Every tool call the CLI runs or denies is appended to ``trail`` (the recipe's entry shape, status
    ``ok``, ``error`` or ``denied``; an MCP tool by its declared key), and the record says so
    (``trail``: :data:`TRAIL_RECORDED`): the harness returns the turn's slice as
    ``agent_state['tool_trail']`` and installs no hooks of its own. A call the CLI answers before any
    hook fires is added at turn end from the CLI's session transcript (:func:`reconcile_trail`).

    Deny-by-default: a built-in runs ungranted only when it is on :data:`READ_ONLY_BUILTINS`. The
    ungranted ones of :data:`HOST_EFFECT_TOOLS` are confined in place or switched off by name, so are
    the agent's other recorded built-ins (``agent.native_tools``), and a hook denies any other
    built-in the CLI offers."""
    from claude_agent_sdk import HookMatcher  # noqa: PLC0415 — the agent's own dependency, never plurai's

    if not hasattr(options, 'strict_mcp_config'):
        raise HarnessFault("the agent's claude_agent_sdk has no strict_mcp_config — upgrade claude-agent-sdk")
    plugin_servers = _plugin_mcp_servers(options)
    # what the harness's own options switch off, read before confine edits them
    offered = getattr(options, 'tools', None)
    harness_off = getattr(options, 'disallowed_tools', None) or []  # confine adds to a new list, never this one
    grant = unsandboxed_grant(workdir)  # a grant it refuses leaves the options untouched
    declared = declared_host_requirements(workdir) or ()
    # the CLI applies its user settings' env over the environment it starts with
    user_settings = _operator_user_settings(options)
    operator_env = _settings_env(user_settings)
    cli_env = {**os.environ, **(options.env or {}), **operator_env}
    route = _cli_auth_route(cli_env, user_settings)
    cli_config, config_env = _cli_config_env(route, cli_env)
    # a sandbox setting that loosens the sandbox, a program excluded from it without a grant, or a
    # channel that would replace or widen it stops the run before it starts: from the agent's own
    # options and from every settings document its CLI loads
    documents = _settings_documents(options, run_dir, Path(config_env['CLAUDE_CONFIG_DIR']))
    blocks = [("the agent's own sandbox options", options.sandbox)] if options.sandbox is not None else []
    blocks += [(where, settings['sandbox']) for where, settings in documents if 'sandbox' in settings]
    if findings := [*_channel_findings(options, documents), *(f for where, block in blocks for f in _sandbox_findings(where, block, grant))]:
        raise HarnessFault('; '.join(findings))
    _carry_operator_sign_in(user_settings, cli_env, Path(config_env['CLAUDE_CONFIG_DIR']), route)
    options.strict_mcp_config = True
    options.mcp_servers = {**plugin_servers, **options.mcp_servers} if plugin_servers else options.mcp_servers
    read_only = frozenset(READ_ONLY_BUILTINS['claude_agent_sdk'])
    natives = [n for n in _declared_native_tools(workdir) if not n.startswith(_MCP_PREFIX) and n not in read_only and n not in HOST_EFFECT_TOOLS]
    candidates = [*HOST_EFFECT_TOOLS, *sorted(set(natives))]
    grants = host_effects_granted(workdir)
    granted = grants & set(candidates)
    options.cwd = str(run_dir)
    # what the agent reads becomes trail, and the trail leaves the machine: scoped whatever was granted
    scope = host_read_scope(run_dir=run_dir, repo=repo, harness=workdir)
    # the shell runs in the sandbox unless Bash is granted live or no sandbox can start (then there is no Bash)
    sandboxed = 'Bash' not in granted and _sandbox_available()
    options.env = {**(options.env or {}), **operator_env, **scope.agent_shell_env(cli_env=cli_env, sandbox_grant=grant.names if sandboxed else None)}
    shell_home = cast(Path, scope._shell_home)  # set by agent_shell_env above
    options.env = {**options.env, **config_env}
    # the CLI process runs npm itself, outside the Bash sandbox and with the operator's HOME: its cache
    # and logs go to the shell's own home, where the agent's sandboxed npm keeps them too
    options.env = {**options.env, 'npm_config_cache': str(shell_home / '.npm')}
    reads_observed = {**dict.fromkeys(READ_LANDINGS, 0), 'calls': 0}
    observe, read_refused = _read_observer(scope, str(shell_home), reads_observed)
    observer: Any = observe  # claude_agent_sdk HookCallbacks
    hooks: dict[str, list[Any]] = dict(options.hooks or {})
    for event, matchers in cli_hooks(scope, trail=trail, on_read_denied=read_refused).items():
        hooks[event] = [*hooks.get(event, []), *matchers]  # the agent's own first
    tool_keys = declared_tool_keys(workdir)
    for event, status in (('PostToolUse', 'ok'), ('PostToolUseFailure', 'error')):
        recorder: Any = _trail_recorder(trail, status, tool_keys)  # claude_agent_sdk HookCallbacks
        hooks[event] = [*hooks.get(event, []), HookMatcher(hooks=[observer, recorder])]
    options.hooks = hooks
    denied = {}
    confined_in_place: set[str] = set()  # ungranted, yet callable under the confinement applied below
    sandboxed_executables: list[str] = []  # set only when Bash runs sandboxed below
    for tool in candidates:
        if tool in granted:
            continue
        if tool in _WRITE_TOOLS and options.permission_mode != 'bypassPermissions':
            options.allowed_tools = [t for t in options.allowed_tools if t != tool]
            options.permission_mode = 'acceptEdits'
            denied[tool] = 'auto-accepted only inside the scenario cwd'  # where they landed: observed_writes
            confined_in_place.add(tool)
        elif tool == 'Bash' and _sandbox_available():
            # the CLI's temp root, short: past ~30 bytes it falls back to the shared /tmp/claude-<uid>
            tmp_root = _scenario_root('sc-')
            scope._allow_back(tmp_root)
            options.env = {**(options.env or {}), 'CLAUDE_CODE_TMPDIR': str(tmp_root)}
            # the agent's own Edit/Write may write there too: acceptEdits auto-accepts inside the working directories
            # only, and a file the model unpacks into the temp folder it was handed would otherwise be refused
            options.add_dirs = [*(options.add_dirs or []), str(tmp_root)]
            # the CLI adds default write paths of its own (some under the home): every deny root that holds
            # none of this scenario's folders is write-denied too
            filesystem: dict[str, Any] = scope.sandbox_filesystem()
            own_forms = _both_forms([run_dir, shell_home, tmp_root])
            filesystem['denyWrite'] = [str(d) for d in scope.deny if not any(_under(c, d) for c in own_forms)]
            filesystem['allowWrite'] = [str(p) for p in own_forms]  # the scenario cwd, its shell home and its own temp folder
            if sandbox_profile() == SANDBOX_PROFILE_PERMISSIVE:
                filesystem['allowWrite'] += _SOFFICE_IPC_WRITES
            # pinned: with no value here the CLI takes its filesystem layer's switch from the operator's
            # own settings, and a relaxed layer drops every denyRead at wrap time
            filesystem['disabled'] = False
            # the agent's own block passed the profile's rules above (_sandbox_rules): under strict what is
            # kept of it only tightens; under permissive it may open what leaves the run intact
            own = options.sandbox or {}
            kept = {k: v for k, v in own.items() if k not in ('enabled', 'allowUnsandboxedCommands', 'filesystem', 'excludedCommands')}
            permissive = sandbox_profile() == SANDBOX_PROFILE_PERMISSIVE
            if permissive:
                # the agent's shell behaves as in the operator's own terminal: macOS services (a desktop
                # program like soffice looks up coreservicesd/configd/diskarbitrationd at launch), local
                # sockets and ports, Apple events and the network. What protects the run stays: no read of
                # the harness, other scenarios or the operator's credentials, no write outside the
                # scenario's own folders.
                # the shared /tmp stays closed to writes: it holds every scenario's own folders
                # (/tmp/plurai-<uid>), and the CLI's denyWrite beats a deeper allowWrite, so it cannot be
                # opened around them. The shell's TMPDIR is its own temp folder, which is writable.
                own_network = kept.get('network', {})
                network = {
                    **own_network,
                    'allowMachLookup': list(dict.fromkeys([*_OPEN_MACH_LOOKUP, *(own_network.get('allowMachLookup') or [])])),
                    'allowAllUnixSockets': True,
                    'allowLocalBinding': True,
                }
                # every host: the CLI's proxy denies a host its allowedDomains does not list, whatever the permission mode
                network['allowedDomains'] = list(dict.fromkeys(['*', *(own_network.get('allowedDomains') or [])]))
                network.pop('strictAllowlist', None)
                kept['allowAppleEvents'] = True
                confined_to = (
                    'sandboxed (permissive): network open; writes limited to the scenario cwd, its shell home and its own temp '
                    'folder; no reads of the harness, other scenarios or your credentials'
                )
            else:
                # a host outside allowedDomains is denied without asking; left unset, bypassPermissions answers yes
                network = {**kept.get('network', {}), 'strictAllowlist': True}
                confined_to = 'sandboxed: network denied, writes limited to the scenario cwd, its shell home and its own temp folder'
            options.sandbox = {
                **kept,
                'enabled': True,
                'allowUnsandboxedCommands': False,
                # a sandbox that cannot start stops the CLI; left unset, the CLI runs every command unsandboxed
                'failIfUnavailable': True,
                'network': network,
                'filesystem': filesystem,
            }
            # permissive: the agent's own exclusions, from its options AND from every settings document its CLI
            # loads (the CLI applies those either way), named here so the record marks each not_confined
            own_excluded = [str(n) for _where, block in blocks if isinstance(block, dict) for n in (block.get('excludedCommands') or []) if n]
            excluded = list(dict.fromkeys([*grant.names, *(own_excluded if permissive else [])]))
            if excluded:
                options.sandbox['excludedCommands'] = excluded
            if grant.names:  # the operator's grant: the documented path out of the sandbox
                denied[tool] = (
                    f'{confined_to} — except a command made only of {", ".join(grant.names)} calls, which runs outside the '
                    'sandbox by your grant, with full read, write and network access to this machine'
                )
            elif grant.state == 'unreadable':
                denied[tool] = f'{confined_to} — {BENCH_MANIFEST_FILENAME} {UNSANDBOXED_GRANT_KEY} could not be read, so no program runs outside it'
            else:
                denied[tool] = confined_to
            sandboxed_executables = sorted({r.name for r in declared if r.kind == 'executable'} - set(grant.names))
            confined_in_place.add(tool)
        else:
            options.disallowed_tools = [*options.disallowed_tools, tool]
            denied[tool] = 'disallowed: no sandbox on this host' if tool == 'Bash' else 'disallowed'
    allowed = frozenset(read_only | grants | confined_in_place)
    options.hooks = {**options.hooks, 'PreToolUse': [*options.hooks.get('PreToolUse', []), _hook_matcher(_builtin_gate(allowed, trail=trail))]}
    # registered after this scenario's roots, so it runs before their removal (``atexit`` runs last-in first)
    atexit.register(_end_cli_processes)
    return {
        'granted': sorted(granted),
        'denied': denied,
        # the tools this session does not have, by who switched each off: a call to one is a refusal, not an error
        SWITCHED_OFF_KEY: switched_off_tools(
            [*read_only, *candidates],
            offered=offered if isinstance(offered, list | tuple) else None,
            harness_off=harness_off,
            confine_off=options.disallowed_tools or (),  # the harness's own first: switched_off_tools keeps their owner
        ),
        'cwd': str(run_dir),
        'shell_home': str(shell_home),
        'auth': route,
        'cli_config': cli_config,
        'read_surfaces': _read_surfaces(options, granted),
        'reads_observed': reads_observed,  # the observer's own dict: it fills as the calls happen
        'trail': TRAIL_RECORDED,  # every call lands in ``trail``: the harness returns its turn's slice as ``tool_trail``
        'mcp': 'declared_only',
        # moot when Bash itself is granted live, and recorded all the same
        'unsandboxed_grant': {'state': grant.state, 'names': list(grant.names)},
        # declared programs that ran inside the Bash sandbox with no grant: a declaration joined with the grant, not a detection
        'sandboxed_executables': sandboxed_executables,
    }


#: ``read_surfaces`` values: the scope is applied to the surface, or nothing plurai applied holds it.
READ_CONFINED = 'confined'
READ_NOT_CONFINED = 'not_confined'


def _read_surfaces(options: Any, granted: frozenset[str]) -> dict[str, str]:
    """Per surface that can read the host for the agent, whether :func:`confine` scoped it.

    ``read_tools`` (the gate) always; ``bash`` when it runs — sandboxed, or granted live and so not
    confined — and ``bash:<program>`` for each program the sandbox leaves out. The agent's MCP
    servers (``mcp:<name>``) run as processes of their own, or in the harness's, outside the scope:
    each is named ``not_confined``, as is a config-file ``mcp_servers`` whose servers plurai cannot see.
    """
    surfaces = {'read_tools': READ_CONFINED}
    if 'Bash' in granted:
        surfaces['bash'] = READ_NOT_CONFINED
    elif 'Bash' not in (options.disallowed_tools or ()):
        surfaces['bash'] = READ_CONFINED
        surfaces.update({f'bash:{name}': READ_NOT_CONFINED for name in (options.sandbox or {}).get('excludedCommands') or ()})
    servers = options.mcp_servers
    if isinstance(servers, dict):
        surfaces.update({f'mcp:{name}': READ_NOT_CONFINED for name in servers})
    elif servers:
        surfaces['mcp'] = READ_NOT_CONFINED
    return dict(sorted(surfaces.items()))
