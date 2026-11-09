"""Deterministic projection of ``code_surface.yaml`` for the ``code`` action-channel.

The ``code`` skill action-channel runs a Claude-skill the way production does — the
skill writes and runs real vendor code — while a generic runtime in the SDK
intercepts the production call onto the frozen ``/mock/execute`` wire. That runtime is
fully data-driven: the only per-customer artifact it reads is ``code_surface.yaml``.
There are two sub-modes, each with its own authored surface:

* **SDK sub-mode** (``modules``) — the skill calls a vendor SDK method
  (``import resend; resend.Emails.send(...)``); a Python ``meta_path`` / Node
  ``--require`` hook shadows the named module.
* **CLI sub-mode** (``cli_http``) — the skill runs a vendored ``scripts/cli.*`` action
  verb; a ``fetch``-shadow ``--import`` preload routes the subcommand's HTTP egress onto
  the mock wire and fails closed on anything unmapped.

**The SDK owns this projection — the backend does not emit it:** the SDK both PROJECTS the
surface from ``plurai.config.yaml`` *and* CONSUMES it (the import-hook + ``fetch_shadow``).
The plugin is the source of knowledge: the LLM
(having read the skill's own reference docs) authors the semantic resolution — the
per-descriptor ``when:`` key-presence discriminator for a shared SDK method, and the
per-subcommand ``original_command`` + ``http_mapping`` for the CLI verb. This module's
job is to (a) STRUCTURALLY validate those authored fields, (b) run the generation-time
FIELD-CORRECTNESS check that the runtime loader cannot (it needs the config ``args``,
which only exist here), and (c) project the authored fields verbatim into
``code_surface.yaml`` — stamping only the *compiled* fields (resolved positional
indices, return-envelope plumbing, the CLI runtime knobs) reserved for the projector.
A vendor method backing >1 tool MUST carry an authored ``when:`` or generation fails
closed.

The core is a **pure deterministic projection** of the parsed ``plurai.config.yaml``
descriptors — NOT an LLM step and NOT a parse of the generated ``mocks.py``. Validation
is fail-closed: any authored shape that cannot be projected to an unambiguous surface
raises :class:`CodeSurfaceGenerationError` rather than emitting a silently-wrong map.

The public surface is dict-in / dict-out so it has no dependency on the backend's pydantic
models: :func:`build_code_surface` takes the loaded ``plurai.config.yaml`` as a plain
dict, a thin in-module parser converts it to lightweight :class:`_ToolDescriptor`
objects mirroring only the fields the projection reads, and :func:`emit_code_surface`
reads/writes the YAML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from plurai.render_actions import RenderActionsConfigError, RenderSurface, parse_render_actions

logger = logging.getLogger(__name__)

CODE_SURFACE_FILENAME = 'code_surface.yaml'

# Schema version emitted per sub-mode, in lock-step with the SDK's dual-loader
# (``_SUPPORTED_SURFACE_VERSIONS``). SDK ``modules``-only surfaces are v1; the CLI
# ``cli_http`` surface is a v2 feature.
CODE_SURFACE_VERSION_SDK = 1
CODE_SURFACE_VERSION_CLI = 2

# Per-module SDK language → the native return envelope the import-hook shapes the mock
# response into. Python (the v1 surface) returns the mock's plain dict unchanged; Node
# wraps it as ``{data, error}``. Recorded per-method in the surface (not globally) so a
# future mixed-language module stays expressible; for now every method on a module inherits
# its module's language.
_RETURN_ENVELOPE_BY_LANGUAGE = {
    'python': 'python_dict',
    'node': 'node_data_error',
}

# Default SDK language when ``agent.code_languages`` is absent from the config. An absent
# or empty list defaults to python — a config that never names a language is a Python
# bench; a declared list is honoured verbatim (see :func:`_resolve_language`).
_DEFAULT_LANGUAGE = 'python'

# The CLI sub-mode's fixed channel knobs — COMPILED fields (not LLM-authored). The plugin
# authors only the semantic per-subcommand ``original_command`` + ``http_mapping``; the
# projector stamps the runtime plumbing reserved for it, matching the schema the
# SDK's ``fetch_shadow`` loader (``_parse_cli_http``) reads.
_CLI_RUNTIME = 'fetch_shadow'
_CLI_RETURN_ENVELOPE = 'node_cli_json'
_CLI_UNMAPPED_EGRESS = 'fail_closed'
# Path the harness's ``--import`` preload is shipped at by the SDK. The plugin's harness
# template resolves the absolute path itself; the surface records this contract value (the loader
# only structurally requires a non-empty string).
_CLI_PRELOAD = 'scripts/fetch_shadow.mjs'

# The two authored ``when:`` polarities. Key-presence ONLY — a single distinguishing key
# the snippet either includes (``present``) or omits (``absent``); no value comparisons or
# boolean expressions (the runtime resolver is one-sided).
_WHEN_PRESENT = 'present'
_WHEN_ABSENT = 'absent'


class CodeSurfaceGenerationError(Exception):
    """An authored descriptor set can't be projected to an unambiguous ``code_surface.yaml``.

    Raised fail-closed (no silent fallback) on a malformed/incoherent authored surface: a
    shared SDK method missing a ``when:`` on a sharing descriptor, a ``when:`` whose field
    is not present-in-one/absent-in-other across the tools' declared ``args``, a
    half-populated vendor mapping, an ambiguous channel signal (both SDK and CLI), or a
    malformed CLI ``http_mapping``.
    """


@dataclass
class _ToolDescriptor:
    """The slice of the Plurai backend's tool descriptor the surface projection reads.

    A lightweight in-module mirror so this module has no dependency on the backend's pydantic
    models. Only the fields :func:`build_code_surface` and its helpers consume are
    carried; everything else in the config descriptor is ignored.
    """

    name: str
    mode: str = 'mock'
    original_module: str | None = None
    original_function: str | None = None
    original_positional_args: list[str] = field(default_factory=list)
    original_command: str | None = None
    http_mapping: dict | None = None
    when: str | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ParsedConfig:
    """The slice of the Plurai backend's parsed config the surface projection reads."""

    code_languages: list[str] = field(default_factory=list)
    tools: list[_ToolDescriptor] = field(default_factory=list)
    #: The render / action surface ``render_actions:`` declares, TYPED: ``None`` when the key is
    #: absent, an empty surface for ``[]``. Not projected — the code channel intercepts SDK and
    #: CLI calls, and a render verb is neither — but parsed here rather than skipped, so a
    #: malformed declaration fails this reading of the config the way a malformed descriptor does
    #: instead of riding through the untyped mapping unseen.
    render_actions: RenderSurface | None = None


def _parse_config(config: dict) -> _ParsedConfig:
    """Convert a loaded ``plurai.config.yaml`` dict to the thin descriptor view.

    Mirrors the relevant slice of the Plurai backend's config reader (``agent.code_languages`` +
    the per-tool ``descriptor`` fields the projection reads) — the same key mapping and the
    same falsy-empty normalisation (``http_mapping: {}`` → ``None``, ``when: ''`` → ``None``),
    so both readings of one config see identical input — plus the typed ``render_actions``
    surface (:mod:`plurai.render_actions`), whose malformed shape surfaces as
    :class:`CodeSurfaceGenerationError` like every other authored-shape defect here.
    """
    agent = config.get('agent') or {}
    tools_raw = config.get('tools') or {}
    tools: list[_ToolDescriptor] = []
    for name, raw_entry in tools_raw.items():
        entry = raw_entry or {}
        desc = entry.get('descriptor') or {}
        tools.append(
            _ToolDescriptor(
                name=name,
                mode=entry.get('mode', 'mock'),
                original_module=desc.get('original_module'),
                original_function=desc.get('original_function'),
                original_positional_args=list(desc.get('original_positional_args') or []),
                original_command=desc.get('original_command'),
                http_mapping=desc.get('http_mapping') or None,
                when=desc.get('when') or None,
                args=desc.get('args') or {},
            )
        )
    try:
        render_actions = parse_render_actions(config)
    except RenderActionsConfigError as exc:
        raise CodeSurfaceGenerationError(f'plurai.config.yaml: {exc}') from exc
    return _ParsedConfig(
        code_languages=list(agent.get('code_languages') or []),
        tools=tools,
        render_actions=render_actions,
    )


def build_code_surface(config: dict) -> dict:
    """Project a loaded ``plurai.config.yaml`` dict into the ``code_surface.yaml`` document.

    Pure function (no I/O) so it's unit-testable without touching disk. Parses the config
    dict into the thin descriptor view, then determines the ``code`` sub-mode from the
    mock tools' authored signals and dispatches to the matching builder — the projector
    VALIDATES + COPIES the authored fields, it does not derive them:

    * a mock tool carrying ``original_module`` + ``original_function`` is an **SDK** signal;
    * a mock tool carrying ``original_command`` + ``http_mapping`` is a **CLI** signal;
    * **both** signals across the tool set → FAIL CLOSED (ambiguous, matches the dispatcher);
    * **neither** → no ``code`` surface (an empty ``modules`` map — the ``tools`` channel).

    Only ``mode='mock'`` tools participate (a ``mode='real'`` tool reaches the live vendor
    adapter on purpose — intercepting it would reroute that real call to the mock).
    """
    parsed = _parse_config(config)
    mock_tools = [t for t in parsed.tools if t.mode == 'mock']

    # Catch a half-populated CLI mapping (exactly one of original_command / http_mapping) on
    # ANY mock tool BEFORE channel dispatch — otherwise such a tool is neither a full CLI
    # signal nor an SDK signal and would be silently skipped, leaving its vendor verb
    # un-intercepted. (The SDK half-populated guard lives in the SDK builder, where the group
    # is known; this one must run channel-independently.)
    for tool in mock_tools:
        if bool(tool.original_command) != bool(tool.http_mapping):
            raise CodeSurfaceGenerationError(
                f'mock tool {tool.name!r} has a half-populated CLI mapping '
                f'(original_command={tool.original_command!r}, http_mapping={tool.http_mapping!r}); '
                'a CLI op needs BOTH or NEITHER. Complete or remove the mapping.'
            )

    has_sdk = any(_is_sdk_signal(t) for t in mock_tools)
    has_cli = any(_is_cli_signal(t) for t in mock_tools)

    if has_sdk and has_cli:
        sdk_names = sorted(t.name for t in mock_tools if _is_sdk_signal(t))
        cli_names = sorted(t.name for t in mock_tools if _is_cli_signal(t))
        raise CodeSurfaceGenerationError(
            f'ambiguous code action-channel: the tool set carries BOTH an SDK signal '
            f'(original_module/function: {sdk_names}) AND a CLI signal '
            f'(original_command/http_mapping: {cli_names}). A skill is one or the other; '
            'split the surface or correct the descriptors. Refusing to guess.'
        )

    if has_cli:
        return _build_cli_surface(parsed, mock_tools)
    # SDK signal, or neither (empty modules map = the `tools` channel, back-compat).
    return _build_sdk_surface(parsed, mock_tools)


def _is_sdk_signal(tool: _ToolDescriptor) -> bool:
    """True if the tool declares the SDK module-shadow signal (module AND function)."""
    return bool(tool.original_module) and bool(tool.original_function)


def _is_cli_signal(tool: _ToolDescriptor) -> bool:
    """True if the tool declares the CLI signal (a subcommand AND its HTTP mapping)."""
    return bool(tool.original_command) and bool(tool.http_mapping)


# --- SDK sub-mode: validate + copy the authored `modules` surface --------------------


def _build_sdk_surface(config: _ParsedConfig, mock_tools: list[_ToolDescriptor]) -> dict:
    """Project the SDK-signal tools into the ``modules`` surface (validate + copy).

    Groups tools by ``(original_module, original_function)`` and emits one method entry per
    group: a lone tool → ``tool: <name>``; >1 tool sharing the method → a ``discriminator``
    projected from each descriptor's AUTHORED ``when:`` (validated for field-correctness by
    :func:`_resolve_discriminator`). ``positional`` comes from ``original_positional_args``;
    the per-module ``language`` (and the ``return_envelope`` it selects) from
    ``agent.code_languages``.

    Fail-closed on a half-populated vendor mapping (exactly one of module/function set) —
    silently dropping it would leave the vendor call un-intercepted and reaching the real API.
    """
    language = _resolve_language(config)
    return_envelope = _RETURN_ENVELOPE_BY_LANGUAGE[language]

    modules: dict[str, dict[str, list[_ToolDescriptor]]] = {}
    mapped_names: list[str] = []
    skipped_names: list[str] = []
    for tool in mock_tools:
        module = tool.original_module
        function = tool.original_function
        if bool(module) != bool(function):
            raise CodeSurfaceGenerationError(
                f'mock tool {tool.name!r} has a half-populated vendor mapping '
                f'(original_module={module!r}, original_function={function!r}); a mock tool '
                'needs BOTH or NEITHER. Dropping it would leave the vendor call un-intercepted '
                'and reaching the real API. Complete or remove the mapping.'
            )
        if not module or not function:
            skipped_names.append(tool.name)
            continue
        mapped_names.append(tool.name)
        modules.setdefault(module, {}).setdefault(function, []).append(tool)

    # Accounting so a silently-missing interception is visible in the env-gen log.
    logger.info(
        'code_surface[sdk]: %d mock tool(s) mapped %s, %d skipped (no vendor mapping) %s',
        len(mapped_names),
        sorted(mapped_names),
        len(skipped_names),
        sorted(skipped_names),
    )

    out_modules: dict[str, dict] = {}
    for module_name in sorted(modules):
        methods: dict[str, dict] = {}
        for method_path in sorted(modules[module_name]):
            tools = modules[module_name][method_path]
            entry: dict = {}
            if len(tools) == 1:
                entry['tool'] = tools[0].name
            else:
                entry['discriminator'] = _resolve_discriminator(module_name, method_path, tools)
            positional = _derive_positional(module_name, method_path, tools)
            if positional:
                entry['positional'] = positional
            entry['return_envelope'] = return_envelope
            methods[method_path] = entry
        out_modules[module_name] = {'language': language, 'methods': methods}

    return {'version': CODE_SURFACE_VERSION_SDK, 'modules': out_modules}


def _resolve_language(config: _ParsedConfig) -> str:
    """Resolve the single SDK language that applies to every shadowed module.

    * ``agent.code_languages`` absent/empty → default to ``python`` (back-compat);
    * exactly one declared language → it applies to every module;
    * more than one declared language with no per-module mapping → FAIL CLOSED (a mixed-
      language skill needs a per-module language map before we can pick the right
      ``return_envelope`` per module; guessing would mis-shape half of them).

    An unknown language (not in :data:`_RETURN_ENVELOPE_BY_LANGUAGE`) also fails closed.
    """
    languages = config.code_languages
    if not languages:
        return _DEFAULT_LANGUAGE

    distinct = sorted(set(languages))
    if len(distinct) > 1:
        raise CodeSurfaceGenerationError(
            f'agent.code_languages declares multiple languages {distinct} but the v1 code '
            'surface has no per-module language map to assign them; cannot pick a single '
            'return_envelope per module. Declare one language, or add a per-module mapping.'
        )

    language = distinct[0]
    if language not in _RETURN_ENVELOPE_BY_LANGUAGE:
        raise CodeSurfaceGenerationError(
            f'agent.code_languages declares unsupported language {language!r}; the code surface supports {sorted(_RETURN_ENVELOPE_BY_LANGUAGE)}.'
        )
    return language


def _resolve_discriminator(module_name: str, method_path: str, tools: list[_ToolDescriptor]) -> dict:
    """Project + FIELD-CORRECTNESS-check the authored ``when:`` for >1 tools sharing a method.

    The projector does NOT derive the discriminator by diffing args — the plugin AUTHORS a
    ``when:`` on each sharing descriptor (``'<field> present'`` / ``'<field> absent'``) and
    this validates + copies it. The two tools are sorted by name first so ``present``/
    ``absent`` and the emitted field are **order-independent** (the runtime resolver is
    one-sided, so the projection must not depend on parser list order).

    The generation-time FIELD-CORRECTNESS check (which the runtime loader cannot do — it has
    no ``args``) verifies the authored ``when:`` is semantically sound:

    * exactly two tools share the method (the present/absent contract has no third target);
    * each sharing descriptor carries a ``when:`` (no positional fallback — fail closed);
    * the two ``when:`` name the SAME field with OPPOSITE polarities (one present, one
      absent) — mutually exclusive + exhaustive;
    * the ``when:`` field is **present in the present-tool's ``args`` and absent from the
      absent-tool's ``args``** — key-presence-only, and a CHECK of the authored choice
      rather than a guess at it.
    """
    if len(tools) != 2:
        names = sorted(t.name for t in tools)
        raise CodeSurfaceGenerationError(
            f'{module_name}.{method_path} is backed by {len(tools)} tools ({names}); '
            'the code-surface discriminator supports exactly two tools sharing a method '
            '(one present/one absent). Split the descriptors or add a tools-channel-only mapping.'
        )

    # Sort by name so "first"/"second" — and the discriminator we emit — never depend on the
    # order the parser listed the tools in.
    first, second = sorted(tools, key=lambda t: t.name)

    missing_when = sorted(t.name for t in tools if not t.when)
    if missing_when:
        names = sorted(t.name for t in tools)
        raise CodeSurfaceGenerationError(
            f'{module_name}.{method_path} is shared by tools {names} but {missing_when} carry no '
            'authored `when:` discriminator. A vendor method backing >1 tool MUST carry a '
            '`when: <field> present|absent` on EACH sharing descriptor — the projector does not '
            'derive one. Author the `when:` (key-presence) or split the tools onto distinct methods.'
        )

    first_field, first_polarity = _parse_when(module_name, method_path, first)
    second_field, second_polarity = _parse_when(module_name, method_path, second)

    # Same field, opposite polarity — mutually exclusive + exhaustive over the two tools.
    if first_field != second_field:
        raise CodeSurfaceGenerationError(
            f'{module_name}.{method_path}: the two `when:` discriminators name DIFFERENT fields '
            f'({first.name}->{first_field!r}, {second.name}->{second_field!r}); a key-presence '
            'discriminator must test ONE shared field (present on one tool, absent on the other). '
            'Author both `when:` on the same field with opposite polarities.'
        )
    if first_polarity == second_polarity:
        raise CodeSurfaceGenerationError(
            f'{module_name}.{method_path}: both `when:` discriminators use the SAME polarity '
            f'({first_polarity!r}); the pair must be mutually exclusive — one `present`, one '
            f'`absent` — to pick a unique tool. Tools: {first.name}, {second.name}.'
        )

    field_name = first_field
    present_tool = first.name if first_polarity == _WHEN_PRESENT else second.name
    absent_tool = first.name if first_polarity == _WHEN_ABSENT else second.name

    # FIELD-CORRECTNESS: the authored field must actually distinguish the pair by KEY
    # PRESENCE in the declared args — present in the present-tool, absent from the absent-tool.
    # (This is the semantic guard the runtime loader cannot make; it needs these args.)
    present_args = set(_tool_by_name(tools, present_tool).args)
    absent_args = set(_tool_by_name(tools, absent_tool).args)
    if field_name not in present_args:
        raise CodeSurfaceGenerationError(
            f'{module_name}.{method_path}: `when: {field_name} present` names {present_tool!r} as the '
            f'present tool, but {field_name!r} is NOT in its declared args {sorted(present_args)}. The '
            "authored discriminator field must be a real arg key the present tool's call carries."
        )
    if field_name in absent_args:
        raise CodeSurfaceGenerationError(
            f'{module_name}.{method_path}: `when: {field_name} absent` names {absent_tool!r} as the '
            f'absent tool, but {field_name!r} IS in its declared args {sorted(absent_args)}. The field '
            'must be absent from the absent tool, or it cannot distinguish the two by key presence.'
        )

    return {'field': field_name, 'present': present_tool, 'absent': absent_tool}


def _parse_when(module_name: str, method_path: str, tool: _ToolDescriptor) -> tuple[str, str]:
    """Parse an authored ``when: '<field> present|absent'`` into ``(field, polarity)``.

    Key-presence ONLY: exactly two whitespace-separated tokens, the second being ``present``
    or ``absent``. Anything else (value comparison, boolean expression, missing polarity)
    fails closed — the runtime resolver is one-sided key-presence.
    """
    raw = (tool.when or '').strip()
    parts = raw.split()
    if len(parts) != 2 or parts[1] not in (_WHEN_PRESENT, _WHEN_ABSENT):
        raise CodeSurfaceGenerationError(
            f'{module_name}.{method_path}: tool {tool.name!r} has a malformed `when:` {tool.when!r}; '
            "expected exactly '<field> present' or '<field> absent' (key-presence only — no value "
            'comparisons or boolean expressions).'
        )
    field_name, polarity = parts
    return field_name, polarity


def _tool_by_name(tools: list[_ToolDescriptor], name: str) -> _ToolDescriptor:
    """Return the descriptor named ``name`` from ``tools`` (always present by construction)."""
    for tool in tools:
        if tool.name == name:
            return tool
    raise CodeSurfaceGenerationError(f'internal: tool {name!r} not found in group {[t.name for t in tools]}.')


def _derive_positional(module_name: str, method_path: str, tools: list[_ToolDescriptor]) -> list[str]:
    """Project the positional-arg binding from the tools' ``original_positional_args``.

    A COMPILED field (resolved indices, not LLM-authored semantics). Returns the binding
    when the tools agree on a single non-empty positional list, ``[]`` when none declare one
    (the caller omits ``positional`` then). Tools sharing a method that disagree on their
    binding are a fail-closed error — the runtime hook binds by index and can't honour two
    conflicting orders for one method.
    """
    declared = {tuple(t.original_positional_args) for t in tools if t.original_positional_args}
    if not declared:
        return []
    if len(declared) > 1:
        names = sorted(t.name for t in tools)
        raise CodeSurfaceGenerationError(
            f'{module_name}.{method_path} tools {names} declare conflicting '
            f'original_positional_args {sorted(declared)}; one method can have only one '
            'positional binding. Reconcile the descriptors.'
        )
    return list(next(iter(declared)))


# --- CLI sub-mode: validate + copy the authored `cli_http` surface -------------------


def _build_cli_surface(config: _ParsedConfig, mock_tools: list[_ToolDescriptor]) -> dict:
    """Project the CLI-signal tools into the ``cli_http`` surface (validate + copy).

    Each CLI op authors ``original_command`` (the subcommand) + ``http_mapping``
    (``{method, url_pattern}``); the projector validates them and copies each into one
    ``cli_http.subcommands.<command>: {tool, egress_guard: {method, url_pattern}}`` entry —
    the schema the SDK's ``fetch_shadow`` loader reads. Subcommand → tool is 1:1.
    The projector stamps the COMPILED top-level knobs (``runtime`` / ``preload`` /
    ``return_envelope`` / ``unmapped_egress``) reserved for it.

    Fail-closed on a half-populated CLI mapping (exactly one of command/http_mapping set), a
    malformed ``http_mapping`` (missing method/url_pattern), or two ops claiming the same
    subcommand.
    """
    subcommands: dict[str, dict] = {}
    mapped_names: list[str] = []
    for tool in mock_tools:
        command = tool.original_command
        mapping = tool.http_mapping
        # Half-populated CLI mappings are already caught channel-independently in
        # build_code_surface; a tool with NEITHER CLI field has no vendor verb to intercept
        # here (a mixed SDK+CLI set fails closed earlier) and is legitimately skipped.
        if not command or not mapping:
            continue

        method = mapping.get('method')
        url_pattern = mapping.get('url_pattern')
        if not isinstance(method, str) or not method:
            raise CodeSurfaceGenerationError(
                f'CLI tool {tool.name!r} (command {command!r}) has an http_mapping with a missing/'
                f'empty `method` (got {method!r}); the egress guard needs an HTTP method.'
            )
        if not isinstance(url_pattern, str) or not url_pattern:
            raise CodeSurfaceGenerationError(
                f'CLI tool {tool.name!r} (command {command!r}) has an http_mapping with a missing/'
                f'empty `url_pattern` (got {url_pattern!r}); the egress guard needs a URL pattern.'
            )
        if command in subcommands:
            raise CodeSurfaceGenerationError(
                f'CLI subcommand {command!r} is claimed by more than one tool '
                f'(already {subcommands[command]["tool"]!r}, now {tool.name!r}); subcommand → tool '
                'is 1:1. A multiplexing subcommand is still ONE tool — do not split it.'
            )
        subcommands[command] = {
            'tool': tool.name,
            'egress_guard': {'method': method, 'url_pattern': url_pattern},
        }
        mapped_names.append(tool.name)

    if not subcommands:
        raise CodeSurfaceGenerationError(
            'CLI code surface has no mapped subcommands; expected at least one mock tool with '
            'original_command + http_mapping. Refusing to emit an empty cli_http surface.'
        )

    logger.info('code_surface[cli]: %d subcommand(s) mapped %s', len(mapped_names), sorted(mapped_names))

    return {
        'version': CODE_SURFACE_VERSION_CLI,
        'cli_http': {
            'runtime': _CLI_RUNTIME,
            'preload': _CLI_PRELOAD,
            'return_envelope': _CLI_RETURN_ENVELOPE,
            'unmapped_egress': _CLI_UNMAPPED_EGRESS,
            'subcommands': dict(sorted(subcommands.items())),
        },
    }


def emit_code_surface(config_path: str | Path, out_path: str | Path) -> Path:
    """Read ``plurai.config.yaml`` from ``config_path``, build the surface, write ``out_path``.

    Deterministic side-effecting wrapper around :func:`build_code_surface`: loads the
    config YAML, builds the document, dumps it with block style + insertion order preserved,
    and writes it to ``out_path``. Returns ``out_path`` as a :class:`~pathlib.Path`.
    """
    config_path = Path(config_path)
    out_path = Path(out_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    surface = build_code_surface(config)
    out_path.write_text(
        yaml.safe_dump(surface, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding='utf-8',
    )
    channel = 'cli_http' if 'cli_http' in surface else 'modules'
    count = len(surface.get('cli_http', {}).get('subcommands', {})) if channel == 'cli_http' else len(surface.get('modules', {}))
    logger.info('emitted code_surface.yaml: %s channel, %d %s', channel, count, 'subcommand(s)' if channel == 'cli_http' else 'module(s)')
    return out_path
