"""Generic Python import hook for the `code` action channel.

Installs a :class:`importlib.abc.MetaPathFinder` that *shadows* named vendor modules
(here: ``resend``) with synthetic modules whose attribute / method access is resolved
against a ``code_surface.yaml`` and
dispatched through :func:`plurai.code_intercept_call` over the frozen ``/mock/execute``
wire. The skill's production code — ``import resend; resend.Emails.send(...)`` — runs
unchanged, but the call lands on the mock world-state instead of the real vendor API.

**Generic by construction.** Nothing here is resend-specific: the finder shadows
whatever ``modules:`` the loaded ``code_surface`` names, and a method call
``<module>.<Class>.<method>(args)`` is resolved purely from the surface data
(``methods``, ``discriminator``, ``return_envelope``). The same hook serves any vendor.

The synthetic module is built from a generic attribute-path **proxy**: attribute access
accumulates a dotted path (``resend`` → ``.Emails`` → ``.send``) and a *call* resolves
that path against the surface's ``methods`` map, applies the arg-shape discriminator for
multivalued methods, dispatches, and shapes the native return.

This module is the Python half of the hook; the Node ``--require`` / ``--import`` half
lives in ``plurai/_mocking/node/``. The ``node_data_error`` return-envelope value is
honoured here too, so the shaping stays data-driven rather than language-branched.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import yaml

from plurai._mocking.code_intercept import code_intercept_call
from plurai._mocking.interception_watch import watch_interception


class CodeSurfaceError(RuntimeError):
    """Raised when a call cannot be resolved against the loaded code_surface.

    Fail-closed: an unmapped method or an ambiguous/undeterminable tool must error
    rather than silently no-op or reach the real vendor SDK.
    """


@dataclass(frozen=True)
class _Discriminator:
    """The arg-shape discriminator for a multivalued method.

    A typed replacement for the bare ``{field, present, absent}`` dict so the one-of
    invariant (a rule is single-valued XOR discriminated) is expressible in the type
    system and a malformed surface raises :class:`CodeSurfaceError` at load — never a
    bare ``KeyError`` at first dispatch. All three fields are non-empty strings (enforced
    in :func:`_load_code_surface`): ``field`` is the arg name to test, ``present`` /
    ``absent`` the tool to use when it is / is not in the call's args.
    """

    field: str
    present: str
    absent: str


@dataclass(frozen=True)
class _MethodRule:
    """Resolved rule for one vendor method path (e.g. ``Emails.send``).

    Exactly one of ``tool`` (single-valued) or ``discriminator`` (multivalued) is set —
    the one-of invariant, enforced at load. ``return_envelope`` selects the native return
    shape applied to the mock response. ``positional`` binds positional call args (by
    index) to arg names — needed for vendor methods called with a bare value rather than a
    params dict (``resend.Emails.get("email_abc123")`` → ``{"id": "email_abc123"}``).
    """

    tool: str | None
    discriminator: _Discriminator | None
    return_envelope: str
    positional: tuple[str, ...]

    def __post_init__(self) -> None:
        """Enforce the one-of (``tool`` XOR ``discriminator``) invariant on construction.

        The type refuses to construct a both-set / neither-set rule, so a malformed rule can
        never reach :meth:`resolve_tool` (which can assume exactly one is set). A rule that
        violates the invariant raises :class:`CodeSurfaceError` here rather than misrouting at
        dispatch.
        """
        if (self.tool is None) == (self.discriminator is None):
            raise CodeSurfaceError('method rule must set exactly one of `tool` (single-valued) or `discriminator` (multivalued).')

    def resolve_tool(self, args: dict[str, Any]) -> str:
        """Pick the tool for this call's ``args`` (applies the arg-shape discriminator)."""
        if self.tool is not None:
            return self.tool
        # The one-of invariant (enforced in __post_init__) guarantees a discriminator here;
        # the assert is a type-narrowing no-op, not a runtime guard (mirrors _subprocess.py).
        assert self.discriminator is not None  # noqa: S101
        if self.discriminator.field in args:
            return self.discriminator.present
        return self.discriminator.absent


@dataclass
class _ModuleSurface:
    """The resolved surface for one shadowed module: language + per-method rules."""

    language: str
    methods: dict[str, _MethodRule] = field(default_factory=dict)

    def _is_namespace_prefix(self, path: str) -> bool:
        """True if ``path`` is a strict dotted prefix of some known method path."""
        if path == '':
            return True
        prefix = f'{path}.'
        return any(method.startswith(prefix) for method in self.methods)

    def is_known_attribute(self, parent_path: str, name: str) -> bool:
        """Whether attribute ``name`` may be walked from a proxy at ``parent_path``.

        Drives the shadow proxy's fail-closed attribute access. Access is allowed when:

        * the resulting path is a known method (``Emails.send``);
        * the resulting path is a namespace — a strict prefix of some method
          (``Emails`` → ``Emails.send``), so the skill can walk into it; or
        * the *parent* is a known namespace — so an unmapped LEAF method under a real
          namespace (``Emails.cancel``) is still reachable and fails at *call* with a
          clear ``CodeSurfaceError`` rather than being masked here.

        It is rejected when neither the parent nor the result is a modeled namespace —
        e.g. ``resend.exceptions`` (``exceptions`` is not a namespace at all). That
        prevents ``except resend.exceptions.ResendError`` binding a proxy in place of an
        exception class and silently never matching.
        """
        child = f'{parent_path}.{name}' if parent_path else name
        if child in self.methods:
            return True
        if self._is_namespace_prefix(child):
            return True
        # Parent must be a NAMED namespace (non-empty). The root ('') is a namespace prefix
        # by construction, but a bare top-level attribute off the root (e.g. `resend.exceptions`)
        # is only valid if it is itself a modeled namespace/method (handled above).
        return bool(parent_path) and self._is_namespace_prefix(parent_path)


@dataclass(frozen=True)
class _EgressGuard:
    """The HTTP egress guard for one cli subcommand: the (method, url_pattern) it may hit.

    The fetch_shadow runtime (``plurai/_mocking/node/fetch_shadow.mjs``) matches a CLI
    subprocess's outbound request against this guard before mapping it to ``tool``; a request
    that does not match is fail-closed (``unmapped_egress: fail_closed``). Both fields are
    non-empty strings, enforced at load so a malformed guard raises :class:`CodeSurfaceError`
    here rather than letting an unguarded egress through at runtime.
    """

    method: str
    url_pattern: str


@dataclass(frozen=True)
class _CliSubcommand:
    """Resolved mapping for one CLI subcommand: which tool it backs + its egress guard.

    The loader READS this from the authored surface (it never derives the mapping). The
    ``tool`` is the mock tool the subcommand dispatches to; ``egress_guard`` constrains the
    outbound HTTP the subprocess may make for that subcommand.
    """

    tool: str
    egress_guard: _EgressGuard


@dataclass(frozen=True)
class _CliHttpSurface:
    """The resolved ``cli_http`` section: the CLI/fetch_shadow channel's authored contract.

    Carries the per-subcommand tool+egress mappings plus the fixed channel knobs the
    runtime needs: ``runtime`` (``fetch_shadow``), ``preload`` (the path to the
    fetch_shadow preload module), ``return_envelope`` (``node_cli_json``), and
    ``unmapped_egress`` (``fail_closed``). All are READ from the surface and STRUCTURALLY
    validated at load; the loader never derives them.
    """

    runtime: str
    preload: str
    return_envelope: str
    unmapped_egress: str
    subcommands: dict[str, _CliSubcommand] = field(default_factory=dict)


@dataclass(frozen=True)
class _SurfaceDocument:
    """A fully-loaded surface document: the module-shadow surfaces + the cli_http section.

    A v1 surface yields ``modules`` only (``cli_http is None``); a v2 surface may carry
    either or both. At least one is always present (enforced at load — a surface with
    neither fails closed).
    """

    modules: dict[str, _ModuleSurface] = field(default_factory=dict)
    cli_http: _CliHttpSurface | None = None


# Schema versions this loader understands. Bump in lock-step with the Node loader
# (parseCodeSurface) and the generator. A surface declaring any other version is a
# fail-closed error: a newer schema may carry shapes (e.g. value-based discriminators)
# this loader would silently misread.
#
# v1: `modules`-only module-shadow surfaces.
# v2: adds the top-level `cli_http` section (the CLI/fetch_shadow channel) and makes
#     `modules` optional when `cli_http` is present. v1 surfaces still load unchanged.
_SUPPORTED_SURFACE_VERSIONS = frozenset({1, 2})

# The native return-envelope values a method / cli subcommand may declare. `python_dict`
# and `node_data_error` are the SDK-shadow shapes; `node_cli_json` is the CLI channel's
# envelope (the subprocess emits the mock response as JSON on stdout).
_RETURN_ENVELOPES = frozenset({'python_dict', 'node_data_error', 'node_cli_json'})

# The languages a shadowed module may declare. `language` selects which native return
# envelope applies; an unknown value is a generation bug we catch at load (mirroring the
# `return_envelope` check) rather than carrying a meaningless language into dispatch. Kept
# lock-step with the Node loader, which validates the same set.
_MODULE_LANGUAGES = frozenset({'python', 'node'})

# The cli_http channel's fixed top-level field values. The loader READS these from the
# authored surface and validates they are present + well-formed; it never derives them.
_CLI_HTTP_RUNTIME = 'fetch_shadow'
_CLI_HTTP_RETURN_ENVELOPE = 'node_cli_json'
_CLI_HTTP_UNMAPPED_EGRESS = 'fail_closed'


def _require_nonempty_str(value: Any, what: str, code_surface_path: str) -> str:
    """Return ``value`` if it is a non-empty string, else raise :class:`CodeSurfaceError`.

    Load-time guard: a ``tool: ''`` / ``present: null`` slips a ``None``/empty tool name
    into dispatch (``{tool: None}`` misroute) unless caught here.
    """
    if not isinstance(value, str) or not value:
        raise CodeSurfaceError(f'{code_surface_path}: {what} must be a non-empty string (got {value!r}).')
    return value


def _load_surface_document(code_surface_path: str) -> _SurfaceDocument:
    """Parse + STRUCTURALLY validate a ``code_surface.yaml`` into a :class:`_SurfaceDocument`.

    Reads the authored surface and validates its SHAPE — it never derives the mapping.
    The validation order is load-bearing and kept in lock-step with the Node loader:

    1. the document is a mapping;
    2. the schema ``version`` is present and supported (fail closed on absent or unknown) —
       this runs BEFORE the per-section presence check, because a `cli_http`-only v2
       surface has no ``modules`` and a presence check run first would reject it;
    3. at least one of ``modules`` / ``cli_http`` is present (a surface with neither fails
       closed); and
    4. each present section is well-formed — module methods (one-of ``tool`` xor
       ``discriminator``, non-empty discriminator/tool, ``positional`` a list of strings)
       and the ``cli_http`` channel (each subcommand a ``tool`` + an egress guard with
       ``method`` + ``url_pattern``; the channel knobs ``runtime`` / ``preload`` /
       ``return_envelope`` / ``unmapped_egress`` present and well-formed).

    A malformed surface is a generation bug we want to catch at install time, not at first
    dispatch.
    """
    with open(code_surface_path, encoding='utf-8') as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise CodeSurfaceError(f'{code_surface_path}: surface must be a mapping (got {type(raw).__name__}).')

    # Version FIRST, before any per-section presence check: a `cli_http`-only v2 surface has
    # no `modules`, so an unconditional `modules`-required guard ahead of the version dispatch
    # would throw first and shadow it. A projector-stamped surface ALWAYS carries a
    # `version` (build_code_surface / emit_code_surface stamp it); an absent version is
    # anomalous, so we fail closed rather than guessing — defaulting to the highest supported
    # version is the unsafe direction (a v1-only loader would silently misread a v2 surface).
    # Lock-step with the Node loader, which raises on the same absent-version case.
    if 'version' not in raw or raw['version'] is None:
        raise CodeSurfaceError(
            f'{code_surface_path}: surface is missing a `version`. A projector-stamped surface always '
            'declares one; refusing to guess at an unstamped schema.'
        )
    version = raw['version']
    if version not in _SUPPORTED_SURFACE_VERSIONS:
        supported = ', '.join(str(v) for v in sorted(_SUPPORTED_SURFACE_VERSIONS))
        raise CodeSurfaceError(
            f'{code_surface_path}: unsupported code_surface schema version {version!r} '
            f'(this loader understands version(s) {supported}). Refusing to guess at an unknown schema.'
        )

    has_modules = 'modules' in raw and raw['modules'] is not None
    has_cli_http = 'cli_http' in raw and raw['cli_http'] is not None
    if not (has_modules or has_cli_http):
        raise CodeSurfaceError(
            f'{code_surface_path}: surface must declare at least one of top-level `modules` '
            '(SDK module-shadow channel) or `cli_http` (CLI/fetch_shadow channel).'
        )

    modules = _parse_modules(raw['modules'], code_surface_path) if has_modules else {}
    cli_http = _parse_cli_http(raw['cli_http'], code_surface_path) if has_cli_http else None
    return _SurfaceDocument(modules=modules, cli_http=cli_http)


def _parse_modules(raw_modules: Any, code_surface_path: str) -> dict[str, _ModuleSurface]:
    """Parse + validate the ``modules`` section into ``{module_name -> _ModuleSurface}``.

    Validates the one-of (``tool`` xor ``discriminator``) per method, that each
    discriminator carries non-empty ``field`` / ``present`` / ``absent`` strings, that
    ``tool`` (when set) is a non-empty string, and that ``positional`` (if present) is a
    list of arg-name strings.
    """
    if not isinstance(raw_modules, dict):
        raise CodeSurfaceError(f'{code_surface_path}: top-level `modules` must be a mapping (got {raw_modules!r}).')
    surfaces: dict[str, _ModuleSurface] = {}
    for module_name, module_spec in raw_modules.items():
        language = module_spec.get('language', 'python')
        if language not in _MODULE_LANGUAGES:
            allowed = ', '.join(sorted(_MODULE_LANGUAGES))
            raise CodeSurfaceError(f'{code_surface_path}: module {module_name} `language` must be one of {{{allowed}}} (got {language!r}).')
        surface = _ModuleSurface(language=language)
        for method_path, method_spec in (module_spec.get('methods') or {}).items():
            where = f'{module_name}.{method_path}'
            tool = method_spec.get('tool')
            disc_raw = method_spec.get('discriminator')
            if (tool is None) == (disc_raw is None):
                raise CodeSurfaceError(
                    f'{code_surface_path}: method {where} must set exactly one of `tool` (single-valued) or `discriminator` (multivalued).'
                )
            if tool is not None:
                tool = _require_nonempty_str(tool, f'method {where} `tool`', code_surface_path)
            discriminator: _Discriminator | None = None
            if disc_raw is not None:
                if not isinstance(disc_raw, dict):
                    raise CodeSurfaceError(f'{code_surface_path}: discriminator for {where} must be a mapping (got {disc_raw!r}).')
                missing = [k for k in ('field', 'present', 'absent') if k not in disc_raw]
                if missing:
                    raise CodeSurfaceError(f'{code_surface_path}: discriminator for {where} is missing {missing}.')
                discriminator = _Discriminator(
                    field=_require_nonempty_str(disc_raw['field'], f'discriminator `field` for {where}', code_surface_path),
                    present=_require_nonempty_str(disc_raw['present'], f'discriminator `present` for {where}', code_surface_path),
                    absent=_require_nonempty_str(disc_raw['absent'], f'discriminator `absent` for {where}', code_surface_path),
                )
            positional = method_spec.get('positional') or []
            if not (isinstance(positional, list) and all(isinstance(name, str) for name in positional)):
                raise CodeSurfaceError(f'{code_surface_path}: `positional` for {where} must be a list of arg-name strings.')
            return_envelope = method_spec.get('return_envelope', 'python_dict')
            if return_envelope not in _RETURN_ENVELOPES:
                allowed = ', '.join(sorted(_RETURN_ENVELOPES))
                raise CodeSurfaceError(f'{code_surface_path}: `return_envelope` for {where} must be one of {{{allowed}}} (got {return_envelope!r}).')
            surface.methods[method_path] = _MethodRule(
                tool=tool,
                discriminator=discriminator,
                return_envelope=return_envelope,
                positional=tuple(positional),
            )
        surfaces[module_name] = surface
    return surfaces


def _parse_cli_http(raw_cli_http: Any, code_surface_path: str) -> _CliHttpSurface:
    """Parse + STRUCTURALLY validate the ``cli_http`` section into a :class:`_CliHttpSurface`.

    Validates the fixed channel knobs (``runtime`` == ``fetch_shadow``, ``preload`` a
    non-empty path, ``return_envelope`` == ``node_cli_json``, ``unmapped_egress`` ==
    ``fail_closed``) and each subcommand (a non-empty ``tool`` and an ``egress_guard``
    mapping with non-empty ``method`` + ``url_pattern``). STRUCTURAL only: it does NOT
    check that a discriminator field actually distinguishes the two tools per their args —
    that needs the config args and lives at GENERATION time, not in the runtime loader.
    The whole section is READ from the authored surface; nothing here is derived.
    """
    if not isinstance(raw_cli_http, dict):
        raise CodeSurfaceError(f'{code_surface_path}: top-level `cli_http` must be a mapping (got {raw_cli_http!r}).')

    runtime = _require_nonempty_str(raw_cli_http.get('runtime'), '`cli_http.runtime`', code_surface_path)
    if runtime != _CLI_HTTP_RUNTIME:
        raise CodeSurfaceError(
            f'{code_surface_path}: `cli_http.runtime` must be {_CLI_HTTP_RUNTIME!r} (got {runtime!r}).'
        )
    preload = _require_nonempty_str(raw_cli_http.get('preload'), '`cli_http.preload`', code_surface_path)
    return_envelope = _require_nonempty_str(raw_cli_http.get('return_envelope'), '`cli_http.return_envelope`', code_surface_path)
    if return_envelope != _CLI_HTTP_RETURN_ENVELOPE:
        raise CodeSurfaceError(
            f'{code_surface_path}: `cli_http.return_envelope` must be {_CLI_HTTP_RETURN_ENVELOPE!r} (got {return_envelope!r}).'
        )
    unmapped_egress = _require_nonempty_str(raw_cli_http.get('unmapped_egress'), '`cli_http.unmapped_egress`', code_surface_path)
    if unmapped_egress != _CLI_HTTP_UNMAPPED_EGRESS:
        raise CodeSurfaceError(
            f'{code_surface_path}: `cli_http.unmapped_egress` must be {_CLI_HTTP_UNMAPPED_EGRESS!r} (got {unmapped_egress!r}).'
        )

    raw_subcommands = raw_cli_http.get('subcommands')
    if not isinstance(raw_subcommands, dict) or not raw_subcommands:
        raise CodeSurfaceError(
            f'{code_surface_path}: `cli_http.subcommands` must be a non-empty mapping of subcommand -> {{tool, egress_guard}}.'
        )
    subcommands: dict[str, _CliSubcommand] = {}
    for name, spec in raw_subcommands.items():
        where = f'cli_http.subcommands.{name}'
        if not isinstance(spec, dict):
            raise CodeSurfaceError(f'{code_surface_path}: {where} must be a mapping (got {spec!r}).')
        tool = _require_nonempty_str(spec.get('tool'), f'{where} `tool`', code_surface_path)
        guard_raw = spec.get('egress_guard')
        if not isinstance(guard_raw, dict):
            raise CodeSurfaceError(f'{code_surface_path}: {where} `egress_guard` must be a mapping (got {guard_raw!r}).')
        egress_guard = _EgressGuard(
            method=_require_nonempty_str(guard_raw.get('method'), f'{where} `egress_guard.method`', code_surface_path),
            url_pattern=_require_nonempty_str(guard_raw.get('url_pattern'), f'{where} `egress_guard.url_pattern`', code_surface_path),
        )
        subcommands[name] = _CliSubcommand(tool=tool, egress_guard=egress_guard)

    return _CliHttpSurface(
        runtime=runtime,
        preload=preload,
        return_envelope=return_envelope,
        unmapped_egress=unmapped_egress,
        subcommands=subcommands,
    )


def _load_code_surface(code_surface_path: str) -> dict[str, _ModuleSurface]:
    """Parse a ``code_surface.yaml`` into ``{module_name -> _ModuleSurface}`` (modules only).

    Backward-compatible accessor for callers that only need the SDK module-shadow surfaces
    (the import hook). Delegates to :func:`_load_surface_document` (so the full document —
    including ``cli_http`` — is validated) and returns just its ``modules`` map.
    """
    return _load_surface_document(code_surface_path).modules


def _shape_native_return(response: Any, return_envelope: str) -> Any:
    """Shape the mock's native return value into the vendor SDK's native shape.

    - ``python_dict``: resend-python returns the plain dict the mock already produced.
    - ``node_data_error``: resend-node returns ``{data, error}``; recorded per-method so the
      Node hook is data-driven, and honoured here as well so one shaping rule serves both.
    """
    if return_envelope == 'node_data_error':
        return {'data': response, 'error': None}
    return response


class _CallProxy:
    """A dotted attribute path on a shadowed module that resolves to a tool on call.

    ``resend.Emails`` is a ``_CallProxy(path='Emails')``; ``.send`` extends it to
    ``_CallProxy(path='Emails.send')``; calling it resolves ``Emails.send`` against the
    module's surface, applies the discriminator, dispatches via ``code_intercept_call``,
    and shapes the native return. A path with no matching method rule that is *called*
    raises :class:`CodeSurfaceError` (fail-closed).

    Attribute access is itself fail-closed: it extends the path only when the result stays
    a known method path or a prefix of one (so nested namespaces like ``A.B.method`` work
    generically), and raises :class:`AttributeError` otherwise. Without this guard
    ``except resend.exceptions.ResendError`` would bind ``resend.exceptions.ResendError``
    to a proxy — not an exception class — and silently never match, masking real errors.
    """

    __slots__ = ('_surface', '_path', '_dispatch')

    def __init__(self, surface: _ModuleSurface, path: str, dispatch: Callable[[str, dict[str, Any]], Any]) -> None:
        self._surface = surface
        self._path = path
        self._dispatch = dispatch

    def __getattr__(self, name: str) -> _CallProxy:
        if not self._surface.is_known_attribute(self._path, name):
            child = f'{self._path}.{name}' if self._path else name
            raise AttributeError(
                f'{child!r} is not a known code_surface namespace/method on this shadowed module '
                f'(known methods: {sorted(self._surface.methods)}). The `code` action-channel shadow '
                'only models the vendor methods named in the surface; unmodeled attributes (e.g. an '
                'exception class) are not proxied so they fail loudly instead of silently misbehaving.'
            )
        child = f'{self._path}.{name}' if self._path else name
        return _CallProxy(self._surface, child, self._dispatch)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        rule = self._surface.methods.get(self._path)
        if rule is None:
            raise CodeSurfaceError(f'no code_surface mapping for method path {self._path!r} (known: {sorted(self._surface.methods)}).')
        call_args = _merge_call_args(rule, self._path, args, kwargs)
        tool = rule.resolve_tool(call_args)
        response = self._dispatch(tool, call_args)
        return _shape_native_return(response, rule.return_envelope)


def _merge_call_args(rule: _MethodRule, method_path: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Flatten a vendor call's positional + keyword args into a single dict.

    Two positional shapes coexist in the resend surface:
      * a params (and optional options) dict — ``Emails.send(params, {"idempotency_key": k})``;
      * a bare value bound by name — ``Emails.get("email_abc123")`` → ``{"id": …}``, driven by
        the method's ``positional: [id]`` rule.

    So each positional is handled by index: if the rule binds index ``i``
    (``i < len(rule.positional)``), ``args[i]`` is bound to ``rule.positional[i]``; otherwise
    a dict positional is merged (the params/options shape) and a **non-dict positional with
    no binding slot raises** :class:`CodeSurfaceError` — fail-closed, because silently
    dropping it would lose the id and dispatch an empty-arg call.
    kwargs are overlaid last (they win on key collision).

    Arity is fail-closed both ways: too FEW positionals (``len(args) < len(rule.positional)``)
    would leave a declared name (e.g. ``id``) unbound and dispatch a partial arg map, so it
    raises here rather than silently sending an incomplete call. A declared name may still
    be supplied as a kwarg, which is checked after the merge.
    """
    if len(args) < len(rule.positional):
        supplied_via_kwargs = [name for name in rule.positional[len(args) :] if name in kwargs]
        if len(args) + len(supplied_via_kwargs) < len(rule.positional):
            missing = [name for name in rule.positional[len(args) :] if name not in kwargs]
            raise CodeSurfaceError(
                f'{method_path!r} declares {len(rule.positional)} positional arg(s) '
                f'{list(rule.positional)} but was called with {len(args)} positional and no '
                f'kwarg for {missing}. Refusing to dispatch a partial arg map.'
            )
    merged: dict[str, Any] = {}
    for i, a in enumerate(args):
        if i < len(rule.positional):
            merged[rule.positional[i]] = a
        elif isinstance(a, dict):
            merged.update(a)
        else:
            raise CodeSurfaceError(
                f'positional arg {i} ({a!r}) to {method_path!r} has no binding: the method declares '
                f'{len(rule.positional)} positional name(s) {list(rule.positional)} and the arg is not a '
                'params dict. Add it to the method `positional` list in the code_surface.'
            )
    merged.update(kwargs)
    return merged


class _ShadowLoader(importlib.abc.Loader):
    """Builds the synthetic module: a root ``_CallProxy`` exposing the surface's classes."""

    def __init__(self, module_name: str, surface: _ModuleSurface, dispatch: Callable[[str, dict[str, Any]], Any]) -> None:
        self._module_name = module_name
        self._surface = surface
        self._dispatch = dispatch

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None  # default module creation

    def exec_module(self, module: ModuleType) -> None:
        root = _CallProxy(self._surface, '', self._dispatch)
        module.__getattr__ = root.__getattr__  # type: ignore[attr-defined]
        module.__plurai_code_shadow__ = True  # type: ignore[attr-defined]


class CodeImportHook(importlib.abc.MetaPathFinder):
    """``meta_path`` finder that shadows the vendor modules named in a code_surface.

    One instance is installed on ``sys.meta_path`` by :func:`install_code_hook`. It claims
    only the module names present in the loaded surface (so unrelated imports are
    untouched) and serves each a synthetic module via :class:`_ShadowLoader`.
    """

    def __init__(self, surfaces: dict[str, _ModuleSurface], dispatch: Callable[[str, dict[str, Any]], Any]) -> None:
        self._surfaces = surfaces
        self._dispatch = dispatch

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        surface = self._surfaces.get(fullname)
        if surface is None:
            return None  # not a shadowed module — defer to the normal import machinery
        loader = _ShadowLoader(fullname, surface, self._dispatch)
        return importlib.machinery.ModuleSpec(fullname, loader)


def install_code_hook(
    code_surface_path: str,
    *,
    dispatch: Callable[[str, dict[str, Any]], Any] = code_intercept_call,
) -> CodeImportHook:
    """Install the generic shadow hook for every module named in ``code_surface_path``.

    Loads + validates the surface, evicts any already-imported real vendor module so a
    later ``import resend`` resolves to the shadow, prepends the finder to
    ``sys.meta_path``, and returns the installed :class:`CodeImportHook` (pass it to
    :func:`uninstall_code_hook`).

    ``dispatch`` is the function each resolved ``(tool, args)`` is routed through; it
    defaults to :func:`plurai.code_intercept_call` (the real frozen-wire caller) and is a
    DI seam so tests can capture ``(tool, args)`` without a live mock.
    """
    surfaces = _load_code_surface(code_surface_path)
    for module_name in surfaces:
        sys.modules.pop(module_name, None)  # drop an already-imported real module
    hook = CodeImportHook(surfaces, dispatch)
    sys.meta_path.insert(0, hook)
    return hook


def uninstall_code_hook(hook: CodeImportHook) -> None:
    """Remove an installed hook and evict its shadow modules from ``sys.modules``.

    Symmetric with :func:`install_code_hook` so a test (or a finished bench) can restore
    the normal import machinery. Tolerant of a hook that is not currently installed.
    """
    try:
        sys.meta_path.remove(hook)
    except ValueError:
        pass
    for module_name in hook._surfaces:  # noqa: SLF001 — symmetric teardown of our own hook
        mod = sys.modules.get(module_name)
        if mod is not None and getattr(mod, '__plurai_code_shadow__', False):
            del sys.modules[module_name]


# Path to the bench's code_surface.yaml, injected into the skill's Bash subprocess env
# (alongside PLURAI_SESSION_ID / PLURAI_MOCK_URL) by the plugin. The grandchild
# ``python send.py`` process reads it via install_code_hook_from_env() so the shadow is
# active before its ``import resend``.
_ENV_CODE_SURFACE = 'PLURAI_CODE_SURFACE'


def install_code_hook_from_env(
    *,
    dispatch: Callable[[str, dict[str, Any]], Any] = code_intercept_call,
) -> CodeImportHook:
    """Install the hook using the ``code_surface.yaml`` path in ``PLURAI_CODE_SURFACE``.

    The Python analog of Node's ``--require``: a ``sitecustomize`` placed on the
    grandchild's ``PYTHONPATH`` calls this at interpreter start, so the shadow is active
    before the skill snippet's ``import resend`` runs. Reads the surface path from the
    environment (the plugin injects it next to ``PLURAI_SESSION_ID`` /
    ``PLURAI_MOCK_URL``) and delegates to :func:`install_code_hook`.

    Raises :class:`CodeSurfaceError` when ``PLURAI_CODE_SURFACE`` is unset — fail-closed,
    matching the runtime: a code-channel snippet with no surface wiring must error, not
    fall through to a real vendor import.
    """
    code_surface_path = os.environ.get(_ENV_CODE_SURFACE)
    if not code_surface_path:
        raise CodeSurfaceError(
            f'{_ENV_CODE_SURFACE} is not set — cannot install the code-channel import hook. '
            'The skill subprocess must inject the path to the bench code_surface.yaml '
            '(the plugin sets it alongside PLURAI_SESSION_ID / PLURAI_MOCK_URL).'
        )
    return install_code_hook(code_surface_path, dispatch=dispatch)


def sitecustomize_source() -> str:
    """Return the ``sitecustomize.py`` body that auto-installs the hook at interpreter start.

    Python imports a top-level ``sitecustomize`` module automatically during startup if one
    is found on ``sys.path`` (here: a dir the plugin puts FIRST on the subprocess
    ``PYTHONPATH``). That module calls :func:`install_code_hook_from_env`, so any later
    ``python send.py`` doing ``import resend`` gets the shadow — no snippet-side change.

    Fail-open ONLY on a genuinely-absent code channel: if ``PLURAI_CODE_SURFACE`` is unset
    the file does nothing (so a plain Python subprocess that happens to share the dir is
    unaffected); when it IS set, an install failure propagates (fail-closed on a real
    code-channel run). Returned as a string so the plugin can ``write_text`` it into the shim dir;
    :func:`write_sitecustomize` is the convenience wrapper.
    """
    return (
        '# Auto-generated by plurai — installs the code-channel import hook at startup.\n'
        '# Place this dir FIRST on PYTHONPATH for the skill subprocess; Python imports\n'
        '# `sitecustomize` automatically before running the snippet.\n'
        'import os\n'
        '\n'
        "if os.environ.get('PLURAI_CODE_SURFACE'):\n"
        '    from plurai import install_code_hook_from_env\n'
        '\n'
        '    install_code_hook_from_env()\n'
    )


def write_sitecustomize(target_dir: str) -> str:
    """Write :func:`sitecustomize_source` into ``target_dir/sitecustomize.py``; return its path.

    Convenience for the plugin's setup: create the shim dir, drop the ``sitecustomize.py``,
    and prepend ``target_dir`` to the subprocess ``PYTHONPATH``. Separate from the source
    accessor so callers that template their own bootstrap can still reuse the body.

    The written file is recorded (:func:`~plurai._mocking.interception_watch.watch_interception`),
    so a run whose agent rewrites it says so.
    """
    path = os.path.join(target_dir, 'sitecustomize.py')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(sitecustomize_source())
    watch_interception(path)
    return path
