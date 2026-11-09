"""Schema-driven mock stubs for the SDK.

When a tool wrapped with ``mock_wrap`` is called and no ``@plurai.mock``
handler exists for it (either because the customer hasn't written one yet,
or because the generated mock env didn't cover this tool), the mock-env
returns a generic sentinel ``{"status": "ok", "mock": True, "tool": <name>}``.
That sentinel has the same shape regardless of what the real tool would
return — so downstream agent code that does ``result["commits"]`` or similar
either crashes or hallucinates.

This module replaces the generic sentinel with a **structurally-correct**
stub matched to the tool's declared return type. Empty list for
``list[Commit]``, zero-filled ``Commit`` (as a dict) for ``Commit``, ``{}``
for ``dict[str, Any]``, etc. The stub doesn't fabricate content — that's
the customer-handler's job — but it gives the agent the right shape, which
is what unblocks day-zero "wrap everything, write zero handlers" runs.

Resolution table (caller treats ``_NO_USEFUL_TYPE`` return as "fall back to
the generic sentinel"):

  str / int / bool / float / bytes  → "" / 0 / False / 0.0 / b""
  list / list[T]                    → []
  dict / dict[K, V]                 → {}
  tuple[...] / set / frozenset      → () / set() / frozenset()
  Optional[T] / T | None            → None
  Union[T, U] (no None arm)         → stub(T)
  Pydantic BaseModel                → model_dump() of construct()-ed instance
  TypedDict                         → {field: stub(annotation), ...}
  Anything else / Any / NoneType    → _NO_USEFUL_TYPE
"""

from __future__ import annotations

import typing
from typing import Any, get_args, get_origin

try:
    from pydantic import BaseModel

    _PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover — pydantic is a hard plurai dep, but be defensive
    BaseModel = None  # type: ignore[assignment,misc]
    _PYDANTIC_AVAILABLE = False


class _NoUsefulType:
    """Sentinel returned by ``schema_stub_from_annotation`` when the
    annotation can't be mapped to a useful stub. Caller substitutes the
    generic ``{"status": "ok", "mock": True, "tool": <name>}`` shape, so
    untyped tools get the generic sentinel.
    """

    _instance: _NoUsefulType | None = None

    def __new__(cls) -> _NoUsefulType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return '_NO_USEFUL_TYPE'


_NO_USEFUL_TYPE: _NoUsefulType = _NoUsefulType()


# ---------------------------------------------------------------------------
# Annotation extraction — best-effort, covers raw functions + LangChain tools.
# ---------------------------------------------------------------------------


def extract_return_type(tool_or_func: Any) -> Any:
    """Pull the return-type annotation off a tool/function, or return ``None``.

    Resolution order — first attribute whose ``return`` hint resolves to a
    usable type wins:
      1. ``tool.func``       — LangChain ``@tool`` sync (returns ``None`` on async tools)
      2. ``tool.coroutine``  — LangChain ``@tool`` async (returns ``None`` on sync tools)
      3. ``tool._run``       — custom ``BaseTool`` subclass
      4. ``tool._arun``      — custom async ``BaseTool``
      5. ``tool_or_func`` itself — plain Python function (or a callable that
         carries no LangChain plumbing)

    Each candidate is read via ``typing.get_type_hints`` (not
    ``__annotations__`` directly) so PEP-563 string annotations — the
    default under ``from __future__ import annotations`` — are resolved
    against the candidate's own globals into real type objects.

    Returns ``None`` when the resolved annotation is missing,
    ``typing.Any``, or ``type(None)``. Those signal "no useful shape" and
    the caller falls back to the generic sentinel.
    """
    if tool_or_func is None:
        return None

    candidates: list[Any] = []
    # LangChain ``@tool`` sync wraps the function as ``.func``; async as ``.coroutine``.
    func_attr = getattr(tool_or_func, 'func', None)
    if func_attr is not None and callable(func_attr):
        candidates.append(func_attr)
    coro_attr = getattr(tool_or_func, 'coroutine', None)
    if coro_attr is not None and callable(coro_attr):
        candidates.append(coro_attr)
    # Custom ``BaseTool`` subclasses define ``_run`` / ``_arun``.
    run_attr = getattr(tool_or_func, '_run', None)
    if run_attr is not None and callable(run_attr):
        candidates.append(run_attr)
    arun_attr = getattr(tool_or_func, '_arun', None)
    if arun_attr is not None and callable(arun_attr):
        candidates.append(arun_attr)
    # Fall through to the object itself (plain function path).
    candidates.append(tool_or_func)

    for cand in candidates:
        annotation = _read_return_annotation(cand)
        if annotation is not None:
            return annotation
    return None


def _read_return_annotation(func: Any) -> Any:
    """Return ``func``'s declared return annotation, or ``None`` when missing
    / empty / ``Any`` / ``NoneType`` / not introspectable.

    Uses ``typing.get_type_hints`` so PEP-563 string annotations (the
    default under ``from __future__ import annotations``) are resolved to
    real type objects against the function's own globals — without this,
    ``-> dict[str, int]`` arrives as the string ``'dict[str, int]'`` and
    no downstream introspection works.
    """
    try:
        hints = typing.get_type_hints(func)
    except (NameError, TypeError, ValueError, AttributeError):
        # Forward refs we can't resolve, builtins without __globals__,
        # exotic callables — fall through to "no useful annotation".
        return None
    if 'return' not in hints:
        return None
    annotation = hints['return']
    if annotation is typing.Any:
        return None
    if annotation is type(None):
        return None
    return annotation


# ---------------------------------------------------------------------------
# Schema-stub generator.
# ---------------------------------------------------------------------------


# Primitive map. Keys are the types themselves (not strings) so subclass
# checks can short-circuit cleanly.
_PRIMITIVES: dict[type, Any] = {
    str: '',
    int: 0,
    bool: False,
    float: 0.0,
    bytes: b'',
}


def schema_stub_from_annotation(annotation: Any) -> Any:
    """Produce a structurally-valid stub for ``annotation``.

    Returns ``_NO_USEFUL_TYPE`` for unrepresentable types — caller substitutes
    the generic sentinel in that case. See module docstring for the full
    type→stub table.
    """
    # Direct primitive hit (e.g., ``str``, ``int``, ``bool``).
    if isinstance(annotation, type) and annotation in _PRIMITIVES:
        return _PRIMITIVES[annotation]

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Container origins: list / List, dict / Dict, tuple, set, frozenset.
    if origin in (list, typing.List):  # noqa: UP006 — runtime-typing requires both
        return []
    if origin in (dict, typing.Dict):  # noqa: UP006
        return {}
    if origin in (tuple, typing.Tuple):  # noqa: UP006
        return ()
    if origin in (set, typing.Set):  # noqa: UP006
        return set()
    if origin in (frozenset, typing.FrozenSet):  # noqa: UP006
        return frozenset()

    # Bare container types (``list`` / ``dict`` without parameterization).
    if annotation is list:
        return []
    if annotation is dict:
        return {}
    if annotation is tuple:
        return ()
    if annotation is set:
        return set()
    if annotation is frozenset:
        return frozenset()

    # Optional[T] / T | None: prefer None (zero-effort, semantically valid).
    # Union[T, U] without None: stub the first arm.
    if origin is typing.Union or _is_pep604_union(annotation):
        non_none = [a for a in args if a is not type(None)]
        if not non_none or len(non_none) < len(args):
            # ``None`` was a member of the union → stub as None.
            return None
        return schema_stub_from_annotation(non_none[0])

    # Pydantic models: zero-fill required fields, keep optional defaults,
    # return a JSON-shaped dict (consumers expect dicts on the wire).
    if _PYDANTIC_AVAILABLE and isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _stub_pydantic_model(annotation)

    # TypedDict: walk declared fields, recurse on each annotation.
    if _is_typed_dict(annotation):
        result: dict[str, Any] = {}
        hints = typing.get_type_hints(annotation)
        for key, sub_annotation in hints.items():
            sub_stub = schema_stub_from_annotation(sub_annotation)
            # Substitute None for the generic sentinel at the field level
            # — TypedDicts can't carry the full sentinel cleanly.
            result[key] = None if sub_stub is _NO_USEFUL_TYPE else sub_stub
        return result

    # Anything else (custom non-pydantic class, ``Any``, exotic generics)
    # → caller falls back to the generic sentinel.
    return _NO_USEFUL_TYPE


def _stub_pydantic_model(model_cls: type) -> dict[str, Any]:
    """Build a zero-filled instance of ``model_cls`` and dump it to a dict.

    Public surface — returns the JSON-shaped dict consumers expect on the
    wire. Nested ``BaseModel`` fields are kept as model **instances**
    internally (via ``_stub_pydantic_instance``) so the outer ``model_dump``
    produces a clean dict tree without Pydantic serializer warnings about
    "expected ModelX, got dict".
    """
    return _stub_pydantic_instance(model_cls).model_dump()


def _stub_pydantic_instance(model_cls: type) -> Any:
    """Recursive helper: returns a ``model_construct``-ed instance with each
    required field resolved via the order below; optional fields fall
    back to their declared default / ``default_factory``.

    Resolution order for required fields (first match wins):
      1. ``Field(examples=[...])`` — first non-None example wins.
         Pydantic's canonical "here's what this field looks like in
         production" hint; using it produces stubs rich enough for agent
         reasoning that depends on field content (not just shape). A
         leading ``None`` example is treated as "no usable hint" and
         falls through to (2)/(3) so we don't substitute ``None`` into
         a typed required field.
      2. Nested ``BaseModel`` annotation — recurse with this same helper
         (returning an instance so the parent's ``model_dump`` produces a
         clean dict tree without serializer warnings).
      3. Anything else — delegate to ``schema_stub_from_annotation``;
         ``_NO_USEFUL_TYPE`` falls through to ``None``.
    """
    values: dict[str, Any] = {}
    for field_name, field_info in model_cls.model_fields.items():
        if not field_info.is_required():
            values[field_name] = field_info.get_default(call_default_factory=True)
            continue

        # Customer-declared examples win over both nested-recursion and
        # zero-fill — they're the most specific intent the field carries.
        # ``examples=[None]`` is treated as "no useful example" rather than
        # "use None": substituting None for a typed required field (e.g.
        # ``id: str``) would silently break agents that do
        # ``result['id'].lower()``. The customer either declared examples
        # by accident or means to signal absence — either way, falling
        # through to the structural stub is safer.
        examples = getattr(field_info, 'examples', None)
        if examples and examples[0] is not None:
            values[field_name] = examples[0]
            continue

        annotation = field_info.annotation
        if _PYDANTIC_AVAILABLE and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            values[field_name] = _stub_pydantic_instance(annotation)
        else:
            sub_stub = schema_stub_from_annotation(annotation)
            values[field_name] = None if sub_stub is _NO_USEFUL_TYPE else sub_stub
    return model_cls.model_construct(**values)


def _is_pep604_union(annotation: Any) -> bool:
    """Detect ``T | None`` PEP-604 unions — they have a different ``get_origin``
    return on 3.10+ (``types.UnionType``) than ``typing.Union``.
    """
    import types

    return isinstance(annotation, types.UnionType)


def _is_typed_dict(annotation: Any) -> bool:
    """TypedDict detection via ``typing.is_typeddict`` (available since 3.10)."""
    return bool(typing.is_typeddict(annotation))


# ---------------------------------------------------------------------------
# Sentinel detection.
# ---------------------------------------------------------------------------


def is_no_handler_sentinel(response: Any, tool_name: str) -> bool:
    """True iff ``response`` is the LocalMockService no-handler sentinel for
    this tool.

    Stable shape emitted by the ``else`` branch of the ``/mock/execute``
    handler in ``LocalMockService._handle`` whenever no ``@plurai.mock``
    handler is registered for ``tool_name``: ::

        {"status": "ok", "mock": True, "tool": "<tool_name>"}

    Any subclass that delegates to ``LocalMockService._handle`` for the
    no-handler case (e.g. a generated ``ExposedMockService``,
    which uses ``super()._handle()``) emits the same shape — one predicate
    covers both. ``mock`` must be exactly ``True`` (not just truthy) so a
    real handler returning, e.g., ``{"status": "ok", "mock": "v2", ...}``
    doesn't trip the fallback.
    """
    if not isinstance(response, dict):
        return False
    return response.get('status') == 'ok' and response.get('mock') is True and response.get('tool') == tool_name
