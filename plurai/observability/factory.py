"""Single-active-sink selection: the slim observability factory.

Given a resolved :class:`~plurai.settings.Settings` carrier, build the ONE active
:class:`~plurai.observability.sink_protocol.ObservabilitySink` instance the
orchestrator holds for its lifetime. The four per-vendor configs live in their own
sink modules; the global ``Settings`` (in :mod:`plurai.settings`) nests them and
owns the env resolution + single-active validation.

CYCLE AVOIDANCE: this module does NOT import :class:`~plurai.settings.Settings` at
runtime — ``plurai.settings`` imports the sink modules, so importing it back here
(directly or via the package ``__init__``) would close a cycle. The type hint is a
``TYPE_CHECKING``-only forward reference; at runtime the factory only reads
``settings.active`` and resolves the matching config via a typed ``SinkName``-keyed
lookup (direct attribute access, NOT ``getattr`` on a dynamic field name).

Single active sink
------------------
At most ONE sink may be active per run (two live OTel/callback sinks share one
tracer context, which breaks trace trees). The contract FAILS FAST at settings
parse: :class:`~plurai.settings.Settings` RAISES (its ``_compose`` after-validator)
when more than one ``<SINK>__ENABLED`` flag is explicitly set — that is the SOLE
enforcement site of the single-active invariant. The factory therefore ASSUMES the
settings invariant (``settings.active`` is at most one name) and does NOT re-assert
it here: it reads ``settings.active`` and constructs that one sink. Zero enabled ->
a :class:`~plurai.observability.sink_protocol.NoOpSink` so call sites never branch
on None/disabled.

Adding a new vendor (single-module drop-in)
-------------------------------------------
Selection is REGISTRY-DRIVEN — no per-sink ``if/elif``. To add a vendor ``foo``:
write ``foo.py`` with a ``FooSinkConfig`` + ``FooSink`` class, add a ``foo`` field
to :class:`~plurai.settings.Settings`, and add ONE row to :data:`_SINK_CLASSES`
below. The four sink classes import at module scope here (cheap — each vendor SDK is
imported lazily INSIDE the sink methods, never at sink-module scope), so selection is
a direct ``{name: Class}`` dict with no ``importlib`` registry indirection, keeping
the conformance table static so a type checker can verify every row.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plurai.observability.arize import ArizeSink
from plurai.observability.braintrust import BraintrustSink
from plurai.observability.langfuse import LangfuseSink
from plurai.observability.langsmith import LangSmithSink
from plurai.observability.sink_protocol import NoOpSink, ObservabilitySink
from plurai.settings import Settings, SinkName

# Direct sink name -> sink CONSTRUCTOR table — the SINGLE source for both selection
# and the static conformance gate. The four classes import at module scope (the
# vendor SDKs stay lazy inside the sink methods), so there is no ``importlib``
# indirection and no per-vendor ``if/elif``. Typed ``Callable[[Any], ObservabilitySink]``
# (each sink's ``__init__`` takes its own ``*SinkConfig``): the ``ObservabilitySink``
# RETURN type is itself the conformance gate — pyright FAILS to assign a sink class
# whose instances miss a protocol method, a TYPE ERROR here at the registry rather
# than a vanished mid-run ``AttributeError``. (The runtime half is the ``isinstance``
# backstop in ``create_observability_sink``.)
_SINK_CLASSES: dict[SinkName, Callable[[Any], ObservabilitySink]] = {
    'langsmith': LangSmithSink,
    'langfuse': LangfuseSink,
    'arize': ArizeSink,
    'braintrust': BraintrustSink,
}


def create_observability_sink(settings: Settings) -> ObservabilitySink:
    """Build the ONE active observability sink instance (single active sink)."""
    name = settings.active
    if name is None:
        return NoOpSink()
    sink_cls = _SINK_CLASSES[name]
    # Resolve the matching config via a typed ``SinkName``-keyed lookup with DIRECT
    # attribute access (no ``getattr`` on a dynamic field name) — pyright checks each
    # config attribute exists and the key set matches ``SinkName``.
    config = {
        'langsmith': settings.langsmith,
        'langfuse': settings.langfuse,
        'arize': settings.arize,
        'braintrust': settings.braintrust,
    }[name]
    sink = sink_cls(config)
    # RUNTIME BACKSTOP: the contract is a runtime_checkable Protocol (not an ABC),
    # so there is no instantiation-time abstractmethod check. Assert conformance
    # here at construction (startup) — a sink missing a role method must fail with
    # a clear error now, not vanish at call time inside the best-effort swallow.
    if not isinstance(sink, ObservabilitySink):
        raise TypeError(
            f'create_observability_sink built {sink_cls.__module__}.{sink_cls.__qualname__} '
            f'for sink {name!r}, but it does not satisfy the ObservabilitySink contract '
            '(missing one or more of warm_once / prepare_run / post_feedback / '
            'record_transcript / drain_run / close / sync_scenarios). Fix the sink class so a malformed sink fails '
            'at startup rather than silently mid-run (the best-effort guard swallows call-time errors).'
        )
    return sink
