"""Client-side observability for the plurai Orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from plurai.observability.sink_protocol import NoOpSink, ObservabilitySink
from plurai.observability.arize import ArizeRunContext, ArizeSink, ArizeSinkConfig
from plurai.observability.braintrust import BraintrustRunContext, BraintrustSink, BraintrustSinkConfig
from plurai.observability.langfuse import LangfuseRunContext, LangfuseSink, LangfuseSinkConfig
from plurai.observability.langsmith import LangSmithRunContext, LangSmithSink, LangSmithSinkConfig
from plurai.observability.errors import swallowable_error_types
from plurai.observability.rollup import RollupMetric

if TYPE_CHECKING:
    from plurai.observability.factory import create_observability_sink

__all__ = [
    'ArizeRunContext',
    'ArizeSink',
    'ArizeSinkConfig',
    'BraintrustRunContext',
    'BraintrustSink',
    'BraintrustSinkConfig',
    'LangfuseRunContext',
    'LangfuseSink',
    'LangfuseSinkConfig',
    'LangSmithRunContext',
    'LangSmithSink',
    'LangSmithSinkConfig',
    'NoOpSink',
    'ObservabilitySink',
    'RollupMetric',
    'create_observability_sink',
    'swallowable_error_types',
]


def __getattr__(name: str) -> Any:
    # The factory reads plurai.settings, and settings reads the sink configs from this package: resolving the
    # factory on first use keeps either module importable first.
    if name == 'create_observability_sink':
        from plurai.observability.factory import create_observability_sink

        return create_observability_sink
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
