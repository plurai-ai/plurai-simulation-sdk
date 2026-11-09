"""Plurai Simulation SDK: the client runtime for simulating conversations against your agent.

The package root is the supported surface for a ``simulate.py``: ``@plurai.mock``, ``plurai.configure``,
``plurai.MockToolNode`` and the rest of ``__all__``. Nothing else is importable from it. The Orchestrator and
the other REST-client types live in ``plurai.orchestrator``; the observability sinks in ``plurai.observability``.

Names resolve on first access, so ``import plurai`` is cheap (see :mod:`plurai._surface`).
"""

import importlib.metadata as _metadata
import typing as _typing

from plurai import _surface

if _typing.TYPE_CHECKING:
    from plurai_mockenv import MockService, SimulationContext, get_simulation_context, mock, observe, world_state
    from plurai_mockenv.descriptor_schema import ToolDescriptor

    from plurai._mocking.client import MockServiceClient
    from plurai._mocking.code_hook import (
        install_code_hook,
        install_code_hook_from_env,
        sitecustomize_source,
        uninstall_code_hook,
        write_sitecustomize,
    )
    from plurai._mocking.code_intercept import code_intercept_call
    from plurai._mocking.code_surface import CodeSurfaceGenerationError, build_code_surface, emit_code_surface
    from plurai._mocking.interception import MockToolNode, apply_mocks, mock_wrap
    from plurai._mocking.interception_watch import watch_interception
    from plurai._mocking.result import MockToolError, mock_result
    from plurai._mocking.session import (
        active_session_id,
        configure,
        get_events,
        get_full_state,
        mock_service_url,
        start_session,
        world_state_snapshot,
    )
    from plurai._protocol import AgentFault, HarnessFault
    from plurai.file_rows import file_row
    from plurai.host_effects import cli_hooks, confine, host_effects_granted, host_read_scope
    from plurai.input_context import model_input_context
    from plurai.world.seed_rows import store_record

# The installed distribution's version; an uninstalled source tree reports 0.0.0.
try:
    __version__ = _metadata.version('plurai-simulation-sdk')
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = '0.0.0'

# The 1.0 contract with the root: exactly these names and the version. ``tests/test_public_surface.py`` holds the
# root to this list and to ``plurai._surface``.
__all__ = [
    # Writing simulate.py: mock interception and the simulation context
    'mock',
    'observe',
    'MockToolNode',
    'MockToolError',
    'mock_result',
    'mock_wrap',
    'apply_mocks',
    'configure',
    'SimulationContext',
    'get_simulation_context',
    'world_state',
    'world_state_snapshot',
    'active_session_id',
    'mock_service_url',
    'cli_hooks',
    'confine',
    'host_effects_granted',
    'host_read_scope',
    'file_row',
    'store_record',
    'model_input_context',
    # Faults a harness or an agent reports
    'HarnessFault',
    'AgentFault',
    # Tool and state contract schema
    'ToolDescriptor',
    # Mock service client and engine
    'MockServiceClient',
    'MockService',
    'start_session',
    'get_full_state',
    'get_events',
    # The code action channel: SDK-call interception over the mock wire
    'code_intercept_call',
    'install_code_hook',
    'install_code_hook_from_env',
    'uninstall_code_hook',
    'sitecustomize_source',
    'write_sitecustomize',
    'watch_interception',
    # The code action channel: surface projection from plurai.config.yaml
    'build_code_surface',
    'emit_code_surface',
    'CodeSurfaceGenerationError',
    '__version__',
]


def __getattr__(name: str) -> object:
    return _surface.resolve(name, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
