"""Where each root name of ``plurai`` lives.

The root resolves its names on first access (``plurai.__getattr__``), so ``import plurai`` loads none of the mock
runtime, the orchestrator or LangChain, and a command that never touches a ``simulate.py`` surface (``plurai login``,
``plurai --version``, ``plurai capabilities``) stays light. This table is the single list of those names: the root's
``__all__`` is derived from it, so a name cannot be advertised without a home or homed without being advertised.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: Module -> the public names it provides to the root.
HOMES: dict[str, tuple[str, ...]] = {
    'plurai_mockenv': (
        'MockService',
        'SimulationContext',
        'get_simulation_context',
        'mock',
        'observe',
        'world_state',
    ),
    'plurai_mockenv.descriptor_schema': ('ToolDescriptor',),
    'plurai._protocol': (
        'AgentFault',
        'HarnessFault',
    ),
    'plurai._mocking.client': ('MockServiceClient',),
    'plurai._mocking.code_hook': (
        'install_code_hook',
        'install_code_hook_from_env',
        'sitecustomize_source',
        'uninstall_code_hook',
        'write_sitecustomize',
    ),
    'plurai._mocking.code_intercept': ('code_intercept_call',),
    'plurai._mocking.code_surface': (
        'CodeSurfaceGenerationError',
        'build_code_surface',
        'emit_code_surface',
    ),
    'plurai._mocking.interception': (
        'MockToolNode',
        'apply_mocks',
        'mock_wrap',
    ),
    'plurai._mocking.interception_watch': ('watch_interception',),
    'plurai._mocking.result': (
        'MockToolError',
        'mock_result',
    ),
    'plurai._mocking.session': (
        'active_session_id',
        'configure',
        'get_events',
        'get_full_state',
        'mock_service_url',
        'start_session',
        'world_state_snapshot',
    ),
    'plurai.host_effects': (
        'cli_hooks',
        'confine',
        'host_effects_granted',
        'host_read_scope',
    ),
    'plurai.file_rows': ('file_row',),
    'plurai.world.seed_rows': ('store_record',),
    'plurai.input_context': ('model_input_context',),
}

#: Name -> the module that provides it.
MODULE_OF: dict[str, str] = {name: module for module, names in HOMES.items() for name in names}


def resolve(name: str, namespace: dict[str, Any]) -> Any:
    """The root's ``__getattr__``: import the module that provides ``name`` and bind the value in ``namespace``."""
    module = MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module 'plurai' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    namespace[name] = value
    return value
