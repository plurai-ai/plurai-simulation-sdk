"""``tools.<name>.exposed_in``: the agent's own scopes (subagent, topic, graph node) that offer a tool.

An agent whose source scopes its tools per unit — each unit holding its own action list — says, per tool, in
which of those units the model can call it. The scan writes it beside ``mode:`` (a hosted-agent adapter reads it
off the bundle's per-unit action blocks) and only when the source really scopes tools that way; a single tool loop
declares nothing. Scenario generation reads it to type which flows run where a knowledge base's tool is at hand.

A SUB-AGENT the agent runs as a call names its scope by that call, so the scope name is also a call on the trail: the
scenario writer stamps the map (:data:`TOOL_SCOPES_KEY`) and retrieval scoring reads "the delegation ran" off it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The ``scenario_metadata`` key carrying ``{tool: [scope, …]}`` from the config's ``exposed_in`` declarations. Stamped
#: by the scenario writer beside ``observation_lanes``, only when something was declared.
TOOL_SCOPES_KEY = 'tool_scopes'


class MalformedToolScopeError(ValueError):
    """A declared ``exposed_in`` that is not a list of scope names."""


def declared_tool_scopes(plurai_config_yaml: str | None) -> dict[str, frozenset[str]]:
    """``{tool: scopes}`` for every ``tools.<name>`` entry declaring a non-empty ``exposed_in`` (a sibling of
    ``descriptor:``), in config order. Empty for an absent or unparseable config.

    RAISES :class:`MalformedToolScopeError` for a value that is not a list of non-empty strings: a malformed
    declaration is refused, never read as "exposed everywhere" or "nowhere".
    """
    import yaml  # noqa: PLC0415

    from plurai.scan_contract._safe_yaml import safe_yaml_load  # noqa: PLC0415

    if not plurai_config_yaml or not plurai_config_yaml.strip():
        return {}
    try:
        raw = safe_yaml_load(plurai_config_yaml)
    except yaml.YAMLError:
        return {}
    tools = raw.get('tools') if isinstance(raw, dict) else None
    out: dict[str, frozenset[str]] = {}
    for name, entry in (tools or {}).items() if isinstance(tools, dict) else ():
        value = entry.get('exposed_in') if isinstance(entry, dict) else None
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(scope, str) and scope.strip() for scope in value):
            raise MalformedToolScopeError(f'tools.{name}.exposed_in {value!r} is not a list of scope names')
        if value:
            out[str(name)] = frozenset(scope.strip() for scope in value)
    return out


def stamped_tool_scopes(scenario_metadata: Any) -> dict[str, frozenset[str]]:
    """``{tool: scopes}`` as the scenario writer stamped it (:data:`TOOL_SCOPES_KEY`); empty when nothing was stamped.

    A malformed stamp reads as empty: it widens nothing, since every reader of it can only withhold a verdict.
    """
    raw = scenario_metadata.get(TOOL_SCOPES_KEY) if isinstance(scenario_metadata, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, frozenset[str]] = {}
    for tool, scopes in raw.items():
        if isinstance(scopes, list | tuple) and all(isinstance(scope, str) and scope.strip() for scope in scopes) and scopes:
            out[str(tool)] = frozenset(scope.strip() for scope in scopes)
    return out
