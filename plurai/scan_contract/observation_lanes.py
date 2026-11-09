"""The CLOSED vocabulary of ``tools.<name>.observation_lane``, and what each lane lets a run conclude.

A tool's entry in ``plurai.config.yaml`` may declare, beside ``mode:``, the channel through which a run
observes its calls. The scan writes it (a hosted-agent adapter computes it structurally). The backend types it on
the env-generator's ``ToolDescriptor``, and scenario generation stamps the declared map onto each scenario
(:data:`OBSERVATION_LANES_KEY`) so SCORING can read it.

Scoring dispatches on two properties, each a separate question:

* :data:`ABSENCE_IS_EVIDENCE`, about CALLS: does NO recorded call on this lane mean the tool was not called? Where
  it does not, a check that reads a zero count for that tool is not measured (the code_eval sandbox) and a judge
  is told the count is a lower bound; a recorded call is evidence either way.
* :data:`CONTENT_IS_COMPLETE`, about a recorded call's CONTENT: does it carry everything the call returned? Where
  it does not, what a recorded call shows is a lower bound, so a check that fails on a record MISSING from it
  (retrieval recall) is not measured.

Keyed on the property, never on a lane's name, so a new lane is one row in each table and no rule changes.

Served to clients as the ``observation_lanes`` contract block; the plugin vendors it and checks the lane words it
writes against it.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, get_args

#: ``tool_events``: the mock / callout interception records every call.
#: ``tool_trail``: the harness self-reports every call (``agent_state['tool_trail']``).
#: ``tool_trail_cited``: the harness self-reports only a call whose reply CITED a record, so the observed
#: calls are a LOWER BOUND.
#: ``none``: no channel records the tool at all (a ``mode: real`` tool with no callout seam).
ObservationLane = Literal['tool_events', 'tool_trail', 'tool_trail_cited', 'none']

OBSERVATION_LANES: tuple[ObservationLane, ...] = get_args(ObservationLane)

#: Does NO recorded call to a tool on this lane mean the tool was not called? ``False`` means zero observed
#: calls is UNKNOWN: a check may credit a recorded call, but must never FAIL (or arm) on zero of them.
ABSENCE_IS_EVIDENCE: Mapping[ObservationLane, bool] = MappingProxyType(
    {
        'tool_events': True,
        'tool_trail': True,
        'tool_trail_cited': False,
        'none': False,
    }
)


#: Does a RECORDED call on this lane carry everything that call returned? ``False`` means its recorded content is a
#: LOWER BOUND (a cited-only self-report keeps only the records the reply cited; on ``none`` nothing is a complete
#: record): a check may credit what was recorded, but must never FAIL on a record missing from it.
CONTENT_IS_COMPLETE: Mapping[ObservationLane, bool] = MappingProxyType(
    {
        'tool_events': True,
        'tool_trail': True,
        'tool_trail_cited': False,
        'none': False,
    }
)


class UnknownObservationLaneError(ValueError):
    """A declared ``observation_lane`` that is not a contract lane."""


def normalize_lane(value: object) -> object:
    """A string lane stripped and lower-cased (the spelling the scan's YAML may carry); anything else as-is."""
    return value.strip().lower() if isinstance(value, str) else value


def contract_lane(value: object) -> ObservationLane | None:
    """The contract lane a declared value spells (normalized), or ``None`` for a word outside the contract."""
    normalized = normalize_lane(value)
    for known in OBSERVATION_LANES:
        if known == normalized:
            return known
    return None


def absence_is_evidence(lane: str | None) -> bool:
    """:data:`ABSENCE_IS_EVIDENCE` for a declared lane.

    ``None`` (no lane declared) is today's default: call presence is treated as an ordinary observable
    premise. An unknown word RAISES :class:`UnknownObservationLaneError` — it is never read as observable.
    """
    if lane is None:
        return True
    known = contract_lane(lane)
    if known is not None:
        return ABSENCE_IS_EVIDENCE[known]
    raise UnknownObservationLaneError(f'observation_lane {lane!r} is not one of {list(OBSERVATION_LANES)}')


def content_is_complete(lane: str | None) -> bool:
    """:data:`CONTENT_IS_COMPLETE` for a declared lane.

    ``None`` (no lane declared) is today's default: a recorded call's content is read as whole. An unknown word
    RAISES :class:`UnknownObservationLaneError` — it is never read as complete.
    """
    if lane is None:
        return True
    known = contract_lane(lane)
    if known is not None:
        return CONTENT_IS_COMPLETE[known]
    raise UnknownObservationLaneError(f'observation_lane {lane!r} is not one of {list(OBSERVATION_LANES)}')


#: The ``scenario_metadata`` key carrying ``{tool: lane}`` for every tool the agent's config declares a lane
#: for. Stamped by the scenario writer beside ``real_mode_tools``, only when something was declared.
OBSERVATION_LANES_KEY = 'observation_lanes'


def declared_lanes(plurai_config_yaml: str | None) -> dict[str, ObservationLane]:
    """``{tool: lane}`` for every ``tools.<name>`` entry declaring ``observation_lane`` (a sibling of
    ``descriptor:``), in config order. Empty for an absent or unparseable config.

    RAISES :class:`UnknownObservationLaneError` for a declared word outside the contract: an unknown lane is
    refused, never read as observable.
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
    out: dict[str, ObservationLane] = {}
    for name, entry in (tools or {}).items() if isinstance(tools, dict) else ():
        value = entry.get('observation_lane') if isinstance(entry, dict) else None
        if value is None:
            continue
        lane = contract_lane(value)
        if lane is None:
            raise UnknownObservationLaneError(f'tools.{name}.observation_lane {value!r} is not one of {list(OBSERVATION_LANES)}')
        out[str(name)] = lane
    return out


def unobserved_absence_tools(scenario_metadata: Any) -> dict[str, str]:
    """``{tool: lane}`` for the stamped tools whose lane does NOT make zero recorded calls evidence.

    A stamped word outside the contract is included: it is never read as observable. Empty when nothing was
    stamped (every scenario generated before the stamp, and every config declaring no lane).
    """
    lanes = scenario_metadata.get(OBSERVATION_LANES_KEY) if isinstance(scenario_metadata, Mapping) else None
    if not isinstance(lanes, Mapping):
        return {}
    out: dict[str, str] = {}
    for tool, lane in lanes.items():
        try:
            if absence_is_evidence(lane if isinstance(lane, str) else str(lane)):
                continue
        except UnknownObservationLaneError:
            pass
        out[str(tool)] = str(lane)
    return out


def partial_content_tools(scenario_metadata: Any) -> dict[str, str]:
    """``{tool: lane}`` for the stamped tools whose lane does NOT record a call's whole content.

    A stamped word outside the contract is included: it is never read as complete. Empty when nothing was
    stamped (every scenario generated before the stamp, and every config declaring no lane).
    """
    lanes = scenario_metadata.get(OBSERVATION_LANES_KEY) if isinstance(scenario_metadata, Mapping) else None
    if not isinstance(lanes, Mapping):
        return {}
    out: dict[str, str] = {}
    for tool, lane in lanes.items():
        try:
            if content_is_complete(lane if isinstance(lane, str) else str(lane)):
                continue
        except UnknownObservationLaneError:
            pass
        out[str(tool)] = str(lane)
    return out
