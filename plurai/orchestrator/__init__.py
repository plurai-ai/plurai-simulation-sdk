"""REST-driven simulation orchestration for plurai.

Every name in ``__all__`` is importable from ``plurai.orchestrator``. ``SeedStateCallable`` is deliberately absent:
there is no ``Orchestrator(seed_state=...)`` kwarg for it to type, because the child auto-discovers ``seed_state``
from the customer's ``simulate.py`` directly.

Names resolve on first access. Importing ``plurai.orchestrator.models`` (the wire types) therefore does not load
the orchestrator, whose own imports reach back into ``plurai.observability``, and neither import order can form a
cycle. Internal dependency DAG (import order): ``models`` -> ``client`` -> ``orchestrator``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

#: Module -> the names it provides to this package.
_HOMES: dict[str, tuple[str, ...]] = {
    'plurai.orchestrator.models': (
        'AgentAction',
        'CreateThreadRequest',
        'CreateThreadResponse',
        'EvalSetPoliciesResponse',
        'FinalizeRequest',
        'FinalizeResponse',
        'GetThreadResponse',
        'ScenarioSetFetchResponse',
        'ScenarioUploadEntry',
        'SerializedMessage',
        'UserTurnRequest',
        'UserTurnResponse',
        'FailResponse',
    ),
    'plurai.orchestrator.client': (
        'OrchestratorClient',
        'UserTurn',
    ),
    'plurai.orchestrator.orchestrator': (
        'ADAPTER_DEGRADED_REASON',
        'AGENT_CLOSED_REASONS',
        'AGENT_ENDED_REASON',
        'GIVE_UP_CAUSE_AGENT_CANNOT',
        'GIVE_UP_CAUSE_AGENT_DIVERGED',
        'GIVE_UP_CAUSE_AGENT_REFUSED',
        'GIVE_UP_CAUSE_AGENT_REPEATED',
        'GIVE_UP_CAUSE_REPLY_NOT_VISIBLE',
        'HANDED_OFF_REASON',
        'PLAN_EXHAUSTED_REASON',
        'POLICY_TERMINATED_REASON',
        'REPLAY_CUT_REASON',
        'SIM_DEGENERATE_REASON',
        'TIME_BUDGET_REASON',
        'TURN_CAP_REASON',
        'USER_GAVE_UP_REASON',
        'AGENT_RAISED_REASON',
        'AGENT_RECURSION_REASON',
        'PROVIDER_INFRA_REASON',
        'TURN_BUDGET_REASON',
        'HARNESS_CUT_REASONS',
        'Orchestrator',
        'ThreadResult',
    ),
    'plurai._protocol': (
        'FinalizePollTimeout',
        'AgentTurnTimeout',
        'FinalizeJobFailed',
        'GraphRecursionWrap',
        'HarnessChildExited',
        'HarnessFaultRaised',
        'HarnessResetContract',
        'HarnessSeedStateContract',
        'MockStubFallback',
        'MockUnreachableError',
        'OrchestratorAdapterError',
        'ScenarioTimeout',
    ),
    'plurai.orchestrator.redrive': (
        'RedriveProvenance',
        'RedriveTrigger',
    ),
}
_MODULE_OF: dict[str, str] = {name: module for module, names in _HOMES.items() for name in names}

if TYPE_CHECKING:
    from plurai.orchestrator.models import (
        AgentAction,
        CreateThreadRequest,
        CreateThreadResponse,
        EvalSetPoliciesResponse,
        FinalizeRequest,
        FinalizeResponse,
        GetThreadResponse,
        ScenarioSetFetchResponse,
        ScenarioUploadEntry,
        SerializedMessage,
        UserTurnRequest,
        UserTurnResponse,
        FailResponse as FailResponse,
    )
    from plurai.orchestrator.client import (
        OrchestratorClient,
        UserTurn,
    )
    from plurai.orchestrator.orchestrator import (
        ADAPTER_DEGRADED_REASON,
        AGENT_CLOSED_REASONS,
        AGENT_ENDED_REASON,
        GIVE_UP_CAUSE_AGENT_CANNOT,
        GIVE_UP_CAUSE_AGENT_DIVERGED,
        GIVE_UP_CAUSE_AGENT_REFUSED,
        GIVE_UP_CAUSE_AGENT_REPEATED,
        GIVE_UP_CAUSE_REPLY_NOT_VISIBLE,
        HANDED_OFF_REASON,
        PLAN_EXHAUSTED_REASON,
        POLICY_TERMINATED_REASON,
        REPLAY_CUT_REASON,
        SIM_DEGENERATE_REASON,
        TIME_BUDGET_REASON,
        TURN_CAP_REASON,
        USER_GAVE_UP_REASON,
        AGENT_RAISED_REASON,
        AGENT_RECURSION_REASON,
        PROVIDER_INFRA_REASON,
        TURN_BUDGET_REASON,
        HARNESS_CUT_REASONS,
        Orchestrator,
        ThreadResult,
    )
    from plurai._protocol import (
        FinalizePollTimeout,
        AgentTurnTimeout,
        FinalizeJobFailed,
        GraphRecursionWrap,
        HarnessChildExited,
        HarnessFaultRaised,
        HarnessResetContract,
        HarnessSeedStateContract,
        MockStubFallback,
        MockUnreachableError,
        OrchestratorAdapterError,
        ScenarioTimeout,
    )
    from plurai.orchestrator.redrive import (
        RedriveProvenance,
        RedriveTrigger,
    )

__all__ = [
    # The typed ``ThreadResult.ended_early`` vocabulary — the reasons a dialog ended for
    # something other than the user reaching their goal. Three are the harness's own (it counted
    # its own turns, held its own clock, and knows which replies it manufactured); the fourth,
    # ``USER_GAVE_UP_REASON``, is the SIMULATOR's and arrives reported over the wire,
    # and the last two are the AGENT's own, reported the same way — it said it was
    # ending or transferring the conversation, on the action channel. ``AGENT_CLOSED_REASONS``
    # is the set of those two, for the one question a consumer asks of them: is the other end of
    # this dialog still there? Exported for the same reason RedriveTrigger is: a consumer
    # branches on these, never on a turn count, a wall-clock reading or a message.
    # ``SIM_DEGENERATE_REASON`` is the backend's own harness fault — its user simulator produced
    # no turn — and it is the one value here that says NOTHING ABOUT THE AGENT WAS MEASURED, so a
    # consumer that reads a run's outcome has to be able to name it.
    'ADAPTER_DEGRADED_REASON',
    'AGENT_CLOSED_REASONS',
    'AGENT_ENDED_REASON',
    'AgentAction',
    'HANDED_OFF_REASON',
    'PLAN_EXHAUSTED_REASON',
    'POLICY_TERMINATED_REASON',
    'REPLAY_CUT_REASON',
    'SIM_DEGENERATE_REASON',
    'TIME_BUDGET_REASON',
    'TURN_CAP_REASON',
    'USER_GAVE_UP_REASON',
    'AGENT_RAISED_REASON',
    'AGENT_RECURSION_REASON',
    'PROVIDER_INFRA_REASON',
    'TURN_BUDGET_REASON',
    'HARNESS_CUT_REASONS',
    # The simulator's stated CAUSE for a give-up (``ThreadResult.ended_early_cause``), the values
    # this build routes on. Same rule: a consumer branches on these, never on the transcript.
    'GIVE_UP_CAUSE_AGENT_CANNOT',
    'GIVE_UP_CAUSE_AGENT_DIVERGED',
    'GIVE_UP_CAUSE_AGENT_REFUSED',
    'GIVE_UP_CAUSE_AGENT_REPEATED',
    'GIVE_UP_CAUSE_REPLY_NOT_VISIBLE',
    'AgentTurnTimeout',
    'CreateThreadRequest',
    'CreateThreadResponse',
    'EvalSetPoliciesResponse',
    'FinalizeJobFailed',
    'FinalizeRequest',
    'FinalizeResponse',
    'FinalizePollTimeout',
    'GetThreadResponse',
    'GraphRecursionWrap',
    'HarnessChildExited',
    'HarnessFaultRaised',
    'HarnessResetContract',
    'HarnessSeedStateContract',
    'MockStubFallback',
    'MockUnreachableError',
    'Orchestrator',
    'OrchestratorAdapterError',
    'OrchestratorClient',
    'RedriveProvenance',
    'RedriveTrigger',
    'ScenarioSetFetchResponse',
    'ScenarioTimeout',
    'ScenarioUploadEntry',
    'SerializedMessage',
    'ThreadResult',
    'UserTurn',
    'UserTurnRequest',
    'UserTurnResponse',
]


def __getattr__(name: str) -> Any:
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_MODULE_OF})
