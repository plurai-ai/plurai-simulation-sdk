"""A turn the agent's provider INFRASTRUCTURE ended — typed on the side, never in place of the agent's words.

Only infrastructure is not the agent's to be measured on: a 5xx, a 529 overload, a 429 rate limit, a dropped
connection, a sign-in, credential, account or billing failure. Everything else the provider returns — a
safety refusal, a context overflow, an oversized file — is what the agent's end user sees, and is graded as
the agent's reply. The marker's shape is framework-neutral so every framework's adapter can stamp the same
fact; ``plurai.child`` stamps it today from the Claude Code CLI's ``StopFailure`` hook.
"""

from __future__ import annotations

from typing import Any

#: The SDK-owned ``agent_state`` key stamped on a turn that infrastructure ended — beside the harness's own
#: keys, never inside them. the backend reads it to demote a judge that found content absent on that turn.
PROVIDER_TURN_FAULT_KEY = 'provider_turn_fault'

#: The SDK-owned ``agent_state`` key for a turn whose last CLI query started and never reported its end within
#: the child's wait: how the turn ended is unknown, so nothing about it is guessed and the reply is kept.
TURN_END_UNOBSERVED_KEY = 'turn_end_unobserved'


def provider_turn_fault_marker(fault_class: str, *, source: str, terminal: bool) -> dict[str, Any]:
    """The value of ``agent_state[PROVIDER_TURN_FAULT_KEY]``: ``fault_class`` is an ``infra_*`` class,
    ``source`` names what observed it, ``terminal`` whether repeating the call cannot change it."""
    if not fault_class.startswith('infra_'):
        raise ValueError(f'a provider turn fault is infrastructure only (infra_*), not {fault_class!r}')
    return {'class': fault_class, 'source': source, 'terminal': terminal}
