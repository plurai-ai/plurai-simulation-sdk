"""What the agent's own code added to its model input this turn: ``plurai.model_input_context``.

A harness reports content the agent puts into its OWN model input (outside the user turns and the tool calls) on
the turn's ``input_context``: a list of ``{source, content}`` entries the judges read as context the agent HAD,
never as something it said and never as a user turn (the backend renders it as its AGENT INPUT CONTEXT block).

Listing those injections BY HAND is how one goes missing: an injection the agent function prepends only under some
condition (say, a rule-state update sent when the rules changed) never reaches the judges, and they fail the agent
for not reading content it was handed.

:func:`model_input_context` derives the entries from what the model call RECEIVED instead: every message of the
model input that the conversation does not carry, in order. Agent-side messages (the agent's own replies and tool
results) are never input context, and neither is a LEADING system message, the agent's own instructions. It reads
both lists, calls nothing, and never re-reads a source (a re-read can differ from what the agent was given).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: Roles (either spelling) of a message the AGENT side produced: its own reply, a tool's result.
_AGENT_SIDE_ROLES = frozenset({'assistant', 'ai', 'tool', 'function'})
_SYSTEM_ROLES = frozenset({'system', 'developer'})


def _role(msg: Any) -> str:
    raw = msg.get('role') or msg.get('type') if isinstance(msg, Mapping) else getattr(msg, 'role', None) or getattr(msg, 'type', None)
    return str(raw or '').lower()


def _text(msg: Any) -> str:
    content = msg.get('content') if isinstance(msg, Mapping) else getattr(msg, 'content', None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else p.get('text') for p in content if isinstance(p, str) or (isinstance(p, Mapping) and p.get('type') == 'text')]
        return '\n'.join(p for p in parts if isinstance(p, str)).strip()
    return ''


def model_input_context(model_input: Iterable[Any], conversation: Iterable[Any], *, source: str) -> list[dict[str, str]]:
    """The ``input_context`` entries for one turn: each ``model_input`` message the ``conversation`` does not carry.

    ``model_input`` is the message list the agent's model call received this turn (dicts with ``role``/``type`` and
    ``content``, or message objects with those attributes; ``content`` a string or a list of text parts).
    ``conversation`` is the messages the harness itself handed the agent (the user turns and the history it keeps).
    ``source`` names the read the entries came from (e.g. the model-call hook the harness wrapped).

    A message is left out when its role is agent-side (``assistant``/``ai``/``tool``/``function``), when it opens the
    input as a ``system``/``developer`` message (the agent's instructions), when its text is blank, or when its text
    equals a conversation message's text. Each remaining text is returned once, in model-input order, as
    ``{'source': source, 'content': text}``. Nothing injected returns ``[]``, so the turn returns no ``input_context``.
    """
    carried = {_text(m) for m in conversation}
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, msg in enumerate(model_input):
        role = _role(msg)
        if role in _AGENT_SIDE_ROLES or (idx == 0 and role in _SYSTEM_ROLES):
            continue
        text = _text(msg)
        if not text or text in carried or text in seen:
            continue
        seen.add(text)
        out.append({'source': source, 'content': text})
    return out
