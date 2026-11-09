"""``issue_turn.json`` — the recorded bad reply a regression folder carries beside its pin.

A regression folder replays one confirmed agent issue: its pin (or opener) is the conversation cut
BEFORE the issue's turn, and this file is the turn itself — what the user said on it, what the agent
replied and, for an agent that acts, the calls and results of that turn the reply rests on. The
backend replays the reply against the check the eval set names as the owner of that regression, so
a check that no longer fails a reply already proven bad is caught before a run.

The plugin's ``record-regression`` writes it; :mod:`plurai.ingest.cli` reads it once more before a
bundle leaves the machine; the backend reads it from the staged bundle. It is agent evidence under the
same posture as the pin — transcript content, no new disclosure class.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from plurai.orchestrator.models import TurnWorld

#: The file's name inside a bundle folder.
ISSUE_TURN_FILENAME = 'issue_turn.json'


class IssueTurnWorld(BaseModel):
    """The recorded world through the issue turn: the run's captured entity ``types``, their rows at the
    start (``initial``, by entity id), and each turn's ``turn_world`` up to and including the cut.

    The plugin's recorder copies it from the run; the backend reads
    exactly these three keys and folds them to grade the reply. It never seeds a build."""

    model_config = ConfigDict(extra='forbid', strict=True)

    types: list[str]
    initial: dict[str, dict[str, Any]]
    turn_worlds: list[TurnWorld]


class IssueTurnDocument(BaseModel):
    """Three required keys and two optional ones, and nothing else."""

    model_config = ConfigDict(extra='forbid', strict=True)

    #: The turn the issue is ON, in the numbering the run's judges and ``--cut-turn`` use.
    turn_index: Annotated[int, Field(ge=0)]
    #: What the user said on that turn; ``None`` when the agent opened the conversation.
    user_message: str | None
    #: The agent's recorded reply — the bad output the owning check must fail.
    agent_reply: Annotated[str, Field(min_length=1)]
    #: The turn's recorded agent-side chain (tool calls, tool results, the reply) as the run wrote it,
    #: when the turn carried one. A reply that claims an outcome is judged against these.
    turn_messages: list[dict[str, Any]] | None = None
    #: The recorded world through this turn, when the run captured any world types.
    world: IssueTurnWorld | None = None


def issue_turn_problem(path: Path) -> str | None:
    """Why ``path`` is not an issue turn the backend can replay, or ``None`` when it is."""
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return f'cannot be read as JSON ({exc})'
    try:
        document = IssueTurnDocument.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = '.'.join(str(part) for part in first['loc']) or '(document)'
        return f'is not an issue turn: {where}: {first["msg"]}'
    if not document.agent_reply.strip():
        return 'holds a blank agent_reply, so there is no recorded reply to replay'
    return None


__all__ = ['ISSUE_TURN_FILENAME', 'IssueTurnDocument', 'IssueTurnWorld', 'issue_turn_problem']
