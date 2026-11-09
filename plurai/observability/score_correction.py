"""Overwrite an eval's recorded observability score with its corrected verdict.

A final-run eval cell that the audit ruled a false red (or a rescore re-judged) is corrected after the run, in the
conversation's ``verdicts.json`` (:mod:`plurai.verdicts`). The
correction reaches the observability platform as an OVERWRITE of the score that run posted,
never a second score beside it. The corrected cell is scored through the SAME mapping a run
uses (:func:`~plurai.observability.eval_mapping.map_eval_score` /
:func:`~plurai.observability.eval_mapping.per_turn_results`), so the overwritten score is
exactly what the run would have posted for that cell. Only per-eval scores are overwritten;
the scenario rollups stay as the backend produced them.

A sink takes part by implementing :class:`ScoreOverwriteSink` (Langfuse and LangSmith do);
any other active sink answers ``unsupported``. Nothing here raises to the caller: a platform
error is an outcome, because the correction is already in the report either way.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from plurai.observability.eval_mapping import TypedScore, eval_scope, map_eval_score, per_turn_results

logger = logging.getLogger(__name__)

OverwriteStatus = Literal['written', 'not_recorded', 'unsupported', 'failed']


class CorrectedEval(BaseModel):
    """One corrected eval: the policy name and its corrected cell, in the backend PolicyResult shape."""

    model_config = ConfigDict(frozen=True)

    policy: str
    cell: dict[str, Any]


class ScoreOverwrite(BaseModel):
    """One thread's corrected evals (:func:`overwrites_from_verdicts`).

    No run or dataset name is needed to find a score: Langfuse looks it up by the thread's session,
    LangSmith derives its ids from the thread id.
    """

    model_config = ConfigDict(frozen=True, extra='ignore')

    thread_id: str
    evals: list[CorrectedEval]


class ScoreOverwriteOutcome(BaseModel):
    """What happened to one eval's score on one thread.

    ``written``: every score the run posted for it that was found now carries the corrected
    verdict. ``not_recorded``: the platform holds no score for it (observability was off for
    that run, or the run predates overwritable scores). ``unsupported``: the active sink cannot
    overwrite. ``failed``: the platform refused or was unreachable; ``detail`` says why.
    """

    model_config = ConfigDict(frozen=True)

    thread_id: str
    name: str
    status: OverwriteStatus
    detail: str = ''


@runtime_checkable
class ScoreOverwriteSink(Protocol):
    """A sink that can overwrite the eval scores a run posted. Must never raise."""

    def overwrite_scores(self, items: Sequence[ScoreOverwrite]) -> list[ScoreOverwriteOutcome]: ...


def overwrites_from_verdicts(docs: Iterable[Mapping[str, Any]]) -> list[ScoreOverwrite]:
    """The scores to overwrite, off conversations' ``verdicts.json`` documents: per graded attempt with a thread, every
    eval whose current state differs from the run's own (``history[0].states``) and that still has a cell to score.
    An attempt nothing changed on contributes nothing."""
    items: list[ScoreOverwrite] = []
    for doc in docs:
        thread_id = (doc.get('attempt') or {}).get('thread_id')
        history = doc.get('history') or [{}]
        at_run: Mapping[str, Any] = history[0].get('states') or {}
        evals: Mapping[str, Mapping[str, Any]] = doc.get('evals') or {}
        changed = [
            CorrectedEval(policy=name, cell=entry['cell'])
            for name, entry in sorted(evals.items())
            if entry.get('state') != at_run.get(name) and isinstance(entry.get('cell'), dict)
        ]
        if doc.get('graded') and isinstance(thread_id, str) and thread_id and changed:
            items.append(ScoreOverwrite(thread_id=thread_id, evals=changed))
    return items


def eval_score_placements(policy: str, cell: dict[str, Any]) -> list[tuple[int | None, TypedScore]]:
    """The scores a run posts for one eval cell, each with the 0-based turn it sits on (None = scenario).

    The placement every sink uses at run time: a turn-scoped cell with a per-turn stream posts
    one score per turn entry; any other cell posts its rolled-up verdict once at scenario level.
    """
    per_turn = per_turn_results(policy, cell) if eval_scope(cell) == 'turn' else []
    if per_turn:
        return [(ts.turn_idx, ts.score) for ts in per_turn]
    return [(None, map_eval_score(policy, cell))]


def eval_outcome(thread_id: str, policy: str, *, written: int, failures: Sequence[str]) -> ScoreOverwriteOutcome:
    """Fold one eval's per-score results into its outcome: any failure wins, then any write."""
    if failures:
        return ScoreOverwriteOutcome(thread_id=thread_id, name=policy, status='failed', detail=failures[0])
    return ScoreOverwriteOutcome(thread_id=thread_id, name=policy, status='written' if written else 'not_recorded')


def every_eval(items: Sequence[ScoreOverwrite], status: OverwriteStatus, detail: str) -> list[ScoreOverwriteOutcome]:
    """The same outcome for every eval of every item."""
    return [ScoreOverwriteOutcome(thread_id=item.thread_id, name=ev.policy, status=status, detail=detail) for item in items for ev in item.evals]


def overwrite_scores(items: Sequence[ScoreOverwrite], sink: object) -> list[ScoreOverwriteOutcome]:
    """Overwrite each corrected eval's score on ``sink``; one outcome per (thread, eval). Never raises."""
    if not isinstance(sink, ScoreOverwriteSink):
        return every_eval(items, 'unsupported', f'the {getattr(sink, "name", type(sink).__name__)} sink cannot overwrite scores')
    try:
        return sink.overwrite_scores(items)
    except Exception as exc:
        logger.exception('Score overwrite failed on the %s sink.', getattr(sink, 'name', type(sink).__name__))
        return every_eval(items, 'failed', f'{type(exc).__name__}: {exc}')


__all__ = [
    'CorrectedEval',
    'OverwriteStatus',
    'ScoreOverwrite',
    'ScoreOverwriteOutcome',
    'ScoreOverwriteSink',
    'eval_outcome',
    'eval_score_placements',
    'every_eval',
    'overwrite_scores',
    'overwrites_from_verdicts',
]
