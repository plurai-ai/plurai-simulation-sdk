"""THE reader of the backend's per-scenario ``scenario_verdict`` block — one owner, three readers.

The backend computes the scenario verdict at ``/finalize`` (and on ``score_run``) and rides it on
``policy_evaluation_results.scenario_verdict``. The SDK never derives that verdict itself: a
client-side rule such as "any policy violated ⇒ fail, else pass" scores an EMPTY policy set as a
pass, and a scenario whose only judge crashed as a clean pass. The client cannot see what the
backend can: which policy OWNS the scenario's goal, whether the goal-owning check reached a
verdict at all, and whether the measurement itself was clean. So this module reads the answer,
and every surface (``rollup.compute_rollup_metrics``, ``feedback.build_scenario_summary``,
``run_simulation._family_row``) reads it from here.

A LEAF at import time: stdlib only, no sibling observability imports, so ``rollup`` (which the sinks
import), ``feedback`` (which deliberately does not import ``rollup``) and ``run_simulation`` can all
depend on it without an import cycle. The pass/fail rule itself is :mod:`plurai.scoring`'s, imported
when a block is read.

THE STATUS is the whole point. ``passed``/``failed`` are the only two that say something about
the agent; the other three each name a DIFFERENT reason there is no verdict, and none of them
may ever be reported as a pass:

===============  ==================================================================
``passed``       the goal was achieved and the conversation rule passes it
``failed``       the backend says the goal was not achieved, a catastrophic or
                 failing-tier red, or the conversation rule (:mod:`plurai.scoring`)
                 fails it
``not_evaluated``  nothing owned the goal, or the owner reached no conclusion
``unclean``      the measurement itself is not trustworthy (a judge errored, a
                 guard raised, a required observation is missing)
``unknown``      no verdict block at all — an older backend, or a shape this
                 version does not understand
===============  ==================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

ScenarioStatus = Literal['passed', 'failed', 'not_evaluated', 'unclean', 'unknown']

#: The wire key the block rides under, on ``policy_evaluation_results``.
VERDICT_KEY: Final = 'scenario_verdict'

#: ``goal_source`` when the eval set declared NO goal owner. Such a scenario is excluded from the
#: "N of M goals achieved" denominator and counted as not-evaluated instead — a customer whose
#: eval set merely predates the goal owner must not read ``0 of 7`` on every dashboard.
GOAL_SOURCE_NONE: Final = 'none'

#: The goal owners the backend declares today. NOT a gate: an unrecognized source still names an
#: owner, so it stays measurable — it is REPORTED by name in :attr:`ScenarioVerdict.note` (and
#: from there into every sink comment) rather than silently swallowed or silently trusted.
KNOWN_GOAL_SOURCES: Final = frozenset({'deterministic', 'target_judge', GOAL_SOURCE_NONE, 'pipeline_error', 'agent_closed'})

#: OPTIONAL block key the backend sets when the situation a scenario's target is written against never arose,
#: so neither axis decided the goal. Absent on every other row.
TARGET_PREMISE_UNSHOWN_KEY: Final = 'target_premise_unshown'
#: The note such a not-evaluated row carries, in the reader's words.
TARGET_PREMISE_UNSHOWN_NOTE: Final = "the situation this scenario tests never came up in the conversation, so its goal wasn't measured"

#: Sentinel telling "the key is absent" from "the key is present and null". The distinction is
#: the whole difference between ``unknown`` and ``not_evaluated``: the backend STATING ``goal_achieved:
#: null`` is a measurement ("nothing concluded"), while a block with no ``goal_achieved`` key at all
#: is a shape this version does not understand, and reading it either way would be a guess.
_MISSING: Final = object()


@dataclass(frozen=True, slots=True)
class ScenarioVerdict:
    """One scenario's verdict, as the backend stated it — never as a consumer re-derived it."""

    status: ScenarioStatus
    #: The backend's tri-state goal answer. ``None`` is "not measurable", NEVER a pass.
    goal_achieved: bool | None
    #: Which check owned the goal, verbatim. ``None`` when there was no readable block.
    goal_source: str | None
    #: The conversation's weighted score (:mod:`plurai.scoring`) as a 0-1 rate; ``None`` when no stamped eval was
    #: graded. Independent of :attr:`status` — a catastrophic red leaves it high, so ``failed`` at 0.9 is normal.
    score: float | None
    #: Catastrophic policies this scenario violated, in the order the backend named them.
    catastrophic_violations: tuple[str, ...]
    #: EVERY violation ``/finalize`` FAILED the run on — the catastrophic set UNION the
    #: ``terminate_values`` blocking set. It is a SECOND axis, not a longer spelling of the
    #: first, and it diverges from the goal axis in both directions (see :func:`read_scenario_verdict`).
    #: Non-empty is the server's own "this run failed", and no surface may render it as a pass.
    decisive_violations: tuple[str, ...]
    #: The backend's own ``measurement_integrity.clean``. ``None`` = nobody said, or said it
    #: unreadably. Never read ``None`` as a positive cleanliness claim: an absent claim has
    #: never been checked, and :attr:`note` says so on every status.
    measurement_clean: bool | None
    #: Why this status, in one line. Rides into every sink comment and the report, because a
    #: bare ``not_evaluated`` sends a reader hunting the wrong thing.
    note: str
    #: The backend's confident HIGH-severity reds beside the goal (never the goal owner, never a
    #: catastrophic one), verbatim. ``None`` when the block served no list — unknown, never ``()``.
    high_severity_violations: tuple[str, ...] | None

    @property
    def conclusive(self) -> bool:
        """Did this scenario reach a verdict at all? (``passed`` or ``failed``.)

        Note this is a STATUS question, so a row failed on ``decisive_violations`` alone counts
        — the server decided it, even though the goal axis had nothing to say.
        """
        return self.status in ('passed', 'failed')

    @property
    def measurable(self) -> bool:
        """Does this scenario belong in the "N of M goals achieved" DENOMINATOR?

        True iff something owned the goal. A scenario whose eval set declared no owner is not a
        failure to achieve a goal — there was no goal to achieve — so it leaves the denominator
        and is counted beside it as not-evaluated.

        The headline that divides by this lives in the plugin's run report, the only surface
        with a scenario-set-wide denominator; this property is the SDK half of that one rule.
        """
        return self.goal_source is not None and self.goal_source != GOAL_SOURCE_NONE


def _unwrap(policy_results: dict[str, Any] | None, key: str) -> Any:
    """``policy_results[key]``, tolerating the nested ``aggregated_results`` envelope.

    The same dual-envelope rule ``rollup.eval_results`` applies to ``evaluation_results``: some
    callers hold the flat policy-results mapping and some hold it wrapped. Looking in one place
    only would read a present block as absent, which lands as ``unknown`` — a silent downgrade
    of a run that WAS measured.
    """
    if not isinstance(policy_results, dict):
        return _MISSING
    if key in policy_results:
        return policy_results[key]
    nested = policy_results.get('aggregated_results')
    return nested[key] if isinstance(nested, dict) and key in nested else _MISSING


def _integrity(policy_results: dict[str, Any] | None) -> tuple[bool | None, str | None]:
    """``(clean, unreadable)`` — the backend's ``measurement_integrity.clean``, and why not.

    Tri-state on purpose, and READ rather than re-derived — the backend joins the channels only it
    can see (an errored judge, a raising applicability guard, a crashed per-turn pass, a broken
    deterministic target check) into this one bool.

    THREE cases. ``clean`` absent is ``(None,
    None)``: nobody checked. ``clean`` a real bool is that bool. ``clean`` PRESENT but not a bool
    (``'false'``, ``0``, a mis-keyed shape) is ``(None, <description>)`` — the backend tried to
    say something this version cannot read, and reading it as silence sends it down the same
    path as "nobody checked", which ends in a pass. Every caller must treat an unreadable claim
    as NOT clean, the same way the sibling ``goal_achieved`` field is named by value and type rather
    than defaulted.
    """
    block = _unwrap(policy_results, 'measurement_integrity')
    if not isinstance(block, dict) or 'clean' not in block:
        return None, None
    clean = block['clean']
    if isinstance(clean, bool):
        return clean, None
    return None, f'measurement_integrity.clean is {clean!r} ({type(clean).__name__}), not a bool'


def measurement_clean(policy_results: dict[str, Any] | None) -> bool | None:
    """The backend's own ``measurement_integrity.clean``, or ``None`` when it did not say.

    ``None`` covers both "no claim" and "a claim this version cannot read"; :func:`_integrity`
    separates them, and :func:`read_scenario_verdict` treats the second as NOT clean. Never read
    ``None`` here as a positive cleanliness claim.
    """
    return _integrity(policy_results)[0]


def _names(raw: Any) -> tuple[str, ...]:
    """A published policy-name list as a tuple of strings, or ``()``. Total: the block is
    ``dict[str, Any]`` on the wire model, so nothing validated inside it."""
    return tuple(str(v) for v in raw) if isinstance(raw, list) else ()


def _scored(policy_results: dict[str, Any] | None) -> dict[str, Any]:
    """THE conversation rule's result over this payload (:func:`plurai.scoring.payload_result`): its score and its
    fail reasons, from the cells' stamped bands and the goal. Imported at call time: :mod:`plurai.scoring` reads
    states through :mod:`plurai.verdicts`, which imports this module."""
    from plurai import scoring  # noqa: PLC0415 - the one copy of the rule; a module-level import would cycle

    return scoring.payload_result(policy_results if isinstance(policy_results, dict) else {})


def read_scenario_verdict(policy_results: dict[str, Any] | None) -> ScenarioVerdict:
    """Read the backend's ``scenario_verdict`` block. Total — never raises, never guesses.

    The backend publishes the FACTS — ``goal_achieved`` (tri-state), ``goal_source``, the confident red lists
    (``catastrophic_violations``, ``decisive_violations`` = catastrophic UNION the ``terminate_values`` STOP set,
    ``high_severity_violations`` = failing-tier reds) — and no ``passed`` / ``score`` of its own. The
    pass/fail and the score come from THE conversation rule, :mod:`plurai.scoring` (weights and cutoff live only
    there), over the same payload's cells; the red lists still fail the row by themselves, so a payload whose cells
    carry no stamped band (an older backend) keeps its tier verdicts.

    STATUS PRECEDENCE, in order, each with the reason it outranks the next:

    1. ``unknown`` — no block, or a block whose ``goal_achieved`` is absent or a non-boolean OTHER THAN null (a
       stated ``null`` is rule 4; a missing field is a shape we cannot read). Inventing a pass for it is the
       false green the block exists to close, and inventing a FAILURE is the manufactured red on the other side.
    2. ``failed`` — the goal was not achieved, OR a red list is non-empty, OR the conversation rule fails it (a low
       score, a critical or failing-tier red). An achieved or unmeasured goal can read ``failed``; the note then
       names the check and says what the goal axis said. A stated red stands even when the measurement is
       unclean: uncleanliness is a reason not to trust a PASS, never a reason to erase a finding.
    3. ``unclean`` — ``measurement_integrity.clean`` is ``false``, OR present and unreadable. Outranks
       ``not_evaluated`` because a breakage is louder than a declaration, and ``passed`` because an unclean row is
       never a pass on any surface. An ABSENT claim does not reach here — nobody checked, which the note says.
    4. ``not_evaluated`` — ``goal_achieved: null``: nothing owned the goal, or the owner reached no conclusion,
       and nothing failed. Not a failure and not a pass.
    5. ``passed`` — the goal achieved, nothing failed, the measurement not stated unclean.

    :attr:`ScenarioVerdict.score` is the rule's weighted score as a 0-1 rate (``None`` when no stamped eval was
    graded).
    """
    clean, unreadable = _integrity(policy_results)
    block = _unwrap(policy_results, VERDICT_KEY)
    raw_goal = block.get('goal_achieved', _MISSING) if isinstance(block, dict) else _MISSING
    if not (raw_goal is None or isinstance(raw_goal, bool)):
        # Absent, or present but unreadable: name what was actually there. A closed set that
        # swallows what fell outside it is how a mis-keyed field becomes a silent default.
        if not isinstance(block, dict):
            seen = 'absent' if block is _MISSING else type(block).__name__
            note = f'no readable {VERDICT_KEY} block ({seen}) — this run was scored by a backend that does not publish one'
        elif raw_goal is _MISSING:
            note = f'{VERDICT_KEY}.goal_achieved is missing'
        else:
            note = f'{VERDICT_KEY}.goal_achieved is {raw_goal!r} ({type(raw_goal).__name__}), not a bool or null'
        return ScenarioVerdict(
            status='unknown',
            goal_achieved=None,
            goal_source=None,
            score=None,
            catastrophic_violations=(),
            decisive_violations=(),
            measurement_clean=clean,
            note=f'{note}; {unreadable}' if unreadable else note,
            high_severity_violations=None,
        )

    goal_achieved: bool | None = raw_goal
    goal_source = block.get('goal_source')
    goal_source = goal_source if isinstance(goal_source, str) and goal_source else GOAL_SOURCE_NONE
    violations = _names(block.get('catastrophic_violations'))
    decisive = _names(block.get('decisive_violations'))
    high = _names(block.get('high_severity_violations'))
    rule = _scored(policy_results)
    score = rule['score'] / 100.0 if rule['score'] is not None else None

    if goal_achieved is False or violations or decisive or high or rule['result'] == 'fail':
        status: ScenarioStatus = 'failed'
        note = f'goal not achieved (owner: {goal_source})'
        if goal_achieved is not False:
            # Failed beside an achieved or unmeasured goal, so "goal not achieved" would be false. Name the
            # check and what the goal axis said — never the owner's internal id.
            if goal_achieved is True:
                goal = 'the goal was achieved'
            elif isinstance(block.get(TARGET_PREMISE_UNSHOWN_KEY), str) and block[TARGET_PREMISE_UNSHOWN_KEY]:
                goal = f'the goal was not measured: {TARGET_PREMISE_UNSHOWN_NOTE}'
            else:
                goal = 'the goal was not measured'
            failing = high or tuple(rule['failing_failed'])
            if failing:
                note = f'high-severity check failed: {", ".join(failing)} ({goal})'
            else:
                note = f'the conversation failed the scoring rule: {", ".join(rule["fail_reasons"])} ({goal})'
        if violations or rule['critical_failed']:
            note = f'catastrophic violation: {", ".join(violations or rule["critical_failed"])}'
        if decisive and set(decisive) != set(violations):
            # The server failed the run on something the catastrophic set does not name — a
            # turn-scoped STOP floor. Saying only "goal not achieved" would send the reader to
            # the wrong policy, and saying nothing would render this row as a pass.
            note = f'the run was FAILED on: {", ".join(decisive)}'
            if goal_achieved is not False:
                note = f'{note} (the goal axis said goal_achieved={goal_achieved!r}; the server failed the run anyway)'
    elif clean is False or unreadable:
        status = 'unclean'
        note = unreadable or 'the measurement itself is not trustworthy (measurement_integrity.clean=false)'
        note = f'{note} — never read as a pass'
    elif goal_achieved is None:
        status = 'not_evaluated'
        if isinstance(block.get(TARGET_PREMISE_UNSHOWN_KEY), str) and block[TARGET_PREMISE_UNSHOWN_KEY]:
            # The backend did not let either axis decide: the situation the target is written against never
            # arose. The owner may well have answered, so "reached no conclusion" would be false.
            note = TARGET_PREMISE_UNSHOWN_NOTE
        else:
            note = (
                'the eval set declared no goal owner for this scenario'
                if goal_source == GOAL_SOURCE_NONE
                else f'the goal owner ({goal_source}) reached no conclusion'
            )
    else:
        status = 'passed'
        note = f'goal achieved (owner: {goal_source})'

    if goal_source not in KNOWN_GOAL_SOURCES:
        # A closed set reports what fell OUTSIDE it. Still measurable (it names an owner);
        # the reader is simply told this SDK did not recognize the name.
        note = f'{note}; unrecognized goal_source {goal_source!r}'
    if goal_source == GOAL_SOURCE_NONE and goal_achieved is not None:
        # The two axes contradict each other: nothing owned the goal, yet a verdict was stated.
        # The stated verdict is honoured (it is the more specific claim) and the oddity is named
        # — silently trusting one of two disagreeing fields is how a default gets introduced.
        note = f'{note}; goal_source says no owner yet goal_achieved={goal_achieved!r}'
    if unclassified := _names(block.get('unclassified_violations')):
        # The backend's OWN residue reporter: a confidently-violated cell whose tier could not be
        # ASKED about (a policy since removed from the set). It is not a red — nobody classified
        # it — but a verdict computed over an unclassified violation is not a complete one, and
        # this note is the only place a sink reader would ever see it.
        note = f'{note}; tier unclassified for: {", ".join(unclassified)}'
    if block.get('polarity_untyped'):
        # ABSENCE IS NOT FULFILMENT. Nothing was polarity-gated, so every
        # action-expecting policy graded this scenario under an assumption nobody typed.
        note = f'{note}; scenario polarity untyped — no policy was gated on it'
    if clean is None and not unreadable:
        # A pass over an UNSTATED integrity is not a checked pass. Said on every status, because
        # the reader of a sink comment cannot see which channels the backend joined.
        note = f'{note}; measurement integrity unstated (nobody checked)'
    if block.get('goal_disagreement'):
        # The backend only sends this when BOTH tiers were conclusive and disagreed, which is a
        # defect in the eval set — invisible unless it rides the note into the sinks.
        note = f'{note}; goal_disagreement={block["goal_disagreement"]}'

    return ScenarioVerdict(
        status=status,
        goal_achieved=goal_achieved,
        goal_source=goal_source,
        score=score,
        catastrophic_violations=violations,
        decisive_violations=decisive,
        measurement_clean=clean,
        note=note,
        high_severity_violations=_names(high) if isinstance(high := block.get('high_severity_violations'), list) else None,
    )
