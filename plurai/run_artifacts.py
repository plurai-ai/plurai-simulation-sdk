"""ONE reader for a run's ``transcript.json`` — the artifact every audit re-derives by hand.

The file carries SIX overlapping views of the same conversation (``user_messages``,
``chatbot_messages``, ``turns``, ``turn_count``, ``user_turn_cycles``,
``per_turn_user_step_info``), and a reader that projects them by hand gets a different answer
from each. Three traps in particular:

* ``chatbot_messages`` is the RICH chain when the orchestrator has one — tool-calling
  ``AIMessage``s **and** ``ToolMessage``s — and the plain per-turn replies otherwise. A reader
  that treats it as "what the agent said" folds tool RESULTS into the agent's words: a judge
  reading that sees the mock's payload as the agent's answer.
* ``turn_count`` is ``len(turns)`` by definition and ``user_turn_cycles`` is the
  server-authoritative ``/user-turn`` count. They differ by one whenever the run has a closing
  reply, which is most runs. Both stay in
  ``raw`` under their own names; nothing here collapses them into one number.
* ``per_turn_user_step_info`` is keyed by turn index as a STRING, so ``step_info[3]`` finds
  nothing and reads as "the backend sent none". :attr:`Transcript.step_info_by_turn` keys it by
  int.

ONE NAME, TWO THINGS. ``user_messages`` appears at two levels of this file and
they are NOT the same view: the top-level one is the flat list of what the user said, turn by
turn, and is what :func:`_user_texts` reads; ``turns[i].user_messages`` is ONE turn's user
content as a list of standard content PARTS, present only when that turn actually carried parts
(a turn whose content was a plain string has the key absent, not empty). A reader that reaches
for the nested one by the top-level spelling gets a list of parts where it expected a list of
turns.

Stdlib only, and every field DEGRADES: a truncated, half-written or foreign file loads as an
empty :class:`Transcript` rather than raising. A reader that crashes on a bad artifact is a
reader nobody puts in an audit script.

Idiom: :func:`plurai.observability.rollup.chatbot_messages_of` — best-effort projection over
whatever shape the artifact happens to carry, never a schema assertion.

ONE-SHOT (pipeline) TRANSCRIPTS. A transcript carrying ``scenario_shape: 'pipeline'`` records one
``invoke_pipeline`` call instead of a conversation: ``turns`` is ``[]`` and both counts are 0 —
which is ALSO what a dialog that died before its first turn looks like, so that key is the only
thing that tells them apart. Its hop rides ``pipeline_io``, and THE READ RULE is::

    pipeline_inputs  := transcript['injected_inputs']['initial_agent_context']
    pipeline_outputs := transcript['pipeline_io']['outputs']
    pipeline_error   := transcript['pipeline_io']['error']      # None ⇒ the hop returned

The inputs are NOT in ``pipeline_io``, and looking for them there is the one mistake this rule
exists to prevent: ``/finalize`` sends the recorded ``initial_agent_context`` as ``pipeline_inputs``
(a served document as its reference), so the injected-inputs block already IS the hop's input
record — equal by construction rather than by two writers agreeing. Both keys are ABSENT from a dialog transcript; neither is ever an empty placeholder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from plurai.content_parts import render_parts

#: Stamped into every transcript this SDK writes (``run_simulation._write_success``). An artifact
#: with NO stamp reads as 1. Bump only for a change a reader cannot absorb by tolerance.
#:
#: ONE message vocabulary. ``user_messages`` / ``chatbot_messages`` spell ``type`` with the
#: LangChain discriminator (:data:`LC_MESSAGE_TYPES`) in every branch, including the messages an
#: agent returns as DICTS: ``plurai/child.py`` normalizes those from whichever of ``type`` /
#: ``role`` they declare, leaving the rest of the dict untouched. A dict naming NEITHER (or a
#: role this SDK does not recognize) passes through as written, which is why :func:`_is_agent`
#: reads ``role`` as well.
#:
#: Additive or nullable fields do not bump the version, because an existing reader already
#: tolerates them: the nullable ``turns[].user_message`` (an agent-first run's turn 0 has no user
#: message and the row carries ``None`` rather than ``''``) and the optional
#: ``turns[].agent_actions`` (written only for a turn that carried actions; readers ``.get()`` it
#: and read absent as ``[]``). Absent ``agent_actions`` means either: this turn carried none, or the
#: producer could not observe its agent's actions. A version number cannot separate those; only a
#: typed channel marker could.
#:
#: The transcript holds NO verdicts. ``policy_evaluation_results`` and the ``eval`` rollup live in the
#: attempt's ``verdicts.json`` (:data:`VERDICTS_FILENAME`, written only by :mod:`plurai.verdicts`); the transcript
#: carries ``finalize_record``, the conversation facts the backend's re-score reads back (``dialog_outcome``,
#: ``user_sim_integrity``, …).
TRANSCRIPT_SCHEMA = 4

#: The message ``type`` vocabulary this SDK writes, by role. LangChain's own discriminator (what
#: ``BaseMessage.model_dump()`` emits and what ``convert_to_messages`` accepts) — a class name is
#: NOT accepted by that loader, which is why it is not written. READ, not merely documented:
#: ``run_simulation`` writes the two flat arrays through it and ``child._lc_message_type``
#: resolves an agent-supplied role against its values.
LC_MESSAGE_TYPES = {'user': 'human', 'agent': 'ai', 'tool': 'tool', 'system': 'system'}

# A turn's chain in ``transcript.json`` and the backend's ``agent_messages`` hold DIFFERENT things, and
# this is the one place that says so, because reading either as the other is a silent wrong answer:
#
# * HERE (``transcript.json``) each ``turns`` entry holds that turn's chain under
#   ``turn_messages``, as serialized DICTS: agent side only, but narration, tool calls and tool
#   results as well as the reply. No transcript key is named ``agent_messages``.
# * In the backend's eval sandbox ``agent_messages`` is a code_eval NAMESPACE bound to BaseMessage-LIKE
#   OBJECTS, role-filtered to the agent's replies with
#   ``content`` projected to rendered text. Policy expressions read ``.content`` there; the dict
#   idioms that work on this file (``m.get('content')``) raise on an object.
#
# Both of the backend's readers normalize a class-name ``type`` to the lowercase form.

#: What :attr:`Transcript.schema` reports when the artifact could not be loaded at all. NOT a
#: writer version: it is the value that keeps a path typo from reading as a healthy empty run.
SCHEMA_NOT_LOADED = 0

#: Where the SDK's own reading of the restore lands inside a resumed run's ``resume_fidelity``.
#: Namespaced so it cannot collide with the harness's freeform keys, and so a reader can tell the
#: two apart: every other key is the customer's claim about what it restored, this one is what the
#: run OBSERVED. Named here, beside the transcript's other reader-facing names, because the
#: orchestrator writes it and the trace converter reads it back off disk.
SEED_RESTORE_KEY = 'sdk_seed_restore'

#: Beside :data:`SEED_RESTORE_KEY`, in a RESUMED run's ``resume_fidelity``: the harness's ``seed_state``
#: returned a report that cannot be carried verbatim (``{'returned': <type name>}``). Any key but the
#: runner's own reading is a reduced restore to every reader, which is the conservative reading of a
#: report nobody can read.
SEED_REPORT_UNREADABLE_KEY = 'sdk_seed_report_unreadable'

#: The harness's own key inside ``resume_fidelity`` naming the seed's ``agent_state`` keys it had no
#: slot for and skipped. A skip, not a reduced restore: the conversation was restored whole, and a
#: seed cut from a run always carries keys the turn reported about itself. Mirrored by value in the
#: plugin's ``scripts/regressions.py``, which pins the two equal.
RESUME_STATE_DROPPED_KEY = 'resume_state_dropped'

#: The folder a resumed regression keeps a Claude Code CLI's own recorded session in, for the local
#: harness to resume. It holds the CLI's full session record, so no upload ever carries it
#: (:func:`plurai.ingest.cli.bundle_units`). Mirrored by value in the plugin's ``scripts/regressions.py``.
CLI_SESSION_DIRNAME = 'cli_session'

#: The folder-name prefix of a recorded regression scenario (the plugin's recorder,
#: ``regressions.py`` ``FOLDER_PREFIX``). One spelling for the three readers that key on it: the
#: ingest folder-name pass-through, the bundle walk, and :mod:`plurai.trials`' replay rates.
REGRESSION_SCENARIO_PREFIX = 'regression_'

#: The record-time rate the plugin writes INTO a recorded regression folder (``regressions.py
#: confirm``), from :func:`plurai.trials.record_time_rate`. It describes the folder and is never one
#: of its records: :func:`plurai.ingest.cli.bundle_units` never uploads it.
REGRESSION_REPLAYS_FILENAME = 'replays.json'

#: A SINGLE-HOP regression replay (``plurai-run-simulation --replay-turn-only``): the ``ended_early``
#: reason the run sends on ``/finalize`` when it stopped by design after the turn under test, and the
#: ``not_measured_cause`` the backend stamps on an end-of-run cell that abstained because of that cut. Both
#: are the backend's (its client-contract ``replay_cut_vocabulary`` block; a parity check holds the
#: two equal), spelled once here for the two readers: the orchestrator
#: and :mod:`plurai.trials`' replay rates.
REPLAY_CUT_REASON = 'replay_cut'
CUT_AT_ISSUE_TURN_CAUSE = 'cut_at_issue_turn'
#: The block's ``trigger_absent``: the ONLY two N/A shapes a single-hop replay reads as "not caught" — every
#: ``guard_suppressions`` record the backend wrote for the policy has this ``kind`` (its guard read the replayed
#: turn and found no trigger), or its cell / turn entry has exactly this ``inconclusive_reason`` (its judge
#: said the premise did not arise). Any other N/A measured nothing.
TRIGGER_ABSENT_GUARD_KIND = 'false'
TRIGGER_ABSENT_REASON = 'not_applicable'

#: How one conversation attempt was JUDGED (:mod:`plurai.verdicts`, its ONE writer): the only place a verdict is
#: read from. Every attempt writes one — a FAILED one too, beside its ``error.json``, and that one leaves with the
#: error record.
VERDICTS_FILENAME = 'verdicts.json'

#: The files ONE completed scenario attempt writes into its own folder. Named here, in the module
#: that owns reading them, because three unrelated places act on this exact list — the runner
#: writes it, the runner's re-run hygiene displaces it, and ``plurai ingest-run`` decides from it
#: whether a folder is uploadable — and a private copy per site is how they drift.
#:
#: ``pipeline_io.json`` is written only by a ONE-SHOT (pipeline) scenario and is listed here
#: anyway, because this tuple is the DISPLACEMENT contract rather than an inventory of what every
#: attempt writes: a name missing from it is a file the re-run hygiene never stages, never
#: displaces into ``previous_attempt/`` and never calls stale — so a hop whose second attempt
#: failed would leave the FIRST attempt's outputs at the top level beside a fresh transcript,
#: and the folder would read as one attempt while describing two.
#:
#: ``verdicts.json`` is listed because it moves, uploads and goes stale with the attempt it judged.
SCENARIO_ARTIFACT_FILES: tuple[str, ...] = ('transcript.json', 'final_world_state.json', 'tool_events.json', 'pipeline_io.json', VERDICTS_FILENAME)

#: The turns a FAILED attempt reached, written beside its ``error.json``. Evidence, never a
#: measurement: not ``transcript.json`` (every scored-row reader keys on that name) and not in the
#: tuple above, which is also the upload list — so the row stays ``error`` everywhere it is read.
ERROR_DIALOG_FILENAME = 'error_dialog.json'

#: ONE error record in two files: whatever drops a stale ``error.json`` drops both together.
ERROR_RECORD_FILES: tuple[str, ...] = ('error.json', ERROR_DIALOG_FILENAME)

#: The names in the displacement contract above that only SOME shapes write — the whole of the
#: difference between that tuple and the required one below, stated once so the two cannot drift.
SHAPE_CONDITIONAL_ARTIFACT_FILES: tuple[str, ...] = ('pipeline_io.json',)

#: The subset EVERY completed attempt writes WHATEVER ITS SHAPE — i.e. the names whose absence is
#: worth telling an operator about. The sibling above is the DISPLACEMENT contract and deliberately
#: a superset: it must name every file an attempt COULD leave behind, or the re-run hygiene strands
#: one. Read as an inventory it says something false about a dialog folder, which has never written
#: ``pipeline_io.json`` and never will — and that difference is exactly what the two names are for.
#:
#: DERIVED from that superset minus the shape-conditional names, never a second hand-list — the
#: same posture ``SILENT_PART_KINDS`` takes towards ``PROVIDER_BLOCK_KINDS``
#: (:mod:`plurai.content_parts`). Two literal tuples sharing three names drift in EITHER
#: direction: the next shape-conditional artifact added to both reports a dialog folder as
#: incomplete, and a name dropped from this one silently stops being reported on every folder.
REQUIRED_ARTIFACT_FILES: tuple[str, ...] = tuple(name for name in SCENARIO_ARTIFACT_FILES if name not in SHAPE_CONDITIONAL_ARTIFACT_FILES)

#: What a FAILED dialog attempt leaves beside its ``error.json``: the conversation and the tool
#: events it had produced when it failed. ``error.json`` stays the record of the failure; these
#: are the evidence behind it.
#:
#: Deliberately NOT ``transcript.json`` / ``tool_events.json``. Their PRESENCE is read as "this
#: scenario ran to the end" by the plugin's regression picker (a folder holding a
#: ``transcript.json`` is a gradeable candidate), by its run builder (it counts the tool calls in
#: any ``tool_events.json``) and by ``plurai ingest-run`` (it uploads a ``tool_events.json`` beside
#: an ``error.json``), so those names would turn a failed attempt into a measured one. For the
#: same reason they stay out of :data:`SCENARIO_ARTIFACT_FILES`: never uploaded, never displaced
#: into ``previous_attempt/``, and removed before the next attempt writes into the same folder, or
#: from ``stability_retry/`` together with that folder's ``error.json``.
PARTIAL_TRANSCRIPT_FILE = 'transcript.partial.json'
PARTIAL_TOOL_EVENTS_FILE = 'tool_events.partial.json'
PARTIAL_EVIDENCE_FILES: tuple[str, ...] = (PARTIAL_TRANSCRIPT_FILE, PARTIAL_TOOL_EVENTS_FILE)

#: Where a scenario folder keeps the artifacts of the attempt a RE-RUN superseded.
#:
#: The superseded attempt is moved here instead of deleted, so a re-run that crashes (or is
#: killed, or is still in flight) cannot strip the folder bare and destroy a measurement already
#: taken. The top level describes exactly ONE attempt, which keeps a fresh ``error.json`` from
#: sitting beside a stale ``transcript.json``.
PREVIOUS_ATTEMPT_DIRNAME = 'previous_attempt'

#: Where a scenario folder keeps the STABILITY RETRY's artifacts.
#:
#: Its own directory, and never the top level, because the retry must not become the answer.
#: Retrying only FAILURES is statistically biased — passes are never re-tested, so "pass if
#: either attempt passed" strictly inflates a suite score — so the first attempt stays
#: authoritative and the second exists to say whether the failure was STABLE or FLAKY. Writing
#: it beside the first would make the folder describe two attempts at once, which is the exact
#: condition ``PREVIOUS_ATTEMPT_DIRNAME`` above exists to prevent.
#:
#: It never holds an ``error.json``, for the same reason ``previous_attempt/`` never does: the
#: documented failure sweep is ``find … -name error.json``, and a retry that also failed would
#: otherwise be counted as a second failed scenario.
STABILITY_RETRY_DIRNAME = 'stability_retry'

#: Trial ``k >= 2`` of a ``--trials N`` run is a complete run of its own at ``$OUT/trial_<k>/``;
#: trial 1 IS ``$OUT``. Readers enumerate trial directories ONLY from ``run_summary.json`` →
#: ``trials.dirs`` (``plurai.trials``), never by globbing this prefix: a directory an older run left
#: behind is not one of this run's trials.
TRIAL_DIR_PREFIX = 'trial_'

#: The run-LEVEL file (one per ``$OUT``, beside the scenario folders, NOT inside one): the run's
#: own machine-readable verdict — counts, ``scored``, ``ended_early_scenarios``,
#: ``measurement_clean`` / ``degraded_scenarios`` / ``measurement_integrity``. Written by
#: ``run_simulation._write_run_summary``.
#:
#: Named HERE, in the module both the runner and ``plurai ingest-run`` already read their other
#: artifact names from, for the same reason as the two above: the writer and the uploader must
#: agree on the filename by construction. The Plurai backend stores the uploaded file under the same
#: name, so the filename is part of the upload contract.
RUN_SUMMARY_FILENAME = 'run_summary.json'


def declared_message_type(message: Any) -> str:
    """The message's declared type/role, normalized ('' when it names none).

    The ONE spelling-resolution kernel: the object form reads ``.type`` (a live LangChain
    ``BaseMessage``), the dict form reads ``type`` then ``role``, and a class-name spelling
    (``AIMessage``) loses its suffix. Shared by :func:`_is_agent` and by ``child.py``'s role
    reader and type stamper so the sites cannot drift on what a spelling means. The
    orchestrator's wire stamp stays OUT deliberately: it preserves the agent's own spelling
    verbatim, and normalizing there would rewrite what transcripts record.
    """
    declared = getattr(message, 'type', None)
    if declared is None and isinstance(message, dict):
        declared = message.get('type') or message.get('role')
    return str(declared or '').strip().lower().removesuffix('message')


#: NORMALIZED message types that are NOT the agent speaking — membership is checked on
#: :func:`declared_message_type` output only, so ``ToolMessage`` / ``Tool`` / ``function``
#: all resolve here. ``tool`` is the load-bearing one: it is the mock's payload, and it
#: lives in the same array as the agent's own turns.
_NON_AGENT_TYPES = frozenset({'tool', 'function', 'human', 'user', 'system'})


def _text_of(message: Any) -> str | None:
    """The human-readable text of one message, or None when the shape is not understood.

    The distinction is load-bearing: ``''`` means "this message was genuinely empty" and None
    means "I could not read this one". Collapsing them would let an unrecognized content shape (a
    list of bare strings, a dict) be DROPPED from ``agent_texts`` as though it were an empty turn —
    a real agent answer vanishing with no trace, which is the loss this module prevents.
    """
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return None
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # A STRUCTURALLY recognisable part list goes through the one projection this SDK shares
        # with the backend: every renderable part in order, and a kind this build cannot render
        # NAMED rather than dropped. Keeping only ``type == 'text'`` blocks would turn a ``json``
        # part — the shape a structured reply arrives as — into ``''`` and the agent's data
        # answer would reach the judges as silence.
        if all(isinstance(b, dict) and 'type' in b for b in content):
            return render_parts(content)
        # A list this reader cannot structurally recognise is UNREADABLE, not empty, and that
        # distinction matters: text blocks if any are salvageable, else None so the caller
        # labels the text a guess instead of the agent's words.
        texts = [t for b in content if isinstance(b, dict) and b.get('type') == 'text' and isinstance(t := b.get('text'), str)]
        return ''.join(texts) if texts else None
    if content is None and 'content' not in message:
        return None
    return ''


def _is_agent(message: Any) -> bool:
    """Is this the AGENT speaking?

    ``role`` is the other spelling a serialized message carries. ``plurai/child.py`` stamps a
    ``type`` onto an agent's dict message when it declares a role this SDK recognizes, but a dict
    naming neither passes through as written. Reading ``type`` alone would let the mock's ``role='tool'`` payload AND the
    simulated user's ``role='user'`` words land in ``agent_texts``, which is the trap this module
    exists to close (``orchestrator.py`` carries the same rule for the same
    reason). A ``tool_call_id`` is a tool result whatever it calls its role.
    """
    if isinstance(message, str):
        return True
    if not isinstance(message, dict):
        return False
    if message.get('tool_call_id') is not None:
        return False
    return declared_message_type(message) not in _NON_AGENT_TYPES


@dataclass(frozen=True)
class Transcript:
    """One run's conversation, normalized. Every field degrades to empty, never to a raise."""

    #: What the simulated user said, in order.
    user_texts: list[str] = field(default_factory=list)
    #: What the AGENT said, in order — ToolMessages EXCLUDED. This is the field a judge or an
    #: audit means by "the agent's answers", and the one the raw array does not give you.
    agent_texts: list[str] = field(default_factory=list)
    #: The turn-ordered log (``turns``), each entry as the artifact carries it.
    turns: list[dict] = field(default_factory=list)
    #: Per-turn user-simulator step snapshots, keyed by INT turn index (the artifact keys them
    #: with strings, so ``step_info[3]`` on the raw dict finds nothing).
    step_info_by_turn: dict[int, dict] = field(default_factory=dict)
    #: ``{reply_source: turns}`` over this transcript — how many replies were the agent's own
    #: words, a tool result a graph ended on, a typed interrupt, an empty turn, or a shape the
    #: classifier could not read (``plurai.child.REPLY_SOURCES``). A turn that carries no label
    #: counts under ``unlabelled``, never under ``assistant``: this is the projection a
    #: run-level summary or an audit sums, and it must not report a producer's silence as a claim.
    reply_sources: dict[str, int] = field(default_factory=dict)
    #: The artifact's ``transcript_schema``; 1 when unstamped, and **0 when nothing was loaded**
    #: (missing / unreadable / not a JSON object). 0 is what distinguishes "a path typo" from
    #: "a healthy empty run of the current shape".
    schema: int = TRANSCRIPT_SCHEMA
    #: The parsed file, verbatim — for anything this projection does not cover.
    raw: dict = field(default_factory=dict)


def load_transcript(path: str | Path) -> Transcript:
    """Read ``transcript.json`` into a :class:`Transcript`. Never raises."""
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8-sig', errors='replace'))
    except (OSError, ValueError):
        return Transcript(schema=SCHEMA_NOT_LOADED, raw={})
    if not isinstance(raw, dict):
        return Transcript(schema=SCHEMA_NOT_LOADED, raw={})

    user = raw.get('user_messages')
    user_texts = [_text_of(m) or '' for m in user] if isinstance(user, list) else []

    chatbot = raw.get('chatbot_messages')
    agent_msgs = [m for m in chatbot if _is_agent(m)] if isinstance(chatbot, list) else []
    # A tool-calling AIMessage carries an empty text beside its tool_use; keeping it would put
    # blank "answers" in the list a judge reads. So an empty text is dropped ONLY when the message
    # is one of those — never merely because it read as empty, which would also swallow a shape
    # this reader does not understand (see _text_of) and a genuinely empty agent turn, itself a
    # signal ("did the agent answer at all?").
    # An unreadable shape keeps its SLOT as '' rather than vanishing: `len(agent_texts)` has to
    # stay the number of agent turns, or "did the agent answer at all?" cannot be asked of it.
    # The ONLY thing dropped is a tool-calling message with no prose, which would otherwise put
    # blank "answers" in the list a judge reads.
    agent_texts = [
        text or ''
        for text, msg in ((_text_of(m), m) for m in agent_msgs)
        if text != '' or not (isinstance(msg, dict) and (msg.get('tool_calls') or msg.get('tool_use')))
    ]

    turns = raw.get('turns')
    turns = [t for t in turns if isinstance(t, dict)] if isinstance(turns, list) else []

    step_info: dict[int, dict] = {}
    declared = raw.get('per_turn_user_step_info')
    if isinstance(declared, dict):
        for key, value in declared.items():
            if not isinstance(value, dict):
                continue
            try:
                step_info[int(key)] = value
            except (TypeError, ValueError):
                continue

    schema = raw.get('transcript_schema')
    schema = schema if isinstance(schema, int) and not isinstance(schema, bool) else 1

    return Transcript(
        user_texts=user_texts,
        agent_texts=agent_texts,
        turns=turns,
        step_info_by_turn=step_info,
        reply_sources=reply_source_counts(turns),
        schema=schema,
        raw=raw,
    )


def reply_source_counts(turns: list[dict]) -> dict[str, int]:
    """``{reply_source: turns}`` over a transcript's ``turns``.

    Its own function, and public, because the run-LEVEL question ("how many replies across this
    whole run were not the agent's own words") is asked by a different caller from the one that
    loads a transcript, and both must count it the same way. An unlabelled turn is its own bucket:
    a turn with no label says nothing, and silence is never read as ``assistant``.
    """
    counts: dict[str, int] = {}
    for turn in turns:
        label = turn.get('reply_source') if isinstance(turn, dict) else None
        key = str(label) if isinstance(label, str) and label else 'unlabelled'
        counts[key] = counts.get(key, 0) + 1
    return counts
