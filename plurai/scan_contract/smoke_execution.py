"""Fire the derived scan-time smoke probes at the LIVE agent, and record what came back.

:mod:`plurai.scan_contract.smoke_probes` derives the asks; this module is the only code in the
package that sends one anywhere. Three properties are the whole design:

**Consent gates execution, not derivation.** Without consent every probe is recorded as
``not_run`` with that reason. The report then shows the asks that were withheld, which is what a
customer granting consent later needs to see.

**Read-only asks only, and the guard is a REFUSER — not a prover.** A probe fires before anyone
has approved a run, so an ask that instructs the agent to DO something would create a side effect
on the customer's real systems. :func:`read_only_refusal` refuses the recognizable mutating
shapes (an imperative clause opening on a state-changing verb) and clears everything else; it
cannot prove an arbitrary sentence is read-only, and says so rather than implying a proof. Its
bias is deliberate: a false refusal costs one unfired probe, recorded with its reason, while a
false clear fires a mutation nobody approved.

**Its vocabulary is PER LANGUAGE, and where there is none it says which.** A guard that reads
English only must refuse every ask it cannot read as English, because silently clearing a Hebrew
"delete my account" is how a mutation gets fired. Applied to a non-English agent, that refuses
EVERY author-supplied ``probe_ask`` and EVERY derived ask (built from ``fact.claim``, in the
corpus's language), so no probe runs. The tables are :data:`MUTATING_VERBS_BY_LANGUAGE` /
:data:`POLITE_OPENERS_BY_LANGUAGE`, and where a language has no table the refusal NAMES that — a
permanent silent refusal and a named, extensible unsupported state are the same behaviour with two
different futures.

**Not-run is never a pass.** Every probe ends with a finding — answered, or not-run with the
reason. A shorter findings list than probe list, or a not-run probe rendering like a clean one,
is the coverage theatre this package exists to remove.

**No verdict is reached here.** The package calls no model, so a reply is EVIDENCE and nothing
more. Whether a fabrication probe's reply invented the absent thing is decided downstream (the
absence probe's own compliance-form judge already exists — :meth:`probes.AbsenceProbe.as_policy`),
never by string-matching a reply here.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess  # noqa: S404 - the agent transport is a customer-supplied command; see subprocess_transport
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from plurai.scan_contract.facts import dominant_scripts, scripts_present
from plurai.scan_contract.smoke_probes import (
    DEFAULT_SMOKE_BUDGET,
    FINDING_ANSWERED,
    FINDING_GUARD_UNSUPPORTED,
    FINDING_NOT_RUN,
    ProbeFinding,
    SmokeProbe,
    SmokeProbeSet,
)

if TYPE_CHECKING:
    from plurai.scan_contract.report import PreGenerationReport

#: One ask in, the agent's reply out. Deliberately the whole transport contract: the backend does not
#: own the customer's agent handle — the scan runs in their repo, against their process — so the
#: caller supplies the two-line adapter and this module owns the gate, the budget and the record.
ProbeTransport = Callable[[str], str]

#: How much of a reply is kept as evidence. Replies are transcripts, so the cap is generous; it
#: exists because a probe artifact is read by humans and can be committed, not to save bytes.
MAX_REPLY_CHARS = 4000

#: Marker appended to a truncated reply. Truncation is never silent: a reply that was cut and one
#: that ended there must not read alike when someone judges fabrication off the transcript.
TRUNCATION_MARKER = '\n…[reply truncated by the scan-time probe executor]'

#: Default seconds one probe may take before the transport is abandoned. A probe is one cheap
#: turn; a transport that hangs must not hold the scan open.
DEFAULT_PROBE_TIMEOUT = 120.0

#: Generic verbs whose imperative form asks the agent to CHANGE something, PER LANGUAGE. Generic by
#: construction — no agent-derived word appears here, so the guard is the same rule for every
#: customer. Not exhaustive in any language, and cannot be: see :func:`read_only_refusal`.
#:
#: One table per language, rather than one merged set, so that a language this guard has NO
#: vocabulary for is nameable: the refusal can say which table is missing instead of telling the
#: author to write English.
#:
#: The backend's shared text-primitives leaf is NOT imported for the script work these tables need:
#: this package is VENDORED into the client SDK as stdlib-self-contained, and the dependency runs
#: one way — that leaf imports from :mod:`plurai.scan_contract.facts`, never the reverse. The
#: script helpers come from this package's own ``facts`` module instead.
MUTATING_VERBS_BY_LANGUAGE: dict[str, frozenset[str]] = {
    'english': frozenset(
        {
            'activate',
            'add',
            'approve',
            'archive',
            'assign',
            'book',
            'buy',
            'cancel',
            'change',
            'close',
            'commit',
            'create',
            'deactivate',
            'delete',
            'deploy',
            'disable',
            'downgrade',
            'drop',
            'edit',
            'email',
            'enable',
            'erase',
            'grant',
            'install',
            'invite',
            'issue',
            'merge',
            'modify',
            'move',
            'order',
            'pay',
            'place',
            'post',
            'publish',
            'purchase',
            'push',
            'register',
            'reject',
            'remove',
            'rename',
            'reschedule',
            'reserve',
            'reset',
            'restore',
            'revoke',
            'save',
            'schedule',
            'send',
            'set',
            'share',
            'sign',
            'submit',
            'subscribe',
            'transfer',
            'truncate',
            'uninstall',
            'unsubscribe',
            'update',
            'upgrade',
            'upload',
            'wipe',
            'write',
        }
    ),
    # The Latin-script tables below are read together for a Latin-script ask; see MUTATING_VERBS.
    'spanish': frozenset(
        {
            'actualiza',
            'actualizar',
            'apaga',
            'apagar',
            'borra',
            'borrar',
            'cancela',
            'cancelar',
            'crea',
            'crear',
            'elimina',
            'eliminar',
            'envia',
            # The ACCENTED spelling, which is how the word is actually written. The unaccented
            # twin above covers text that dropped the accent.
            'envía',
            'enviar',
            'publica',
            'publicar',
        }
    ),
    'french': frozenset({'annule', 'annuler', 'efface', 'envoie', 'envoyer', 'supprime', 'supprimer'}),
    'italian': frozenset({'cancella', 'crea', 'elimina', 'invia'}),
    'german': frozenset({'entferne', 'loesche'}),
    'portuguese': frozenset({'apaga', 'apagar', 'cancela', 'cancelar', 'envia', 'enviar', 'excluir', 'publica', 'publicar'}),
}

#: The union of every table. It is this module's published vocabulary and it IS the effective table
#: for a Latin-script ask: no test separates Spanish from English prose by script, so a Latin ask is
#: read against every Latin table at once.
MUTATING_VERBS: frozenset[str] = frozenset().union(*MUTATING_VERBS_BY_LANGUAGE.values())

#: Openers that carry no meaning of their own and can precede the verb of an imperative request
#: ("please cancel it", "can you cancel it", "go ahead and cancel it"), PER LANGUAGE. Stripped
#: repeatedly from the head of a clause before the first word is tested, longest first so "could
#: you please" is consumed as one step rather than leaving "please" to a second.
#:
#: An opener table only ever HELPS the guard reach the verb, so a language whose openers are absent
#: degrades to "the verb had better be first" rather than to a wrong verdict \u2014 which is why the
#: non-English entries here are short and honest rather than a translation exercise.
POLITE_OPENERS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    'english': (
        'i would like you to please',
        'i would like you to',
        "i'd like you to",
        '\u2019d like you to',
        'i would like to',
        "i'd like to",
        'would you mind',
        'do you mind',
        'proceed to',
        'make sure to',
        'be sure to',
        'immediately',
        "let's",
        '\u2019s',
        'lets',
        'could you please',
        'would you please',
        'can you please',
        'i want you to',
        'i need you to',
        'go ahead and',
        'please could you',
        'i would like',
        'could you',
        'would you',
        'can you',
        'i want to',
        'i need to',
        'please',
        'now',
        'then',
        'also',
        'kindly',
        'just',
    ),
    'spanish': ('por favor', 'me gustaria que', 'necesito que', 'quiero que', 'podrias', 'puedes', 'ahora'),
    'french': ('pourriez-vous', "s'il te plait", "s'il vous plait", 'peux-tu', 'pouvez-vous', 'veuillez', 'maintenant'),
    'italian': ('per favore', 'potresti', 'potrebbe', 'vorrei che', 'puoi', 'adesso'),
    'german': ('bitte', 'koenntest du', 'kannst du', 'jetzt'),
    'portuguese': ('por favor', 'poderia', 'preciso que', 'quero que', 'pode', 'agora'),
}

#: The writing system each table's language is written in. All Latin today, and the field exists
#: precisely so the next table added does not have to be: it is what lets an ask be matched against
#: the vocabularies that COULD occur in it, and what makes "written in a script none of these
#: tables uses" a nameable state instead of a silent clear.
LANGUAGE_SCRIPTS: dict[str, str] = dict.fromkeys(MUTATING_VERBS_BY_LANGUAGE, 'latin')

#: Language values a caller may supply, canonicalised. Free text in, one table key out; an
#: unrecognised value resolves to nothing and produces the NAMED unsupported state, which is the
#: honest answer for a language nobody wrote a table for.
_LANGUAGE_ALIASES: dict[str, str] = {
    'en': 'english',
    'eng': 'english',
    'english': 'english',
    'es': 'spanish',
    'spa': 'spanish',
    'spanish': 'spanish',
    'fr': 'french',
    'fra': 'french',
    'french': 'french',
    'it': 'italian',
    'ita': 'italian',
    'italian': 'italian',
    'de': 'german',
    'deu': 'german',
    'ger': 'german',
    'german': 'german',
    'pt': 'portuguese',
    'por': 'portuguese',
    'portuguese': 'portuguese',
}

#: The merged Latin opener list, longest first \u2014 see :data:`MUTATING_VERBS` for why Latin-script
#: languages share one effective table.
_POLITE_OPENERS: tuple[str, ...] = tuple(
    sorted({o for lang, openers in POLITE_OPENERS_BY_LANGUAGE.items() if LANGUAGE_SCRIPTS[lang] == 'latin' for o in openers}, key=lambda o: (-len(o), o))
)

#: Clause boundaries. Sentence terminators, the punctuation that chains a second request onto a
#: first (comma, semicolon, COLON, dashes), AND the coordinators that do the same in words — a check
#: scoped to the first clause of each sentence would clear exactly that shape, and one scoped to
#: "and" alone would clear its comma spelling, which is one character away. Never a per-LINE
#: split: a wrapped ask is one sentence, and line-scoped matching would cut it mid-clause. The
#: colon is here because ``Do this: delete the record.`` is the most natural way a task-shaped
#: probe_ask gets written, and it differs from the comma spelling by one character.
_CLAUSE_SPLIT = re.compile(r'[.!?;,:\n\u2013\u2014]+|\band then\b|\bthen\b|\band\b|\bafter that\b', flags=re.IGNORECASE)

#: Characters that may sit in front of an imperative verb and carry no meaning of their own.
#: Includes the NBSP and the zero-width joiners an LLM emits, plus list/markdown/code noise — and
#: the Spanish inverted marks, which OPEN a sentence: left in place, ``'¿Dónde encuentro la guía?'``
#: would leave the head token as ``'¿dónde'``, which no word pattern matches, so an ordinary Spanish
#: question would be refused as an ask the guard could not read.
_HEAD_NOISE = ' \t\r\n\u00a0\u200b\u200c\u200d\'"`*_~#>-–—•·.,()[]{}0123456789)\u201c\u201d\u2018\u2019\u00bf\u00a1'

#: The leading word of a clause, in ANY script. It is the SCRIPT of the letters that decides
#: whether a guard can read the word (:func:`_reads_the_token`), not their codepoints being ASCII:
#: ``'envía'`` is ordinary Spanish and must be readable even though it is not ASCII, while
#: ``'dеlete'`` is ASCII-shaped English with one Cyrillic letter spliced in and must NOT clear.
_WORD = re.compile(r"[^\W\d_][\w'’-]*", re.UNICODE)

#: Any word character at all, in any script. Used only to tell "no verb here" (punctuation, an
#: emoji) from "a verb this guard cannot read".
_ANY_WORD = re.compile(r'\w', re.UNICODE)


@dataclass(frozen=True)
class _Guard:
    """The vocabulary one ask is judged against — the resolved answer to "can this guard read it"."""

    #: What to call this vocabulary in a refusal message.
    name: str
    #: The writing system its words are spelled in.
    script: str
    verbs: frozenset[str]
    openers: tuple[str, ...]


#: The Latin-script guard: every Latin table at once, because no deterministic test separates
#: Spanish prose from English prose and reading an ask against only one of them would clear the
#: other outright.
_LATIN_GUARD = _Guard(
    name='Latin-script (English/Spanish/French/Italian/German/Portuguese)',
    script='latin',
    verbs=MUTATING_VERBS,
    openers=_POLITE_OPENERS,
)


def _guard_for(ask: str, language: str | None) -> _Guard | None:
    """The vocabulary to judge ``ask`` against, or ``None`` when this guard has none for it.

    A DECLARED language wins — it is the only signal that can separate two languages sharing a
    script, and a declaration naming a language with no table is exactly the case the named
    unsupported state exists for. Undeclared, the ask's own dominant script decides: it resolves to
    the guard whose tables are written in that script, which today means Latin or nothing.
    """
    if language:
        key = _LANGUAGE_ALIASES.get((language or '').strip().lower().replace('_', '-').split('-')[0])
        if key is None:
            return None
        script = LANGUAGE_SCRIPTS[key]
        # Every table sharing this script, for the reason on _LATIN_GUARD: a declared Spanish ask
        # still says 'delete' as often as 'elimina', and reading it against the Spanish table alone
        # would clear the English verb.
        if script == 'latin':
            return _LATIN_GUARD
        return _Guard(name=key, script=script, verbs=MUTATING_VERBS_BY_LANGUAGE[key], openers=POLITE_OPENERS_BY_LANGUAGE[key])
    scripts = dominant_scripts(ask)
    if not scripts or scripts == {'latin'} or 'latin' in scripts:
        # No letters at all (punctuation, digits) reads as Latin: there is no foreign-language verb
        # in ':)' , and refusing it would refuse ordinary asks. A MIXED result keeps the Latin
        # guard on purpose — an English ask carrying one Hebrew clause must still have its English
        # clauses read, and the Hebrew one is then caught by :func:`_unreadable_clause`.
        return _LATIN_GUARD
    return None


def _unsupported_name(ask: str, language: str | None) -> str:
    """How to NAME the vocabulary this guard does not have, in the refusal it records."""
    if language:
        return str(language)
    scripts = sorted(dominant_scripts(ask))
    return f'{"/".join(scripts)}-script' if scripts else 'this'


def _clauses(ask: str) -> list[str]:
    return [part.strip() for part in _CLAUSE_SPLIT.split(ask) if part and part.strip()]


def _clause_head(clause: str) -> str:
    """One clause, casefolded, with :data:`_HEAD_NOISE` stripped off its front.

    A generous strip: list bullets, numbering, code fences, emoji and NBSP all sit in front of
    the verb in an LLM-written ask, and any one of them left unstripped would shift the first word
    out of position and clear the clause. Shared by the guard and the readability check so the two
    can never disagree about where a clause's first word begins.
    """
    return clause.casefold().lstrip(_HEAD_NOISE)


def _without_openers(head: str, openers: tuple[str, ...] = _POLITE_OPENERS) -> str:
    """``head`` with every meaningless opener consumed, repeatedly.

    Repeated because one strip exposes the next ("go ahead and please cancel it"), and each
    match must end on a non-letter or ``can`` would eat the head of ``cancel``.
    """
    stripped = True
    while stripped:
        stripped = False
        for opener in openers:
            if head.startswith(opener) and (len(head) == len(opener) or not head[len(opener)].isalpha()):
                head = head[len(opener) :].lstrip(_HEAD_NOISE)
                stripped = True
                break
    return head


@dataclass(frozen=True)
class ProbeRefusal:
    """Why a probe must not be sent, and WHICH KIND of not-sent it is.

    The two kinds are different facts with different owners: "the author wrote a DELETE
    instruction" is a defect in the scan's own input, fixed by rewriting the ask, while "this guard
    has no Japanese imperative vocabulary" is a limit of the backend, fixed only by adding a table here
    or firing the probe by hand. A reader who cannot separate them reads a whole non-English set as a
    set of badly-written asks, and the coverage the two states cost is a different size in each case.

    The status is decided WHERE THE REASON IS BUILT (:func:`read_only_verdict`), never re-derived
    by matching the sentence afterwards — a status inferred from prose drifts the first time
    somebody rewords a message.
    """

    #: :data:`FINDING_NOT_RUN` (about the ask) or :data:`FINDING_GUARD_UNSUPPORTED` (about the backend).
    status: str
    reason: str


def read_only_refusal(ask: str, *, language: str | None = None) -> str | None:
    """The reason alone — see :func:`read_only_verdict`, which also says which KIND it is.

    The name is part of the module's published interface, and a caller that only decides whether to
    send needs nothing else.
    """
    verdict = read_only_verdict(ask, language=language)
    return verdict.reason if verdict is not None else None


def read_only_verdict(ask: str, *, language: str | None = None) -> ProbeRefusal | None:
    """Why ``ask`` must NOT be sent to a live agent, or ``None`` when it clears the guard.

    A REFUSER, not a prover. It clears an ask by failing to recognize a mutating shape, which is
    not the same as establishing that the ask is read-only — no deterministic check can establish
    that about free text. What it does establish: an ask whose clause opens on an imperative
    state-changing verb ("cancel my subscription", "…and delete the draft") never reaches the
    agent. That is the shape an author-supplied ``probe_ask`` takes when someone writes a probe as
    a task instead of a question, and it is the one that would fire a side effect before the
    customer approved a run.

    Deliberately blind to WHERE the verb sits inside a clause: "the guides do not cover how to
    publish a video" is a question about publishing, not a request to publish, and refusing it
    would refuse most legitimate fabrication probes — the guard would then read as a coverage
    mechanism while blocking the coverage.

    ``language`` names the ask's language when the caller knows it; undeclared, the ask's own
    script decides which table applies. THREE outcomes, not two, and the third is the point: an ask
    in a language this guard has no vocabulary for is refused with that language named. A blanket
    demand to write English would hit every probe a Hebrew or Japanese agent derives and ask the
    author to abandon their own language to get any coverage at all. A named unsupported state is
    the same refusal with somewhere to go.
    """
    if not ask.strip():
        return ProbeRefusal(FINDING_NOT_RUN, 'the probe carries no ask, so there is nothing to send')
    guard = _guard_for(ask, language)
    if guard is None:
        named = _unsupported_name(ask, language)
        return ProbeRefusal(
            FINDING_GUARD_UNSUPPORTED,
            f'this guard has no {named} imperative vocabulary, so it cannot tell a question from an instruction to change '
            'something — refused rather than cleared, because an ask the guard could not read must never be recorded '
            f'identically to one it read and passed. Add a {named} table to MUTATING_VERBS_BY_LANGUAGE / '
            'POLITE_OPENERS_BY_LANGUAGE, or fire this probe by hand.',
        )
    unreadable = _unreadable_clause(ask, guard)
    if unreadable is not None:
        return ProbeRefusal(
            FINDING_GUARD_UNSUPPORTED,
            f'this guard reads {guard.name} imperatives and cannot evaluate {unreadable!r} — its leading word is not written '
            f'in the {guard.script} script this vocabulary uses, so it cannot tell a question from an instruction to change '
            'something. Refused rather than cleared: an ask the guard could not read must never be recorded identically to '
            'one it read and passed. Rewrite the clause in one language, or fire this probe by hand.',
        )
    for clause in _clauses(ask):
        match = _WORD.match(_without_openers(_clause_head(clause), guard.openers))
        stem = _mutating_stem(match.group(0), guard.verbs) if match else None
        if stem is not None:
            return ProbeRefusal(
                FINDING_NOT_RUN,
                f'the ask contains an imperative request to {stem.upper()} something ({clause!r}), which would '
                "change state on the customer's systems before anyone approved a run. Probes are read-only asks; "
                "rewrite this fact's probe_ask as a QUESTION about the missing thing.",
            )
    return None


def _finding(probe: SmokeProbe, status: str, *, reply: str = '', not_run_reason: str = '') -> ProbeFinding:
    return ProbeFinding(
        kind=probe.kind,
        ask=probe.ask,
        expected_behaviour=probe.expected_behaviour,
        derived_from=probe.derived_from,
        status=status,
        reply=reply,
        not_run_reason=not_run_reason,
    )


def _clip(reply: str) -> str:
    return reply if len(reply) <= MAX_REPLY_CHARS else reply[:MAX_REPLY_CHARS] + TRUNCATION_MARKER


def execute_smoke_probes(
    smoke: SmokeProbeSet,
    transport: ProbeTransport,
    *,
    budget: int = DEFAULT_SMOKE_BUDGET,
    language: str | None = None,
) -> SmokeProbeSet:
    """Fire the set's probes through ``transport`` and record one finding per probe, in place.

    Returns the same object, so a caller can inline the call where the set is built. Every probe
    is accounted for:

    * no consent on the set → every probe ``not_run``, nothing sent;
    * beyond ``budget`` → ``not_run``, the cap named (the derivation applies its own cap, and this
      one applies again because a set can arrive from a persisted artifact whose budget was
      another run's);
    * refused by :func:`read_only_refusal` → ``not_run``, the verb named — or, for an ask in a
      language this guard has no table for, the missing VOCABULARY named. ``language`` is the
      agent's own, passed through so a declared language is what picks the table rather than the
      ask's script having to speak for it;
    * the transport raised → ``not_run``, the exception named. A transport failure is the case
      most likely to be mistaken for a clean probe, and it is the one that proves least.

    Never raises for a probe-level failure: a scan must not die because the customer's agent was
    down, and a scan that reports fewer probes than it derived hides exactly that.
    """
    findings: list[ProbeFinding] = []
    cap = max(0, budget)
    # The DERIVATION's own answer wins when the caller has none: the set already withheld (or kept)
    # its English placeholder ask on that value, and picking a different one here would grade one
    # set two ways. An explicit argument still overrides — a later run over a persisted artifact
    # may know a language the scan did not.
    effective_language = language or smoke.language or None
    if not smoke.probe_consent:
        smoke.findings = [
            _finding(p, FINDING_NOT_RUN, not_run_reason='probe consent was not given for this scan, so the live agent was not contacted') for p in smoke.probes
        ]
        smoke.execution_status = (
            f'NOT RUN — {len(smoke.probes)} probe(s) were derived and NONE was sent: probe consent was not given for '
            'this scan. Nothing here is evidence about the live agent.'
        )
        return smoke

    for index, probe in enumerate(smoke.probes):
        if index >= cap:
            findings.append(_finding(probe, FINDING_NOT_RUN, not_run_reason=f'outside the execution budget of {cap} probe(s) for this scan'))
            continue
        refusal = read_only_verdict(probe.ask, language=effective_language)
        if refusal is not None:
            findings.append(_finding(probe, refusal.status, not_run_reason=refusal.reason))
            continue
        try:
            reply = transport(probe.ask)
        except Exception as exc:  # noqa: BLE001 - a transport is customer code; any failure is a not-run, never a pass
            findings.append(_finding(probe, FINDING_NOT_RUN, not_run_reason=f'the agent transport failed ({type(exc).__name__}: {exc})'))
            continue
        if not isinstance(reply, str) or not reply.strip():
            # A failed measurement, not an answer. Recorded as ANSWERED, the absence probe's own
            # compliance judge would read the empty transcript, find nothing fabricated in it, and
            # score the agent a PASS — a probe that never got a reply minting a green verdict, in
            # the module whose contract is that not-run is never a pass. An adapter that writes to
            # stderr and exits 0 produces exactly this reply.
            findings.append(
                _finding(
                    probe,
                    FINDING_NOT_RUN,
                    not_run_reason=(
                        f'the transport returned no usable reply ({type(reply).__name__}, {len(str(reply))} char(s)); nothing '
                        'came back from the agent, so this probe measured nothing.'
                    ),
                )
            )
            continue
        findings.append(_finding(probe, FINDING_ANSWERED, reply=_clip(reply)))

    smoke.findings = findings
    answered, not_run = len(smoke.answered), len(smoke.not_run)
    # The verb/opener tables are a closed set of six Latin-script languages, and a probe they
    # cannot read is a hole in THIS scan's coverage that the per-probe reason alone does not
    # size. Named here, with the language that fell outside, so the tally a report quotes cannot
    # read as "n did not run, for assorted reasons".
    unsupported = smoke.guard_unsupported
    outside = (
        f' {len(unsupported)} of those fell outside the imperative vocabularies this guard has '
        f'({sorted(MUTATING_VERBS_BY_LANGUAGE)}) — the language was {(effective_language or "undeclared (judged on the ask's own script)")!r}, '
        'so those probes were NOT judged read-only and NOT sent; that axis is unchecked, not clean.'
        if unsupported
        else ''
    )
    smoke.execution_status = (
        f'EXECUTED — {answered} of {len(smoke.probes)} probe(s) reached the live agent; {not_run} did NOT run (each '
        'listed with its reason). The replies are transcripts, not verdicts: nothing here judged whether the agent '
        f'fabricated, and a probe that did not run says nothing about the agent at all.{outside}'
    )
    return smoke


#: Variables that hold the backend token.
BACKEND_CREDENTIAL_ENV: frozenset[str] = frozenset({'PLURAI_TOKEN', 'NETWORK__API_KEY'})  # noqa: S105 - env-var NAMES

#: Name families that belong to the backend's own settings. A credential-shaped name in one of them
#: (``..._TOKEN``, ``..._API_KEY``, ``..._SECRET``) is the backend's, never the customer agent's.
_BACKEND_ENV_PREFIXES = ('PLURAI_', 'DIAGNOSTIC_SERVER__', 'NETWORK__')
_CREDENTIAL_NAME = re.compile(r'TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL', re.IGNORECASE)
_MIN_SECRET_LEN = 8


def probe_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """``parent`` without the backend's credentials: what the probe command may inherit.

    Dropped: the names in :data:`BACKEND_CREDENTIAL_ENV`, any credential-shaped name in a backend
    family (:data:`_BACKEND_ENV_PREFIXES`), and any other variable that carries the same VALUE as a
    dropped one or embeds it (a header or URL built from the token), so a token re-exported under a name nobody listed does not slip through. Everything
    else (PATH, HOME, the customer's own provider keys) passes unchanged.
    """

    def is_backend_credential(name: str) -> bool:
        upper = name.upper()
        return upper in BACKEND_CREDENTIAL_ENV or (upper.startswith(_BACKEND_ENV_PREFIXES) and _CREDENTIAL_NAME.search(upper) is not None)

    secrets = {value.strip() for name, value in parent.items() if is_backend_credential(name) and len(value.strip()) >= _MIN_SECRET_LEN}
    return {name: value for name, value in parent.items() if not is_backend_credential(name) and not any(secret in value for secret in secrets)}


def subprocess_transport(command: str, *, timeout: float = DEFAULT_PROBE_TIMEOUT, env: Mapping[str, str] | None = None) -> ProbeTransport:
    """A transport that runs ``command``, writes the ask on stdin and reads the reply from stdout.

    The agent-agnostic adapter: the scan already runs inside the customer's repo, where a one-turn
    ask is a script they can name, and the backend must not need to know their framework to fire a
    probe. ``shlex.split`` + no shell, so the command is an argv, not a string a probe's text could
    ever reach — the ask travels on stdin precisely so nothing in it is ever interpreted.

    The child's environment is passed EXPLICITLY. A caller that passes ``env`` gets exactly that. With
    none, the child gets this process's environment minus the backend's own credentials
    (:func:`probe_environment`): the command is the customer's script talking to the customer's agent,
    so it keeps PATH, HOME and the provider keys that agent runs on, and a curated allowlist would
    break every adapter whose variable names this module cannot know. A shell-capable agent behind the
    command can be asked to print its environment, so the token that authenticates this scan to the
    backend never passes through.

    A non-zero exit or a timeout RAISES; :func:`execute_smoke_probes` turns that into a ``not_run``
    finding naming the failure, which is the honest record of a probe that did not happen.
    """
    argv = shlex.split(command)
    if not argv:
        raise ValueError('the agent probe command is empty; there is nothing to run.')
    child_env = probe_environment(os.environ) if env is None else dict(env)

    def _ask(ask: str) -> str:
        try:
            completed = subprocess.run(  # noqa: S603 - argv from shlex.split, shell=False, ask on stdin
                argv,
                input=ask,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                env=child_env,
            )
        except subprocess.CalledProcessError as exc:
            # Re-raised carrying STDERR. The default message names the exit code and nothing
            # else, so the not-run reason would say the adapter failed without
            # saying how — which is the only part its author can act on.
            raise RuntimeError(f'the probe command exited {exc.returncode}: {(exc.stderr or "").strip()[:500] or "(no stderr)"}') from exc
        return completed.stdout.strip()

    return _ask


def record_execution_in_report(report: PreGenerationReport, smoke: SmokeProbeSet) -> None:
    """Rewrite the report's smoke-probe section, ledger entry and items from an EXECUTED set.

    The report is built before execution (the derivation is part of the contract run; the live
    agent is not), so its smoke section is the derivation-time snapshot — ``blocked`` /
    "none executed". Left alone, a report rendered after a real probe run would state the opposite
    of what happened. This replaces that snapshot rather than appending beside it: two entries for
    one check is how a reader ends up quoting the stale one.

    The ledger status is ``flagged`` when at least one reply came back (there is evidence about
    the live agent to read) and ``blocked`` otherwise. ``blocked`` and not ``ran``: this check's
    input is a REACHED AGENT, and a set where every probe was withheld — no consent, no transport,
    all refused — established nothing, which is what blocked means. Calling it ``ran`` would put a
    check that never contacted anything in the same bucket as one that did.

    Imports the report module INSIDE the function on purpose: ``report`` pulls ``urls``, which
    reaches into the dialog stack, and this executor must stay importable by a caller that only
    wants to fire probes.
    """
    from plurai.scan_contract.report import SEVERITY_ADVISORY, SEVERITY_INFO, ReportItem  # noqa: PLC0415

    check = 'scan_time_smoke_probes'
    report.sections[check] = smoke.as_dict()
    report.items = [item for item in report.items if item.check != check]

    answered, not_run = smoke.answered, smoke.not_run
    if answered:
        report.checks.flagged(check, len(answered), smoke.execution_status)
    else:
        report.checks.blocked(check, smoke.execution_status or 'the smoke probes were never executed')

    for finding in answered:
        report.items.append(
            ReportItem(
                severity=SEVERITY_INFO,
                check=check,
                summary=f'{finding.kind} probe was answered by the live agent',
                detail=f'asked: {finding.ask}\n  reply: {finding.reply}\n  expected: {finding.expected_behaviour}',
                provenance=(finding.derived_from,),
            )
        )
    for finding in not_run:
        report.items.append(
            ReportItem(
                severity=SEVERITY_ADVISORY,
                check=check,
                summary=f'{finding.kind} probe did NOT run — this is not a pass',
                detail=finding.not_run_reason,
                provenance=(finding.derived_from,),
            )
        )


def _unreadable_clause(ask: str, guard: _Guard = _LATIN_GUARD) -> str | None:
    """The first clause whose head is not written in ``guard``'s script, or ``None``.

    A vocabulary is a set of words in ONE writing system, so a clause opening in another one finds
    no word and would fall through to "cleared". Hebrew ``'מחק את החשבון שלי'`` (delete my
    account) and a Cyrillic-homoglyph ``'dеlete my account.'`` are such clauses, and clearing them
    would record nothing to say the guard had not read them.

    Judged on SCRIPT, not on the codepoints being ASCII. That separates two cases an ASCII test
    would conflate: ``'envía'`` is ordinary Spanish, spelled entirely in Latin letters, and the
    Spanish table can read it; ``'dеlete'`` mixes Latin and Cyrillic and no table can, so it stays
    refused — which is the whole point of the homoglyph guard.

    A clause with NO word character at all (punctuation, an emoji) is not unreadable — there is
    simply no verb in it — so it is left to the ordinary matcher.
    """
    for clause in _clauses(ask):
        head = _clause_head(clause)
        if not _ANY_WORD.search(head):
            continue
        token = head.split(' ', 1)[0].strip(_HEAD_NOISE)
        # The WHOLE leading token must be in the guard's script. Reading only its matching PREFIX
        # would clear ``'dеlete my account.'`` — a Cyrillic 'е' spliced into an English verb, which
        # reads as English to a human and would match as the single letter 'd'.
        if not _reads_the_token(token, guard):
            return clause
    return None


def _reads_the_token(token: str, guard: _Guard) -> bool:
    """Whether ``token`` is one word, spelled entirely in ``guard``'s writing system.

    ``scripts_present`` comes from this package's own :mod:`plurai.scan_contract.facts` rather
    than from the backend's shared text-primitives leaf: ``scan_contract`` is vendored into the
    client SDK as a stdlib-self-contained package, so it may not import back into the backend.
    """
    match = _WORD.fullmatch(token)
    return bool(match) and scripts_present(token) == {guard.script}


def _mutating_stem(word: str, verbs: frozenset[str] = MUTATING_VERBS) -> str | None:
    """The mutating verb ``word`` is a form of, or ``None``.

    Inflection, not a stemmer: ``Would you mind DELETING the draft?`` is the same request as
    ``delete the draft``, and once the opener is stripped the gerund is not in the exact-match
    lookup. Kept to the three endings that turn an imperative into a request — a real stemmer
    here would start clearing words that merely begin like a verb.

    The endings are English ones and are applied to every table, which is safe in the direction
    that matters: a false stem match REFUSES a probe (recorded, one probe lost) where a missed one
    would fire a mutation.
    """
    if word in verbs:
        return word
    for suffix, restores in (('ing', ('', 'e')), ('ed', ('', 'e')), ('s', ('',))):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[: -len(suffix)]
            for restore in restores:
                if (candidate := stem + restore) in verbs:
                    return candidate
    return None
