"""The agent's RENDER / ACTION surface: declared, carried on the scan artifact, and checked.

Some agents act through a channel that is not a tool call: a UI render (``show_table``,
``show_image``), a navigation, a hand-off verb. There is no dispatch to mock, so such an agent
declares ``tools: {}`` by design. The surface therefore needs its own typed declaration: without
one, no completeness check can see it, the scan carries nothing about it, and an agent that invents
a ``show_video`` it does not have is graded by nobody.

The plugin's ``verify_config.py`` gates the key's SHAPE at authoring time and its ``lint_prd.py``
reads the key against the PRD; the client SDK reads it the same way (``plurai/render_actions.py``).
This module is the backend's reading of the same key, in the same vocabulary — one key, one
meaning — and it is what both shells of the checker (the backend's CLI and the client's ``plurai-scan-contract``) carry onto
``scan_contract.json``:

* File: ``plurai.config.yaml``. Key: top-level ``render_actions``, a list of the verb names the
  agent emits. Found under a root at the two rungs every workspace declaration is looked for
  (:data:`CONFIG_CANDIDATE_RELPATHS`).
* ABSENT key: the config says nothing about a render surface. :func:`parse_render_actions` returns
  ``None``, and nothing downstream may read that as "no render actions". ``render_actions: []``
  is a DECLARATION — "none, by declaration" — and reads as an empty surface. The two are different
  facts and stay different all the way onto the scan artifact and into the check.
* MALFORMED — not a list, a non-string or blank entry, a verb listed twice:
  :class:`RenderActionsConfigError`, fail closed. A malformed declaration read as "undeclared"
  puts the surface right back where this key exists to take it from — invisible to every check.
* Two configs under one root that both declare and disagree: an error, not a precedence puzzle.
  The surface is env-scoped.

On the artifact the surface is the top-level ``render_actions`` field, ``{"verbs": [...],
"source": <config path>}`` — ABSENT when no config declared one (never an empty record).

The CHECK (:func:`render_surface_check`) is the backend half: it reads that field off the artifact
a generation job was given and compares the render verbs the agent's own replies carry against the
declared list. A verb outside the surface is a typed finding. When the field is absent the check is
NOT EXECUTED and its ledger says so — an undeclared surface is not a surface of zero verbs.

VENDORED into the client SDK: stdlib + pyyaml + this package only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from plurai.scan_contract._safe_yaml import safe_yaml_load

from plurai.scan_contract._suite_findings import CheckStatus
from plurai.scan_contract.facts import artifact_fact_verdicts

#: Top-level ``plurai.config.yaml`` key. ONE vocabulary with the plugin's
#: ``verify_config._KNOWN_ROOT_KEYS`` entry, ``lint_prd.RENDER_ACTIONS_KEY`` and the SDK's
#: ``plurai.render_actions.RENDER_ACTIONS_KEY`` — and the name of the field the scan artifact
#: carries the surface under.
RENDER_ACTIONS_KEY = 'render_actions'

#: The one key of the artifact's ``render_actions`` field when a declaration EXISTED and could not be
#: read on the scanning machine: ``{"unreadable": <why>}``. Never absent in that case — an absent
#: field reads as ``undeclared`` downstream, which sends the operator to add a declaration they
#: already wrote; this one is ``unusable`` ("present and cannot be read"), the remedy that is true.
RENDER_ACTIONS_UNREADABLE = 'unreadable'

#: The RESERVED ``subject`` of the PRD fact that CLOSES the surface (``kind: tool_surface``,
#: ``predicate: omits`` in the plugin's ``agent_facts.md``): its absence ``check.terms`` are the
#: verbs the agent does NOT have — the ones an agent in its domain would plausibly invent. The
#: scan verifies that sweep against the corpus like any absence fact; this module reads the same
#: terms off the published verdict as the vocabulary to look for in the agent's replies.
RENDER_SURFACE_SUBJECT = 'RENDER_ACTIONS'

#: Workspace config file that carries the env's declarations.
CONFIG_FILENAME = 'plurai.config.yaml'
#: The workspace layout: the roots every committed artifact is looked for under, relative to a
#: source root and in the order they are read. The SDK's ``ingest.workspace_config`` states the
#: same tuple; the two must agree or the two shells discover different files.
WORKSPACE_CANDIDATE_ROOTS: tuple[Path, ...] = (Path(), Path('plurai_sim'))
#: Config discovery paths, relative to a source root, in the order they are read.
CONFIG_CANDIDATE_RELPATHS: tuple[Path, ...] = tuple(root / CONFIG_FILENAME for root in WORKSPACE_CANDIDATE_ROOTS)


class RenderActionsConfigError(ValueError):
    """``render_actions`` exists but cannot testify to a surface (fail closed).

    A present-but-broken declaration must never degrade to "nothing declared": undeclared is the
    exact state this key exists to end, so a malformed one has to stop the reader, not be dropped.
    Also raised for a config that exists and cannot be read at all — an existing config that
    cannot testify never silently defaults.
    """


@dataclass(frozen=True, slots=True)
class RenderSurface:
    """The declared render / action surface: the verbs, and the config that declared them."""

    verbs: tuple[str, ...]
    #: The config file the declaration was read from; ``None`` for an in-memory document.
    source: Path | None = None

    def as_record(self) -> dict[str, Any]:
        """The wire shape. An empty ``verbs`` is "none, by declaration" — a consumer that tests the
        record's truthiness sees a declaration either way, which is why this is a mapping and not
        the bare list."""
        return {'verbs': list(self.verbs), 'source': str(self.source) if self.source is not None else None}

    @classmethod
    def from_record(cls, record: Any) -> RenderSurface:
        """Read the wire shape back. Raises :class:`RenderActionsConfigError` on anything that is
        not ``{"verbs": [<verb>, ...], "source": <path or null>}`` — the same rule the config is
        parsed under, so a field somebody hand-edited fails the same way a hand-edited config does.
        The :data:`RENDER_ACTIONS_UNREADABLE` marker raises with the scanning machine's own reason."""
        if isinstance(record, Mapping) and RENDER_ACTIONS_UNREADABLE in record:
            raise RenderActionsConfigError(
                f'a `{RENDER_ACTIONS_KEY}` declaration exists under the scanned roots and could not be read there: {record[RENDER_ACTIONS_UNREADABLE]}'
            )
        if not isinstance(record, Mapping) or 'verbs' not in record:
            raise RenderActionsConfigError(f"the artifact's `{RENDER_ACTIONS_KEY}` field must be a mapping with a `verbs` list, got {type(record).__name__}.")
        source = record.get('source')
        if source is not None and not isinstance(source, str):
            raise RenderActionsConfigError(f"the artifact's `{RENDER_ACTIONS_KEY}.source` must be a path string or null, got {type(source).__name__}.")
        surface = parse_render_actions({RENDER_ACTIONS_KEY: record['verbs']}, source=Path(source) if source else None)
        assert surface is not None  # noqa: S101 - the key is present by construction
        return surface


def parse_render_actions(document: Mapping[str, Any], *, source: Path | None = None) -> RenderSurface | None:
    """The surface one loaded config declares: ``None`` when the key is absent, else the verbs.

    Verbs are stripped, must be non-blank strings and must be unique. Raises
    :class:`RenderActionsConfigError` on a malformed value. ``source`` is named in every error so
    the operator is sent to the file, not to a key.
    """
    if RENDER_ACTIONS_KEY not in document:
        return None
    where = f'{source}: ' if source is not None else ''
    raw = document[RENDER_ACTIONS_KEY]
    if not isinstance(raw, list):
        raise RenderActionsConfigError(
            f'{where}`{RENDER_ACTIONS_KEY}` must be a list of the verb names the agent emits, got {type(raw).__name__}. '
            f'Omit the key when the agent has no render/action surface; `{RENDER_ACTIONS_KEY}: []` declares that it has none.'
        )
    verbs: list[str] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, str) or not entry.strip():
            raise RenderActionsConfigError(
                f'{where}`{RENDER_ACTIONS_KEY}[{index}]` must be a non-blank verb name (the exact token the agent emits), got {entry!r}.'
            )
        verb = entry.strip()
        if verb in verbs:
            raise RenderActionsConfigError(f'{where}`{RENDER_ACTIONS_KEY}` lists {verb!r} more than once — the surface is a set of verbs.')
        verbs.append(verb)
    return RenderSurface(verbs=tuple(verbs), source=source)


def discover_config_documents(workspace_root: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    """``(path, document)`` for every workspace config that exists under ``workspace_root``.

    Absent files are skipped silently — that is the only "didn't say". A file that exists but
    cannot be read as a YAML mapping raises :class:`RenderActionsConfigError`: an existing config
    that cannot testify must never be treated as silence. An empty file parses to an empty mapping
    (it exists, it is well-formed, it simply declares nothing). The SDK's
    ``ingest.workspace_config.discover_config_documents`` is the same procedure.
    """
    documents: list[tuple[Path, dict[str, Any]]] = []
    for relpath in CONFIG_CANDIDATE_RELPATHS:
        candidate = Path(workspace_root) / relpath
        if not candidate.exists() and not candidate.is_symlink():
            continue  # genuinely absent — contributes nothing
        if not candidate.is_file():
            raise RenderActionsConfigError(
                f'{candidate} exists but is not a regular file; it cannot testify to a declaration. '
                'Fix or remove it — an existing config that cannot testify never silently defaults.'
            )
        try:
            raw = candidate.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError) as exc:
            raise RenderActionsConfigError(f'{candidate} exists but could not be read: {exc}. Fix or remove it (an unreadable config cannot testify).') from exc
        try:
            data = safe_yaml_load(raw)
        except yaml.YAMLError as exc:
            raise RenderActionsConfigError(
                f'{candidate} is not parseable YAML ({exc}). Fix or remove it — an existing config that cannot be parsed never silently defaults.'
            ) from exc
        if data is None:
            data = {}  # empty file → a config that declares nothing
        if not isinstance(data, dict):
            raise RenderActionsConfigError(f'{candidate} must be a YAML mapping to declare anything; got {type(data).__name__}.')
        documents.append((candidate, data))
    return documents


def discover_render_surface(roots: Sequence[Path | str]) -> RenderSurface | None:
    """The surface declared under ``roots``, or ``None`` when no config under them declares one.

    Each root is read at the two rungs every workspace declaration is (a config that exists and
    cannot be read raises, whether or not it carries this key). A file reached from two roots — the
    workdir given directly and the repo root it sits under, which is how the plugin's scan step
    passes them — is read once. Two DIFFERENT files that both declare and disagree raise: the
    surface is env-scoped, and a discovery order does not get to settle which one the env meant.
    """
    seen: set[Path] = set()
    declared: list[RenderSurface] = []
    for root in roots:
        for path, document in discover_config_documents(root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            surface = parse_render_actions(document, source=path)
            if surface is not None:
                declared.append(surface)
    if not declared:
        return None
    if len({frozenset(s.verbs) for s in declared}) > 1:
        sources = '; '.join(f'{s.source} → {list(s.verbs)}' for s in declared)
        raise RenderActionsConfigError(
            f'conflicting `{RENDER_ACTIONS_KEY}` declarations ({sources}); the render surface is env-scoped and must agree. '
            f'Reconcile the {CONFIG_FILENAME} files (or remove one) so the env declares a single surface.'
        )
    return declared[0]


def declared_render_surface(artifact: Mapping[str, Any]) -> RenderSurface | None:
    """The surface a published ``scan_contract.json`` carries, or ``None`` when it carries none.

    ``None`` means the artifact has NO ``render_actions`` field — no config under the scanned
    roots declared one. It is never returned for a present field: an empty declaration comes back
    as a surface with no verbs, and a field that cannot be read raises
    :class:`RenderActionsConfigError` rather than reading as undeclared.
    """
    if RENDER_ACTIONS_KEY not in artifact:
        return None
    return RenderSurface.from_record(artifact[RENDER_ACTIONS_KEY])


# =====================================================================================
# The check: the declared surface against the verbs the agent's replies carry
# =====================================================================================

#: WHY a check is ``blocked`` — typed, because "not executed" has four different remedies and a
#: consumer reading prose to tell them apart is the collapse this ledger exists to prevent.
BLOCKED_NO_ARTIFACT = 'no_artifact'
#: The artifact carries no ``render_actions`` field: nothing declared a surface. The remedy is a
#: `render_actions:` line in plurai.config.yaml — NOT "no render actions", which `[]` declares.
BLOCKED_UNDECLARED = 'undeclared'
#: The field is present and cannot be read — a defect somebody has to fix.
BLOCKED_UNUSABLE = 'unusable'
#: A surface is declared but the artifact carries no answered agent reply to compare against.
BLOCKED_NO_REPLIES = 'no_replies'
#: A surface is declared (none, by declaration) and no ``RENDER_ACTIONS`` fact names a verb to
#: sweep the replies for, so no invented verb could be observed at all.
BLOCKED_NO_VOCABULARY = 'no_vocabulary'
#: A NON-EMPTY surface is declared and no ``RENDER_ACTIONS`` fact names a verb to sweep for. It is
#: its own kind rather than a flavour of the one above, because the check would otherwise run GREEN
#: BY CONSTRUCTION: the vocabulary is non-empty (the declared verbs), so the no-vocabulary guard
#: does not fire; and ``outside = observed − declared`` while ``observed ⊆ declared ∪ swept``, so
#: with ``swept`` empty ``outside`` is empty ARITHMETICALLY — over any replies, from any agent,
#: whatever it invented. ``status='ran'`` with the detail "every render verb observed is on the
#: declared surface" would be a statement about the closing fact's absence and not about the agent
#: at all. The two kinds share a remedy (write the closing fact) and differ in what the operator
#: is looking at: no_vocabulary is an agent with no render story on either side, no_sweep is a
#: declared surface nobody ever swept against.
BLOCKED_NO_SWEEP = 'no_sweep'

#: The smoke-probe finding status whose ``reply`` is a transcript of the live agent
#: (``smoke_probes.FINDING_ANSWERED``). Restated rather than imported: this module is read by the
#: generation jobs, and the smoke package is the derivation side of the scan.
_FINDING_ANSWERED = 'answered'


@dataclass(frozen=True)
class RenderSurfaceCheck:
    """What the render-surface check established — or, typed, why it could not run.

    ``status`` is the house three-valued ledger: ``blocked`` (did not execute; ``blocked_kind``
    says why), ``ran`` (executed, every observed verb is on the surface), ``flagged`` (executed,
    ``outside`` is non-empty). ``declared`` is ``None`` only when the surface is undeclared —
    distinct from ``()``, which is "none, by declaration".
    """

    status: CheckStatus
    detail: str
    blocked_kind: str | None = None
    declared: tuple[str, ...] | None = None
    source: str | None = None
    #: Render verbs the agent's replies carried, in the order first seen — on OR off the surface.
    observed: tuple[str, ...] = ()
    #: The observed verbs the declaration does not license. Each is one typed finding.
    outside: tuple[str, ...] = ()
    #: How many answered replies were read. Zero with ``status == 'blocked'``.
    replies_read: int = 0

    @property
    def executed(self) -> bool:
        return self.status != 'blocked'

    #: The ``blocked_kind`` values that are ORDINARY SILENCE rather than a defect: nobody ran a
    #: scan, or nobody declared a surface. Both are the default state of an agent that never asked
    #: for this check, so :meth:`notes` stays quiet on them — a warning on every such job trains
    #: its reader to skip warnings, the same trade the batch absence probes make for an absent
    #: artifact.
    _QUIET_BLOCKS = frozenset({BLOCKED_NO_ARTIFACT, BLOCKED_UNDECLARED})

    def notes(self) -> list[str]:
        """The BLOCKED half of :meth:`findings` — one line when the check DID NOT EXECUTE for a
        reason somebody has to fix, and none otherwise.

        A not-executed check must not look clean in any channel an operator reads: ``findings()``
        is empty for every ``blocked`` state by construction, and the typed ledger this check
        writes (``absence_probes['render_surface']`` / ``absence_probe_merge['render_surface']``)
        is not an operator-facing channel. A surface the customer DECLARED and this run never
        compared anything against — because the artifact cannot be parsed, because no fact names a
        verb to sweep for, or because no smoke probe ever fired — is a check that silently did not
        happen while the set reports no render finding at all. Those three say so; the two
        ordinary silences (:data:`_QUIET_BLOCKS`) do not.
        """
        if self.status != 'blocked' or self.blocked_kind in self._QUIET_BLOCKS:
            return []
        return [f'the render-surface check DID NOT RUN ({self.blocked_kind}) — {self.detail}']

    def findings(self) -> list[str]:
        """One line per verb outside the surface, in the pre-generation-finding voice. Never
        empty when there is one, and never non-empty when the check did not execute — a blocked
        check speaks through :meth:`notes` instead, which is the whole reason that method exists."""
        if not self.outside:
            return []
        licensed = ', '.join(f'`{v}`' for v in self.declared) if self.declared else 'none, by declaration'
        where = f' (declared in {self.source})' if self.source else ''
        return [
            f'the agent invented `{verb}` in a scan-time reply — a render verb OUTSIDE its declared surface '
            f'({licensed}{where}). Either the agent has this render and the declaration is incomplete, or it does not '
            'and a judge must grade the invention.'
            for verb in self.outside
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            'status': self.status,
            'executed': self.executed,
            'blocked_kind': self.blocked_kind,
            'detail': self.detail,
            'declared': list(self.declared) if self.declared is not None else None,
            'source': self.source,
            'observed': list(self.observed),
            'outside': list(self.outside),
            'replies_read': self.replies_read,
        }


def _blocked(kind: str, detail: str, surface: RenderSurface | None = None) -> RenderSurfaceCheck:
    return RenderSurfaceCheck(
        status='blocked',
        detail=detail,
        blocked_kind=kind,
        declared=surface.verbs if surface is not None else None,
        source=str(surface.source) if surface is not None and surface.source is not None else None,
    )


def observed_render_verbs(texts: Iterable[str], vocabulary: Iterable[str]) -> tuple[str, ...]:
    """The verbs of ``vocabulary`` that occur in ``texts`` as whole tokens, in the order first seen.

    A render verb is the EXACT token the agent emits (``show_table``), so the match is
    case-sensitive and bounded by non-word characters on both sides: ``show_tables`` is not
    ``show_table``, and a sentence that merely says "show the table" names no verb. No vocabulary
    is inferred from the text — every verb looked for was declared or swept by a typed fact.
    """
    patterns = [(verb, re.compile(rf'(?<!\w){re.escape(verb)}(?!\w)')) for verb in dict.fromkeys(v for v in vocabulary if v)]
    seen: list[str] = []
    for text in texts:
        for verb, pattern in patterns:
            if verb not in seen and pattern.search(text):
                seen.append(verb)
    return tuple(seen)


def check_render_surface(surface: RenderSurface | None, observed: Iterable[str], *, replies_read: int = 0) -> RenderSurfaceCheck:
    """Compare ``observed`` render verbs against the declared ``surface``. Source-agnostic: a
    scoring-time caller can hand it the verbs a run's tool events carried, the artifact reader
    below hands it the verbs the scan-time replies carried.

    ``None`` is NOT EXECUTED, whatever was observed: an undeclared surface licenses nothing and
    forbids nothing, and deciding "outside" against it would be deciding against a list nobody
    wrote. An empty surface is a declaration, so every observed verb is outside it.
    """
    if surface is None:
        return _blocked(
            BLOCKED_UNDECLARED,
            f'NOT EXECUTED — the scan artifact carries no `{RENDER_ACTIONS_KEY}` field: no {CONFIG_FILENAME} under the scanned '
            'roots declared a render surface, so nothing states which render verbs the agent has and an invented one cannot '
            f'be told from a real one. This is not "no render actions" (`{RENDER_ACTIONS_KEY}: []` declares that); declare the '
            'list and re-scan.',
        )
    seen = tuple(dict.fromkeys(v for v in observed if v))
    outside = tuple(v for v in seen if v not in surface.verbs)
    licensed = ', '.join(surface.verbs) or 'none, by declaration'
    source = str(surface.source) if surface.source is not None else None
    if outside:
        return RenderSurfaceCheck(
            status='flagged',
            detail=f'{len(outside)} render verb(s) the agent used are outside its declared surface ({licensed}): {", ".join(outside)}. '
            f'Read over {replies_read} answered scan-time reply(ies).',
            declared=surface.verbs,
            source=source,
            observed=seen,
            outside=outside,
            replies_read=replies_read,
        )
    return RenderSurfaceCheck(
        status='ran',
        detail=f'every render verb observed ({", ".join(seen) or "none"}) is on the declared surface ({licensed}). '
        f'Read over {replies_read} answered scan-time reply(ies).',
        declared=surface.verbs,
        source=source,
        observed=seen,
        replies_read=replies_read,
    )


def _answered_replies(artifact: Mapping[str, Any]) -> list[str]:
    """The live agent's replies the artifact carries: the ANSWERED scan-time smoke findings."""
    report = artifact.get('report')
    sections = report.get('sections') if isinstance(report, Mapping) else None
    section = sections.get('scan_time_smoke_probes') if isinstance(sections, Mapping) else None
    findings = section.get('findings') if isinstance(section, Mapping) else None
    if not isinstance(findings, list):
        return []
    return [
        str(row['reply'])
        for row in findings
        if isinstance(row, Mapping) and row.get('status') == _FINDING_ANSWERED and isinstance(row.get('reply'), str) and row['reply'].strip()
    ]


def swept_render_verbs(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    """The verbs the ``RENDER_ACTIONS`` fact(s) SWEEP for — what the agent does NOT have.

    Read off the published verdicts (``report.sections.fact_verification.verdicts``) by the
    reserved subject, taking the executable ``check.terms`` (a hand-authored row states the same
    key at the top level, and both spellings are read, as ``load_scan_facts`` reads them). Empty
    when no such fact was published.
    """
    verbs: list[str] = []
    for row in artifact_fact_verdicts(artifact) or []:
        if not isinstance(row, Mapping) or str(row.get('subject') or '') != RENDER_SURFACE_SUBJECT:
            continue
        check = row.get('check')
        terms = check.get('terms') if isinstance(check, Mapping) else row.get('terms')
        for term in terms if isinstance(terms, list) else []:
            if isinstance(term, str) and term.strip() and term.strip() not in verbs:
                verbs.append(term.strip())
    return tuple(verbs)


def render_surface_check(artifact: Mapping[str, Any] | None, *, absent_reason: str = '') -> RenderSurfaceCheck:
    """The check, read off one published ``scan_contract.json``. NEVER raises: this runs inside a
    generation job, and every way it could not look becomes ``blocked`` with its typed reason.

    The observed side is the render vocabulary — the declared verbs plus the verbs the
    ``RENDER_ACTIONS`` fact sweeps for — found as whole tokens in the answered scan-time replies
    the artifact carries. Nothing else is read as a verb: a term neither declared nor swept is not
    one this check can name, and inferring one from prose would be an agent-derived rule.
    """
    if not isinstance(artifact, Mapping):
        return _blocked(BLOCKED_NO_ARTIFACT, absent_reason or 'no scan contract reached this generation, so no render surface could be read.')
    try:
        surface = declared_render_surface(artifact)
    except RenderActionsConfigError as exc:
        return _blocked(BLOCKED_UNUSABLE, f'the scan artifact carries a `{RENDER_ACTIONS_KEY}` field this check cannot read: {exc}')
    if surface is None:
        return check_render_surface(None, ())
    swept = swept_render_verbs(artifact)
    vocabulary = (*surface.verbs, *(v for v in swept if v not in surface.verbs))
    licensed = ', '.join(surface.verbs) or 'none, by declaration'
    # THE SWEPT HALF IS THE WHOLE CHECK, and its absence is a block whatever the surface declares.
    # ``outside`` is ``observed − declared`` and ``observed`` is drawn from ``declared ∪ swept``,
    # so an empty ``swept`` makes ``outside`` empty by arithmetic rather than by evidence: the
    # declared verbs are, by definition, the ones that are NOT findings. Gating on the whole
    # vocabulary would let a declared surface with no closing fact run to ``status='ran'`` and
    # publish "every render verb observed is on the declared surface" — an invented
    # ``show_video`` graded by nobody under a green ledger line, on a check
    # ``RenderSurfaceCheck.executed`` reports as executed.
    #
    # Even swept, ``outside ⊆ swept`` holds, so this check can only ever find inventions a human
    # enumerated in the closing fact.
    # The module refuses to infer verbs from prose on purpose (:func:`observed_render_verbs`), and
    # widening the sweep to "every ``render_actions`` verb the scan derived across the tree" would
    # be a no-op — :func:`discover_render_surface` raises on any two disagreeing configs and
    # returns the one declaration, so that union is identically ``declared``. A typed block, which
    # :meth:`RenderSurfaceCheck.notes` then speaks for, is the honest outcome when nothing was
    # swept.
    if not swept:
        return _blocked(
            BLOCKED_NO_VOCABULARY if not vocabulary else BLOCKED_NO_SWEEP,
            f'the surface is declared ({licensed}) but no `{RENDER_SURFACE_SUBJECT}` fact names a verb to sweep the replies for, so '
            'no invented verb could be observed. Write the closing fact (kind tool_surface, predicate omits, check.terms = the '
            'verbs the agent does NOT have).',
            surface,
        )
    replies = _answered_replies(artifact)
    if not replies:
        return _blocked(
            BLOCKED_NO_REPLIES,
            f'the surface is declared ({licensed}) but the artifact carries no answered scan-time reply to compare it against — '
            'no smoke probe was fired at the live agent (it needs --probe-consent and --agent-command). The surface is still '
            'carried for scoring-time judges.',
            surface,
        )
    return check_render_surface(surface, observed_render_verbs(replies, vocabulary), replies_read=len(replies))


def declared_render_verbs(text: str | None) -> list[str] | None:
    """The render verbs a stashed scan contract DECLARES, for stamping onto each scenario — or ``None``.

    ``None`` when no contract was supplied, the contract carries no ``render_actions`` field, or
    either cannot be read; ``[]`` is a declaration of none. NEVER raises: the unreadable cases are
    the scenario job's own warnings (:func:`render_surface_check_from_json`), and stamping nothing
    leaves the scoring reader on prose alone.
    """
    if text is None:
        return None
    try:
        payload = json.loads(text)
        surface = declared_render_surface(payload) if isinstance(payload, Mapping) else None
    except (ValueError, RenderActionsConfigError):
        return None
    return None if surface is None else list(surface.verbs)


def render_surface_check_from_json(text: str | None) -> RenderSurfaceCheck:
    """:func:`render_surface_check` over the stashed contract BYTES. NEVER raises.

    ``None`` is "no contract was supplied", and only that. A supplied document that does not
    parse, or that parses to something that is not a mapping, is ``unusable`` — a defect somebody
    has to fix, not the ordinary silence of an agent that never ran a scan.
    """
    if text is None:
        return render_surface_check(None)
    try:
        payload = json.loads(text)
    except ValueError as exc:
        return _blocked(BLOCKED_UNUSABLE, f'the stashed scan contract is not JSON ({type(exc).__name__}: {exc}); it was supplied and could not be read.')
    if not isinstance(payload, Mapping):
        return _blocked(BLOCKED_UNUSABLE, f'the stashed scan contract is a {type(payload).__name__}, not a mapping; it was supplied and could not be read.')
    return render_surface_check(payload)


__all__ = [
    'BLOCKED_NO_ARTIFACT',
    'BLOCKED_NO_REPLIES',
    'BLOCKED_NO_SWEEP',
    'BLOCKED_NO_VOCABULARY',
    'BLOCKED_UNDECLARED',
    'BLOCKED_UNUSABLE',
    'CONFIG_CANDIDATE_RELPATHS',
    'CONFIG_FILENAME',
    'RENDER_ACTIONS_KEY',
    'RENDER_ACTIONS_UNREADABLE',
    'RENDER_SURFACE_SUBJECT',
    'WORKSPACE_CANDIDATE_ROOTS',
    'RenderActionsConfigError',
    'RenderSurface',
    'RenderSurfaceCheck',
    'declared_render_verbs',
    'check_render_surface',
    'declared_render_surface',
    'discover_config_documents',
    'discover_render_surface',
    'observed_render_verbs',
    'parse_render_actions',
    'render_surface_check',
    'render_surface_check_from_json',
    'swept_render_verbs',
]
