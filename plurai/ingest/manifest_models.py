"""Typed single-archive ingest manifest — the SDK↔backend wire contract.

The ``plurai ingest-*`` CLI bundles ALL anonymized units into ONE streamed ``.tar.gz`` (tar member
name == each unit's ``key``), uploads it as ONE storage object (→ ONE server-minted lease), and
prints THIS manifest to stdout. The plugin hands it verbatim to the backend's ``*_start(manifest=...)``
MCP tool, which validates it against its own ``LeaseManifest`` model. The SDK does not import the
backend; this model mirrors the backend's field set and is pinned to it by a two-sided contract
test.

Wire shape — ARCHIVE-ONLY (snake_case; the backend accepts it via ``populate_by_name=True`` + the
``lease_id`` alias choice)::

    {"lease_id": "<uuid>", "checksum": "<sha256 of the .tar.gz>", "bytes": <archive size>,
     "posture": "raw" | "anonymized", "anon_key_fingerprint": "<hex>" | null,
     "provenance": {"sdk_version": "...", "eval_set_id": "...", ...}}   # ingest-run only, present keys only

There are NO per-member entries: the whole-archive ``checksum`` pins every member's bytes WITHIN a
single upload — it is computed over the exact archive that is PUT, and the server re-verifies it on
PUT + fetch — which makes per-member ``{key, checksum, bytes}`` descriptors redundant. The archive is
byte-deterministic (sorted members, zero times, no gzip file name or time), so the checksum is also
the archive's content address. There is likewise NO ``id``/``set_id``/``job_id`` — the server mints
the job/set id.

**Run provenance on the wire.** A ``plurai ingest-run`` manifest also carries a ``provenance``
block (:class:`RunProvenance`) — which eval set / scenario set the run was scored against, an opaque
harness fingerprint, the agent's build tag, and the SDK's own version — so a later REPLAY of those
artifacts can tell whether they still describe the current world. Every key is optional and absent
keys are OMITTED rather than nulled. The backend model ignores unknown keys, so a backend that does
not know the block drops it instead of failing the upload.

**Posture on the wire.** Every manifest records the ``posture`` its archive ran under (and, when
``anonymized``, the ``anon_key_fingerprint``) so a backend env can refuse artifacts that mix
postures — a raw↔pseudonym mix breaks grounding joins silently. The backend ignores unknown keys,
so these snake_case keys validate without being acted on there. The SDK-side per-workspace audit check
(:func:`plurai.ingest.disclosure.assert_posture_consistent`) enforces posture consistency, and
``manifest.posture`` is the durable cross-machine seam a backend check can read.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator, model_serializer, model_validator

from plurai.ingest.posture import AnonymizationPosture


#: Values that MEAN absence even though a caller supplied them. Kept beside the validator that
#: honours them so the list cannot drift from its rationale.
#:
#: ``unknown`` is `plurai-run-simulation --agent-version`'s own git-failure fallback; ``none`` and
#: ``null`` are what a shell default produces when a driver interpolates an unset variable through
#: a template. All three are a caller saying "I do not know", and recording them as data makes a
#: replay compare a real value against a placeholder.
_ABSENT_SENTINELS = frozenset({'unknown', 'none', 'null'})


class RunProvenance(BaseModel):
    """What an uploaded run IS OF — the ids + versions a later reader needs to decide whether these
    artifacts still describe the current world.

    Exists for the REPLAY path: the backend re-judges saved transcripts (``score_run``) instead of
    paying for another scored run, which is only sound while the harness that produced them, and the
    scenario/eval sets they were graded against, are still the current ones.

    The SDK COMPUTES only ``sdk_version`` (the running build, read from package metadata).
    ``harness_fingerprint`` is an OPAQUE string the driver hashes out of its own harness file
    manifest — the SDK neither derives nor interprets it, so what counts as "the harness" can change
    without a client release.

    Every field but ``sdk_version`` is optional, and an absent value stays ABSENT: a blank flag
    resolves to ``None`` and a ``None`` never reaches the wire (see :meth:`_omit_absent`). That
    distinction is load-bearing — the backend WARNS on provenance it does not have and can only
    refuse on provenance that DISAGREES, so a placeholder (``''`` / ``'unknown'``) stamped where an
    id belongs would convert "nobody said" into "said, and wrong". It is exactly where these flags
    diverge from ``plurai-run-simulation --agent-version``, which does fall back to the literal
    ``"unknown"`` because its create-thread wire requires a non-empty tag.

    Two fields the client deliberately does NOT carry: ``gt_revision`` (only the backend that owns
    the scenario set knows its current grading-GT edit counter) and ``backend_build_id`` (a
    client-quoted build id names whichever backend the client last spoke to, not the one importing
    this run). The server enriches both itself at finalize.
    """

    # extra='forbid' + frozen: built once from the parsed flags, dumped, and never mutated — an
    # unexpected key here is a client bug, not forward-compat.
    model_config = ConfigDict(extra='forbid', frozen=True)

    #: The running SDK build (``plurai.__version__``), always stamped — never a hardcoded literal.
    sdk_version: str = Field(min_length=1)
    #: The backend handles this run was driven against. Optional and absent-stays-absent, like
    #: every field below them.
    eval_set_id: str | None = None
    scenario_set_id: str | None = None
    #: Opaque driver-computed digest of the harness file manifest. Never parsed here.
    harness_fingerprint: str | None = None
    #: WHICH SDK actually ran, from the LOADED module (:func:`plurai.identity.sdk_identity`).
    #: ``sdk_version`` above is dist metadata, which can disagree with the code that actually ran
    #: (e.g. a stale editable install). This block names the commit (or the content hash) and the
    #: client contract version, so a replay can say which client SHAPE produced the transcripts it
    #: is about to re-judge.
    sdk_identity: dict[str, Any] | None = None
    agent_version: str | None = None

    @field_validator('eval_set_id', 'scenario_set_id', 'harness_fingerprint', 'agent_version', mode='before')
    @classmethod
    def _blank_is_absent(cls, value: object) -> object:
        """``--eval-set-id ''`` (an unset shell variable) is ABSENCE, not an empty id.

        A caller interpolating a marker file that does not exist passes an empty string; stamping
        that would tell the backend an id was supplied and is blank — a mismatch it can act on —
        where the truth is that nobody supplied one.
        """
        if isinstance(value, str):
            stripped = value.strip()
            # The runner's OWN fallback, treated as the absence it stands for. `run_simulation`
            # returns the literal "unknown" whenever git cannot answer, and a caller naturally
            # forwards that run's recorded agent_version into this flag. Stamping it would turn
            # "nobody could say" into present-and-wrong, which the backend can act on.
            return None if stripped.casefold() in _ABSENT_SENTINELS else (stripped or None)
        return value

    @model_serializer(mode='wrap')
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize only the fields that actually have a value — no nulls on the wire."""
        return {key: value for key, value in handler(self).items() if value is not None}


class IngestManifest(BaseModel):
    """The single-call manifest the SDK prints: the ONE archive lease + the archive's own integrity
    (``checksum``/``bytes``) + the ``posture`` (and, when anonymized, the ``anon_key_fingerprint``)
    it ran under. No per-member entries, no set/job id — the server mints the id and re-verifies
    member bytes from the whole-archive checksum."""

    # extra='forbid': the SDK constructs this itself, so an unexpected field is a bug, not
    # forward-compat — fail loud rather than silently smuggle a stray key onto the wire.
    # frozen=True: construct-once (built + dumped with nothing between), so pin it immutable.
    model_config = ConfigDict(extra='forbid', frozen=True)

    lease_id: UUID
    checksum: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    posture: AnonymizationPosture
    #: Present iff posture == 'anonymized' (the sha256(anon.key) fingerprint); null under raw.
    anon_key_fingerprint: str | None = None
    #: The run's identity block (``plurai ingest-run`` only) — OMITTED entirely when nothing stamped
    #: it, so an absent block reads as "nobody said" rather than "asked and answered, nothing there".
    provenance: RunProvenance | None = None

    @model_validator(mode='after')
    def _posture_fingerprint_invariant(self) -> IngestManifest:
        """anonymized ⇒ a fingerprint is present; raw ⇒ there is none. Any other combo is a bug."""
        if self.posture == 'anonymized' and not self.anon_key_fingerprint:
            raise ValueError("posture 'anonymized' requires anon_key_fingerprint (the sha256(anon.key) the scrub ran under)")
        if self.posture == 'raw' and self.anon_key_fingerprint is not None:
            raise ValueError("posture 'raw' must not carry an anon_key_fingerprint (no key is minted under raw)")
        return self

    @model_serializer(mode='wrap')
    def _omit_absent_provenance(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Drop ``provenance`` from the dump when there is none, leaving every other key untouched.

        Only this ONE key is dropped: ``anon_key_fingerprint`` still serializes as an explicit null
        under raw posture (its null is the posture/fingerprint invariant's other half, not a missing
        value), so a non-run ingest's manifest carries no ``provenance`` key at all.
        """
        dumped = handler(self)
        if dumped.get('provenance') is None:
            dumped.pop('provenance', None)
        return dumped


__all__ = ['IngestManifest', 'RunProvenance']
