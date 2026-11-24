"""The online rules of ``plurai ci run`` and ``plurai ci validate --online``: the committed ids exist and link to each other
within the recorded lineage, the org has credit, and every smoke name is in the committed scenario set.

Each rule that stops the run raises :class:`StopRun` carrying the :class:`~plurai.report.Stopped` that earns its heading and exit
code. The errors of the backend pass through for the caller to map: :class:`~plurai.ci.backend.TokenRejected`, ``TokenRefused``
and ``UpgradeRefused`` (a 401, a 403 and a 426, never retried) and ``BackendUnreachable`` (no answer after the retries).
:func:`stop_for` is the one map from the first two and the last to a stop; a 426 carries a typed refusal each caller reports in
its own way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from plurai.ci.backend import Artifact, Backend, BackendUnreachable, TokenRefused, TokenRejected
from plurai.ci.config import SuiteIds
from plurai.report import Stopped
from plurai.report.render import IDS_FIX
from plurai.report.run import TIERS

#: A lineage longer than this is a loop or a corrupt record, not a history.
_MAX_LINEAGE = 64

#: An eval set whose scenario set link left the committed lineage: the evals were made for another set, and regenerating the
#: scenario set from scratch is what breaks the link.
_RELINK_FIX = f'relink the eval set to the committed scenario set, or regenerate the evals for it, and commit the new eval set id. Otherwise {IDS_FIX}'


class StopRun(Exception):  # noqa: N818 - a stop is a result, not an error condition
    """A step ended the run; ``stopped`` is its heading, fix and exit code."""

    def __init__(self, stopped: Stopped) -> None:
        super().__init__(stopped.reason)
        self.stopped = stopped


def stop_for(error: TokenRejected | TokenRefused | BackendUnreachable, *, repo: str | None = None) -> Stopped:
    """The stop a backend error ends a command with. ``repo`` names the repository in the fix of a rejected token."""
    if isinstance(error, TokenRejected):
        return Stopped.token_rejected(repo)
    if isinstance(error, TokenRefused):
        return Stopped.token_refused(error.route)
    return Stopped.not_measured('infra', 'the backend did not answer', 'run it again once the backend answers.', detail=str(error))


@dataclass(frozen=True)
class Suite:
    """The four committed artifacts as the backend reads them."""

    scenarios: Artifact
    evals: Artifact
    env: Artifact
    corpus: tuple[Artifact, ...]

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        return (self.scenarios, self.evals, self.env, *self.corpus)

    @property
    def scenario_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.scenarios.scenarios)

    @property
    def policies(self) -> tuple[tuple[str, str], ...]:
        """The policies of the eval set with their tiers; :func:`check_ids` has refused a read that has none or cannot place one."""
        return self.evals.policy_tiers or ()

    def digests(self) -> dict[str, str]:
        """Each committed id's pinned digest, for the ids that carry one."""
        return {a.id: a.pinned_digest for a in self.artifacts if a.pinned_digest}

    def drifted(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.artifacts if a.pinned_digest_drift)


class Ancestry:
    """Reads artifacts once each and follows the single copied-from id a read returns."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._cache: dict[str, Artifact | None] = {}

    def read(self, artifact_id: str) -> Artifact | None:
        if artifact_id not in self._cache:
            self._cache[artifact_id] = self._backend.artifact(artifact_id)
        return self._cache[artifact_id]

    def chain(self, artifact_id: str) -> frozenset[str]:
        """``artifact_id`` and every ancestor the recorded lineage reaches, one read per step. A step the backend no longer
        holds ends the chain."""
        seen = [artifact_id]
        current = self.read(artifact_id)
        while current is not None and current.lineage and current.lineage not in seen and len(seen) < _MAX_LINEAGE:
            seen.append(current.lineage)
            current = self.read(current.lineage)
        return frozenset(seen)


def _require(ancestry: Ancestry, kind: str, artifact_id: str) -> Artifact:
    artifact = ancestry.read(artifact_id)
    if artifact is None:
        raise StopRun(Stopped.not_on_backend(kind, artifact_id))
    if artifact.status != 'done':
        raise StopRun(
            Stopped.ids_unavailable(
                f'the committed {kind} is not published', f'the committed {kind} {artifact_id} is not published (status `{artifact.status}`)'
            )
        )
    return artifact


def _within(ancestry: Ancestry, owner: Artifact, *, owner_kind: str, role: str, committed: Sequence[str], committed_kind: str) -> None:
    """Every link of ``owner`` with ``role`` is a committed id or an ancestor of one."""
    allowed: frozenset[str] = frozenset().union(*(ancestry.chain(committed_id) for committed_id in committed))
    for link in owner.links:
        if link.role == role and link.id not in allowed:
            outside = (
                f'which is outside the lineage of the committed {committed_kind} {", ".join(committed)}'
                if committed
                else f'but no {committed_kind} is committed'
            )
            raise StopRun(
                Stopped.ids_unavailable(
                    f'the {owner_kind} links a {committed_kind} that is not committed',
                    f'the {owner_kind} {owner.id} links {committed_kind} {link.id}, {outside}',
                    fix=_RELINK_FIX if role == 'scenario_set' else IDS_FIX,
                )
            )


def check_ids(backend: Backend, ids: SuiteIds) -> Suite:
    """Every committed id exists and is published, and the links agree with the recorded lineage.

    Raises :class:`StopRun` (not measured: suite, naming the pair) otherwise.
    """
    ancestry = Ancestry(backend)
    scenarios = _require(ancestry, 'scenario set', ids.scenarios)
    evals = _require(ancestry, 'eval set', ids.evals)
    env = _require(ancestry, 'env', ids.env)
    corpus = tuple(_require(ancestry, 'corpus', corpus_id) for corpus_id in ids.corpus)
    _within(ancestry, evals, owner_kind='eval set', role='scenario_set', committed=(ids.scenarios,), committed_kind='scenario set')
    _within(ancestry, evals, owner_kind='eval set', role='corpus', committed=ids.corpus, committed_kind='corpus')
    _within(ancestry, scenarios, owner_kind='scenario set', role='corpus', committed=ids.corpus, committed_kind='corpus')
    _within(ancestry, scenarios, owner_kind='scenario set', role='env', committed=(ids.env,), committed_kind='env')
    if not evals.policy_tiers:
        raise StopRun(
            Stopped.ids_unavailable(
                'the policies of the committed eval set cannot be loaded', f'the policies of the committed eval set {ids.evals} cannot be loaded on the backend'
            )
        )
    unknown = sorted({tier for _, tier in evals.policy_tiers if tier not in TIERS})
    if unknown:
        raise StopRun(
            Stopped.ids_unavailable(
                'the committed eval set has a policy tier this SDK does not know',
                f'the committed eval set {ids.evals} names the tier {", ".join(unknown)}; this SDK knows {", ".join(TIERS)}',
                fix='update the SDK; a policy with a tier it cannot place would not count toward the check.',
            )
        )
    if not scenarios.scenarios:
        raise StopRun(
            Stopped.ids_unavailable('the committed scenario set lists no scenarios', f'the committed scenario set {ids.scenarios} lists no scenarios')
        )
    return Suite(scenarios, evals, env, corpus)


def check_credits(backend: Backend) -> str | None:
    """Stops when the org's credit state is not ``ok``. Returns the state it saw, or ``None`` when the backend exposes none (the
    first thread-route 402 is then the signal)."""
    state = backend.credits()
    if state is None:
        return None
    if state.state != 'ok':
        raise StopRun(Stopped.gateway_402(state.state, state.resets_at))
    return state.state


def check_smoke_names(suite: Suite, smoke: Sequence[str]) -> None:
    """Refuses (65) a smoke name the committed scenario set does not have, naming it."""
    known = set(suite.scenario_names)
    missing = [name for name in smoke if name not in known]
    if missing:
        raise StopRun(
            Stopped.refused(
                'ci.yaml smoke names a scenario the suite does not have',
                f'ci.yaml `smoke` names {", ".join(f"`{name}`" for name in missing)}, which the committed scenario set does not have; remove or rename it',
            )
        )
