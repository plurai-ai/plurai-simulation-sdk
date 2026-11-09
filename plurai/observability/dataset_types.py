"""Vendor-neutral dataset-upload result type + the dataset JOIN KEY.

:class:`ProviderSyncResult` is the shape every sink's ``sync_scenarios`` returns.
It lives in this dependency-free LEAF module (no sink imports) so the four sink
modules can import it EAGERLY at module top without any circular import — the
dataset-upload path consumes the result a sink returns from ``sync_scenarios``.

:func:`dataset_join_key` lives here for the same reason: the three sinks that resolve
a dataset row client-side (LangSmith / Langfuse / Braintrust) need it, and without one
owner each hand-rolls ``folder or name``. Arize does not — it recovers the identity from
the fetched dataset row itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: The metadata key every sink's dataset upload stamps to identify a scenario's row.
#: Uploads write it (``sync_scenarios_to_*``); the per-scenario recording path reads it
#: back to link its run to that row.
#:
#: Named here so the two ends of the LangSmith join cannot drift apart. The
#: ``arize``/``braintrust``/``langfuse`` upload sites spell the same literal, so a rename here
#: must change them too.
SCENARIO_FOLDER_KEY = 'scenario_folder'


def dataset_join_key(*, scenario_folder: str | None, scenario_name: str) -> str:
    """THE single source of the scenario→dataset-row join key.

    A scenario carries TWO identities and they are NOT interchangeable:

    * **folder** — ``scenario_01_onboarding_final_scenario_01``. The
      scenario folder; ``ScenarioUploadEntry.folder_name``, and the
      ``scenario_id`` the runner drives each scenario with. Every sink's dataset
      upload stamps it as :data:`SCENARIO_FOLDER_KEY`. This is the JOIN KEY.
    * **name** — ``onboarding_final_scenario_01``. The server's
      ``scenario_metadata['scenario_name']``, a DISPLAY label: it names the
      record (``scenario:<name>``) and rides along in record metadata.

    They coincide often enough that assuming they ALWAYS do looks safe. They do not,
    and the assumption costs a run its Experiments grid: an upload that keys rows by
    folder against a recording path that looks them up by name misses on every lookup,
    so each scenario mints a DUPLICATE row instead of joining the uploaded one.
    Join on the folder; label with the name.

    ``scenario_folder=None`` means the caller supplies no folder (or a sink is invoked
    directly in a test); fall back to the name, which is still correct whenever the two
    identities coincide. One case where it is never
    correct: ``scenario_name`` itself falls back to the thread id when the server omits it
    (``Orchestrator._scenario_scope``), and a thread id matches no folder in any dataset —
    so that run mints a fresh lean example every time.
    """
    return scenario_folder or scenario_name


@dataclass(frozen=True, slots=True)
class ProviderSyncResult:
    """Outcome of the active sink's dataset upload pass.

    Immutable record returned ALWAYS by the active sink's :meth:`sync_scenarios`
    — ``None`` is never returned; a true no-op (no sink / SDK missing / empty
    batch) is represented as ``ProviderSyncResult(ok=True, uploaded=0, failed=0,
    metadata_updates={}, ...)``.

    The caller merges ``metadata_updates`` into the on-disk dataset-metadata
    mapping (for idempotent re-runs) and may surface ``summary_line`` to the
    operator.

    Success/failure semantics (the CONTRACT ``sync_scenarios`` upholds):

    * ``ok=True`` + ``uploaded > 0`` + ``failed == 0`` ⇒ full success (every
      item uploaded).
    * ``ok=True`` + ``uploaded == 0`` + ``failed == 0`` ⇒ no-op (disabled /
      SDK missing / empty batch — nothing was attempted).
    * ``ok=True`` + ``uploaded > 0`` + ``failed > 0`` ⇒ PARTIAL success — some
      items uploaded, some failed; ``failed`` is the lost-item tally and
      ``errors`` carries short per-failure strings. The dataset-upload caller
      surfaces this as an overall FAILURE (``UploadResult.ok=False``) so a
      lossy run is never reported green.
    * ``ok=False`` ⇒ the attempt FAILED outright (dataset get-or-create transport
      error, an unrecoverable blank dataset handle, or ALL items failed); no
      successfully-uploaded handles are persisted.

    The cross-field contract above is enforced in ``__post_init__``.
    """

    provider: str
    # Keys + values to merge into the dataset-metadata mapping (read back as
    # ``existing_metadata`` on the next run). Each sink's contract is documented
    # on the relevant sync_scenarios_to_* fn. Read-only by callers — annotated
    # as Mapping to prevent unintentional mutation of the produced dict.
    metadata_updates: Mapping[str, Any]
    # Plain (no Rich markup) human-readable line — sink label, item count, and
    # dataset name/id — for an operator-facing confirmation log/print.
    summary_line: str
    # False ⇒ the attempt FAILED outright (no items landed / unrecoverable
    # dataset handle). True ⇒ at least the dataset op succeeded; consult
    # ``uploaded`` / ``failed`` for whether items actually landed.
    ok: bool = True
    # Count of items successfully uploaded in this pass (0 for no-ops and
    # outright failures). Combined with ``failed`` gives the full X/Y picture.
    uploaded: int = 0
    # Count of items that failed to upsert inside an otherwise-successful pass
    # (best-effort per-item loops tally failures rather than aborting).
    failed: int = 0
    # Short per-failure strings (truncated) for the operator log; never the full
    # tracebacks.
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.uploaded < 0:
            raise ValueError(f'ProviderSyncResult.uploaded must be >= 0, got {self.uploaded!r}')
        if self.failed < 0:
            raise ValueError(f'ProviderSyncResult.failed must be >= 0, got {self.failed!r}')
        if not self.ok and self.uploaded != 0:
            raise ValueError(f'ProviderSyncResult invariant violated: ok=False implies uploaded==0, but got uploaded={self.uploaded!r}')
        if not self.ok and not self.errors:
            raise ValueError('ProviderSyncResult invariant violated: ok=False requires at least one error string in errors')

    @property
    def is_noop(self) -> bool:
        """True when the sink was disabled/skipped and nothing was attempted."""
        return self.ok and self.uploaded == 0 and self.failed == 0

    @property
    def is_partial(self) -> bool:
        """True when some items uploaded and some failed (lossy run)."""
        return self.uploaded > 0 and self.failed > 0

    @property
    def is_full_success(self) -> bool:
        """True when all attempted items uploaded without any failures."""
        return self.ok and self.uploaded > 0 and self.failed == 0
