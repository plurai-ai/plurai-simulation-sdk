"""Client-side dataset upload: fetch -> gather -> dispatch -> persist.

Dataset upload is a CLIENT concern, not a backend one: observability
lives client-side, so this module owns the end-to-end flow:

  1. FETCH    — pull the scenario set from the backend via the KEY-FREE
                ``GET /scenario-sets/{id}`` endpoint (the server holds no sink
                credentials).
  2. GATHER   — adapt the fetched payload into the
                ``{folder_name, inputs, metadata}`` dicts the dispatcher expects
                (consuming the fetched set, never local dirs).
  3. DISPATCH — push the batch to the SINGLE active sink by calling
                :meth:`sink.sync_scenarios` on the already-built active sink the
                orchestrator holds (single-active-sink selection belongs to the
                factory that built that sink, not here).
  4. PERSIST  — round-trip the provider idempotency handles (e.g. langsmith
                example ids / *_dataset_items) into a local JSON file next to the
                run artifacts so a re-run reuses the prior items instead of
                creating duplicates. The CLIENT owns that mapping.

Everything here is BEST-EFFORT for the upload itself: a sink/transport failure
returns a failure :class:`UploadResult` rather than raising, so an upload run can
report a problem without aborting. The FETCH step does surface a clear error
(404 -> unknown set id) because the caller explicitly asked to upload a set.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plurai.observability.sink_protocol import ObservabilitySink
    from plurai.orchestrator import ScenarioSetFetchResponse

logger = logging.getLogger(__name__)

#: ``NoOpSink.name`` — the sentinel provider meaning "no observability sink configured",
#: as opposed to a real sink that uploaded nothing. Duplicated rather than imported to
#: keep this module free of a sink_protocol import cycle.
_NOOP_PROVIDER = 'noop'

# Filename for the client-side idempotency mapping, written next to the run
# artifacts (the same ``<out>/`` dir ``plurai-run-simulation`` writes
# per-scenario folders into). Holds the provider handles a re-run reuses
# (langsmith_examples / langfuse_dataset_items / braintrust_dataset_items /
# arize_dataset_id). The CLIENT owns this mapping.
DATASET_UPLOAD_METADATA_FILENAME = 'dataset_upload_metadata.json'

__all__ = [
    'DATASET_UPLOAD_METADATA_FILENAME',
    'UploadResult',
    'gather_upload_scenarios',
    'load_upload_metadata',
    'save_upload_metadata',
    'upload_scenarios_from_fetch',
]


class UploadResult:
    """Outcome of one dataset-upload pass (a plain result, never raised).

    ``provider`` / ``summary_line`` come from the active sink's
    ``ProviderSyncResult``; on a TRUE no-op (no sink enabled / nothing to upload)
    both are ``None`` and ``uploaded`` is ``False`` (``ok=True``).

    ``ok`` is ``False`` when the upload was ATTEMPTED but failed — either an
    outright failure (transport / sink error, captured in ``error``) OR a PARTIAL
    success where some items uploaded and some did not (``ProviderSyncResult.failed
    > 0``). In the partial case ``uploaded`` stays ``True`` (handles WERE persisted)
    but ``ok`` is ``False`` so a lossy run is never reported green; ``summary_line``
    then reads like "uploaded 42/50 item(s) (8 failed — see log)". The caller
    surfaces all of this without aborting the run.

    ``skipped`` counts scenarios dropped at the GATHER boundary (missing
    ``folder_name`` or empty ``inputs``) before any upload attempt.  These are
    informational — they do NOT affect ``ok``.

    ``selected`` / ``set_size`` are set only when the pass was scoped to a subset of the
    fetched set (``scenario_ids``): how many of the set's scenarios the pass covered and how
    many the set holds. Scenarios left out by that scope are not ``skipped`` — that field
    means incomplete entries — so the scope is stated apart from it.
    """

    __slots__ = ('error', 'metadata_path', 'ok', 'provider', 'selected', 'set_size', 'skipped', 'summary_line', 'uploaded')

    def __init__(
        self,
        *,
        ok: bool,
        uploaded: bool,
        provider: str | None = None,
        summary_line: str | None = None,
        metadata_path: Path | None = None,
        error: str | None = None,
        skipped: int = 0,
        selected: int | None = None,
        set_size: int | None = None,
    ) -> None:
        self.ok = ok
        self.uploaded = uploaded
        self.provider = provider
        self.summary_line = summary_line
        self.metadata_path = metadata_path
        self.error = error
        self.skipped = skipped
        self.selected = selected
        self.set_size = set_size

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f'UploadResult(ok={self.ok!r}, uploaded={self.uploaded!r}, '
            f'provider={self.provider!r}, summary_line={self.summary_line!r}, '
            f'metadata_path={self.metadata_path!r}, error={self.error!r}, '
            f'skipped={self.skipped!r}, selected={self.selected!r}, set_size={self.set_size!r})'
        )

    def result_line(self) -> str:
        """A single operator-facing status line for this upload pass."""
        skipped_note = f' (skipped {self.skipped} incomplete)' if self.skipped else ''
        # Said only when the pass covered fewer scenarios than the set holds, so a count in the
        # sink's own summary is never read as the size of the set.
        if self.selected is not None and self.set_size is not None and self.selected < self.set_size:
            skipped_note += f' — {self.selected} of {self.set_size} scenario(s) in the set: the ones this run covers'
        if self.error is not None:
            return f'[dataset-upload] FAILED: {self.error}{skipped_note}'
        # Partial success: items WERE uploaded but some failed (ok is False). Surface
        # the sink's "X/Y (Z failed)" summary, flagged so the operator sees the loss.
        if self.uploaded and not self.ok:
            return f'[dataset-upload] PARTIAL: {self.summary_line}{skipped_note}'
        if self.uploaded:
            return f'[dataset-upload] {self.summary_line}{skipped_note}'
        # Nothing uploaded. Only claim "no sink enabled" when that is actually why — a
        # REAL provider that uploaded zero items is a different story (an empty or wholly
        # incomplete set), and for a sink whose experiment needs a populated dataset up
        # front it is a genuine problem, not a benign skip. ``NoOpSink`` IS the
        # no-sink-configured case, so it says so.
        if self.provider and self.provider != _NOOP_PROVIDER:
            return f'[dataset-upload] nothing uploaded to {self.provider}: {self.summary_line or "no complete scenarios in the set"}{skipped_note}'
        return f'[dataset-upload] no observability sink enabled / nothing to upload — skipped.{skipped_note}'


def _in_scope(fetched: ScenarioSetFetchResponse, scenario_ids: Sequence[str] | None) -> list[Any]:
    """The fetched entries a pass covers: all of them, or those whose ``folder_name`` is in ``scenario_ids``."""
    entries = fetched.scenarios or []
    if scenario_ids is None:
        return list(entries)
    wanted = set(scenario_ids)
    return [entry for entry in entries if entry.folder_name in wanted]


def gather_upload_scenarios(fetched: ScenarioSetFetchResponse, *, scenario_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Adapt a fetched scenario set into the dispatcher's scenario-dict list.

    Consumes the FETCHED payload, never local dirs: the server already does the
    disk read + incomplete-folder skip + scenario-block flatten, so each fetched
    entry is a complete ``ScenarioUploadEntry``. This only reshapes it into the plain
    ``{folder_name, inputs, metadata}`` dicts ``sink.sync_scenarios``
    expects (``folder_name`` keys the idempotency mapping).

    Entries with a falsy ``folder_name`` or empty ``inputs`` are skipped
    (defensive: the server already filters these, but a corrupt payload could
    slip through). A warning is emitted when any are skipped so the operator
    knows the true denominator before the upload attempt.

    ``scenario_ids`` narrows the batch to the entries whose ``folder_name`` is listed
    (``None`` keeps the whole set); entries outside it are not counted as skipped.
    """
    return _adapt_entries(_in_scope(fetched, scenario_ids))


def _adapt_entries(entries: list[Any]) -> list[dict[str, Any]]:
    """Reshape in-scope fetched entries into dispatcher dicts (see :func:`gather_upload_scenarios`)."""
    scenarios: list[dict[str, Any]] = []
    skipped = 0
    total = len(entries)
    for entry in entries:
        # Defensive: the server skips incomplete folders, but a folder with no
        # inputs would key an empty example, so it is skipped.
        if not entry.folder_name or not entry.inputs:
            skipped += 1
            continue
        scenarios.append(
            {
                'folder_name': entry.folder_name,
                'inputs': entry.inputs,
                'metadata': entry.metadata or {},
            }
        )
    if skipped:
        logger.warning(
            'Dataset upload: skipped %d/%d incomplete scenario(s) (missing folder_name or empty inputs).',
            skipped,
            total,
        )
    return scenarios


def load_upload_metadata(out_dir: Path) -> dict[str, Any]:
    """Load the client-side idempotency mapping from ``<out>/<filename>``.

    Returns ``{}`` when the file is absent (first run) or unparseable — a
    corrupt/empty mapping degrades to "create fresh items", which is safe (it
    just re-creates examples rather than reusing them). Never raises.
    """
    path = out_dir / DATASET_UPLOAD_METADATA_FILENAME
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('Could not read dataset-upload metadata at %s (%s); treating as first run.', path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def save_upload_metadata(out_dir: Path, metadata: dict[str, Any]) -> Path:
    """Persist the idempotency mapping to ``<out>/<filename>`` and return its path.

    Written with ``indent=2`` to match the JSON artifact convention in
    ``plurai.run_simulation``. Best-effort: a write failure is logged and the
    path is still returned (the caller treats the mapping as not-persisted).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / DATASET_UPLOAD_METADATA_FILENAME
    try:
        path.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    except OSError as exc:
        logger.warning('Could not write dataset-upload metadata to %s (%s); re-runs may create duplicate items.', path, exc)
    return path


async def upload_scenarios_from_fetch(
    fetched: ScenarioSetFetchResponse,
    *,
    sink: ObservabilitySink,
    out_dir: Path,
    scenario_ids: Sequence[str] | None = None,
) -> UploadResult:
    """Run gather -> dispatch -> persist for an already-fetched scenario set.

    Loads the prior idempotency mapping from ``out_dir`` BEFORE dispatch and
    saves the merged mapping AFTER, so a second upload of the same set reuses the
    prior provider handles (no duplicate creates). Best-effort: any sink /
    transport failure is caught and returned as a failed :class:`UploadResult`
    rather than raised — the upload never aborts the surrounding run.

    ``sink`` is the already-built active sink the orchestrator holds (selection
    happened when the factory built it); the dispatch calls its
    :meth:`sync_scenarios` directly. ``sink.sync_scenarios`` ALWAYS returns a
    :class:`~plurai.observability.dataset_types.ProviderSyncResult` — a no-op
    (NoOpSink / SDK missing / disabled) returns ``ok=True, uploaded=0``, never
    ``None``.

    Async: the blocking ``sink.sync_scenarios`` (drives the sink SDKs
    synchronously) runs inside via ``asyncio.to_thread`` so the caller awaits this
    directly without an outer thread hop.

    ``scenario_ids`` uploads only the scenarios a run covers: the dataset is the
    experiment's input, so examples the run never executes would show as empty rows,
    inflate the count the operator sees and send customer scenario content that nothing
    reads. The upload is idempotent by folder (the persisted mapping), so a later run
    over other scenarios adds its examples to the same dataset. ``None`` uploads the
    whole set. Scoping the upload does not scope what reads the dataset: a vendor-driven
    experiment (Arize ``run_experiment_set``) is handed every accumulated example and must
    skip the ones outside its own ``scenario_ids``.
    """
    entries = _in_scope(fetched, scenario_ids)
    scenarios = _adapt_entries(entries)
    in_scope = len(entries)
    skipped = in_scope - len(scenarios)
    # Recorded on the result only for a scoped pass; ``None`` leaves the result line unqualified.
    selected = in_scope if scenario_ids is not None else None
    set_size = len(fetched.scenarios or []) if scenario_ids is not None else None
    if not scenarios:
        logger.info('Dataset upload: no complete scenarios in set %s; nothing to upload.', fetched.scenario_set_id)
        return UploadResult(ok=True, uploaded=False, skipped=skipped, selected=selected, set_size=set_size)

    # Load-before: the prior mapping carries each provider's handles
    # (langsmith_examples / *_dataset_items / arize_dataset_id) so the dispatcher
    # upserts in place instead of creating duplicates.
    existing_metadata = load_upload_metadata(out_dir)

    try:
        result = await asyncio.to_thread(
            sink.sync_scenarios,
            dataset_name=fetched.dataset_name,
            dataset_description=fetched.dataset_description or None,
            # ``gather_upload_scenarios`` produces the plain ``{folder_name, inputs,
            # metadata}`` dicts the sinks duck-type; ``ScenarioUploadEntry`` (the
            # tightened param type) mirrors exactly that shape — the dispatch reads
            # only those keys, so the dict batch is the intended input here.
            scenarios=scenarios,  # type: ignore[arg-type]
            existing_metadata=existing_metadata,
        )
    except Exception as exc:  # noqa: BLE001 — upload is best-effort; never abort the run
        logger.exception('Dataset upload failed for set %s.', fetched.scenario_set_id)
        return UploadResult(ok=False, uploaded=False, error=f'{type(exc).__name__}: {exc}', skipped=skipped, selected=selected, set_size=set_size)

    if not result.ok:
        # Attempted but FAILED outright (dataset get-or-create transport error /
        # unrecoverable blank dataset id / ALL items failed). Nothing usable landed,
        # so persist nothing and report the FAILED branch.
        err = result.summary_line or '; '.join(result.errors) or f'{result.provider} dataset upload failed'
        return UploadResult(ok=False, uploaded=False, provider=result.provider, error=err, skipped=skipped, selected=selected, set_size=set_size)

    if result.uploaded == 0:
        # True no-op: no sink configured / SDK missing / disabled / empty batch.
        # ``metadata_updates`` will be empty; nothing to persist.
        return UploadResult(
            ok=True,
            uploaded=False,
            provider=result.provider or None,
            summary_line=result.summary_line or None,
            skipped=skipped,
            selected=selected,
            set_size=set_size,
        )

    # Save-after: merge the provider handles back into the mapping so the next
    # upload reuses them. On a PARTIAL success (failed > 0) ``metadata_updates``
    # already carries only the successfully-uploaded handles (each per-item loop
    # records a result entry only on success), so failed folders are NOT persisted
    # as if done.
    existing_metadata.update(result.metadata_updates)
    metadata_path = save_upload_metadata(out_dir, existing_metadata)
    # Any per-item failure makes the overall result NOT-ok (a lossy upload is never
    # reported green), even though some handles WERE uploaded + persisted.
    return UploadResult(
        ok=result.failed == 0,
        uploaded=True,
        provider=result.provider,
        summary_line=result.summary_line,
        metadata_path=metadata_path,
        skipped=skipped,
        selected=selected,
        set_size=set_size,
    )
