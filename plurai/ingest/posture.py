"""Resolve the env-scoped **anonymization posture** from workspace config.

An env is either ``raw`` (default — data egresses unscrubbed under the customer's DPA
with Plurai) or ``anonymized`` (the opt-in local scrub runs fail-closed before egress).
The posture is a durable **per-env declaration**, never a per-run flag, so it lives in
workspace config; the first plugin ingest skill writes it once, from the operator's answer, and every
ingest run reads it.

The posture governs only the data uploaded while the env is built. What a simulation run produced
(:data:`RUN_KINDS`) always uploads as-is, stamped ``raw``, and never reads the posture.

Config contract (verbatim) — the first plugin ingest skill writes this, from the operator's answer; the SDK reads it:

* File: ``plurai.config.yaml``. Key: top-level ``anonymize: true | false`` (strict YAML bool).
* Discovery relative to ``--workspace-root``: (1) ``<workspace_root>/plurai.config.yaml``,
  (2) ``<workspace_root>/plurai_sim/plurai.config.yaml``.
* Merge: zero declarations → posture ``raw`` (default; the ingest CLI still refuses to upload the
  customer's own data on it — ``cli._dispatch(declaration_required=True)``); one → its value
  (``true`` → ``anonymized``, ``false`` → ``raw``); two agreeing → that value; two
  disagreeing → :class:`PostureConfigError` (fail closed); an existing-but-unparseable
  file or a non-bool value → :class:`PostureConfigError` (an existing config that cannot
  testify never silently defaults to raw).
* There is NO CLI flag, NO env var, and NO override of any kind — the posture is a
  human's once-per-env consent act, not something a run can flip.
* Edge cases: an explicit ``anonymize: null`` (or any other non-bool) fails closed — only
  a genuinely absent key/file "didn't say" and defaults to raw. "Strict bool" means the
  YAML-1.1 boolean type: unquoted ``yes``/``no``/``on``/``off`` parse as booleans and are
  accepted; quoted strings refuse. A candidate path that exists but is not a regular file
  (directory, broken symlink, FIFO) also fails closed — it cannot testify.

Finding the config files is not this module's business and lives in
:mod:`plurai.ingest.workspace_config`. What ``anonymize`` *means* — strict bool, non-bool fails
closed, and how two files declaring it are collapsed into one answer — stays here.

**Enforcement seam.** This module only *resolves* the declared posture; it does not
enforce consistency. The mixed-posture fail-closed check lives at the egress boundary in
:func:`plurai.ingest.disclosure.assert_posture_consistent`, and the
per-upload posture is pinned onto the wire via
:class:`plurai.ingest.manifest_models.IngestManifest`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from plurai.ingest.workspace_config import CONFIG_FILENAME, WorkspaceConfigError, discover_config_documents

logger = logging.getLogger(__name__)

#: The two postures an env can declare. ``raw`` is the default.
AnonymizationPosture = Literal['raw', 'anonymized']

#: Top-level config key whose strict YAML bool declares the posture.
POSTURE_CONFIG_KEY = 'anonymize'

#: The ingest kinds that carry a simulation run's own output. The posture does not govern them: they
#: upload as-is in every env, and neither fix nor meet the env's posture in the mixed-posture fence.
RUN_KINDS = frozenset({'run-artifacts', 'run-report'})


class PostureConfigError(WorkspaceConfigError):
    """The anonymization posture could not be resolved from config (fail-closed).

    Raised when an existing config file cannot testify to a single posture: it is
    unparseable, its ``anonymize`` value is not a strict YAML bool, or two discovered
    config files disagree. A posture that cannot be resolved must never silently default
    to raw — the operator has to fix the config."""


@dataclass(frozen=True)
class PostureResolution:
    """The resolved posture plus its provenance (``source is None`` ⇒ the default raw)."""

    posture: AnonymizationPosture
    source: Path | None


def resolve_posture(workspace_root: str | Path = '.') -> PostureResolution:
    """Resolve the env's :data:`AnonymizationPosture` from workspace config (see module docstring).

    Reads the discovery paths under ``workspace_root`` and merges their declarations:
    none → default ``raw``; one → its value; agreeing pair → that value; a disagreeing
    pair, an unparseable file, or a non-bool value → :class:`PostureConfigError`.
    """
    declarations: list[tuple[Path, bool]] = []
    for path, document in discover_config_documents(workspace_root, error=PostureConfigError):
        if POSTURE_CONFIG_KEY not in document:
            continue  # no anonymize declaration → contributes nothing
        value = document[POSTURE_CONFIG_KEY]
        if not isinstance(value, bool):
            raise PostureConfigError(
                f'{path}: {POSTURE_CONFIG_KEY!r} must be a YAML boolean (true/false), got {value!r}. '
                'A non-bool posture declaration fails closed rather than silently defaulting to raw.'
            )
        declarations.append((path, value))

    if not declarations:
        logger.info('anonymization posture: raw (default — no %s declaration found under %s)', POSTURE_CONFIG_KEY, Path(workspace_root))
        return PostureResolution(posture='raw', source=None)
    if len({declared for _, declared in declarations}) > 1:
        # The declaration is env-scoped, so two configs under one root cannot both be right: this
        # is an unresolvable env, not a precedence puzzle a discovery order gets to settle.
        sources = ', '.join(f'{path} → {declared!r}' for path, declared in declarations)
        raise PostureConfigError(
            f'conflicting {POSTURE_CONFIG_KEY!r} declarations ({sources}); the posture is env-scoped and must agree. '
            'Reconcile the config files (or remove one) so the env declares a single value.'
        )

    source, value = declarations[0]
    posture: AnonymizationPosture = 'anonymized' if value else 'raw'
    logger.info('anonymization posture: %s (from %s: %s=%r)', posture, source, POSTURE_CONFIG_KEY, value)
    return PostureResolution(posture=posture, source=source)


__all__ = [
    'CONFIG_FILENAME',
    'POSTURE_CONFIG_KEY',
    'RUN_KINDS',
    'AnonymizationPosture',
    'PostureConfigError',
    'PostureResolution',
    'resolve_posture',
]
