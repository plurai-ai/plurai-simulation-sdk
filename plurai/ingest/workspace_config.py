"""Discover and read the env's workspace config — the file every env-scoped DECLARATION lives in.

``plurai.config.yaml`` is where a human writes down, once per env, the facts a run must not
guess. Today exactly one such fact is read here: whether the env anonymizes before egress
(``anonymize:``). This module owns the part that is about the FILE rather than about
that key — *finding* it — so the discovery order and the failure posture are stated once, in one
place, instead of being re-derived by whatever declaration is added next.

What lives here (and why it is not in the declaring module):

* **Layout.** :data:`WORKSPACE_CANDIDATE_ROOTS`, the two roots every committed artifact is looked
  for under. ``plurai_adapters/`` (:mod:`plurai.trace_adapter`) derives its own discovery order
  from the same tuple, so a workspace has one layout to learn rather than one per artifact kind.
* **Discovery.** Two paths relative to ``--workspace-root``, read in this order:
  (1) ``<workspace_root>/plurai.config.yaml``, (2) ``<workspace_root>/plurai_sim/plurai.config.yaml``.
* **Fail-closed reading.** A candidate that exists but cannot testify — not a regular file,
  unreadable, unparseable YAML, or not a mapping — raises rather than being skipped. A
  genuinely absent file is the ONLY thing that "didn't say".

What is NOT here: what a key means, which values are legal, what a legal-but-unknown value does,
and how several files declaring it are collapsed into the env's one answer. That is the declaring
module's business — see :mod:`plurai.ingest.posture`.

Callers pass their own exception class, so a config failure keeps surfacing under the name of
the gate that failed (:class:`plurai.ingest.posture.PostureConfigError` today). It must derive
from :class:`WorkspaceConfigError`, which is fail-closed by construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: Workspace config file that carries the env's declarations.
CONFIG_FILENAME = 'plurai.config.yaml'
#: The workspace layout itself: the roots every COMMITTED artifact is looked for under, relative to
#: ``--workspace-root`` and in the order they are read. Stated once so a workspace has ONE layout to
#: learn rather than one per artifact kind — ``plurai.config.yaml`` below and ``plurai_adapters/``
#: (:mod:`plurai.trace_adapter`) both derive their discovery paths from this tuple.
WORKSPACE_CANDIDATE_ROOTS: tuple[Path, ...] = (Path(), Path('plurai_sim'))
#: Config discovery paths, relative to the workspace root, in the order they are read.
CONFIG_CANDIDATE_RELPATHS: tuple[Path, ...] = tuple(root / CONFIG_FILENAME for root in WORKSPACE_CANDIDATE_ROOTS)


class WorkspaceConfigError(ValueError):
    """A workspace config exists but cannot testify to the declaration being read (fail-closed).

    Base class only — gates raise their own subclass so the operator sees which declaration
    failed, not just that some config is broken."""


def discover_config_documents(
    workspace_root: str | Path = '.',
    *,
    error: type[WorkspaceConfigError] = WorkspaceConfigError,
) -> list[tuple[Path, dict[str, Any]]]:
    """Return ``(path, document)`` for every workspace config that exists under ``workspace_root``.

    Absent files are skipped silently — that is the only "didn't say". A file that exists but
    cannot be read as a YAML mapping raises ``error``: an existing config that cannot testify
    must never be treated as silence. An empty file parses to an empty mapping (it exists, it
    is well-formed, it simply declares nothing).
    """
    documents: list[tuple[Path, dict[str, Any]]] = []
    for relpath in CONFIG_CANDIDATE_RELPATHS:
        candidate = Path(workspace_root) / relpath
        if not candidate.exists() and not candidate.is_symlink():
            continue  # genuinely absent — contributes nothing
        if not candidate.is_file():
            raise error(
                f'{candidate} exists but is not a regular file; it cannot testify to a declaration. '
                'Fix or remove it — an existing config that cannot testify never silently defaults.'
            )
        try:
            raw = candidate.read_text(encoding='utf-8')
        # ``UnicodeDecodeError`` alongside ``OSError``, as in ``trace_convert._read_source`` and
        # ``trace_adapter._load_document``: it is a ``ValueError``, so without its own clause a
        # config saved as UTF-16 escapes this fail-closed family entirely and reaches the operator
        # as a raw traceback — the one presentation this module exists to rule out. A file that
        # cannot be decoded cannot testify, exactly like one that cannot be opened.
        except (UnicodeDecodeError, OSError) as e:
            raise error(f'{candidate} exists but could not be read: {e}. Fix or remove it (an unreadable config cannot testify).') from e
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise error(f'{candidate} is not parseable YAML ({e}). Fix or remove it — an existing config that cannot be parsed never silently defaults.') from e
        if data is None:
            data = {}  # empty file → a config that declares nothing
        if not isinstance(data, dict):
            raise error(f'{candidate} must be a YAML mapping to declare anything; got {type(data).__name__}.')
        documents.append((candidate, data))
    return documents


def bench_config_roots(simulate_path: Path) -> tuple[Path, Path]:
    """The two rungs a declaration beside a harness is read at, in order: the repo root above the
    harness directory, then the harness directory itself (``plurai_sim/``)."""
    return (simulate_path.parent.parent, simulate_path.parent)


__all__ = [
    'CONFIG_CANDIDATE_RELPATHS',
    'CONFIG_FILENAME',
    'WORKSPACE_CANDIDATE_ROOTS',
    'WorkspaceConfigError',
    'bench_config_roots',
    'discover_config_documents',
]
