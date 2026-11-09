"""The agent's RENDER / ACTION surface, as ``plurai.config.yaml`` declares it.

Some agents act through a channel that is not a tool call: a UI render (``show_table``,
``show_image``), a navigation, a hand-off verb. There is no dispatch to mock, so such an agent
declares ``tools: {}`` by design, so without this key its surface is declared nowhere typed: no
completeness check sees it, the scan carries nothing about it, and an agent that invents a
``show_video`` it does not have is graded by nobody.

The plugin's ``verify_config.py`` gates the key's SHAPE at authoring time and its ``lint_prd.py``
reads the key against the PRD; this module is the SDK's reading of the same key, in the same
vocabulary — one key, one meaning:

* File: ``plurai.config.yaml``. Key: top-level ``render_actions``, a list of the verb names the
  agent emits. Found under a root at the two rungs every workspace declaration is looked for
  (:data:`plurai.ingest.workspace_config.CONFIG_CANDIDATE_RELPATHS`).
* ABSENT key: the config says nothing about a render surface. :func:`parse_render_actions` returns
  ``None``, and nothing downstream may read that as "no render actions". ``render_actions: []``
  is a DECLARATION — "none, by declaration" — and reads as an empty surface. The two are different
  facts and stay different all the way onto the scan artifact.
* MALFORMED — not a list, a non-string or blank entry, a verb listed twice:
  :class:`RenderActionsConfigError`, fail closed. A malformed declaration read as "undeclared"
  puts the surface right back where this key exists to take it from — invisible to every check.
* Two configs under one root that both declare and disagree: an error, not a precedence puzzle.
  The surface is env-scoped, the same rule ``anonymize`` follows.

Read by :mod:`plurai._mocking.code_surface` (the typed slice of the config the projector parses)
and by :mod:`plurai.scan_contract_cli`, which carries the surface onto ``scan_contract.json`` so
the backend's ``RENDER_ACTIONS`` fact check has a declared surface to compare invented verbs
against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plurai.ingest.workspace_config import CONFIG_FILENAME, WorkspaceConfigError, discover_config_documents

#: Top-level ``plurai.config.yaml`` key. ONE vocabulary with the plugin's
#: ``verify_config._KNOWN_ROOT_KEYS`` entry and ``lint_prd.RENDER_ACTIONS_KEY`` — and the name of
#: the field the scan artifact carries the surface under.
RENDER_ACTIONS_KEY = 'render_actions'


class RenderActionsConfigError(WorkspaceConfigError):
    """``render_actions`` exists but cannot testify to a surface (fail closed).

    A present-but-broken declaration must never degrade to "nothing declared": undeclared is the
    exact state this key exists to end, so a malformed one has to stop the reader, not be dropped.
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


def parse_render_actions(document: Mapping[str, Any], *, source: Path | None = None) -> RenderSurface | None:
    """The surface one loaded config declares: ``None`` when the key is absent, else the verbs.

    Raises :class:`RenderActionsConfigError` on a malformed value. ``source`` is named in every
    error so the operator is sent to the file, not to a key.
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


def discover_render_surface(roots: Sequence[Path | str]) -> RenderSurface | None:
    """The surface declared under ``roots``, or ``None`` when no config under them declares one.

    Each root is read the way every workspace declaration is (two rungs; a config that exists and
    cannot be read raises, whether or not it carries this key — an existing config that cannot
    testify never silently defaults). A file reached from two roots — the workdir given directly
    and the repo root it sits under, which is how the plugin's scan step passes them — is read
    once. Two DIFFERENT files that both declare and disagree raise: the surface is env-scoped, and
    a discovery order does not get to settle which one the env meant.
    """
    seen: set[Path] = set()
    declared: list[RenderSurface] = []
    for root in roots:
        for path, document in discover_config_documents(root, error=RenderActionsConfigError):
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


__all__ = [
    'RENDER_ACTIONS_KEY',
    'RenderActionsConfigError',
    'RenderSurface',
    'discover_render_surface',
    'parse_render_actions',
]
