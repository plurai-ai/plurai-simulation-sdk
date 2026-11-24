"""The suite diff of a CI run and of ``plurai suite diff``: the committed ids at a git ref against the working tree, and the
backend's content for both.

A failure to read the base or the content is a note, never a stop; only a rejected or refused token (401, 403) or a backend that
refuses this SDK version (426) ends the command.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

import yaml

from plurai.ci.backend import REFUSALS, Backend, BackendError, BackendUnreachable
from plurai.ci.gitrepo import git
from plurai.ci.config import CiConfig, Refusal, SuiteIds, parse_config, read_ids
from plurai.report import SuiteDiff, SuiteSide, render_ids_only, render_suite_diff

__all__ = ['BaseUnreadable', 'Diffed', 'SuiteNames', 'base_ids_at', 'diff_against', 'git_reader', 'ids_map', 'side_of', 'suite_names']


class BaseUnreadable(Exception):  # noqa: N818
    """The ids at the base ref cannot be read: the ref is not in this checkout (a shallow clone), or the files are not in it."""


GitReader = Callable[[str], str | None]


def git_reader(workdir: Path, ref: str) -> GitReader:
    """``read(relative_path)`` for files of ``workdir`` as they are at ``ref``; ``None`` for a file that is absent there.

    Raises :class:`BaseUnreadable` when ``ref`` is not a commit of this checkout.
    """
    if not git(workdir, 'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}').ok:
        raise BaseUnreadable(f'the base commit {ref[:12]} is not in this checkout; fetch it (the checkout may be shallow)')
    prefix = git(workdir, 'rev-parse', '--show-prefix').stdout.strip()

    def read(relative: str) -> str | None:
        shown = git(workdir, 'show', f'{ref}:{prefix}{relative}')
        return shown.stdout if shown.ok else None

    return read


def ids_map(ids: SuiteIds) -> dict[str, str]:
    return {'scenarios': ids.scenarios, 'evals': ids.evals, 'env': ids.env, 'corpus': ids.corpus_text}


def base_ids_at(workdir: Path, ref: str, head_config: CiConfig) -> dict[str, str]:
    """The four committed ids at ``ref``. Raises :class:`BaseUnreadable` when git or the files cannot answer."""
    read = git_reader(workdir, ref)
    text = read('ci.yaml')
    try:
        config = parse_config(yaml.safe_load(text)) if text is not None else head_config
        return ids_map(read_ids(workdir, config, read=read))
    except (Refusal, yaml.YAMLError) as error:
        raise BaseUnreadable(f'the base commit has no complete set of committed ids ({str(error).split(";")[0]})') from error


def _fetch_yaml(backend: Backend, path: str, route: str) -> Any:  # noqa: ANN401
    response = backend.get(path, route=route)
    if response.status_code != 200:
        return None
    try:
        return yaml.safe_load(response.text)
    except yaml.YAMLError:
        return None


def _corpus_documents(backend: Backend, corpus_ids: str) -> dict[str, str] | None:
    """The digest of each document of the committed corpora by name, ``{}`` for no corpus, ``None`` when any corpus has no
    listing to read or cannot be read: the diff then says the documents cannot be compared and still shows everything else. A
    rejected or refused token, or a refused SDK version, is not a corpus problem and passes through.

    A name held by several corpora carries their digests joined, so a change to any one of them still reads as a change.
    """
    digests: dict[str, list[str]] = {}
    for corpus_id in filter(None, corpus_ids.split(',')):
        try:
            artifact = backend.artifact(corpus_id)
        except REFUSALS:
            raise
        except BackendError:
            return None
        if artifact is None or artifact.documents is None:
            return None
        for document in artifact.documents:
            digests.setdefault(document.name, []).append(document.sha256)
    return {name: ','.join(sorted(found)) for name, found in digests.items()}


def side_of(backend: Backend, ids: Mapping[str, str]) -> SuiteSide:
    """The suite with these ids as the backend holds it. Raises :class:`BackendError` when something cannot be read; a token
    or version refusal passes through."""
    scenarios: dict[str, dict[str, Any]] | None = None
    response = backend.get(f'/scenario-sets/{ids["scenarios"]}', route='GET /scenario-sets/{id}')
    if response.status_code != 200:
        raise BackendUnreachable(f'the scenario set answered HTTP {response.status_code}')
    try:
        rows = response.json().get('scenarios')
    except (ValueError, AttributeError) as error:
        raise BackendUnreachable('the scenario set did not answer JSON') from error
    if isinstance(rows, list):
        scenarios = {
            str(r['folder_name']): {'inputs': r.get('inputs') or {}, 'metadata': r.get('metadata') or {}}
            for r in rows
            if isinstance(r, Mapping) and 'folder_name' in r
        }
    policies: dict[str, dict[str, Any]] | None = None
    body = _fetch_yaml(backend, f'/eval-sets/{ids["evals"]}/policies.yaml', 'GET /eval-sets/{id}/policies.yaml')
    artifact = backend.artifact(ids['evals'])
    if artifact is None:
        raise BackendUnreachable('the eval set is not on the backend')
    if isinstance(body, Mapping):
        tiers = dict(artifact.policy_tiers or ())
        policies = {str(name): {**(spec if isinstance(spec, Mapping) else {}), 'effective_tier': tiers.get(str(name))} for name, spec in body.items()}
    env_tools: dict[str, Any] | None = None
    config = _fetch_yaml(backend, f'/artifacts/{ids["env"]}/files/inputs/plurai.config.yaml', 'GET /artifacts/{id}/files/{path}')
    if isinstance(config, Mapping) and isinstance(config.get('tools'), (Mapping, list)):
        tools = config['tools']
        env_tools = (
            {str(k): v for k, v in tools.items()}
            if isinstance(tools, Mapping)
            else {str(t['name']): t for t in tools if isinstance(t, Mapping) and 'name' in t}
        )
    return SuiteSide(ids, scenarios=scenarios, policies=policies, env_tools=env_tools, corpus_documents=_corpus_documents(backend, ids['corpus']))


#: Where a scenario row of the scenario set carries a title of its own, best first. The user's goal is not a name.
_SCENARIO_TITLE_PATHS: Final = (
    ('metadata', 'title'),
    ('metadata', 'scenario_card', 'title'),
    ('inputs', 'title'),
)
#: The scenario card the backend writes in each scenario's folder: the short title that ``artifacts/scenarios.md`` shows.
_SCENARIO_CARD_FILE: Final = 'scenario_card.yaml'
#: The keys of a policy body that name it for a reader, best first: its customer-facing name, then its short rule.
_POLICY_NAME_KEYS: Final = ('display_name', 'rule')


@dataclass(frozen=True)
class SuiteNames:
    """What the suite calls its scenarios and policies, by id, for those it names."""

    scenario_titles: dict[str, str]
    policy_descriptions: dict[str, str]


def _at(row: Mapping[str, Any], path: tuple[str, ...]) -> str | None:
    value: Any = row
    for key in path:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def scenario_title(row: Mapping[str, Any]) -> str | None:
    """The title a row of ``GET /scenario-sets/{id}`` carries, if any."""
    return next((text for path in _SCENARIO_TITLE_PATHS if (text := _at(row, path))), None)


def card_title(text: str) -> str | None:
    """The title of a ``scenario_card.yaml``. A card marked stale no longer describes its scenario and names nothing."""
    try:
        card = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(card, Mapping) or card.get('stale') is True:
        return None
    return _at(card, ('title',))


def policy_description(body: Mapping[str, Any]) -> str | None:
    """The policy's short description from its body in ``policies.yaml``."""
    return next((text for key in _POLICY_NAME_KEYS if (text := _at(body, (key,)))), None)


def _card_title_of(backend: Backend, scenario_set_id: str, folder: str) -> str | None:
    """The title on the card of one scenario of the set, or ``None`` when the set holds no card for it."""
    path = f'/artifacts/{quote(scenario_set_id, safe="")}/files/{quote(folder, safe="")}/{_SCENARIO_CARD_FILE}'
    try:
        response = backend.get(path, route='GET /artifacts/{id}/files/{path}')
    except REFUSALS:
        raise
    except BackendError:
        return None
    return card_title(response.text) if response.status_code == 200 else None


def suite_names(backend: Backend, ids: Mapping[str, str]) -> SuiteNames:
    """The scenario titles and policy descriptions of the suite with these ids. A scenario's title is the one on its card, the
    short title ``artifacts/scenarios.md`` shows; the user's goal never stands in for it. Names are display data: what cannot be read is
    left out and the report shows the id in words. A rejected or refused token, or a refused SDK version, passes through."""
    titles: dict[str, str] = {}
    try:
        response = backend.get(f'/scenario-sets/{ids["scenarios"]}', route='GET /scenario-sets/{id}')
        rows = response.json().get('scenarios') if response.status_code == 200 else None
    except REFUSALS:
        raise
    except (BackendError, ValueError, AttributeError):
        rows = None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping) or not isinstance(folder := row.get('folder_name'), str):
            continue
        if title := _card_title_of(backend, ids['scenarios'], folder) or scenario_title(row):
            titles[folder] = title
    try:
        body = _fetch_yaml(backend, f'/eval-sets/{ids["evals"]}/policies.yaml', 'GET /eval-sets/{id}/policies.yaml')
    except REFUSALS:
        raise
    except BackendError:
        body = None
    specs = body.items() if isinstance(body, Mapping) else ()
    descriptions = {str(name): text for name, spec in specs if isinstance(spec, Mapping) and (text := policy_description(spec))}
    return SuiteNames(titles, descriptions)


@dataclass(frozen=True)
class Diffed:
    diff: SuiteDiff
    #: Lines for the report's notes: what could not be fetched.
    notes: tuple[str, ...] = ()


def diff_against(backend: Backend, head_ids: Mapping[str, str], base_ids: Mapping[str, str], *, base_ref: str | None, include_content: bool) -> Diffed:
    """The diff of the suite at ``head_ids`` against ``base_ids``, from the backend's content. A fetch failure is a note."""
    try:
        head = side_of(backend, head_ids)
    except REFUSALS:
        raise
    except BackendError as error:
        return Diffed(
            render_ids_only(base_ids, head_ids, 'the content diff could not be fetched', base_ref=base_ref), (f'the suite diff could not be fetched ({error})',)
        )
    try:
        base: SuiteSide | None = side_of(backend, base_ids)
    except REFUSALS:
        raise
    except BackendError:
        base = None
    notes = () if base is not None else ('the base suite is no longer readable on the backend',)
    return Diffed(render_suite_diff(base, head, base_ref=base_ref, include_content=include_content, base_ids=base_ids), notes)
