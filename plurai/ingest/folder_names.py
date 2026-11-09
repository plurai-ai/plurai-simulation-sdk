"""Scenario-bundle folder names under the anonymized posture: keyed pseudonyms plus a local map.

A bundle folder's NAME is its scenario's identity on the backend. The backend names the scenario
by the folder's slug, and from there the name reaches every run folder, report row and
``run_summary.json`` id. The name is authored by whoever made the folder, and a real-records folder
is easily named after the customer whose record it holds. The anonymization scrub reads file
CONTENT only, so the folder name needs its own rewrite under ``anonymize: true``.

Under the anonymized posture, ``ingest-bundle`` replaces each folder name with
``b_<12 hex>`` before anything is archived. The hex is an HMAC of the name under the workspace's
existing pseudonym key (``.plurai/anon.key``), so:

* the same folder gets the same pseudonym on every upload from this workspace, which keeps a
  scenario's identity stable across builds;
* the backend only ever sees pseudonyms, so every later upload (run artifacts, run reports) carries
  them too, with no further rewrite;
* the operator reads their own files through a local, never-uploaded map
  (``.plurai/scenario_folder_names.json``, beside the key, under the same self-ignoring
  ``.plurai/``), and ``plurai folder-names`` prints it.

The rewrite touches only the OUTGOING copy: the bundle is staged, and the operator's folders and
files on disk are never modified. Inside the staged copy, the prose units (``ROLES.md``) have every
mention of a renamed folder replaced; a record file has only path uses (``<name>/...``) replaced,
because a value that happens to equal a folder name is the real datum. The set-level ``BUNDLE.md``
does not travel in the archive (the start call carries it inline), so under the anonymized posture
its outgoing copy is always written to :data:`OUTGOING_OVERVIEW_RELPATH`, names rewritten and
content-scrubbed like the archive, for the caller to send instead.

Names the minting code produces pass through unchanged, because other tools match them by name:
the plugin's regression folders, the converter's ``thread-<id>`` folders, the walk's
``scenario_NN``, and an existing pseudonym. This is a SHAPE rule, not PII detection: anything
outside those shapes is pseudonymized, whatever it looks like. A regression slug is words the
recorder authors by rule, which this module does not screen.

Under the raw posture none of this runs.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from plurai import _private_files
from plurai._platform import is_windows
from plurai.ingest.anonymize import KEY_DIR_NAME, keyed_hash
from plurai.run_artifacts import REGRESSION_SCENARIO_PREFIX

#: The pseudonym's fixed prefix. Also the shape a pseudonym is recognised by.
PSEUDONYM_PREFIX = 'b_'
#: Hex characters of the keyed hash kept in a pseudonym (48 bits; collisions within one
#: workspace are negligible, and a collision would be refused as a duplicate archive member).
PSEUDONYM_HEX_LEN = 12
#: The local name map, under ``.plurai/`` beside the pseudonym key. Never uploaded.
NAME_MAP_FILENAME = 'scenario_folder_names.json'
#: The ingest kinds whose units are keyed ``<scenario folder>/<path>`` and whose folder NAME becomes
#: a scenario's identity on the backend. Run artifacts and reports are keyed by the ids the backend
#: minted from these, so they are already pseudonyms and are not renamed again.
PSEUDONYMIZED_FOLDER_KINDS = frozenset({'scenario-bundle'})
#: Where the rewritten set-level overview is written for the start call's ``overview_md``.
OUTGOING_OVERVIEW_RELPATH = f'{KEY_DIR_NAME}/uploads/overview_md.md'
#: Domain separation from the free-text scrub, which hashes the SAME key over bare values.
_HASH_DOMAIN = 'scenario-folder\x00'

_PSEUDONYM = rf'{PSEUDONYM_PREFIX}[0-9a-f]{{{PSEUDONYM_HEX_LEN}}}'
_PSEUDONYM_RE = re.compile(rf'(?<![A-Za-z0-9]){_PSEUDONYM}(?![0-9a-f])')

#: Folder names that pass through verbatim, each the exact shape its minting code produces.
#: ``regression_<slug>`` is the plugin's recorder (``regressions.py``: ``FOLDER_PREFIX`` +
#: ``SLUG_RE``). Its slug is lowercase words the recorder authors by rule (policy + cause word), so
#: it is the one pass-through that CAN carry text a person chose: the rule is the recorder's, and
#: nothing here screens the words. The disclosure line names how many such folders went verbatim.
_REGRESSION = rf'{re.escape(REGRESSION_SCENARIO_PREFIX)}[a-z0-9]+(?:_[a-z0-9]+)*'
_TOOL_MINTED = re.compile(
    '|'.join(
        (
            _PSEUDONYM,  # already a pseudonym
            _REGRESSION,
            r'thread-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',  # the converter, over a UUID thread id
            r'thread-[0-9a-f]{16,}',  # the converter, over a hex thread id
            r'scenario_\d+',  # the walk's own numbering
        )
    )
)
_REGRESSION_RE = re.compile(_REGRESSION)

#: The prose units: text written ABOUT the folder, where every mention of its name is replaced.
#: Every other file is the real record the lane exists to ship, and only a PATH use of the name
#: (``<name>/...``) is rewritten there — a value equal to the name (``"type": "refund"``, an
#: ``account_id`` that is also the folder name) is data, and is left for the content scrub.
PROSE_FILENAMES = frozenset({'ROLES.md', 'BUNDLE.md'})


def is_tool_minted(name: str) -> bool:
    """Whether a folder name is one of the minted shapes that pass through verbatim."""
    return _TOOL_MINTED.fullmatch(name) is not None


def is_regression_folder(name: str) -> bool:
    return _REGRESSION_RE.fullmatch(name) is not None


def folder_pseudonym(name: str, key: bytes) -> str:
    """The keyed pseudonym for a folder name: deterministic per (name, workspace key)."""
    return PSEUDONYM_PREFIX + keyed_hash(_HASH_DOMAIN + name, key)[:PSEUDONYM_HEX_LEN]


def pseudonyms_for(names: Iterable[str], key: bytes) -> dict[str, str]:
    """``{original: pseudonym}`` for every name that is not tool-minted."""
    return {name: folder_pseudonym(name, key) for name in sorted(set(names)) if not is_tool_minted(name)}


def _longest_first(renames: Mapping[str, str]) -> list[str]:
    return sorted((name for name in renames if name), key=len, reverse=True)


def substitute_in_prose(text: str, renames: Mapping[str, str]) -> str:
    """Every mention of a name, in any case, bounded only by letters and digits.

    ``_`` and ``-`` are separators here, so ``Jane_Doe_Refund``, ``jane_doe_refund_v2`` and
    ``jane_doe_refund-2`` are all mentions of ``jane_doe_refund``, while ``refunds`` is not a mention
    of ``refund``.
    """
    for original in _longest_first(renames):
        pattern = re.compile(rf'(?<![A-Za-z0-9]){re.escape(original)}(?![A-Za-z0-9])', re.IGNORECASE)
        text = pattern.sub(lambda _m, p=renames[original]: p, text)
    return text


def substitute_paths(text: str, renames: Mapping[str, str]) -> str:
    """Only PATH uses: the exact name as the FIRST segment of a relative path, ``<name>/...`` or
    ``./<name>/...``, where that path starts a line or follows whitespace, a quote, a backtick or an
    opening bracket. A segment deeper in some other path is not the bundle folder:
    ``https://shop.example/refund/policy`` and ``/api/refund/42`` are left alone."""
    for original in _longest_first(renames):
        pattern = re.compile(rf'(?:^|(?<=[\s"\'`(\[<]))(\./)?{re.escape(original)}(?=/)', re.MULTILINE)
        text = pattern.sub(lambda m, p=renames[original]: (m.group(1) or '') + p, text)
    return text


def _substitute_tree(node: Any, renames: Mapping[str, str]) -> Any:  # noqa: ANN401 - decoded JSON
    if isinstance(node, str):
        return substitute_paths(node, renames)
    if isinstance(node, list):
        return [_substitute_tree(item, renames) for item in node]
    if isinstance(node, dict):
        return {substitute_paths(k, renames) if isinstance(k, str) else k: _substitute_tree(v, renames) for k, v in node.items()}
    return node


def outgoing_text(path: Path, renames: Mapping[str, str]) -> str | None:
    """The rewritten text of one outgoing file, or ``None`` when it stays byte-identical.

    Reads ``path`` and never writes it. A prose unit gets :func:`substitute_in_prose`; any other
    UTF-8 file gets :func:`substitute_paths`, over the decoded JSON strings when it parses as JSON
    (so an escaped non-ASCII name is still found) and over the raw text otherwise.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return None
    if path.name in PROSE_FILENAMES:
        rewritten = substitute_in_prose(text, renames)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            rewritten = substitute_paths(text, renames)
        else:
            substituted = _substitute_tree(data, renames)
            rewritten = text if substituted == data else json.dumps(substituted, ensure_ascii=False)
    return None if rewritten == text else rewritten


def _write_owner_only(path: Path, text: str, *, is_win: bool | None = None) -> None:
    """Atomic owner-only write: a stale temp file's looser mode is reset, not inherited.

    On Windows, which has no modes, the temporary file's access list is cut to the current user before the text is
    written (:func:`plurai._private_files.write_atomic`), and a symlink at ``path`` is refused.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_windows() if is_win is None else is_win:
        _private_files.write_atomic(path, text.encode('utf-8'))
        return
    tmp = path.with_name(f'{path.name}.tmp')
    if not hasattr(os, 'fchmod'):  # no POSIX modes on this platform: the file would be readable by others, so it is not written
        raise OSError(f'cannot restrict {path.name} to its owner on this platform')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write(text)
    tmp.replace(path)


def name_map_path(workspace_root: str | Path) -> Path:
    return Path(workspace_root) / KEY_DIR_NAME / NAME_MAP_FILENAME


def load_name_map(workspace_root: str | Path) -> dict[str, str]:
    """``{pseudonym: original}`` recorded by earlier anonymized bundle uploads (``{}`` when none)."""
    path = name_map_path(workspace_root)
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(loaded, dict):
        raise ValueError(f'{path} is not a JSON object of pseudonym → folder name')
    return {str(k): str(v) for k, v in loaded.items()}


def record_names(workspace_root: str | Path, renames: Mapping[str, str]) -> Path:
    """Merge ``{original: pseudonym}`` into the local map, owner-only, atomically. Returns its path.

    Written BEFORE anything egresses, so a failed upload still leaves the operator able to read
    what a retry will send.
    """
    path = name_map_path(workspace_root)
    merged = load_name_map(workspace_root)
    merged.update({pseudonym: original for original, pseudonym in renames.items()})
    _write_owner_only(path, json.dumps(dict(sorted(merged.items())), ensure_ascii=False, indent=2) + '\n')
    return path


def write_outgoing_overview(workspace_root: str | Path, text: str | None) -> Path | None:
    """Write the outgoing ``overview_md`` (owner-only) and return its path; with ``None``, remove a
    stale one so a later caller cannot send an overview written for a different bundle."""
    path = Path(workspace_root) / OUTGOING_OVERVIEW_RELPATH
    if text is None:
        path.unlink(missing_ok=True)
        return None
    _write_owner_only(path, text)
    return path


def reveal(text: str, name_map: Mapping[str, str]) -> str:
    """Replace every pseudonym in ``text`` with its recorded folder name — including one embedded
    in a longer id such as a set-prefixed scenario name. Unknown pseudonyms stay as they are."""
    return _PSEUDONYM_RE.sub(lambda m: name_map.get(m.group(0), m.group(0)), text)
