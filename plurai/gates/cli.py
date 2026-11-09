"""``--overrule FILE``: the one way a caller lifts a gate finding on an SDK command.

The file is a list of :class:`~plurai.gates.contract.Overrule` records, in either of two shapes:

* a YAML or JSON LIST of ``{gate_id, fingerprint, reason}`` mappings (JSON is read as YAML); or
* the plugin's overrule LEDGER, ``.gate_overrules.jsonl`` — one JSON object per line, the overrule
  under its ``overrule`` key. The ledger is append-only, so the LAST line naming a finding wins.

``plugin.`` ids are the plugin's own gates, decided and applied by the plugin; they are skipped
here (counted, never forwarded). Anything malformed refuses the whole file with
:class:`OverruleFileError` naming the file and the entry: a file that half-loads would read as
"these findings were never overruled", which is a different instruction from the one written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from plurai.gates.contract import Overrule
from plurai.user_path import expand_user_path

OVERRULE_FLAG: Final[str] = '--overrule'

#: The runner's command name, as catalogue ``carriers`` spell it.
RUNNER_COMMAND: Final[str] = 'plurai-run-simulation'

#: The ``plurai`` subcommands that take ``--overrule``: every one that converts or publishes input.
INGEST_OVERRULE_SUBCOMMANDS: Final[tuple[str, ...]] = (
    'convert-trace',
    'ingest-bundle',
    'ingest-corpus',
    'ingest-doc-examples',
    'ingest-run',
    'ingest-run-report',
    'ingest-traces',
)

#: The scan contract's ``check``, as catalogue ``carriers`` spell it. Its overrules are applied INSIDE
#: the vendored checker (``plurai.scan_contract``), so both scan shells lift the same holds.
SCAN_CHECK_COMMAND: Final[str] = 'plurai-scan-contract check'

#: Every command that takes ``--overrule``, as catalogue ``carriers`` spell it.
OVERRULE_COMMANDS: Final[frozenset[str]] = frozenset({RUNNER_COMMAND, SCAN_CHECK_COMMAND, *(f'plurai {name}' for name in INGEST_OVERRULE_SUBCOMMANDS)})

#: Gate ids the plugin owns. The plugin applies them itself; they never reach a command or the backend.
PLUGIN_GATE_PREFIX: Final[str] = 'plugin.'

#: A ledger file, read line by line rather than as one document.
LEDGER_SUFFIX: Final[str] = '.jsonl'

_GATE_ID_RE: Final[re.Pattern[str]] = re.compile(r'[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+')
_FINGERPRINT_RE: Final[re.Pattern[str]] = re.compile(r'[0-9a-f]{64}')


class OverruleFileError(ValueError):
    """The ``--overrule`` file could not be read as a list of overrules. The message names the file."""


@dataclass(frozen=True)
class OverrulesFile:
    """What one ``--overrule`` file carried, with the identity of the bytes it was read from."""

    path: Path
    sha256: str
    overrules: tuple[Overrule, ...]
    #: ``plugin.`` entries the file carried and this command left to the plugin.
    skipped_plugin_ids: int

    def as_record(self) -> dict[str, Any]:
        return {'path': str(self.path), 'sha256': self.sha256, 'skipped_plugin_ids': self.skipped_plugin_ids}


def add_overrule_argument(parser: argparse.ArgumentParser) -> None:
    """The one spelling of the flag, shared by every command in :data:`OVERRULE_COMMANDS`."""
    parser.add_argument(
        OVERRULE_FLAG,
        default=None,
        type=expand_user_path,
        metavar='FILE',
        help="Overrules to apply to this invocation's gate findings: a YAML/JSON list of {gate_id, fingerprint, reason}, or "
        "the plugin's .gate_overrules.jsonl ledger. An overrule binds to the finding's fingerprint, so it lapses when "
        'the evidence changes. Security and invariant gates are never lifted. A malformed file refuses the command.',
    )


def takes_overrule(parser: argparse.ArgumentParser) -> bool:
    """Whether ``parser`` was given the flag — read off the parser, so a report of it cannot drift."""
    return any(OVERRULE_FLAG in action.option_strings for action in parser._actions)


def _entries_from_list(path: Path, text: str) -> list[tuple[str, Any]]:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OverruleFileError(f'{path}: not YAML or JSON ({exc}).') from exc
    if not isinstance(document, list):
        raise OverruleFileError(f'{path}: must be a list of overrules, got {type(document).__name__}.')
    return [(f'entry {index}', item) for index, item in enumerate(document)]


def _entries_from_ledger(path: Path, text: str) -> list[tuple[str, Any]]:
    # ``\n`` only: ``str.splitlines`` also breaks at U+2028, U+2029 and NEL, which a ledger line that
    # is not ASCII-escaped can carry raw inside a reason or a copied finding.
    entries: list[tuple[str, Any]] = []
    for number, line in enumerate(text.split('\n'), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise OverruleFileError(f'{path}:{number}: not JSON ({exc}).') from exc
        if not isinstance(row, dict) or not isinstance(row.get('overrule'), dict):
            raise OverruleFileError(f'{path}:{number}: not an overrule ledger line (no "overrule" object).')
        entries.append((f'line {number}', row['overrule']))
    return entries


def _overrule(path: Path, where: str, item: Any) -> Overrule:
    if not isinstance(item, dict):
        raise OverruleFileError(f'{path}: {where} is a {type(item).__name__}, not a {{gate_id, fingerprint, reason}} mapping.')
    try:
        overrule = Overrule.model_validate(item)
    except ValidationError as exc:
        problems = '; '.join(f'{".".join(str(p) for p in err["loc"]) or "entry"}: {err["msg"]}' for err in exc.errors())
        raise OverruleFileError(f'{path}: {where}: {problems}.') from exc
    if not _GATE_ID_RE.fullmatch(overrule.gate_id):
        raise OverruleFileError(f'{path}: {where}: gate_id {overrule.gate_id!r} is not a dotted lower-case gate id.')
    if not _FINGERPRINT_RE.fullmatch(overrule.fingerprint):
        raise OverruleFileError(f'{path}: {where}: fingerprint {overrule.fingerprint!r} is not 64 lower-case hex characters as a finding prints it.')
    return overrule


def load_overrules(path: str | Path) -> OverrulesFile:
    """Read and validate one ``--overrule`` file. Raises :class:`OverruleFileError` on anything malformed.

    A list may name a finding once: two entries for one ``(gate_id, fingerprint)`` are two
    instructions with no order between them. The ledger names a finding as often as the plugin
    re-decided it, and its order IS the decision order, so its last line stands.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OverruleFileError(f'{path}: cannot be read ({exc.strerror or exc}).') from exc
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise OverruleFileError(f'{path}: not UTF-8 ({exc}).') from exc
    ledger = path.suffix == LEDGER_SUFFIX
    entries = _entries_from_ledger(path, text) if ledger else _entries_from_list(path, text)

    by_key: dict[tuple[str, str], Overrule] = {}
    skipped = 0
    for where, item in entries:
        overrule = _overrule(path, where, item)
        if overrule.gate_id.startswith(PLUGIN_GATE_PREFIX):
            skipped += 1
            continue
        key = (overrule.gate_id, overrule.fingerprint)
        if key in by_key and not ledger:
            raise OverruleFileError(f'{path}: {where} overrules {overrule.gate_id} {overrule.fingerprint} a second time; name each finding once.')
        by_key.pop(key, None)
        by_key[key] = overrule
    return OverrulesFile(path=path, sha256=hashlib.sha256(raw).hexdigest(), overrules=tuple(by_key.values()), skipped_plugin_ids=skipped)
