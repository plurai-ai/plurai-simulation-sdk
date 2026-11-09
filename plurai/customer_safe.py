"""The customer-safe projection of a backend answer: the only form in which the SDK prints or writes one.

The backend projects its own responses, and this module is the second line: whatever a command
shows a user or a run folder keeps passes through here, so a field the projection does not declare
cannot reach a terminal or an artifact whatever the backend sends. Three entry points cover the
three shapes:

* :func:`project_job` is a job envelope. Its top level is limited to the fields the wire model
  :class:`~plurai.orchestrator.models.DiagnosticJobStatusResponse` declares; ``error`` is a scrubbed
  diagnosis; ``detail`` is free-form, so internal-only names are dropped at any depth and its strings
  are scrubbed.
* :func:`project_value` is any other JSON value a command prints (the backend identity, say).
* :func:`describe_failure` is one line for a failed request: the backend's message, scrubbed, and the
  ``error_id`` the user quotes. The body is never echoed raw, and a body that is not JSON contributes
  nothing but the status.

Free-text ``warnings`` lists are emptied: the warnings a client acts on arrive as typed findings in
``detail.gate_authority.findings``. The lists of internal-only names and the scrub patterns mirror the
backend's own, so the two projections agree on what a customer may read.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any, Final

from plurai.host_effects import credential_redactor, redact_cli_credentials, redact_value

#: Names that carry spend, a model identity, a location on the backend host or a stack trace. Dropped at any
#: depth. A closed list, not a pattern: a customer's own data may carry a field called ``cost``.
INTERNAL_ONLY_KEYS: Final[frozenset[str]] = frozenset(
    {
        'total_cost_usd',
        'cost_usd',
        'cost_usd_per_eval',
        'original_cost_usd_per_eval',
        'hybrid_cost_usd_per_eval',
        'cost_reduction_pct',
        'cost_ceiling_usd',
        'unit_ledger',
        'model_usage',
        'judge_model',
        'workdir',
        'cache_path',
        'baseline_out_dir',
        'backup_path',
        'hybrid_artifact_dir',
        '_provider_error',
        'traceback',
        'stack_trace',
    }
)

#: Free-text warning lists, emptied wherever they appear.
_WARNING_KEYS: Final = ('warnings', 'standing_warnings')

#: Keys whose values are identities a finding binds to; their strings are kept exactly.
_VERBATIM_KEYS: Final[frozenset[str]] = frozenset({'evidence', 'fingerprint', 'gate_id', 'subject', 'id', 'draft_ref'})

_TRACEBACK_MARK: Final = 'Traceback (most recent call last)'
#: An absolute path under a directory the backend host's own files live in. Not a general path matcher: a
#: customer's ``/app/agent.py`` in a quoted line survives.
_PATH_RE: Final = re.compile(r"(?<![\w/.:~-])/(?:Users|home|opt|var|tmp|private|srv|mnt|root)(?:/[^\s'\"`<>()\[\]{},;]+)+")
_AMOUNT_RE: Final = re.compile(r'\$\d[\d,]*(?:\.\d+)?')
_FRAME_RE: Final = re.compile(r'^\s*File "[^"]*", line \d+.*$')
_MAX_MESSAGE_CHARS: Final = 2000
_LINE_CHARS: Final = 300
_MESSAGE_FIELDS: Final = ('error', 'message', 'detail')
_NO_MESSAGE: Final = 'no message in the body'


def safe_text(text: str) -> str:
    """``text`` as a customer may read it: one diagnosis, no stack frames, no internal paths, no amounts, no credential.

    Credentials are redacted on the whole text before anything is cut, so a length cap can never leave the
    prefix of a token behind.
    """
    text = redact_cli_credentials(text)
    kept = [line for line in text.partition(_TRACEBACK_MARK)[0].splitlines() if not _FRAME_RE.match(line)]
    cleaned = _PATH_RE.sub('<path>', '\n'.join(kept).strip())
    cleaned = _AMOUNT_RE.sub('<amount>', cleaned)
    if len(cleaned) > _MAX_MESSAGE_CHARS:
        cleaned = cleaned[:_MAX_MESSAGE_CHARS].rstrip() + ' [truncated]'
    return cleaned


def _scrub_string(text: str, redact: Callable[[str], str] = redact_cli_credentials) -> str:
    head, mark, _ = redact(text).partition(_TRACEBACK_MARK)
    return _PATH_RE.sub('<path>', head.rstrip() if mark else head)


#: ``value`` with every credential redacted from its strings and keys at any depth; nothing else changes.
redact_deep = redact_value


def project_value(value: Any, *, verbatim: bool = False) -> Any:
    """``value`` without internal-only names at any depth, with warning lists emptied and every string scrubbed.

    Under an identity-bound key (``verbatim``) a string is only redacted, because a finding binds to its exact
    text; the structure around it is projected the same way.
    """
    return _project(value, verbatim, credential_redactor())


def _project(value: Any, verbatim: bool, redact: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return redact(value) if verbatim else _scrub_string(value, redact)
    if isinstance(value, list):
        return [_project(item, verbatim, redact) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in INTERNAL_ONLY_KEYS:
                continue
            if key in _WARNING_KEYS and isinstance(item, list):
                out[key] = []
                continue
            out[redact(key) if isinstance(key, str) else key] = _project(item, verbatim or key in _VERBATIM_KEYS, redact)
        return out
    return value


def is_job(body: object) -> bool:
    """True when ``body`` has the job-envelope shape: non-empty string ``kind`` and ``status``."""
    return isinstance(body, dict) and all(isinstance(body.get(key), str) and body[key].strip() for key in ('kind', 'status'))


@cache
def job_fields() -> frozenset[str]:
    """The fields of a job envelope a customer may read: those the wire model declares.

    Resolved on first use because the orchestrator package imports this module.
    """
    from plurai.orchestrator.models import DiagnosticJobStatusResponse  # noqa: PLC0415

    return frozenset(DiagnosticJobStatusResponse.model_fields)


def project_job(body: Mapping[str, Any]) -> dict[str, Any]:
    """A job envelope as a customer may read it: declared fields only."""
    out = {key: redact_deep(value) for key, value in body.items() if key in job_fields() and key not in ('error', 'detail')}
    if isinstance(body.get('error'), str):
        out['error'] = safe_text(body['error'])
    elif 'error' in body:
        out['error'] = project_value(body['error'])
    if 'detail' in body:
        out['detail'] = project_value(body['detail'])
    return out


def error_id_of(body: object) -> str | None:
    """The id a user quotes when asking for help, if the body carries one."""
    if isinstance(body, dict) and isinstance(body.get('error_id'), str) and body['error_id'].strip():
        return redact_cli_credentials(body['error_id'].strip())[:80]
    return None


def _validation_message(detail: list[Any]) -> str:
    """The ``loc: msg`` rows of a FastAPI validation error, which name the offending field."""
    rows: list[str] = []
    for item in detail:
        if isinstance(item, dict) and isinstance(item.get('msg'), str):
            loc = item.get('loc')
            where = '.'.join(str(part) for part in loc) if isinstance(loc, list) else ''
            rows.append(f'{where}: {item["msg"]}' if where else item['msg'])
    return '; '.join(rows)


def failure_message(body: object) -> str:
    """The message of a failure body, scrubbed and on one line; empty when the body carries none."""
    if not isinstance(body, dict):
        return ''
    for key in _MESSAGE_FIELDS:
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return ' '.join(safe_text(value).split())
        if isinstance(value, list) and (message := _validation_message(value)):
            return ' '.join(safe_text(message).split())
    return ''


def describe_body(body: object, *, limit: int = _LINE_CHARS, empty: str = _NO_MESSAGE) -> str:
    """One customer-safe line for a failure body: its message cut to ``limit`` characters, then ``(error id: ...)``.

    A body with no message and no id (a proxy's page, an empty answer) reads ``empty``.
    """
    message = failure_message(body)[:limit]
    error_id = error_id_of(body)
    if message and error_id:
        return f'{message} (error id: {error_id})'
    if error_id:
        return f'error id: {error_id}'
    return message or empty


def response_json(response: Any) -> Any:
    """``response``'s JSON body, or ``None``. Never raises: the failure path must not fail.

    Duck-typed on ``.json()`` so that a streamed, unread response, which raises on read, is an empty answer.
    """
    try:
        return response.json()
    except Exception:  # noqa: BLE001 - a body that is not JSON carries no message
        return None


def describe_failure(response: Any, *, limit: int = _LINE_CHARS, empty: str = _NO_MESSAGE) -> str:
    """:func:`describe_body` of ``response``'s JSON body."""
    return describe_body(response_json(response), limit=limit, empty=empty)


__all__ = [
    'INTERNAL_ONLY_KEYS',
    'describe_body',
    'describe_failure',
    'error_id_of',
    'failure_message',
    'is_job',
    'job_fields',
    'project_job',
    'project_value',
    'redact_deep',
    'response_json',
    'safe_text',
]
