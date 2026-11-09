"""Payload codecs: the mock exchanges a declared encoded field in its DECODED form.

A tool mocked at a wire seam (``transport``) or below a client sees what the wire carries: a Gmail
draft's body is ``body.message.raw``, base64url RFC 5322 bytes. The mock's model reads that as an
opaque string, so a URL inside it is invisible. ``tools.<name>.payload_codecs`` in the
``plurai.config.yaml`` beside the harness (a sibling of ``descriptor:``) names the encoded fields::

    payload_codecs:
      - {at: args.body.message.raw, codec: [base64url, mime]}
      - {at: returns.payload.parts[*].body.data, codec: [base64url]}   # a json/mime-terminated chain on returns

:func:`apply_payload_codecs` is the one place a call's args are decoded (in chain order) before the
``/mock/execute`` wire, a ``@plurai.mock`` handler and the event trail see them, and a successful
reply is encoded (in reverse order) before the agent's library sees it. ``_resolve_mock_response``
(``mock_wrap`` and ``MockToolNode``) and ``code_intercept_call`` both call it.

Rules: an absent path is skipped (one transport entry serves many operations); ``[*]`` walks every
item of a list; a value that does not decode is forwarded raw and noted (the real API would answer
it, so it is never a harness fault); a reply is encoded only where its value is a dict or list (a
string there is already in the wire form, and encoding it again would corrupt it); a dict or list there
the codec cannot encode is a ``mock_runtime_fault`` and the agent is served a failed call, never the
decoded shape; stub and error envelopes are never encoded; ``attachments`` are decode-only.

The ``json`` codec refuses an object that repeats a key, because readers disagree on which value wins. The refusal is
exact-match: keys that differ in case or in Unicode normalisation are distinct and are not refused.

Everything the agent or the mock controls (a tool name, an argument key, a reply key) reaches stderr, the egress record and
``run_summary.json`` only as a label: ``[A-Za-z0-9_-]`` per key (every other character, ``.``, ``[``, ``]`` and ``*``
included, is shown as ``?``, so a key never reads as a path separator or a list item), at most 40 characters per key. Each
note kind reports at most 256 distinct notes per process; a kind cut off there says so once on stderr and in the summary.

A tool at ``transport`` or ``client`` with NO codec at a path is scanned for an args leaf that looks
encoded (:func:`encoded_guess`); a hit is a warning only — one stderr line and an
``undeclared_encoded_payload`` row in the run's egress record — and never prints the decoded value.

Stdlib-only, beside the stdlib-only turn context (the code channel's grandchild imports it).
"""

from __future__ import annotations

import base64
import binascii
import email
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from email import policy
from email.header import decode_header, make_header
from email.message import EmailMessage
from typing import Any, Final, Literal, NamedTuple
from urllib.parse import parse_qsl, urlencode

from plurai._mocking.turn_context import note_harness_fault

Direction = Literal['args', 'returns']

#: The closed codec vocabulary.
CODECS: Final[frozenset[str]] = frozenset({'base64', 'base64url', 'json', 'mime', 'form_urlencoded'})
#: Codecs that may close a ``returns.*`` chain: their decoded value is a dict or list, so a reply already in
#: the encoded form (a string) is told apart from one to encode. A bare base64 ending could double-encode.
_RETURN_TERMINALS: Final[frozenset[str]] = frozenset({'json', 'mime', 'form_urlencoded'})
#: Headers the ``mime`` codec derives from the body; never carried in ``headers``.
_MIME_STRUCTURAL: Final[frozenset[str]] = frozenset({'content-type', 'content-transfer-encoding', 'mime-version'})
#: The keys of a decoded message (``attachments`` is decode-only).
_MIME_KEYS: Final[frozenset[str]] = frozenset({'headers', 'body', 'html', 'attachments'})
_FOLD = re.compile(r'\r?\n(?=[ \t])')
#: Wire form of an encoded message: CRLF line ends.
_MIME_WIRE = policy.default.clone(linesep='\r\n')


class CodecError(ValueError):
    """A value the codec cannot convert. The message names the cause, never the value."""


class PayloadCodec(NamedTuple):
    """One declared ``{at, codec}`` row: the path, split, and the chain in decode order."""

    at: str
    root: Direction
    segments: tuple[str, ...]
    chain: tuple[str, ...]


# --- the codecs ----------------------------------------------------------------------------------

_B64 = re.compile(r'[A-Za-z0-9+/]*={0,2}')
_B64URL = re.compile(r'[A-Za-z0-9_-]*={0,2}')


def _text(value: Any, codec: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise CodecError(f'{codec}: input is not UTF-8') from exc
    if isinstance(value, str):
        return value
    raise CodecError(f'{codec}: expected text, got {type(value).__name__}')


def _b64_decode(value: Any, *, url: bool) -> bytes:
    codec = 'base64url' if url else 'base64'
    text = ''.join(_text(value, codec).split())
    if not (_B64URL if url else _B64).fullmatch(text):
        raise CodecError(f'{codec}: characters outside the alphabet')
    stripped = text.rstrip('=')
    if len(stripped) % 4 == 1:
        raise CodecError(f'{codec}: impossible length')
    try:
        return base64.b64decode(stripped + '=' * (-len(stripped) % 4), altchars=b'-_' if url else None, validate=True)
    except binascii.Error as exc:
        raise CodecError(f'{codec}: not decodable') from exc


def _b64_encode(value: Any, *, url: bool) -> str:
    raw = value.encode('utf-8') if isinstance(value, str) else value
    if not isinstance(raw, bytes):
        raise CodecError(f'{"base64url" if url else "base64"}: expected text or bytes, got {type(value).__name__}')
    return (base64.urlsafe_b64encode(raw) if url else base64.b64encode(raw)).decode('ascii')


def _one_of_each_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object whose keys are all distinct: a repeated key is ambiguous (readers disagree on which wins), so it is refused.

    The test is exact string equality, after JSON escapes are decoded and at any depth. Keys that differ only in case
    (``a`` and ``A``) or in Unicode normalisation (``a`` + U+0301 and U+00E1) are different keys here, so a reader that
    folds them (Go struct decoding folds case) can still see one key twice."""
    out: dict[str, Any] = {}
    for key, item in pairs:
        if key in out:
            raise CodecError('json: an object repeats a key')
        out[key] = item
    return out


def _json_decode(value: Any) -> Any:
    try:
        return json.loads(_text(value, 'json'), object_pairs_hook=_one_of_each_key)
    except json.JSONDecodeError as exc:
        raise CodecError('json: not a JSON document') from exc


def _json_encode(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise CodecError('json: value is not JSON-serializable') from exc


def _form_decode(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        pairs = parse_qsl(_text(value, 'form_urlencoded'), keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise CodecError('form_urlencoded: not a form body') from exc
    for key, item in pairs:
        if key not in out:
            out[key] = item
        elif isinstance(out[key], list):
            out[key].append(item)
        else:
            out[key] = [out[key], item]
    return out


def _form_encode(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise CodecError(f'form_urlencoded: expected a mapping, got {type(value).__name__}')
    return urlencode({str(k): v for k, v in value.items()}, doseq=True)


def _lines(text: str) -> str:
    """A decoded text part with ``\\n`` line ends; the wire's CRLF is the encoder's to restore."""
    return text.replace('\r\n', '\n')


def _header_text(raw: str) -> str:
    """A header's value as its source wrote it — unfolded, RFC 2047 words decoded, nothing re-rendered — so
    an address list the structured parser would drop part of (``"a" <b@c>; d@e``) reaches the mock whole."""
    unfolded = _FOLD.sub('', raw)
    try:
        return str(make_header(decode_header(unfolded)))
    except Exception:  # noqa: BLE001 — an undecodable encoded word: the source text is still the value
        return unfolded


def _mime_decode(value: Any) -> dict[str, Any]:
    """RFC 5322 → ``{headers, body, html?, attachments?[{filename, content_type, size}]}``."""
    raw = value if isinstance(value, bytes) else _text(value, 'mime').encode('utf-8')
    try:
        msg = email.message_from_bytes(raw, policy=policy.default)
        if not msg.keys():
            raise CodecError('mime: no RFC 5322 header block')
        headers: dict[str, Any] = {}
        for name, source in msg.raw_items():
            if name.lower() in _MIME_STRUCTURAL:
                continue
            text = _header_text(str(source))
            if name not in headers:
                headers[name] = text
            elif isinstance(headers[name], list):
                headers[name].append(text)
            else:
                headers[name] = [headers[name], text]
        plain = msg.get_body(preferencelist=('plain',))
        html = msg.get_body(preferencelist=('html',))
        out: dict[str, Any] = {'headers': headers, 'body': _lines(plain.get_content()) if plain is not None else ''}
        if html is not None:
            out['html'] = _lines(html.get_content())
        attachments = [
            {'filename': part.get_filename(), 'content_type': part.get_content_type(), 'size': len(part.get_payload(decode=True) or b'')}
            for part in msg.iter_attachments()
        ]
        has_payload = msg.is_multipart() or bool(str(msg.get_payload() or '').strip())
    except CodecError:
        raise
    except Exception as exc:  # noqa: BLE001 — the stdlib parser raises more than ValueError on hostile input
        raise CodecError(f'mime: {type(exc).__name__}') from exc
    if has_payload and not out['body'] and not out.get('html') and not attachments:
        raise CodecError('mime: no readable text body')
    if attachments:
        out['attachments'] = attachments
    return out


def _mime_encode(value: Any) -> bytes:
    if not isinstance(value, Mapping):
        raise CodecError(f'mime: expected a {{headers, body}} mapping, got {type(value).__name__}')
    if value.get('attachments'):
        raise CodecError('mime: attachments are decode-only')
    if unknown := sorted({_segment(k) for k in value if k not in _MIME_KEYS}):
        shown = ', '.join(unknown[:_UNKNOWN_KEYS_SHOWN]) + (f' and {len(unknown) - _UNKNOWN_KEYS_SHOWN} more' if len(unknown) > _UNKNOWN_KEYS_SHOWN else '')
        raise CodecError(f'mime: unknown keys {shown} (expected headers, body, html)')
    if 'body' not in value and 'html' not in value:
        raise CodecError('mime: neither body nor html')
    msg = EmailMessage()
    headers = value.get('headers') or {}
    if not isinstance(headers, Mapping):
        raise CodecError('mime: headers is not a mapping')
    try:
        for name, header in headers.items():
            if str(name).lower() in _MIME_STRUCTURAL:
                continue
            for item in header if isinstance(header, list) else [header]:
                msg[str(name)] = str(item)
        msg.set_content(str(value.get('body') or ''))
        if value.get('html'):
            msg.add_alternative(str(value['html']), subtype='html')
        return msg.as_bytes(policy=_MIME_WIRE)
    except Exception as exc:  # noqa: BLE001 — CR/LF in a header value, a bad name, …: the message cannot be built
        raise CodecError(f'mime: the message could not be built ({type(exc).__name__})') from exc


_DECODERS: Final[dict[str, Callable[[Any], Any]]] = {
    'base64': lambda v: _b64_decode(v, url=False),
    'base64url': lambda v: _b64_decode(v, url=True),
    'json': _json_decode,
    'mime': _mime_decode,
    'form_urlencoded': _form_decode,
}
_ENCODERS: Final[dict[str, Callable[[Any], Any]]] = {
    'base64': lambda v: _b64_encode(v, url=False),
    'base64url': lambda v: _b64_encode(v, url=True),
    'json': _json_encode,
    'mime': _mime_encode,
    'form_urlencoded': _form_encode,
}


def decode(value: Any, chain: tuple[str, ...] | list[str]) -> Any:
    """``value`` decoded through ``chain`` in order. Bytes left at the end must be UTF-8 text."""
    for codec in chain:
        value = _DECODERS[codec](value)
    return _text(value, chain[-1]) if isinstance(value, bytes) else value


def encode(value: Any, chain: tuple[str, ...] | list[str]) -> Any:
    """``value`` encoded through ``chain`` in reverse order; bytes left at the end are returned as text."""
    for codec in reversed(chain):
        value = _ENCODERS[codec](value)
    return _text(value, chain[0]) if isinstance(value, bytes) else value


# --- the declaration -----------------------------------------------------------------------------

_SEGMENT = re.compile(r'([^.\[\]]+)?((?:\[\*\])*)')


def _split_path(at: str) -> tuple[str, ...]:
    segments: list[str] = []
    for piece in at.split('.'):
        match = _SEGMENT.fullmatch(piece)
        if not piece or match is None:
            raise CodecError(f'path {at!r}: malformed segment {piece!r}')
        if match.group(1):
            segments.append(match.group(1))
        segments.extend('[*]' for _ in range(match.group(2).count('[*]')))
    return tuple(segments)


def parse_payload_codecs(raw: Any) -> tuple[list[PayloadCodec], list[str]]:
    """The valid rows of a ``payload_codecs`` value, and a reason for each row refused."""
    rows: list[PayloadCodec] = []
    refused: list[str] = []
    for index, row in enumerate(raw if isinstance(raw, list) else [raw]):
        at = row.get('at') if isinstance(row, Mapping) else None
        chain = row.get('codec') if isinstance(row, Mapping) else None
        chain = [chain] if isinstance(chain, str) else chain
        try:
            if not isinstance(at, str) or not isinstance(chain, list) or not chain:
                raise CodecError('needs `at` (a path) and `codec` (a non-empty list)')
            segments = _split_path(at)
            root = segments[0]
            if root not in ('args', 'returns') or (root == 'args' and len(segments) < 2):
                raise CodecError(f'path {at!r} must start at args.<name> or returns')
            unknown = [c for c in chain if c not in CODECS]
            if unknown:
                raise CodecError(f'codec {unknown[0]!r} is not one of {", ".join(sorted(CODECS))}')
            if 'mime' in chain[:-1]:
                raise CodecError('mime must be the last codec')
            if root == 'returns' and chain[-1] not in _RETURN_TERMINALS:
                raise CodecError(f'a returns chain must end in {", ".join(sorted(_RETURN_TERMINALS))}, so an already-encoded reply is never encoded twice')
        except CodecError as exc:
            refused.append(f'[{index}] {exc}')
            continue
        rows.append(PayloadCodec(at=at, root='args' if root == 'args' else 'returns', segments=segments[1:], chain=tuple(chain)))
    return rows, refused


# --- the walk ------------------------------------------------------------------------------------


def _at(node: Any, segments: tuple[str, ...], fn: Callable[[Any], Any]) -> Any:
    """``node`` with ``fn`` applied at ``segments``, rebuilt only along the path (never mutated). Absent: ``node``."""
    if not segments:
        return fn(node)
    head, rest = segments[0], segments[1:]
    if head == '[*]':
        if not isinstance(node, list):
            return node
        items = [_at(item, rest, fn) for item in node]
        return node if all(new is old for new, old in zip(items, node, strict=True)) else items
    if not isinstance(node, Mapping) or head not in node:
        return node
    child = _at(node[head], rest, fn)
    return node if child is node[head] else {**node, head: child}


# --- the notes -----------------------------------------------------------------------------------

#: The runner's egress-guard spec env var (``plurai.egress_guard.GUARD_ENV``); its ``log`` is the run's record.
_GUARD_ENV: Final = 'PLURAI_EGRESS_GUARD'
_noted: set[tuple[Any, ...]] = set()
#: Distinct notes of ONE kind a process reports; the agent chooses its argument keys, so a kind's set must not grow with
#: them, and a flood of one kind must not hide another (a codec failure invalidates a scenario, an undeclared guess does not).
_NOTED_MAX: Final = 256
#: Egress-record row that says a kind's notes were cut off at :data:`_NOTED_MAX`; the run summary folds it into
#: ``payload_codec_notes_suppressed``.
SUPPRESSED_KIND: Final = 'payload_codec_notes_suppressed'
_LABEL_MAX: Final = 120
_LABEL_SAFE: Final = re.compile(r'[A-Za-z0-9_.\[\]*-]')
#: One key (an object key, a mime key) inside a label. Narrower than the label itself: ``.``, ``[``, ``]`` and ``*``
#: are the path syntax, so a key that carries them could not be told from the structure around it.
_SEGMENT_SAFE: Final = re.compile(r'[A-Za-z0-9_-]')
_SEGMENT_MAX: Final = 40
_UNKNOWN_KEYS_SHOWN: Final = 5
#: A codec failure's cause: printable ASCII only, cut at this many characters.
_REASON_MAX: Final = 200


def _label(value: object) -> str:
    """``value`` as it may appear on stderr and in the run summary: the agent chooses tool names and argument keys, so
    only ``[A-Za-z0-9_.[]*-]`` survives (anything else, control and line-separator characters included, becomes ``?``)
    and the result is capped at :data:`_LABEL_MAX` characters."""
    text = str(value)
    cut = text[:_LABEL_MAX]
    out = ''.join(c if _LABEL_SAFE.fullmatch(c) else '?' for c in cut)
    return out + '...' if len(text) > _LABEL_MAX else out


def _segment(key: object) -> str:
    """One key of a path as a label shows it: ``[A-Za-z0-9_-]`` survives and every other character, ``.``, ``[``, ``]`` and
    ``*`` included, becomes ``?``; the key is cut at :data:`_SEGMENT_MAX` characters. A key can therefore never read as a
    separator or a list item, and a long key cannot carry a sentence."""
    text = str(key)
    out = ''.join(c if _SEGMENT_SAFE.fullmatch(c) else '?' for c in text[:_SEGMENT_MAX])
    return out + '...' if len(text) > _SEGMENT_MAX else out


def _plain(text: str) -> str:
    """``text`` as printable ASCII (anything else becomes ``?``), cut at :data:`_REASON_MAX`: the last guard on a cause before it is printed."""
    out = ''.join(c if ' ' <= c <= '~' else '?' for c in text[:_REASON_MAX])
    return out + '...' if len(text) > _REASON_MAX else out


def _record(row: dict[str, Any]) -> None:
    """Append ``row`` to the run's egress record, when the runner named one."""
    raw = os.environ.get(_GUARD_ENV)
    if not raw:
        return
    try:
        log = str(json.loads(raw)['log'])
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps({**row, 'pid': os.getpid()}) + '\n').encode('utf-8'))
        finally:
            os.close(fd)
    except (OSError, ValueError, KeyError, TypeError):
        pass  # the stderr line stands; the egress record is the runner's to report as unreadable


def _note(kind: str, line: str, **fields: Any) -> None:
    """One stderr line and one row in the run's egress record, once per distinct note per process (at most :data:`_NOTED_MAX`
    per ``kind``; the first note cut off says so on stderr and in the record)."""
    key = (kind, *fields.values())
    if key in _noted:
        return
    if sum(1 for noted in _noted if noted[0] == kind) >= _NOTED_MAX:
        marker = (SUPPRESSED_KIND, kind)
        if marker not in _noted:
            _noted.add(marker)
            print(f'[plurai] further {kind} notes are suppressed.', file=sys.stderr)
            _record({'kind': SUPPRESSED_KIND, 'suppressed': kind})
        return
    _noted.add(key)
    print(f'[plurai] {line}', file=sys.stderr)
    _record({'kind': kind, **fields})


# --- the detector --------------------------------------------------------------------------------

_DETECT_MIN_LEN: Final = 64
#: A leaf longer than this is not probed (an upload, not a field); the detector's cost stays bounded per call.
_DETECT_MAX_LEN: Final = 1 << 20
#: Deeper JSON than this is not parsed by the probe (``json.loads`` recurses).
_JSON_PROBE_DEPTH: Final = 64
_PRINTABLE_SHARE: Final = 0.95
#: An args key with one of these in it carries a credential, which may well be base64 text: never scanned.
_AUTH_WORDS: Final = ('authorization', 'token', 'cookie', 'secret', 'password', 'signature')
_HEADER_LINE = re.compile(r'[A-Za-z][A-Za-z0-9-]*:[^\r\n]*\r?\n')


def _auth_like(key: str) -> bool:
    lowered = key.lower()
    if any(word in lowered for word in _AUTH_WORDS):
        return True
    words = re.split(r'[^a-z0-9]+', re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', key).lower())
    return any(word in ('key', 'apikey') or word.endswith('apikey') for word in words)


#: The guess for a payload no codec in the vocabulary decodes (gzip): only a ``client`` seam reads it plainly.
UNKNOWN_BINARY: Final = 'unknown binary'


def _json_container(text: str) -> bool:
    """Whether ``text`` is a JSON object or array, parsed only when its nesting stays within :data:`_JSON_PROBE_DEPTH`."""
    stripped = text.lstrip()
    if stripped[:1] not in ('{', '['):
        return False
    depth, in_string, escaped = 0, False, False
    for c in stripped:
        if in_string:
            escaped, in_string = (False, True) if escaped else (c == '\\', c != '"')
        elif c == '"':
            in_string = True
        elif c in '{[':
            depth += 1
            if depth > _JSON_PROBE_DEPTH:
                return False
        elif c in '}]':
            depth -= 1
    try:
        return isinstance(json.loads(stripped), dict | list)
    except ValueError:
        return False


def encoded_guess(value: str) -> str | None:
    """What ``value`` decodes as when it is a strictly base64/base64url string of 64 chars to 1 MiB that decodes
    cleanly to gzip (:data:`UNKNOWN_BINARY`) or to ≥95% printable UTF-8 (``base64url+mime``, ``base64+json``,
    ``base64url``, …), else ``None``. Every guess but :data:`UNKNOWN_BINARY` is a chain in the vocabulary."""
    if not _DETECT_MIN_LEN <= len(value) <= _DETECT_MAX_LEN:
        return None
    has_std, has_url = any(c in value for c in '+/'), any(c in value for c in '-_')
    if has_std and has_url:
        return None
    url = has_url or (not has_std and len(value) % 4 != 0)  # standard base64 is padded; an unpadded string is base64url
    try:
        raw = _b64_decode(value, url=url) if (_B64URL if url else _B64).fullmatch(value) else None
    except CodecError:
        return None
    if not raw:
        return None
    alphabet = 'base64url' if url else 'base64'
    if raw[:2] == b'\x1f\x8b':
        return UNKNOWN_BINARY
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        return None
    if sum(1 for c in text if c.isprintable() or c in '\r\n\t') < _PRINTABLE_SHARE * len(text):
        return None
    if _HEADER_LINE.match(text) and ('\n\n' in text or '\r\n\r\n' in text):
        return f'{alphabet}+mime'
    return f'{alphabet}+json' if _json_container(text) else alphabet


#: A path below ``args`` as walked: a mapping key is its text, a list item is ``None`` (so a key spelled ``[*]`` is still a key).
_Walked = tuple[str | None, ...]


def _leaves(node: Any, path: _Walked) -> list[tuple[_Walked, str]]:
    """Every string leaf under ``node`` as ``(path, value)``, skipping auth-like keys."""
    if isinstance(node, str):
        return [(path, node)]
    if isinstance(node, Mapping):
        return [leaf for key, child in node.items() if not _auth_like(str(key)) for leaf in _leaves(child, (*path, str(key)))]
    if isinstance(node, list):
        return [leaf for child in node for leaf in _leaves(child, (*path, None))]
    return []


def _declared_at(declared: tuple[str, ...], walked: _Walked) -> bool:
    """Whether the declared path ``declared`` (``[*]`` for a list item) is ``walked`` or a prefix of it. A list item matches
    only ``[*]`` and a key matches only an equal segment that is not ``[*]``, so a key spelled ``[*]`` is never a wildcard."""
    return len(declared) <= len(walked) and all(
        (seg == '[*]') if step is None else (seg != '[*]' and seg == step) for seg, step in zip(declared, walked, strict=False)
    )


def _spelled(walked: _Walked, show: Callable[[str], str]) -> str:
    return 'args' + ''.join('[*]' if step is None else f'.{show(step)}' for step in walked)


def _detect_undeclared(tool: str, args: Mapping[str, Any], declared: set[tuple[str, ...]]) -> None:
    """Warn-only: a probe that fails on any input is no guess, never an error in the tool call. ``declared`` holds the
    declared paths below ``args`` as segments."""
    try:
        leaves = _leaves(args, ())
    except Exception:  # noqa: BLE001 — args nested past the recursion limit: nothing to say about them
        return
    for walked, value in leaves:
        if any(_declared_at(at, walked) for at in declared):
            continue
        try:
            guess = encoded_guess(value)
        except Exception:  # noqa: BLE001 — a probe that fails is no guess
            guess = None
        if guess is None:
            continue
        shown_tool, shown_path = _label(tool), _label(_spelled(walked, _segment))
        chain = guess.replace('+', ', ')
        if guess == UNKNOWN_BINARY:
            remedy = 'no payload codec decodes it: mock this tool at `seam: client`, where its args are plain values.'
        elif any(step is not None and (not step or any(c in step for c in '.[]')) for step in walked):
            remedy = (
                'a key on this path contains `.`, `[` or `]` (shown as `?`), which a payload_codecs path cannot name: '
                'mock this tool at `seam: client`, where its args are plain values.'
            )
        elif shown_path != _spelled(walked, str):
            remedy = (
                f'If it is, declare tools.{shown_tool}.payload_codecs with a row whose `at` spells this path with the real keys '
                f'(`?` stands for a character not shown, `...` for a cut key) and codec: [{chain}].'
            )
        else:
            remedy = f'If it is, declare tools.{shown_tool}.payload_codecs: [{{at: {shown_path}, codec: [{chain}]}}].'
        _note(
            'undeclared_encoded_payload',
            f'tool {shown_tool}: {shown_path} looks like an encoded payload ({guess}) and no payload_codecs row declares it, '
            f'so the mock reads it encoded. {remedy}',
            tool=shown_tool,
            path=shown_path,
            guess=guess,
        )


# --- the one entry point -------------------------------------------------------------------------

#: Envelope tags of an answer that is not the tool's success (stub, handler error, refusal, runtime fault, refused write).
_NOT_A_REPLY: Final = ('_plurai_stub_error', '_plurai_handler_error', '_plurai_unregistered_tool', '_plurai_runtime_fault', '_plurai_write_refused')


class PayloadEncodeError(Exception):
    """The mock's successful reply has a value at a declared ``returns.*`` path its codec cannot encode. The
    agent's library would get the decoded shape where it reads the encoded one, so the call is the mock
    runtime's fault (recorded by :func:`apply_payload_codecs`), and the caller serves its failed-call stub."""


def _reason(row: PayloadCodec, exc: Exception) -> str:
    """Why a codec failed, never the value: a :class:`CodecError`'s own words, else the chain and the exception's type;
    printable ASCII and capped, whatever the message carried."""
    return _plain(str(exc) if isinstance(exc, CodecError) else f'{"+".join(row.chain)}: {type(exc).__name__}')


def _codec_failed(tool: str, row: PayloadCodec, direction: Direction, reason: str) -> None:
    outcome = 'it was passed through unchanged' if direction == 'args' else 'the agent was served a failed call'
    _note(
        'payload_codec_failed',
        f'tool {_label(tool)}: {_label(row.at)} could not be {"decoded" if direction == "args" else "encoded"} ({reason}); {outcome}.',
        tool=_label(tool),
        path=_label(row.at),
        direction=direction,
        reason=reason,
    )


def _decoder(tool: str, row: PayloadCodec) -> Callable[[Any], Any]:
    def fn(leaf: Any) -> Any:
        if not isinstance(leaf, str | bytes):
            return leaf  # absent, or already in the decoded shape
        try:
            return decode(leaf, row.chain)
        except Exception as exc:  # noqa: BLE001 — any decode failure forwards the raw value; the real API would answer it
            _codec_failed(tool, row, 'args', _reason(row, exc))
            return leaf

    return fn


def _encoder(tool: str, row: PayloadCodec) -> Callable[[Any], Any]:
    def fn(leaf: Any) -> Any:
        if isinstance(leaf, str):
            _note(
                'payload_codec_passthrough',
                f'tool {_label(tool)}: the mock answered {_label(row.at)} as a string, already in the encoded form; it was passed through unchanged.',
                tool=_label(tool),
                path=_label(row.at),
            )
            return leaf
        if not isinstance(leaf, Mapping | list):
            return leaf
        try:
            return encode(leaf, row.chain)
        except Exception as exc:  # noqa: BLE001 — never hand the library the decoded shape
            reason = _reason(row, exc)
            _codec_failed(tool, row, 'returns', reason)
            raise PayloadEncodeError(f'{row.at}: {reason}') from exc

    return fn


def _encode_reply(tool: str, response: Any, rows: list[PayloadCodec]) -> Any:
    def walk(value: Any) -> Any:
        for row in rows:
            value = _at(value, row.segments, _encoder(tool, row))
        return value

    if isinstance(response, Mapping):
        if any(tag in response for tag in _NOT_A_REPLY):
            return response
        status = response.get('status')
        if status == 'error' or (status == 'ok' and response.get('mock') is True):
            return response  # an error envelope, or the untyped no-handler sentinel
        if status == 'success' and 'content' in response:
            content = walk(response['content'])
            return response if content is response['content'] else {**response, 'content': content}
    return walk(response)


def apply_payload_codecs(tool: str, value: Any, direction: Direction) -> Any:
    """``value`` (a call's args, or the mock's answer to it) with ``tool``'s declared payload codecs applied.

    ``args``: each declared ``args.*`` path decoded, and every other string leaf of a ``transport``/``client``
    tool scanned for an undeclared encoded payload (warning only). ``returns``: each declared ``returns.*``
    path encoded on a successful reply; a value there its codec cannot encode is a ``mock_runtime_fault`` (the
    scenario is not valid) and raises :class:`PayloadEncodeError` for the caller to serve its failed-call stub.
    A tool that declares nothing gets ``value`` back, the same object.
    """
    from plurai._mocking.result import declared_payload_codecs, declared_tool_seam  # noqa: PLC0415 — result reads this module's vocabulary

    rows = declared_payload_codecs(tool)
    if direction == 'returns':
        returns = [row for row in rows if row.root == 'returns']
        if not returns:
            return value
        try:
            return _encode_reply(tool, value, returns)
        except PayloadEncodeError as fault:
            note_harness_fault(
                'plurai.PayloadCodecFault',
                f'mocked tool {_label(tool)} answered {fault} in a shape its payload codec cannot encode; the scenario did not measure the agent on it',
                kind='mock_runtime_fault',
                transient=False,
            )
            raise
    if not isinstance(value, Mapping):
        return value
    args_rows = [row for row in rows if row.root == 'args']
    if declared_tool_seam(tool) in ('transport', 'client'):
        _detect_undeclared(tool, value, {row.segments for row in args_rows})
    for row in args_rows:
        value = _at(value, row.segments, _decoder(tool, row))
    return value
