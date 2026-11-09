"""The one reader of a mocked call's answer: :func:`mock_result` and :class:`MockToolError`.

A mocked call answers with the mock envelope (defined by the backend, served as the
``mock_envelope`` client contract). Its ``model`` column says what the agent is handed: the tool's
value, or a failure. :func:`mock_result` executes that column — it returns the value or raises
:class:`MockToolError` — so a harness adapter decides only HOW its agent sees a failure (its own
exception class, an error tool message, or the failure value its real tool returns), never WHETHER
the call failed. ``mock_wrap`` and ``MockToolNode`` read through it.

How the REAL tool reports the failure the mock models is a typed fact, ``tools.<name>.error_channel``
(``raises`` | ``returns``, with ``error_channel_source``: ``trace`` | ``declared`` | ``inferred``) in
the ``plurai.config.yaml`` beside the harness — a sibling of ``descriptor:``, read here the way the
child reads ``world_state.owner``. ``mock_wrap`` raises only for a declared ``raises``; an
undeclared tool gets the response whole.

``tools.<name>.payload_codecs`` (the fields the wire carries encoded) and ``tools.<name>.seam`` are read
the same way, for :func:`plurai._mocking.codecs.apply_payload_codecs`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from plurai._mocking.codecs import PayloadCodec, parse_payload_codecs
from plurai.ingest.workspace_config import WorkspaceConfigError, bench_config_roots, discover_config_documents

ErrorChannel = Literal['raises', 'returns']
ERROR_CHANNELS: tuple[ErrorChannel, ...] = ('raises', 'returns')

#: The mock engine's tag on its refusal of a tool with no handler (the backend's envelope defines the same key).
ENGINE_REFUSAL_KEY: Final = '_plurai_unregistered_tool'
#: The mock engine's tag on a handler that raised; that envelope has no ``status`` (the backend's envelope defines the same key).
_HANDLER_ERROR_KEY = '_plurai_handler_error'
#: The SDK's own error stub carries its cause here (``resolution._stub_or_legacy_sentinel``). The envelope's
#: ``message`` is the agent-visible text the real tool would show; this cause is read only when there is no ``message``.
_STUB_ERROR_KEY = '_plurai_stub_error'


class MockToolError(Exception):
    """A mocked tool call that failed: the envelope's ``model`` view is ``fails``.

    ``message`` is the failure text the envelope carries; ``case`` is the envelope row it is an
    instance of (``error_message``, ``engine_refusal``, ``handler_error``,
    ``drifted_status_with_content``); ``response`` is the envelope as served. Catch it in the
    harness adapter and hand the agent its own failure — never a value that reads as a success.
    """

    def __init__(self, message: str, case: str, response: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.case = case
        self.response = response


def _failure_text(response: Mapping[str, Any], *keys: str) -> str:
    """The failure's own words: the first non-blank string under ``keys``, else the first structured
    diagnostic (a dict or list the model can still act on, as JSON), else a constant naming the status."""
    for key in keys:
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in keys:
        value = response.get(key)
        if not isinstance(value, str) and value not in (None, [], {}):
            return json.dumps(value, ensure_ascii=False, default=str)
    return f'mocked tool failed (status {response.get("status")!r})'


def mock_result(response: Any, channel: ErrorChannel | None = None) -> Any:
    """What a mocked call hands the agent: the tool's value, or :class:`MockToolError`.

    * ``{"status": "success", "content": X, ...}`` → ``X`` (the siblings are bookkeeping, not the return).
    * ``{"status": "error", "content": {map}, ...}`` → the map: a failure the real tool reports through its value
      — unless ``channel`` is ``raises`` (the tool's declared ``error_channel``), which raises it.
    * any other ``error``, a handler that raised, the engine's refusal, or a non-standard status
      token carrying ``content`` → raises :class:`MockToolError`.
    * anything else (a success with no ``content``, a schema stub, a bare list) → the response whole.
    """
    if not isinstance(response, Mapping):
        return response
    if response.get(_HANDLER_ERROR_KEY) is True:
        raise MockToolError(_failure_text(response, 'error', 'message'), 'handler_error', response)
    status = response.get('status')
    if status == 'success':
        return response.get('content', response)
    if status == 'error':
        if response.get(ENGINE_REFUSAL_KEY) is True:
            raise MockToolError(_failure_text(response, 'error', 'message'), 'engine_refusal', response)
        if isinstance(response.get('content'), Mapping) and channel != 'raises':
            return response['content']
        raise MockToolError(_failure_text(response, 'message', _STUB_ERROR_KEY, 'error', 'content'), 'error_message', response)
    if 'content' in response:
        raise MockToolError(_failure_text(response, 'message', 'error', 'content'), 'drifted_status_with_content', response)
    return response


# --- The typed fact: how the real tool reports a failure ------------------------------------------

_harness_path: Path | None = None
_declared_channels: dict[str, ErrorChannel] | None = None
_declared_codecs: dict[str, tuple[PayloadCodec, ...]] | None = None
_declared_seams: dict[str, str] | None = None


def use_harness_config(simulate_path: Path | str | None) -> None:
    """Read ``tools.<name>.error_channel`` (and ``payload_codecs`` / ``seam``) from the config beside
    ``simulate_path``; the child calls this before it imports the harness. ``None`` forgets it, so
    nothing is declared."""
    global _harness_path, _declared_channels, _declared_codecs, _declared_seams  # noqa: PLW0603 — one harness per process, read once
    _harness_path = Path(simulate_path).resolve() if simulate_path is not None else None
    _declared_channels = None
    _declared_codecs = None
    _declared_seams = None


def declared_error_channel(tool: str) -> ErrorChannel | None:
    """The ``error_channel`` the config beside the harness declares for ``tool``, or ``None``.

    A value outside :data:`ERROR_CHANNELS`, or a config that cannot be read, declares nothing and
    says so on stderr: the tool gets the response whole.
    """
    global _declared_channels  # noqa: PLW0603
    if _declared_channels is None:
        _declared_channels = {}
        if _harness_path is not None:
            try:
                for root in bench_config_roots(_harness_path):
                    for path, document in discover_config_documents(root):
                        tools = document.get('tools')
                        for name, entry in tools.items() if isinstance(tools, dict) else ():
                            value = entry.get('error_channel') if isinstance(entry, dict) else None
                            if value is None:
                                continue
                            if value in ERROR_CHANNELS:
                                _declared_channels[str(name)] = value
                            else:
                                print(f'[plurai] {path}: tools.{name}.error_channel {value!r} is not raises or returns; ignored.', file=sys.stderr)
            except WorkspaceConfigError as exc:
                print(f'[plurai] tool error channels unreadable, so none is declared: {exc}', file=sys.stderr)
    return _declared_channels.get(tool)


# --- The typed fact: which fields the wire carries encoded (``plurai._mocking.codecs``) -------------


def _read_codec_declarations() -> tuple[dict[str, tuple[PayloadCodec, ...]], dict[str, str]]:
    """``tools.<name>.payload_codecs`` (valid rows only; a refused row is named on stderr) and ``tools.<name>.seam``.
    A config that cannot be read declares nothing here; :func:`declared_error_channel` already says so on stderr."""
    codecs: dict[str, tuple[PayloadCodec, ...]] = {}
    seams: dict[str, str] = {}
    if _harness_path is None:
        return codecs, seams
    try:
        for root in bench_config_roots(_harness_path):
            for path, document in discover_config_documents(root):
                tools = document.get('tools')
                for name, entry in tools.items() if isinstance(tools, dict) else ():
                    if not isinstance(entry, dict):
                        continue
                    if isinstance(entry.get('seam'), str):
                        seams[str(name)] = entry['seam']
                    if entry.get('payload_codecs') is None:
                        continue
                    rows, refused = parse_payload_codecs(entry['payload_codecs'])
                    for reason in refused:
                        print(f'[plurai] {path}: tools.{name}.payload_codecs{reason}; ignored.', file=sys.stderr)
                    codecs[str(name)] = tuple(rows)
    except WorkspaceConfigError:
        return {}, {}
    return codecs, seams


def declared_payload_codecs(tool: str) -> tuple[PayloadCodec, ...]:
    """The ``payload_codecs`` rows the config beside the harness declares for ``tool``; ``()`` when none."""
    global _declared_codecs, _declared_seams  # noqa: PLW0603
    if _declared_codecs is None or _declared_seams is None:
        _declared_codecs, _declared_seams = _read_codec_declarations()
    return _declared_codecs.get(tool, ())


def declared_tool_seam(tool: str) -> str | None:
    """The ``seam`` the config beside the harness declares for ``tool`` (``transport``, ``client``, …), or ``None``."""
    declared_payload_codecs(tool)
    return (_declared_seams or {}).get(tool)
