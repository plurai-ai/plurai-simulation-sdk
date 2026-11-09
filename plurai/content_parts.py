"""The ONE text projection of a message's CONTENT, for every reader in this SDK.

A turn's content is a ``str`` or a LIST of standard OpenAI-shaped content parts. This module
exists so that the moment a content is a list, every reader in the SDK projects it the SAME way,
and so that a part this build cannot render is NAMED rather than dropped or repr'd.

The Plurai backend renders the same wire for the judges and the simulator; this module renders
it for ``transcript.json`` and the echo split. The two renderers are separate copies, not a
shared import, because the SDK does not depend on the backend. Two tables that disagree produce
one reply with two projections: ``score_saved_run`` re-scores off this side while live judges
grade the other.

The only guard is ``tests/test_renderer_twin_parity.py``: it runs both implementations over one
table of part shapes and diffs the output. It enforces only where a backend checkout is
reachable (located by an environment variable), because requiring the backend to run the SDK's
unit tests would re-create the dependency the copy exists to remove. Where it cannot reach one it SKIPS, naming
what went unchecked; it does NOT skip on a bad path. The wire does not hold the pair together:
it carries the PARTS, and these tables independently turn parts into TEXT.

THREE PROPERTIES:

* **A lookup, never an ``if`` chain.** ``RENDERABLE_PART_KINDS`` maps a part's ``type`` to its
  renderer, so an unknown kind (``json_v2``) is refused BY NAME instead of being flattered into
  the nearest one it resembles.
* **An unrenderable part is a NAMED placeholder, never ``''`` and never a repr.** Dropping it
  loses an agent answer with no trace; a repr puts ``"[{'type': 'json', …}]"`` in front of a
  judge as though the agent had typed it. ONE exception, and it is a different question rather
  than a softening of this one: a PROVIDER-INTERNAL block (:data:`PROVIDER_BLOCK_KINDS` —
  ``thinking``, ``tool_use``, …) is not something the agent SAID, so it is SILENT
  (:data:`SILENT_PART_KINDS`). Naming those would put a placeholder in front of the agent's own
  words on every reply an extended-thinking model produces. The backend's renderer treats both
  classes the same way.
* **Inert on text.** A ``str`` content and a list of only ``text`` parts project to exactly the
  text a plain reader produces, which makes a text-only harness byte-identical on the wire by
  CONSTRUCTION rather than by test.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any


def _render_text(part: Mapping[str, Any]) -> str:
    """``{'type': 'text', 'text': ...}`` — the text itself, the identity projection.

    The ``None`` case is spelled OUT rather than folded into ``or ''`` so only absence becomes
    the empty string: a provider that emitted the key empty (or omitted it) renders as ``''``
    instead of raising a ``TypeError`` inside the join that would swallow the whole message,
    while every other value still renders AS ITSELF. ``or ''`` would collapse every falsy value:
    ``0`` and ``False`` arrive here as text a provider put on the part, and turning them into
    ``''`` deletes a turn's words (a zero-balance reply, ``text: 0``, is a routine answer) rather
    than projecting them. The backend's renderer does the same.
    """
    text = part.get('text')
    return '' if text is None else str(text)


def _render_json(part: Mapping[str, Any]) -> str:
    """``{'type': 'json', 'json': <value>}`` — the value as a fenced ``json`` block.

    ``ensure_ascii=False`` because the data may be Hebrew, Arabic or anything else a customer
    ships and ``\\uXXXX`` escapes are unreadable to a judge and to a simulated user alike.
    ``sort_keys=True`` because this rendering is compared byte-for-byte across a
    serialization round-trip (see ``_text_blocks_joined``): key order that survives one hop
    and not another would give one reply two different projections.
    """
    body = json.dumps(part.get('json'), ensure_ascii=False, sort_keys=True, default=str)
    return f'```json\n{body}\n```'


def _render_image_url(part: Mapping[str, Any]) -> str:
    """``{'type': 'image_url', 'image_url': {'url': ...}}`` — that an image was on the turn.

    A ``data:`` URL is the image's BYTES; it is named by its media type and never emitted.
    """
    spec = part.get('image_url')
    url = str(spec.get('url') or '') if isinstance(spec, Mapping) else str(spec or '')
    if url.startswith('data:'):
        media_type = url[len('data:') :].split(';', 1)[0].split(',', 1)[0] or 'image'
        return f'[image: inline {media_type} data]'
    return f'[image: {url}]' if url else '[image]'


def _render_file(part: Mapping[str, Any]) -> str:
    """``{'type': 'file', 'file': {'filename': ..., 'file_data': ...}}`` — that a file was on
    the turn, and which one.

    ``file_data`` is base64 and is NEVER rendered: the filename (or the ``file_id`` when the
    file was uploaded rather than inlined) is the identity a reader can act on.
    """
    spec = part.get('file')
    if not isinstance(spec, Mapping):
        return '[file]'
    name = str(spec.get('filename') or spec.get('file_id') or '')
    return f'[file: {name}]' if name else '[file]'


def _render_citation(part: Mapping[str, Any]) -> str:
    """``{'type': 'citation', 'citation': {'title': ..., 'url': ...}}`` — a source the reply SHOWED.

    Unbracketed, because the person read it as part of the reply. A payload naming neither a title nor a url, or a non-string, renders ``''``.
    """
    spec = part.get('citation')
    if isinstance(spec, str):
        return f'Source: {spec}' if spec.strip() else ''
    if not isinstance(spec, Mapping):
        return ''
    title, url = (str(spec.get(key) or '').strip() for key in ('title', 'url'))
    if title and url:
        return f'Source: {title} ({url})'
    return f'Source: {title or url}' if title or url else ''


#: Standard content-part ``type`` -> how this build renders it as view text. THE mechanism:
#: a kind absent from this table is refused by name (``part_text`` returns ``None``) and the
#: caller applies its own contract for an unrenderable part. Read-only so a caller cannot
#: widen the vocabulary by mutating it from the outside — a new kind is a code change here,
#: reviewed with the rendering it implies.
RENDERABLE_PART_KINDS: Mapping[str, Callable[[Mapping[str, Any]], str]] = MappingProxyType(
    {
        'text': _render_text,
        'json': _render_json,
        'image_url': _render_image_url,
        'file': _render_file,
        'citation': _render_citation,
    }
)


def part_text(part: Any) -> str | None:
    """One content part as view text, or ``None`` when this build has no renderer for its kind.

    ``None`` is not "empty" — a ``text`` part that is genuinely empty renders as ``''``, and
    the two must stay distinguishable because the callers treat them oppositely (an empty
    render is joined, an unrenderable part is handed back).

    The part is resolved through :func:`declared_part`, which is THE ONE PLACE this SDK says a
    bare string block is text, rather than through a second ``isinstance(str)`` rule here. A
    bare ``str`` is legal in ``SerializedMessage.content``'s list arm and means the agent's
    WORDS; answering ``None`` would name the agent's own sentence an unrenderable part while
    :func:`_opener_block` reads the identical block on the USER's half of the same wire as text.
    The backend reads it as words too, and ``test_renderer_twin_parity`` compares the two.
    """
    declared = declared_part(part)
    if declared is None:
        return None
    render = RENDERABLE_PART_KINDS.get(str(declared.get('type') or ''))
    return render(declared) if render is not None else None


def declared_part(block: Any) -> Mapping[str, Any] | None:
    """One block of a content LIST as a part with a DECLARED ``type``, or ``None`` if it has none.

    THE ONE PLACE THIS SDK SAYS "a bare string block is text". ``SerializedMessage.content`` is
    typed ``str | list[str | dict]`` — the list arm admits plain STRINGS beside parts, and a
    provider that emits one means the agent's words, which is the reading :func:`_opener_block`
    applies to the user's half of the same wire. Stated here so the AGENT's
    half cannot answer differently: a consumer that decides what an over-budget block costs by
    looking up its declared kind must be handed the kind a bare string HAS, not told it has none.

    ``None`` is "no declared kind" and is not a guess: a block that is neither a mapping nor a
    string gets it, and the caller applies its own contract (name it, drop it) rather than being
    handed an invented projection. That is deliberately narrower than a general coercion — see
    :func:`_opener_block`, which additionally unwraps a pydantic model because ITS carrier can
    hold one.
    """
    if isinstance(block, Mapping):
        return block
    return {'type': 'text', 'text': block} if isinstance(block, str) else None


#: Provider-INTERNAL content blocks: the closed vocabulary a provider (Anthropic today) puts in
#: an assistant message BESIDE its words — reasoning, tool plumbing, attachment handles. The backend
#: carries the same set for the same wire, and ``test_renderer_twin_parity`` diffs the two,
#: because a kind one side calls provider-internal and the other calls unknown is one reply with
#: two projections.
#:
#: DECLARED ONCE, DISPATCHED. :data:`SILENT_PART_KINDS` is DERIVED from it below rather than
#: hand-listed, so adding a kind here is the whole change: a second table would be a second
#: vocabulary to keep in step.
PROVIDER_BLOCK_KINDS: frozenset[str] = frozenset(
    {
        'text',
        'thinking',
        'redacted_thinking',
        'tool_use',
        'tool_result',
        'server_tool_use',
        'web_search_tool_use',
        'web_search_tool_result',
        'code_execution_tool_use',
        'code_execution_tool_result',
        'mcp_tool_use',
        'mcp_tool_result',
        'image',
        'document',
        'search_result',
        'container_upload',
    }
)

#: A provider block this build has no renderer for is SILENT in :func:`render_parts` — not named.
#: DERIVED from the table above minus what this build renders, never a second hand-list.
#:
#: WHY SILENT AND NOT NAMED, which is the opposite of the rule for every other unrenderable kind:
#: a provider block is not something the agent SAID. ``thinking`` / ``redacted_thinking`` are an
#: Anthropic extended-thinking agent's reasoning, present on EVERY reply that model produces;
#: naming them would prepend ``[the agent returned a thinking part this build cannot render as text]``
#: to the agent's own words, which then ride ``/user-turn`` as ``agent_response`` for every judge
#: to grade and for the simulated user to read and answer. ``tool_use`` / ``tool_result`` are the
#: same claim about tool plumbing. The backend's renderer DROPS these by contract ("what was
#: SAID"), so silencing them here keeps the two projections equal on a real reply.
#:
#: An unknown NON-provider kind (``json_v2``, whatever ships next) stays NAMED: this side feeds
#: ``transcript.json`` and the echo split, where a dropped part is an agent answer that vanished.
#: The backend names that class too, so the default arm of the two tables gives one answer at
#: every list position.
SILENT_PART_KINDS: frozenset[str] = PROVIDER_BLOCK_KINDS - frozenset(RENDERABLE_PART_KINDS)


def render_parts(blocks: Any) -> str:
    """A LIST content as text: every renderable part in order, unrenderable ones named.

    Order is the message's own. An empty renderable part contributes nothing (a ``text`` part
    carrying ``''`` is not worth a blank line), but an UNRENDERABLE one always contributes its
    placeholder — the whole point is that it cannot pass unnoticed.
    """
    if not isinstance(blocks, list):
        return ''
    out: list[str] = []
    for block in blocks:
        # The kind is read off :func:`declared_part`, the same resolution :func:`part_text` makes
        # one line down, so the silence gate and the renderer cannot disagree about what a block
        # IS. It admits ``Mapping``, not only ``dict``, so a ``tool_use`` in a ``MappingProxyType``
        # or a ``UserDict`` reaches the silence gate, and a bare ``str`` gets the kind it HAS
        # (``text``) instead of being named by its Python type.
        declared = declared_part(block)
        kind = declared.get('type') if declared is not None else None
        if kind in SILENT_PART_KINDS:
            continue
        rendered = part_text(block)
        if rendered is None:
            named = kind if kind is not None else type(block).__name__
            out.append(f'[the agent returned a {named} part this build cannot render as text]')
        elif rendered:
            out.append(rendered)
    return '\n'.join(out)


def _opener_block(block: Any) -> dict[str, Any]:
    """One block of an opener message's LIST content, as a plain content-part dict.

    ``SerializedMessage.content`` is typed ``str | list``, and the list arm admits blocks that
    are plain STRINGS as well as parts, so a block that is not a dict is never assumed to be a
    pydantic model.

    A string becomes a ``text`` part (what it means); anything else with no ``model_dump`` is
    NAMED as a text part rather than dropped, the same contract :func:`render_parts` applies one
    function up — an opener block that vanishes is the user's words vanishing.
    """
    if isinstance(block, dict):
        return block
    if isinstance(block, str):
        # ONE declaration of what a bare string block MEANS, read here and by the wire budget
        # (``orchestrator._budgeted_parts``) — see :func:`declared_part`.
        return dict(declared_part(block) or {})
    dump = getattr(block, 'model_dump', None)
    if callable(dump):
        dumped = dump()
        return dumped if isinstance(dumped, dict) else {'type': 'text', 'text': str(dumped)}
    return {'type': 'text', 'text': f'[the user sent a {type(block).__name__} part this build cannot render as text]'}


def opener_content(messages: Any) -> str | list[dict[str, Any]] | None:
    """The opener's CONTENT, out of the server's ``initial_messages`` MESSAGE list.

    The distinction this exists for: ``initial_messages`` is a list of
    MESSAGES (``{'type': 'human', 'content': …}``), while everything downstream of this carrier —
    ``child.user_turn``, :func:`content_text`, ``transcript.json``'s per-turn record — takes a
    turn's CONTENT (a ``str``, or a list of content PARTS). Handing the message list straight
    through makes every reader treat ``{'type': 'human'}`` as an unrenderable content part and
    render the opener as ``[the agent returned a human part this build cannot render as text]``.

    Returns ``None`` when the list carries nothing usable, so the caller can fall back to the
    string projection rather than invent an empty turn.

    The opener is ONE user turn. A list with several messages is concatenated IN MESSAGE ORDER —
    rather than silently taking the first or the last, because either choice would drop the
    author's words with no trace. Order is the message list's own: an earlier plain-string message
    stays ahead of a later parts message.
    """
    if not isinstance(messages, list) or not messages:
        return None
    out: list[dict[str, Any]] = []
    texts: list[str] = []
    saw_parts = False
    for message in messages:
        content = message.get('content') if isinstance(message, dict) else getattr(message, 'content', None)
        if isinstance(content, list):
            saw_parts = True
            out.extend(_opener_block(block) for block in content)
        elif isinstance(content, str) and content:
            texts.append(content)
            out.append({'type': 'text', 'text': content})
    if saw_parts:
        # A turn that mixes a plain-string message with a parts message keeps both, the strings
        # rendered as text parts so one list carries the whole turn.
        return out
    return '\n'.join(texts) if texts else None


def content_text(content: Any) -> str:
    """A message CONTENT — ``str`` or list of parts — as text.

    The one call every string reader in this SDK makes. A ``str`` is returned unchanged, which is
    what keeps a text-only harness byte-identical: this function is the identity on the shape
    that harness produces.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return render_parts(content)
    return ''


#: The part kinds that carry a document by VALUE, and where: ``file.file_data`` and ``image_url.url``
#: (an ``image_url`` may also be spelled as the bare string).
_DOCUMENT_VALUE_SLOTS: Mapping[str, str] = MappingProxyType({'file': 'file_data', 'image_url': 'url'})


def _document_value(part: Any) -> str | None:
    if not isinstance(part, Mapping) or (slot := _DOCUMENT_VALUE_SLOTS.get(str(part.get('type')))) is None:
        return None
    spec = part.get(str(part['type']))
    value = spec.get(slot) if isinstance(spec, Mapping) else spec
    return value if isinstance(value, str) else None


def document_part_values(content: Any) -> list[str]:
    """The document values a turn's ``file`` and ``image_url`` parts carry, in part order.

    Empty for a ``str`` turn, an absent one, or a turn whose parts carry no document.
    """
    if not isinstance(content, list):
        return []
    return [value for part in content if (value := _document_value(part)) is not None]


def with_part_values(content: Any, replace: Mapping[str, str]) -> Any:
    """``content`` with every ``file`` / ``image_url`` value that is a key of ``replace`` carrying its replacement.

    Every other part, and a ``str`` or absent turn, is returned as it is; ``content`` is not mutated.
    """
    if not isinstance(content, list) or not replace:
        return content
    out: list[Any] = []
    for part in content:
        value = _document_value(part)
        if value is None or value not in replace:
            out.append(part)
            continue
        kind = str(part['type'])
        spec = part[kind]
        out.append({**part, kind: {**spec, _DOCUMENT_VALUE_SLOTS[kind]: replace[value]} if isinstance(spec, Mapping) else replace[value]})
    return out
