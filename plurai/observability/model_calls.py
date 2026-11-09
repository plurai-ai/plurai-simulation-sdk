"""The model calls inside one agent turn, derived from the turn's serialized message chain.

A turn's chain is ``[ai(tool_calls), tool, ai(tool_calls), tool, ..., ai(final)]``: every
assistant message is one call to the model, made with everything before it in the turn as its
context. Observability backends model that as a generation (Langfuse) or an ``llm`` run
(LangSmith) so token usage, model name and the prompt/completion pair render natively, instead
of one opaque ``agent`` node.

Total: no I/O, no sink imports (``sink_protocol`` imports this module for the
:class:`ModelCall` type), and a message shape it cannot read is skipped, never raised on.
A chain without an assistant message yields no calls; none is invented from ``agent_response``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict

from plurai.content_parts import content_text


class ModelCall(TypedDict):
    """One model call within a turn: what it was shown and what it produced.

    * ``input`` — the chat messages visible to the call, in order: the turn's user message
      (when the chain does not start with it) followed by the chain messages before this one.
      Tool calls on those messages use the OpenAI chat spelling (:func:`openai_tool_calls`),
      which is what the vendor chat renderers read.
    * ``output`` — the assistant message's text.
    * ``tool_calls`` — the tool calls the message requested, as ``{name, args, id}``.
    * ``model`` — the model that answered, when the message's metadata names it.
    * ``provider`` — who served it (LangChain's ``response_metadata.model_provider``), when named.
    * ``usage`` — the message's token usage as the framework reports it (LangChain's
      ``usage_metadata``: ``input_tokens`` / ``output_tokens`` / ``total_tokens``).
    """

    input: list[dict[str, Any]]
    output: str
    tool_calls: list[dict[str, Any]]
    model: NotRequired[str]
    provider: NotRequired[str]
    usage: NotRequired[dict[str, Any]]


#: LangChain ``type`` / OpenAI ``role`` spellings of each chat role.
_ROLES: Mapping[str, str] = {
    'ai': 'assistant',
    'aimessagechunk': 'assistant',
    'assistant': 'assistant',
    'human': 'user',
    'humanmessagechunk': 'user',
    'user': 'user',
    'tool': 'tool',
    'function': 'tool',
    'system': 'system',
}


def _field(message: Any, name: str) -> Any:
    """``message[name]`` for a mapping, ``message.name`` for an object, else ``None``."""
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def _role_of(message: Any) -> str | None:
    """The chat role a message declares, whichever of ``role`` / ``type`` carries it."""
    for key in ('role', 'type'):
        declared = _field(message, key)
        if isinstance(declared, str) and (role := _ROLES.get(declared.lower())) is not None:
            return role
    return None


def _args_of(raw: Any) -> dict[str, Any]:
    """A tool call's arguments as a mapping; a JSON string is decoded, text that is not JSON is
    kept as ``{'_unparsed_arguments': raw}`` (the harness helper's spelling), anything else is ``{}``."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return {'_unparsed_arguments': raw}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _tool_calls_of(message: Any) -> list[dict[str, Any]]:
    """The message's ``tool_calls`` as ``{name, args, id}``; an entry that names no tool is dropped.

    Reads both wire spellings: ``{name, args, id}`` and the OpenAI ``{id, function: {name, arguments}}``.
    """
    raw = _field(message, 'tool_calls')
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return []
    calls: list[dict[str, Any]] = []
    for entry in raw:
        function = _field(entry, 'function')
        name = _field(entry, 'name') or _field(function, 'name')
        if not isinstance(name, str) or not name:
            continue
        args = _field(entry, 'args')
        calls.append({'name': name, 'args': _args_of(args if args is not None else _field(function, 'arguments')), 'id': _field(entry, 'id')})
    return calls


def openai_tool_calls(tool_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``{name, args, id}`` tool calls in the OpenAI chat spelling vendor chat renderers read.

    ``{'id', 'type': 'function', 'function': {'name', 'arguments': <JSON text>}}``; arguments
    that do not serialise are carried as their ``str``.
    """
    return [
        {
            'id': call.get('id'),
            'type': 'function',
            'function': {'name': call.get('name'), 'arguments': _arguments_text(call.get('args'))},
        }
        for call in tool_calls
    ]


def _arguments_text(args: Any) -> str:
    """Tool arguments as JSON text; arguments JSON cannot hold (a non-string key) are carried as their ``str``."""
    try:
        return json.dumps(args or {}, default=str)
    except (TypeError, ValueError):
        return str(args)


def _model_of(message: Any) -> str | None:
    """The model name from ``response_metadata`` (``model_name`` / ``model``) or a ``model`` field."""
    metadata = _field(message, 'response_metadata')
    for source, key in ((metadata, 'model_name'), (metadata, 'model'), (message, 'model')):
        value = _field(source, key) if source is not None else None
        if isinstance(value, str) and value:
            return value
    return None


def _provider_of(message: Any) -> str | None:
    """The serving provider from ``response_metadata.model_provider``, ``None`` when unnamed."""
    value = _field(_field(message, 'response_metadata'), 'model_provider')
    return value if isinstance(value, str) and value else None


def _usage_of(message: Any) -> dict[str, Any] | None:
    """The message's ``usage_metadata`` as a plain dict, ``None`` when absent or empty."""
    usage = _field(message, 'usage_metadata')
    return dict(usage) if isinstance(usage, Mapping) and usage else None


def _chat_message(message: Any, role: str) -> dict[str, Any]:
    """One chain message as an OpenAI-style chat message: ``role`` + ``content`` (+ tool fields)."""
    # Text only: every later call of the turn repeats this message in its input, so an attached
    # document or image (a base64 content part) would be re-sent once per call.
    out: dict[str, Any] = {'role': role, 'content': content_text(_field(message, 'content'))}
    if role == 'assistant' and (calls := _tool_calls_of(message)):
        out['tool_calls'] = openai_tool_calls(calls)
    if role == 'tool' and isinstance(call_id := _field(message, 'tool_call_id'), str):
        out['tool_call_id'] = call_id
    return out


def _already_seen(message: Any, seen_message_ids: set[str] | None) -> bool:
    """Whether an earlier turn made a call of this message; records its id when it did not."""
    message_id = _field(message, 'id')
    if seen_message_ids is None or not isinstance(message_id, str) or not message_id:
        return False
    if message_id in seen_message_ids:
        return True
    seen_message_ids.add(message_id)
    return False


def _with_opener(context: list[dict[str, Any]], opener_text: str | None) -> list[dict[str, Any]]:
    """``context`` with the turn's user message placed after its leading system messages.

    Unchanged when there is no user message to add, or when the leading system/user run
    (everything before the first assistant or tool message) already holds one.
    """
    if opener_text is None:
        return list(context)
    lead = 0
    while lead < len(context) and context[lead]['role'] in ('system', 'user'):
        if context[lead]['role'] == 'user':
            return list(context)
        lead += 1
    systems = 0
    while systems < len(context) and context[systems]['role'] == 'system':
        systems += 1
    return [*context[:systems], {'role': 'user', 'content': opener_text}, *context[systems:]]


def derive_model_calls(
    messages: Sequence[Any] | None,
    *,
    input_user_message: str | list[dict[str, Any]] | None,
    seen_message_ids: set[str] | None = None,
) -> list[ModelCall]:
    """One :class:`ModelCall` per assistant message in this turn's ``messages``.

    ``input_user_message`` is the user message the agent was given this turn. It opens every
    call's ``input`` (after any leading system messages) unless the chain's leading
    system/user run already holds a user message of its own; ``None`` (an agent-first turn)
    adds nothing. A message with no readable role is neither a call nor context.

    ``seen_message_ids`` (mutated) holds the ids of assistant messages an earlier turn already
    made calls of: such a message is context here, not a call, so a chain that re-lists earlier
    turns' history does not count their tokens and cost twice. A message without an id is
    always a call.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, str):
        return []
    opener_text = None if input_user_message is None else content_text(input_user_message)
    context: list[dict[str, Any]] = []
    calls: list[ModelCall] = []
    for message in messages:
        role = _role_of(message)
        if role is None:
            continue
        if role == 'assistant' and not _already_seen(message, seen_message_ids):
            call = ModelCall(
                input=_with_opener(context, opener_text),
                output=content_text(_field(message, 'content')),
                tool_calls=_tool_calls_of(message),
            )
            if model := _model_of(message):
                call['model'] = model
            if provider := _provider_of(message):
                call['provider'] = provider
            if usage := _usage_of(message):
                call['usage'] = usage
            calls.append(call)
        context.append(_chat_message(message, role))
    return calls
