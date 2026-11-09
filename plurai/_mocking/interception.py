"""LangGraph mock interception — MockToolNode, mock_wrap, apply_mocks.

Public helpers customers use in simulate.py to route tool calls through the
mock environment.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import sys
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, ToolMessage

from plurai_mockenv import _registry
from plurai_mockenv._registry import _active_session_id

from plurai._stubs import extract_return_type
from plurai._mocking.resolution import _resolve_mock_response
from plurai._mocking.result import MockToolError, declared_error_channel, mock_result

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable as _NodeBase
    from langchain_core.runnables import RunnableConfig
else:
    try:
        from langchain_core.runnables import Runnable as _NodeBase
    except ImportError:  # pragma: no cover - langchain-core's runnables import langsmith; without it the node stays a plain callable
        _NodeBase = object

try:
    from langchain_core.tools.base import TOOL_MESSAGE_BLOCK_TYPES as _TOOL_MESSAGE_BLOCK_TYPES
except ImportError:  # pragma: no cover - same optional-dependency guard as above
    _TOOL_MESSAGE_BLOCK_TYPES = ('text', 'image_url', 'image', 'json', 'search_result', 'custom_tool_call_output', 'document', 'file')

logger = logging.getLogger(__name__)

_MODE_LABELS = {'mock': 'MOCK', 'real': 'REAL', 'observe': 'OBSERVE'}


#: What LangGraph's ``ToolNode`` answers when the model names a tool the node does not hold.
_INVALID_TOOL_NAME_TEMPLATE = 'Error: {requested_tool} is not a valid tool, try one of [{available_tools}].'


def _render_content(value: Any) -> Any:
    """A tool result as ``ToolMessage.content``, the way LangChain renders it.

    A string and a list of content blocks pass through; everything else is JSON
    (``ensure_ascii=False``), falling back to ``str`` for what JSON cannot encode. ``None`` is
    ``'null'`` and a bool is ``'true'`` / ``'false'``, exactly as the real tool's message shows.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(x, dict) and x.get('type') in _TOOL_MESSAGE_BLOCK_TYPES for x in value):
        return list(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - mirrors LangChain: anything JSON cannot encode renders as str()
        return str(value)


def _error_message(name: str, cid: Any, payload: Any) -> ToolMessage:
    return ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=cid, name=name, status='error')


def _is_command(value: Any) -> bool:
    try:
        from langgraph.types import Command
    except ImportError:
        return False
    return isinstance(value, Command)


def _control_flow_exceptions() -> tuple[type[BaseException], ...]:
    """LangGraph's interrupt / parent-command signals, which must reach the graph runtime."""
    try:
        from langgraph.errors import GraphBubbleUp
    except ImportError:
        return ()
    return (GraphBubbleUp,)


def _has_injected_args(tool: Any) -> bool:
    """Whether the tool declares arguments the model does not supply (``InjectedToolCallId`` and kin)."""
    try:
        return bool(set(tool.get_input_schema().model_fields) - set(tool.tool_call_schema.model_fields))
    except Exception:  # noqa: BLE001 - a duck-typed tool without schemas has no injected arguments
        return False


def _needs_envelope(tool: Any) -> bool:
    """Whether the tool must be invoked with its tool-call envelope rather than its bare arguments.

    ``content_and_artifact`` tools put the artifact on the message, a tool that converts its own
    exceptions to text reports them as ``status='error'``, and ``InjectedToolCallId`` is filled from
    the envelope; all three exist only for the tool-call input ``BaseTool`` turns into a ``ToolMessage``.
    """
    return (
        getattr(tool, 'response_format', 'content') == 'content_and_artifact'
        or _is_error_handler_flag(getattr(tool, 'handle_tool_error', False))
        or _is_error_handler_flag(getattr(tool, 'handle_validation_error', False))
        or _has_injected_args(tool)
    )


def _as_runnable_tool(tool: Any) -> Any:
    """A plain callable as a LangChain tool (what ``ToolNode`` does at construction); anything with ``invoke`` as is.

    An ``async def`` callable becomes a coroutine-only tool, so ``ainvoke`` awaits it and a sync ``invoke``
    fails exactly as it does for ``ToolNode``.
    """
    if hasattr(tool, 'invoke') or not callable(tool):
        return tool
    from langchain_core.tools import StructuredTool

    kwargs: dict[str, Any] = {
        'name': getattr(tool, 'name', None) or getattr(tool, '__name__', None),
        'description': inspect.getdoc(tool) or '',
    }
    if inspect.iscoroutinefunction(tool):
        return StructuredTool.from_function(coroutine=tool, **kwargs)
    return StructuredTool.from_function(tool, **kwargs)


def _is_error_handler_flag(flag: Any) -> bool:
    """A ``handle_tool_error`` / ``handle_validation_error`` setting LangChain honours: ``True``, a message or a callable."""
    return flag is True or isinstance(flag, str) or inspect.isfunction(flag) or inspect.ismethod(flag)


class MockToolNode(_NodeBase):
    """A drop-in LangGraph tool node that routes tool calls through your mocks.

    Replace your agent's real ``ToolNode`` with this in ``simulate.py`` so every
    tool call is served by the mock environment (the world-state-backed mock
    service) instead of hitting production systems. Each tool is dispatched per
    ``mock_config`` mode: ``'mock'`` (default) returns the mocked response
    (:func:`plurai.mock_result`; a failed call is a ``status='error'`` ToolMessage),
    ``'observe'`` runs the real tool and records its result to world-state, and
    ``'real'`` runs the tool untouched.

    The messages it returns match ``ToolNode``'s: ``name``, ``tool_call_id``, ``status``
    (``'error'`` on a failure), ``artifact`` and the rendered ``content``, for sync and async
    graphs. Where it deliberately differs (calls run one after another; a tool exception becomes an
    error message instead of failing the run; injected state / store / runtime arguments are not
    supplied) a parity test pins each divergence.

    Example::

        import plurai
        from langgraph.graph import StateGraph

        node = plurai.MockToolNode(
            tools=[get_user_profile, process_refund],
            mock_config={'get_user_profile': 'mock', 'process_refund': 'mock'},
        )
        graph.add_node('tools', node)
    """

    def __init__(self, tools: list, mock_config: dict[str, str] | None = None, *, name: str = 'tools', messages_key: str = 'messages'):
        self.name = name
        self.messages_key = messages_key
        self.tools_by_name = {}
        for t in tools:
            tool_name = getattr(t, 'name', None) or getattr(t, '__name__', str(t))
            self.tools_by_name[tool_name] = t
        self.mock_config = mock_config or dict.fromkeys(self.tools_by_name, 'mock')

    # -- Runnable surface -------------------------------------------------------------------

    def __call__(self, state: Any, config: RunnableConfig | None = None) -> Any:
        return self.invoke(state, config)

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        tool_calls, shape = self._parse_input(input)
        return self._combine([self._call_one(tc, config) for tc in tool_calls], shape)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        tool_calls, shape = self._parse_input(input)
        return self._combine([await self._acall_one(tc, config) for tc in tool_calls], shape)

    # -- input / output shape ---------------------------------------------------------------

    def _parse_input(self, input: Any) -> tuple[list, str]:
        """The tool calls of the latest AI message, and whether the state was a bare message list (``'list'``) or a mapping."""
        if isinstance(input, list):
            messages, shape = input, 'list'
        elif isinstance(input, dict):
            messages, shape = input.get(self.messages_key, []), 'dict'
        else:
            messages, shape = getattr(input, self.messages_key, []), 'dict'
        messages = messages or []
        latest = next((m for m in reversed(messages) if isinstance(m, AIMessage)), messages[-1] if messages else None)
        return list(getattr(latest, 'tool_calls', None) or []), shape

    def _combine(self, outputs: list, shape: str) -> Any:
        """The node's state update: ToolMessages as ``{messages_key: [...]}`` (a list for list state), with any ``Command`` kept as its own update."""
        flat: list = []
        for out in outputs:
            flat.extend(out) if isinstance(out, list) else flat.append(out)
        if not any(_is_command(o) for o in flat):
            return flat if shape == 'list' else {self.messages_key: flat}

        from langgraph.types import Command, Send

        combined: list = []
        parent_sends: list[Send] = []
        for out in flat:
            if not _is_command(out):
                combined.append([out] if shape == 'list' else {self.messages_key: [out]})
            elif out.graph is Command.PARENT and isinstance(out.goto, list) and all(isinstance(s, Send) for s in out.goto):
                parent_sends.extend(out.goto)
            else:
                combined.append(out)
        if parent_sends:
            combined.append(Command(graph=Command.PARENT, goto=parent_sends))
        return combined

    # -- one call ---------------------------------------------------------------------------

    def _call_one(self, tc: dict, config: RunnableConfig | None) -> Any:
        name = tc.get('name', '')
        args = tc.get('args', {})
        cid = tc.get('id', uuid.uuid4().hex[:8])
        mode = self.mock_config.get(name, 'mock')

        if mode == 'mock':
            return self._mock_message(name, args, cid)
        if mode not in ('observe', 'real'):
            return self._unrecognized_mode(name, cid, mode)
        tool = self.tools_by_name.get(name)
        if tool is None:
            return self._unknown_tool(name, cid, mode)
        tool = _as_runnable_tool(tool)
        try:
            envelope, raw = self._invoke_real(tool, {'name': name, 'args': args, 'id': cid, 'type': 'tool_call'}, config)
        except _control_flow_exceptions():
            raise
        except Exception as e:
            return _error_message(name, cid, {'error': str(e)})
        if mode == 'observe':
            self._observe(name, args, raw)
        return self._real_output(envelope, raw, name, cid)

    async def _acall_one(self, tc: dict, config: RunnableConfig | None) -> Any:
        name = tc.get('name', '')
        args = tc.get('args', {})
        cid = tc.get('id', uuid.uuid4().hex[:8])
        mode = self.mock_config.get(name, 'mock')

        if mode == 'mock':
            # The mock service call is blocking HTTP; it runs off the event loop.
            return await asyncio.to_thread(self._mock_message, name, args, cid)
        if mode not in ('observe', 'real'):
            return self._unrecognized_mode(name, cid, mode)
        tool = self.tools_by_name.get(name)
        if tool is None:
            return self._unknown_tool(name, cid, mode)
        tool = _as_runnable_tool(tool)
        try:
            envelope, raw = await self._ainvoke_real(tool, {'name': name, 'args': args, 'id': cid, 'type': 'tool_call'}, config)
        except _control_flow_exceptions():
            raise
        except Exception as e:
            return _error_message(name, cid, {'error': str(e)})
        if mode == 'observe':
            await asyncio.to_thread(self._observe, name, args, raw)
        return self._real_output(envelope, raw, name, cid)

    def _mock_message(self, name: str, args: dict, cid: Any) -> ToolMessage:
        # Sentinel detection + schema-stub fallback live in the shared
        # resolver, which also stamps the event's conversation turn from
        # the shared turn context — the OUTER conversation index, never a
        # per-call counter. ``mock_wrap`` goes through the same resolver,
        # so both interception paths report one turn per turn.
        # The answer is read through ``mock_result`` with the tool's declared ``error_channel``: a
        # failed call (every fails row, and an error-with-content for a declared ``raises`` tool) is a
        # ``status='error'`` message carrying the failure's own text, as ToolNode reports a raise.
        # Only a declared ``returns`` tool gets an error-with-content's map as its value
        # (``status='success'``); an UNDECLARED tool gets the error envelope whole for that row,
        # ``status='error'``.
        return_type = extract_return_type(self.tools_by_name.get(name))
        resolved = _resolve_mock_response(name, args, return_type)
        channel = declared_error_channel(name)
        try:
            value = mock_result(resolved, channel)
        except MockToolError as exc:
            return ToolMessage(content=exc.message, tool_call_id=cid, name=name, status='error')
        if channel is None and isinstance(resolved, dict) and resolved.get('status') == 'error':
            return ToolMessage(content=_render_content(resolved), tool_call_id=cid, name=name, status='error')
        return ToolMessage(content=_render_content(value), tool_call_id=cid, name=name, status='success')

    def _unrecognized_mode(self, name: str, cid: Any, mode: str) -> ToolMessage:
        # Never leave a dangling tool_call_id without a response, which would break the agent loop.
        logger.error('unrecognized mock mode %r for tool %r; emitting error ToolMessage', mode, name)
        return _error_message(name, cid, {'error': f'unrecognized mock mode: {mode}'})

    def _unknown_tool(self, name: str, cid: Any, mode: str) -> ToolMessage:
        logger.error("unknown tool %r in '%s' mode; emitting error ToolMessage", name, mode)
        content = _INVALID_TOOL_NAME_TEMPLATE.format(requested_tool=name, available_tools=', '.join(self.tools_by_name))
        return ToolMessage(content=content, tool_call_id=cid, name=name, status='error')

    def _invoke_real(self, tool: Any, call: dict, config: RunnableConfig | None) -> tuple[Any, Any]:
        """``(envelope, raw)``: the tool's own ToolMessage when it carries more than the return value, and the return value."""
        if _needs_envelope(tool):
            envelope = tool.invoke(call, config)
            return envelope, getattr(envelope, 'content', envelope)
        args = call['args']
        return None, tool.invoke(args, config) if config is not None else tool.invoke(args)

    async def _ainvoke_real(self, tool: Any, call: dict, config: RunnableConfig | None) -> tuple[Any, Any]:
        if not hasattr(tool, 'ainvoke'):
            return await asyncio.to_thread(self._invoke_real, tool, call, config)
        if _needs_envelope(tool):
            envelope = await tool.ainvoke(call, config)
            return envelope, getattr(envelope, 'content', envelope)
        return None, await tool.ainvoke(call['args'], config)

    def _observe(self, name: str, args: dict, result: Any) -> None:
        try:
            if _registry._client is None:
                raise RuntimeError('mock service client not configured; cannot observe tool results to world-state')
            _registry._client.observe_result(_active_session_id(), name, args, result)
        except Exception as obs_exc:
            # observe_result failure must NOT crash the real tool call — the
            # tool already returned a result. Log at ERROR: a failed world-state
            # write means policy eval runs on stale state, which is a
            # high-severity data-quality issue the operator must be aware of.
            logger.error(
                'observe_result failed for tool %r (world-state write failed; policy eval will run on stale state): %r',
                name,
                obs_exc,
            )

    @staticmethod
    def _real_output(envelope: Any, raw: Any, name: str, cid: Any) -> Any:
        """The state update a real tool's result becomes: its own ToolMessage / Command / message list as is, else a rendered ToolMessage."""
        result = envelope if envelope is not None else raw
        if isinstance(result, ToolMessage):
            result.content = _render_content(result.content)
            return result
        if _is_command(result):
            return result
        if isinstance(result, list) and result and all(isinstance(r, ToolMessage) or _is_command(r) for r in result):
            for r in result:
                if isinstance(r, ToolMessage):
                    r.content = _render_content(r.content)
            return result
        return ToolMessage(content=_render_content(result), tool_call_id=cid, name=name, status='success')


def _tool_content(response: Any) -> Any:
    """What ``mock_wrap`` hands a tool with no declared ``error_channel`` (and no ``failure``): a
    success's ``content``, else the response whole. Every typed reading goes through
    :func:`plurai.mock_result`."""
    if isinstance(response, dict) and response.get('status') == 'success' and 'content' in response:
        return response['content']
    return response


def mock_wrap(func: Callable, mode: str = 'mock', failure: Callable[[MockToolError], Any] | None = None) -> Callable:
    """Wrap a single tool callable so its calls route through the mock env.

    The function-level counterpart to :class:`MockToolNode`, for agents that call
    tools directly rather than through a LangGraph tool node. ``mode='mock'``
    (default) returns the mocked response (:func:`plurai.mock_result`), ``mode='observe'``
    runs the real tool and records its result to world-state, and ``mode='real'``
    returns the tool unchanged. The wrapper preserves the tool's ``name`` /
    ``description`` / ``args_schema`` attributes when present.

    A failed mocked call follows the tool's declared ``tools.<name>.error_channel``
    (the config beside the harness): ``raises`` RAISES :class:`plurai.MockToolError`,
    as the real tool does; ``returns`` hands back ``failure(error)`` — the failure
    value the real tool returns — when ``failure`` is given. Undeclared and with
    no ``failure``, the failed call's response is handed back whole.

    Example::

        import plurai

        get_user_profile = plurai.mock_wrap(get_user_profile)  # now mocked
        # tools.save_note.error_channel: returns — the real tool returns its failure
        save_note = plurai.mock_wrap(save_note, failure=lambda e: {'error': e.message})
    """
    if mode == 'real':
        return func
    name = getattr(func, 'name', None) or getattr(func, '__name__', str(func))
    # Capture the return-type annotation once at wrap time. Frozen on the
    # wrapper for the rest of its lifetime; the shared resolver uses it to
    # produce a schema-shaped stub whenever the backend has no handler.
    return_type = extract_return_type(func)

    if mode == 'observe':

        @functools.wraps(func)
        def ow(*args, **kwargs):
            result = func(*args, **kwargs)
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            merged = {params[i]: a for i, a in enumerate(args) if i < len(params)}
            merged.update(kwargs)
            try:
                if _registry._client is None:
                    raise RuntimeError('mock service client not configured; cannot observe tool results to world-state')
                _registry._client.observe_result(_active_session_id(), name, merged, result)
            except Exception as obs_exc:
                # observe_result failure must NOT crash the real tool call — the
                # tool already returned a result. Log at ERROR: a failed world-state
                # write means policy eval runs on stale state.
                logger.error(
                    'observe_result failed for tool %r (world-state write failed; policy eval will run on stale state): %r',
                    name,
                    obs_exc,
                )
            return result

        return ow

    @functools.wraps(func)
    def w(*args, **kwargs):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        merged = {params[i]: a for i, a in enumerate(args) if i < len(params)}
        merged.update(kwargs)
        response = _resolve_mock_response(name, merged, return_type)
        channel = declared_error_channel(name)
        if channel is None and failure is None:
            return _tool_content(response)
        try:
            return mock_result(response, channel)
        except MockToolError as exc:
            if channel == 'raises':
                raise
            return failure(exc) if failure is not None else response

    for attr in ('name', 'description', 'args_schema'):
        if hasattr(func, attr):
            setattr(w, attr, getattr(func, attr))
    return w


def apply_mocks(module: Any, mock_config: dict[str, str]) -> list[str]:
    """Wrap each named callable on ``module`` in place with :func:`mock_wrap`; return the names patched.

    A bulk convenience for ``simulate.py`` when your tools are module-level
    functions: it rebinds every ``name`` in ``mock_config`` on ``module`` to its
    mock-wrapped version, using the per-tool mode (``'mock'`` / ``'observe'`` /
    ``'real'``). Returns the names actually patched.

    Resolution is module-ATTRIBUTE-only, so a tool built inside a factory closure
    (or exported under a different name) cannot be reached here and is NOT mocked.
    The returned list is the authoritative answer to "what is live?", and it is the
    ONLY per-name report: splitting tools across modules means one ``apply_mocks``
    per module each passed the whole config, so on that shape every call legitimately
    fails to find the other modules' tools and naming them would be false alarm at
    the level an operator reads. A call that reached NOTHING is different — nothing
    it was asked for exists here — and says so.

    Example::

        import plurai, my_agent.tools as tools

        plurai.apply_mocks(tools, {'get_user_profile': 'mock', 'charge_card': 'mock'})
    """
    patched: list[str] = []
    missed: list[str] = []
    for name, mode in mock_config.items():
        target = getattr(module, name, None)
        if callable(target):
            setattr(module, name, mock_wrap(target, mode=mode))
            patched.append(name)
            print(f'[plurai] {name}: {_MODE_LABELS.get(mode, mode)}', file=sys.stderr)
        else:
            missed.append(name)

    # An EMPTY config asked for nothing, so nothing is wrong: an all-live topology calls this
    # with the empty result of its own mode filter, and alarming there asserts a defect that is
    # the approved posture.
    if mock_config and not patched:
        where = getattr(module, '__name__', None) or repr(module)[:80]
        print(
            f'[plurai] apply_mocks patched NOTHING on {where} — every tool call will hit the REAL implementation. '
            f'Asked for: {", ".join(missed)}. Tools built inside a factory/closure are not module attributes; wrap '
            'them with plurai.mock_wrap() or route them through plurai.MockToolNode().',
            file=sys.stderr,
        )
    return patched
