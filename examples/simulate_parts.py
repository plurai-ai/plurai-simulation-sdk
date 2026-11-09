"""A harness whose turn is a LIST of messages with content parts, not one string.

IN  — reads the OpenAI content-part vocabulary off ``payload['messages']``: a ``text`` part and a
      ``file`` part. ``examples/plurai.config.yaml`` beside this file declares
      ``agent.io: {input: [text, file], may_open: true}``, so the child projects a parts list to
      this harness rather than refusing it.

OUT — the turn is a LIST of messages, not one string:
        1. an AIMessage whose content is prose            (the SPEECH)
        2. an AIMessage whose content is a content-part LIST: a ``text`` part and a ``json``
           part                                            (the STRUCTURED reply)
        3. an AIMessage carrying a tool call               (the ACT)
      Everything a standard OpenAI chat message list turn can carry.

Keyed on the user's words only because there is no LLM in this file.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

#: The content-part kinds this harness can READ off a user message. Same list
#: ``examples/plurai.config.yaml`` declares on ``agent.io.input``; ``image_url`` is in neither.
_READABLE_PARTS = ('text', 'file')

#: The structured payload this agent returns beside its words — a card list, the UI-first shape.
#: Module-level so the probe can compare what arrived against the source.
CARDS: list[dict[str, Any]] = [{'title': 'Statement 2026-08', 'amount': '240.00 USD', 'status': 'due'}]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    agent_actions: list[dict]


def _read_user_turn(last: Any) -> tuple[str, tuple[str, ...]]:
    """``(the user's words, the filenames they attached)`` — from EITHER content shape."""
    if not isinstance(last, HumanMessage):
        return '', ()
    content = last.content
    if isinstance(content, str):
        return content.lower(), ()
    words: list[str] = []
    attached: list[str] = []
    for part in content if isinstance(content, list) else ():
        kind = part.get('type') if isinstance(part, dict) else None
        if kind == 'text':
            words.append(str(part.get('text') or ''))
        elif kind in _READABLE_PARTS:
            attached.append(str((part.get(kind) or {}).get('filename') or kind))
    return ' '.join(words).lower(), tuple(attached)


def llm_node(state: AgentState) -> dict[str, Any]:
    messages = state['messages']
    last = messages[-1] if messages else None
    if not messages:
        return {'messages': [AIMessage(content='Plurbo billing here. How can I help?')], 'agent_actions': []}
    said, attachments = _read_user_turn(last)

    if attachments:
        # THE SHAPE UNDER TEST: three messages in one turn — prose, a structured content-part
        # message, and an act. A standard OpenAI assistant turn can carry all three.
        return {
            'messages': [
                AIMessage(content=f'I read {", ".join(attachments)}. Here is what it says.'),
                AIMessage(
                    content=[
                        {'type': 'text', 'text': 'Your statement summary:'},
                        {'type': 'json', 'json': {'cards': CARDS}},
                    ]
                ),
                AIMessage(
                    content='',
                    tool_calls=[{'name': 'show_statement', 'args': {'period': '2026-08'}, 'id': 'tc_show'}],
                ),
            ],
            'agent_actions': [],
        }

    if 'summary' in said:
        # STRUCTURED ONLY — no prose anywhere in the turn. The card-only shape.
        return {
            'messages': [AIMessage(content=[{'type': 'json', 'json': {'cards': CARDS}}])],
            'agent_actions': [],
        }

    return {'messages': [AIMessage(content='How can I help you?')], 'agent_actions': []}


_graph = StateGraph(AgentState)
_graph.add_node('llm', llm_node)
_graph.set_entry_point('llm')
_graph.set_finish_point('llm')
app = _graph.compile()


def invoke_agent(payload: dict, config: dict | None = None) -> dict:
    """The dialog hook the child drives."""
    return app.invoke(payload, config=config or {})
