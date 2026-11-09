"""
Example simulate.py
- Inline mocks are DUMB: just WorldState lookup, return entity as-is
- Descriptors are RICH: tell the backend everything about the tool
- plurai_mocks.yaml gets exported for the backend to generate real mocks
"""

import sys
from unittest.mock import MagicMock

# ================================================================
# PHASE 1: STUBS
# ================================================================
sys.modules['production_db'] = MagicMock()

# ================================================================
# PHASE 2: Import / define agent
# ================================================================
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langchain_core.tools import tool


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    user_profile: dict
    permissions: list[str]
    # What the agent DID this turn, declared on the state (and so on the result envelope) rather
    # than as a function call — the second spelling of the ``agent_actions`` channel. Written on
    # EVERY turn, ``[]`` included: ``[]`` says "this agent reports its actions and there were
    # none" while a missing key says "it cannot report them at all", and the two are not the
    # same answer. Not a reducer channel — each turn replaces it.
    agent_actions: list[dict]


@tool
def get_user_profile(user_id: str) -> dict:
    """Fetch user profile from database."""
    raise RuntimeError('Production DB!')


@tool
def process_refund(user_id: str, amount: float) -> dict:
    """Process a refund via payment gateway."""
    raise RuntimeError('Stripe!')


@tool
def get_transactions(user_id: str) -> list:
    """Get user's transaction history."""
    raise RuntimeError('Production DB!')


# The three DIALOG-CONTROL verbs the backend reserves on the ``agent_actions`` channel, and the two
# branches below are what an agent that uses them looks like. They are NOT tools the mock node
# dispatches — the party that ends, transfers or pauses a conversation is the agent itself, so a
# turn carrying one ends on it and the verb rides the turn's action list to the server:
#
#   ``end_conversation``  this agent closed the session (a kiosk / IVR / support flow that
#                         finishes after a transaction). Returned the ordinary OpenAI way, as a
#                         function call, which is how it reaches ``agent_actions`` with no
#                         harness-side plumbing at all.
#   ``handoff``           it transferred the conversation to another party. Shown in the OTHER
#                         spelling — declared on the result envelope (``agent_actions``) by an
#                         agent that already tracks what it did.
#   ``interrupt``         it paused for a human decision. LangGraph's own ``interrupt()`` is the
#                         typed producer, and the SDK child projects it onto the same channel, so
#                         an agent already doing HITL needs no new call.
#
# Keyed on the user's words only because this file has no LLM in it — a real agent decides this
# the way it decides anything else.


#: The content-part kinds this harness can READ off a user message — one list, and the same one
#: `agent.io.input` declares in plurai.config.yaml beside this file. `image_url` is absent from
#: BOTH on purpose: a kind plurai knows how to carry but this agent cannot read is the case the
#: child must refuse rather than drop, and refusing it is only correct while the two agree.
_READABLE_PARTS = ('text', 'file')


def _read_user_turn(last):
    """``(the user's words, the non-text parts they attached)`` — from EITHER content shape.

    ``message.content`` is a plain string on an ordinary turn and an OpenAI content-part list
    when the user attaches something (`agent.io.input` above is what lets the runner send one).
    An agent that only ever does ``content.lower()`` raises ``AttributeError`` on the second
    shape, which is why the declaration and the reader go together.
    """
    if not isinstance(last, HumanMessage):
        return '', ()
    content = last.content
    if isinstance(content, str):
        return content.lower(), ()
    words, attached = [], []
    for part in content if isinstance(content, list) else ():
        kind = part.get('type') if isinstance(part, dict) else None
        if kind == 'text':
            words.append(str(part.get('text') or ''))
        elif kind in _READABLE_PARTS:
            # The FILENAME, not the bytes: what the agent can say back about an attachment is
            # what proves it arrived, and the payload is not something to echo into a transcript.
            attached.append(str((part.get(kind) or {}).get('filename') or kind))
    return ' '.join(words).lower(), tuple(attached)


def llm_node(state):
    messages = state['messages']
    last = messages[-1] if messages else None
    if not messages:
        # THE OPENER. `agent.io.may_open: true`, so the child may call invoke_agent with
        # `{'messages': []}` — nobody has said anything yet and the AGENT speaks first (the IVR /
        # outbound / kiosk shape). An agent that declares may_open and then indexes
        # `payload['messages'][-1]` dies on turn 0 of every scenario; handling it is this branch.
        return {'messages': [AIMessage(content='Hello — Plurbo billing here. How can I help you today?')], 'agent_actions': []}
    said, attachments = _read_user_turn(last)
    if attachments:
        # The CONFIRMATION branch for a declared non-text part: the agent says back WHAT it
        # received, which is the only thing a smoke probe can check from outside the process —
        # a declared `file` that the harness silently ignores looks exactly like one it read.
        return {
            'messages': [AIMessage(content=f'I have the attachment you sent ({", ".join(attachments)}) and I am reading it now.')],
            'agent_actions': [],
        }
    if 'annual fee' in said:
        # The APPROVAL branch. ``interrupt()`` suspends the graph and LangGraph surfaces the
        # payload as ``__interrupt__`` on the result envelope; the child turns that into an
        # ``interrupt`` action carrying the prompt + choices, so the simulated user is asked the
        # question instead of meeting a turn in which the agent said nothing.
        decision = interrupt({'prompt': 'Approve the 240 USD annual fee?', 'choices': ['yes', 'no']})
        return {'messages': [AIMessage(content=f'Noted — you answered {decision}.')], 'agent_actions': []}
    if "that's all" in said or 'that is all' in said:
        # The CLOSE branch: the agent, not the harness and not the budget, ended this dialog.
        # Declared as a FUNCTION CALL — the ordinary OpenAI shape — which is how it reaches
        # ``agent_actions`` with no harness-side plumbing at all.
        return {
            'messages': [
                AIMessage(
                    content='Glad I could help — closing this session.',
                    tool_calls=[{'name': 'end_conversation', 'args': {'reason': 'resolved'}, 'id': 'tc_end'}],
                )
            ],
            'agent_actions': [],
        }
    if 'human' in said or 'a person' in said:
        # The HANDOFF branch, and the OTHER spelling: this one declares what it did on its own
        # RESULT ENVELOPE. An agent that already tracks its actions in state says so here and
        # needs no function-call shape to be heard.
        return {
            'messages': [AIMessage(content='Transferring you to a human colleague now.')],
            'agent_actions': [{'name': 'handoff', 'args': {'target': 'human_queue'}, 'status': 'ok'}],
        }
    if 'refund' in said:
        return {
            'messages': [
                AIMessage(
                    content='Let me check your account.',
                    tool_calls=[{'name': 'get_user_profile', 'args': {'user_id': state.get('user_id', 'user_123')}, 'id': 'tc_001'}],
                )
            ],
            'agent_actions': [],
        }
    return {'messages': [AIMessage(content='How can I help you?')], 'agent_actions': []}


#: The tools the mock node below actually owns. Read off the tool objects rather than re-typed,
#: so a dialog-control verb is "a call this node cannot serve" by construction and not by a
#: second list somebody has to remember to update.
_DISPATCHABLE_TOOLS = {get_user_profile.name, process_refund.name, get_transactions.name}


def should_use_tools(state):
    last = state['messages'][-1] if state['messages'] else None
    calls = getattr(last, 'tool_calls', None) or []
    if any(call.get('name') in _DISPATCHABLE_TOOLS for call in calls):
        return 'tools'
    # No dispatchable call — including a turn whose only call is a dialog-control verb, which
    # ends here. Routing one into the tool node would fail on a tool that does not exist, and
    # the verb's whole point is that the HARNESS acts on it.
    return '__end__'


# ================================================================
# PHASE 3: Replace context-loading nodes
# ================================================================
import plurai


def sim_fetch_context(state):
    ctx = plurai.get_simulation_context()
    state['user_profile'] = ctx.agent_context.get('user_profile', {})
    state['permissions'] = ctx.agent_context.get('permissions', [])
    return state


# ================================================================
# PHASE 4: MOCKS
# ================================================================
# Inline mock functions are DUMB — just return entity from WorldState.
# The descriptor dict is RICH — tells the backend everything it needs
# to generate proper mocks with correct response formats, state
# transitions, business rules, and consistency checks.


@plurai.mock(
    'get_user_profile',
    descriptor={
        'description': 'Fetch user profile from production Postgres users table',
        'original_module': 'our_backend.db.queries',
        'original_function': 'get_user_profile',
        'reads_from': 'Postgres users table joined with accounts table',
        'writes_to': None,
        'stateful': False,
        'args': {
            'user_id': {'type': 'str', 'description': 'UUID of the user'},
        },
        'returns': {
            'type': 'dict',
            'fields': {
                'id': {'type': 'str', 'description': 'user UUID'},
                'name': {'type': 'str', 'description': 'full name'},
                'email': {'type': 'str', 'description': 'email address'},
                'plan': {'type': 'str', 'enum': ['free', 'pro', 'enterprise']},
                'balance': {'type': 'float', 'description': 'account balance in USD'},
                'created_at': {'type': 'str', 'description': 'ISO timestamp'},
            },
        },
        'agent_usage': "Agent accesses result['name'], result['plan'], result['balance'] to determine refund eligibility",
        'depends_on': [],
        'depended_by': ['process_refund', 'check_eligibility'],
        'error_cases': [
            {'condition': 'user not found', 'returns': {'error': 'User not found'}},
        ],
    },
)
def mock_get_user_profile(user_id: str):
    # DUMB: just return the entity as-is from WorldState
    ctx = plurai.get_simulation_context()
    return ctx.world_state.get_entity(user_id) or {'error': 'not found'}


@plurai.mock(
    'process_refund',
    descriptor={
        'description': 'Process a refund through Stripe payment gateway',
        'original_module': 'our_backend.services.payments',
        'original_function': 'process_refund',
        'reads_from': 'Stripe API + Postgres users table',
        'writes_to': 'Stripe (creates refund) + Postgres (updates balance, creates transaction record)',
        'stateful': True,
        'args': {
            'user_id': {'type': 'str', 'description': 'UUID of the user'},
            'amount': {'type': 'float', 'description': 'refund amount in USD'},
        },
        'returns': {
            'type': 'dict',
            'fields': {
                'status': {'type': 'str', 'enum': ['approved', 'rejected', 'pending_review']},
                'refund_id': {'type': 'str', 'description': 'Stripe refund ID'},
                'new_balance': {'type': 'float', 'description': 'balance after refund'},
                'rejection_reason': {'type': 'str', 'description': 'only if rejected'},
            },
        },
        'business_rules': [
            'Premium users: full refund up to $500 without approval',
            'Free users: store credit only, no cash refund',
            'Max 2 refunds per session before requiring review',
            'Balance must be >= refund amount',
        ],
        'state_transitions': {
            'user.balance': 'decremented by amount',
            'user.refund_count': 'incremented by 1',
            'creates': 'transaction record with type=refund',
        },
        'agent_usage': "Agent checks result['status'], shows result['new_balance'] to user",
        'depends_on': ['get_user_profile'],
        'depended_by': ['get_transactions'],
        'error_cases': [
            {'condition': 'user not found', 'returns': {'status': 'rejected', 'rejection_reason': 'User not found'}},
            {'condition': 'insufficient balance', 'returns': {'status': 'rejected', 'rejection_reason': 'Insufficient balance'}},
            {'condition': 'free user', 'returns': {'status': 'rejected', 'rejection_reason': 'Store credit only'}},
        ],
    },
)
def mock_process_refund(user_id: str, amount: float):
    # DUMB: minimal state update, no business rules
    ctx = plurai.get_simulation_context()
    user = ctx.world_state.get_entity(user_id)
    if not user:
        return {'status': 'rejected', 'rejection_reason': 'not found'}
    balance = user.get('balance', 0)
    if balance < amount:
        return {'status': 'rejected', 'rejection_reason': 'insufficient balance'}
    ctx.world_state.apply({user_id: {'balance': balance - amount}})
    return {'status': 'approved', 'new_balance': balance - amount}


@plurai.mock(
    'get_transactions',
    descriptor={
        'description': 'Get transaction history from Postgres transactions table',
        'original_module': 'our_backend.db.queries',
        'original_function': 'get_transactions',
        'reads_from': 'Postgres transactions table filtered by user_id',
        'writes_to': None,
        'stateful': False,
        'args': {
            'user_id': {'type': 'str', 'description': 'UUID of the user'},
        },
        'returns': {
            'type': 'list[dict]',
            'item_fields': {
                'id': {'type': 'str'},
                'amount': {'type': 'float'},
                'type': {'type': 'str', 'enum': ['charge', 'refund', 'subscription']},
                'status': {'type': 'str', 'enum': ['completed', 'pending', 'failed']},
                'created_at': {'type': 'str'},
            },
        },
        'agent_usage': 'Agent iterates results, shows transaction list to user, references specific transactions by ID',
        'depends_on': ['process_refund'],
        'depended_by': [],
    },
)
def mock_get_transactions(user_id: str):
    # DUMB: query WorldState for transaction entities
    ctx = plurai.get_simulation_context()
    return ctx.world_state.query('transaction', user_id=user_id) or []


# ================================================================
# PHASE 5: Rebuild graph and compile
# ================================================================
plurai_tool_node = plurai.MockToolNode(
    tools=[get_user_profile, process_refund, get_transactions],
    mock_config={'get_user_profile': 'mock', 'process_refund': 'mock', 'get_transactions': 'mock'},
)

from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(AgentState)
graph.add_node('fetch_context', sim_fetch_context)
graph.add_node('llm', llm_node)
graph.add_node('tools', plurai_tool_node)
graph.set_entry_point('fetch_context')
graph.add_edge('fetch_context', 'llm')
graph.add_conditional_edges('llm', should_use_tools, {'tools': 'tools', '__end__': '__end__'})
graph.add_edge('tools', 'llm')
app = graph.compile(checkpointer=MemorySaver())

# ================================================================
# PHASE 6: Expose the dialog entry point for the runner
# ================================================================


def invoke_agent(payload, config=None):
    """One dialog turn. The plurai child calls exactly this, once per /user-turn.

    ``payload`` is ``{'messages': [HumanMessage(...)]}`` plus the scenario's ``agent_context``
    on the first turn only; ``config`` carries the stable ``thread_id`` the checkpointer keys
    history by. What may be IN that message, and whether the list may be EMPTY, is what
    ``agent.io`` in the plurai.config.yaml beside this file declares — here ``[text, file]`` and
    ``may_open: true``, which ``_read_user_turn`` and ``llm_node``'s opener branch are the
    reading half of. The RESULT is returned whole — messages, the agent's own state keys, and
    LangGraph's ``__interrupt__`` when the graph suspended — because every one of those is a
    channel the SDK reads: the messages become the transcript, ``agent_actions`` and the
    tool calls on the chain become what the agent DID this turn, and the interrupt becomes the
    question the simulated user is asked.
    """
    return app.invoke(payload, config=config)


# This file IS a customer ``simulate.py``: it defines the agent + its mocks and
# exports ``invoke_agent`` — the one hook the child looks up by name. It is NOT run directly — the
# ``plurai-run-simulation`` console script drives it by ID against a (local or
# hosted) Plurai backend:
#
#     plurai-run-simulation --simulate examples/simulate_smoke.py --scenario <SCENARIO_ID>
#
# The runner spawns ``python -m plurai.child <simulate.py>`` and drives the
# graph over the per-thread orchestration API (LLM user-turns + eval); it
# supplies the world-state / agent-context for each scenario, so the agent reads
# them via ``plurai.get_simulation_context()`` (see ``sim_fetch_context`` above).
