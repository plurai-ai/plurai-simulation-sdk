"""Example simulate.py for an agent whose world is NOT the mock's — the ``reset()`` bench.

Sibling of ``examples/simulate_smoke.py`` and deliberately much smaller: no LangGraph, no LLM,
no checkpointer. It exists to make ONE claim falsifiable — *what a scenario leaves behind is
exactly what the child process does not own* — so it carries three store channels and REPORTS
all three at the top of every turn, before it writes anything:

===========  =============================================  ===================================
channel      where it lives                                 what a second scenario sees
===========  =============================================  ===================================
``mock``     the deployed mock's per-session world           clean — the world is keyed on the
                                                             session, and the backend opens a new
                                                             one per scenario
``module``   a module-level list in THIS process             clean — one child process per
                                                             scenario, so the module dies with
                                                             the scenario. THE NEGATIVE CONTROL.
``file``     a json file the process does not own            DIRTY without ``reset()`` — the
                                                             file outlives the process, the way
                                                             a real CRM, Postgres row or
                                                             server-side conversation store does
===========  =============================================  ===================================

The ``module`` channel is in here on purpose and must stay: an isolation test written on it
alone CANNOT FAIL — the per-scenario child process already isolates it — so it would pass on a
``reset()`` hook that does nothing. It is the control that proves the instrument distinguishes,
and ``file`` is the channel that measures whether ``reset()`` clears what the process does not own.

**The declaration.** An agent with a store like this one owns its own world, so a
``plurai.config.yaml`` beside the harness declares it::

    world_state:
      owner: external          # `mock` (the default) | `agent` | `external`

and a harness declaring anything but ``mock`` must export ``reset()``. The runner calls it in the
child, before every scenario, after the mock session is started and before ``seed_state`` puts the
scenario's preconditions in — so a store the mock does not key starts each scenario empty instead of
carrying its predecessor's writes into the verdicts. A declared-external world with no ``reset()``
is refused at RUN START (``plurai-run-simulation`` exits ``REFUSED``), before a thread is opened, because
every scenario after the first would be scored on the run order.

The config file beside THIS one carries no ``world_state:`` on purpose: a workspace declaration is
read at the harness's directory AND the repo root above it, so ``examples/plurai.config.yaml``
answers for every bench in here — including ``simulate_smoke.py``, which is mock-owned and exports
no ``reset()``, and which the key would therefore stop the runner from starting.

Run it the way any other harness is run::

    plurai-run-simulation --simulate examples/simulate_own_state.py --scenario <SCENARIO_ID>
"""

import json
import os
import pathlib
import tempfile

import plurai

# --- MODULE channel: the NEGATIVE CONTROL -----------------------------------------------------
# A module-level store in the customer's own process. It looks like the leak and is not one: the
# runner spawns one child per scenario, so this list is born and dies inside a single scenario.
_MODULE_STORE: list[str] = []

# --- FILE channel: the store the process does NOT own (the real leak) -------------------------
# Stands in for the external system an own-state agent really writes to — a CRM, a Postgres row,
# a server-side conversation store. Nothing about the scenario boundary reaches it.
_STORE_FILE = pathlib.Path(os.environ.get('OWN_STATE_STORE') or (pathlib.Path(tempfile.gettempdir()) / 'plurai_own_state_store.json'))


def _file_read() -> list[str]:
    return json.loads(_STORE_FILE.read_text(encoding='utf-8')) if _STORE_FILE.exists() else []


def _file_write(rows: list[str]) -> None:
    _STORE_FILE.write_text(json.dumps(rows), encoding='utf-8')


# --- The MOCKED surface: one tool that writes to the mock's world, one that reads it back ------
# Production implementations raise, exactly as in ``simulate_smoke.py`` — reaching them at all
# would mean the interception is off.


def create_record(record_id: str, label: str) -> dict:
    """Create a record in the production CRM."""
    raise RuntimeError('Production CRM!')


def list_records() -> dict:
    """List the records this session created."""
    raise RuntimeError('Production CRM!')


@plurai.mock('create_record')
def mock_create_record(record_id: str, label: str) -> dict:
    ctx = plurai.get_simulation_context()
    ctx.world_state.apply({record_id: {'_entity_type': 'record', 'id': record_id, 'label': label}})
    return {'status': 'created', 'id': record_id}


@plurai.mock('list_records')
def mock_list_records() -> dict:
    ctx = plurai.get_simulation_context()
    return {'session_records': sorted(row.get('id') for row in (ctx.world_state.query('record') or []))}


# The function-level counterpart to ``MockToolNode``, for an agent that calls its tools directly
# rather than through a LangGraph tool node.
create_record = plurai.mock_wrap(create_record)
list_records = plurai.mock_wrap(list_records)


#: What ``seed_state`` was handed for THIS scenario. A real harness would push these into its own
#: store; here they are kept so ``invoke_agent`` can report what it started from.
_SEEDED: dict = {}


def seed_state(initial_agent_context, initial_world_state, resume_seed=None):
    """The hook that SEEDS. It puts this scenario's preconditions in; it clears nothing.

    Returns nothing, which IS the resume-fidelity report for "the seed was applied in full" — a
    reduced restore returns a dict describing what it left out.
    """
    _SEEDED['agent_context'] = dict(initial_agent_context or {})
    _SEEDED['world_state'] = dict(initial_world_state or {})


def reset() -> None:
    """The hook that CLEARS — called by the runner in this child, before every scenario.

    Only the stores this harness owns, and only the ones the mock session does not key. The
    mock's per-session world is not touched here: the backend opens a new session per scenario and
    seeds it, so clearing it from the harness would fight the seeding rather than complete it.
    """
    _MODULE_STORE.clear()
    _file_write([])


def invoke_agent(payload, config=None):
    """One dialog turn: REPORT all three channels, then write to all three."""
    observed = {
        'session_id': plurai.active_session_id(),
        'mock_session_records': list_records().get('session_records'),
        'module_store': list(_MODULE_STORE),
        'file_store': _file_read(),
        'seeded_world_entities': sorted((_SEEDED.get('world_state') or {}).get('entities', {})),
        'pid': os.getpid(),
    }

    # The scenario's own name for this case, from the preconditions the backend seeded — so the rows
    # a later scenario finds say WHICH scenario wrote them.
    tag = str((_SEEDED.get('agent_context') or {}).get('case_id') or 'unlabelled')
    create_record(record_id=f'rec_{tag}', label=tag)
    _MODULE_STORE.append(tag)
    _file_write([*_file_read(), tag])

    return {'messages': [{'role': 'assistant', 'content': 'OBSERVED=' + json.dumps(observed)}]}
