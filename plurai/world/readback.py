"""The seed read-back at each scenario's Init: is the world the child reads back the one the scenario seeded?

Asked right after ``child.init``, before any turn, with the vendored comparator (:mod:`plurai.world.seed_landing`,
the backend's own bytes). A read-back that VOIDS the scenario (``SeedReadback.voids_scenario``: a seeded row or field the read
does not hold, or a harness read failure on a scenario that seeds a row) is the verdict the backend's finalize reaches on the same
pair (``undelivered_seed``). So the runner does not drive that scenario: it finalizes it with zero turns, the backend
records the void, and the run goes on with every other scenario and trial. Driving it would only spend the agent's and
the simulated user's turns on a scenario whose every verdict is void.

``landed``, ``no_read_path`` (no read, or a world the harness declares and does not read back) and a read the backend
did not answer (a timeout, a dropped connection, a 5xx) are driven.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, Final

from plurai.world.seed_landing import SeedReadback, seed_readback

#: The local ``ended_early`` of a scenario not driven because its seed did not read back. Never relayed to the backend:
#: the backend reaches the same verdict on its own from the post-seed read (``undelivered_seed``).
SEED_NOT_LANDED_REASON: Final = 'seed_not_landed'


def seed_readback_at_init(scenario_id: str, post_seed_world_state: Any, scenario_seed: Mapping[str, Any] | None) -> SeedReadback:
    """The scenario's read-back verdict, with one stderr line when the read did not hold its seed or failed.

    The caller drives no turn when the result's ``voids_scenario`` is true."""
    readback = seed_readback(post_seed_world_state=post_seed_world_state, scenario_seed=scenario_seed)
    if readback.voids_scenario:
        print(
            f'[world-state] SEED NOT LANDED: {scenario_id} — {readback.line()}. Not driven: the scenario is finalized with no '
            'turn and reported unmeasured (undelivered_seed); the other scenarios run. Fix the harness read-back.',
            file=sys.stderr,
        )
    elif readback.verdict == 'read_failed':
        why = 'The backend did not answer the read' if readback.backend_transport else 'The scenario seeds nothing'
        print(f'[world-state] {scenario_id}: {readback.line()}. {why}, so it is driven and scored with the world distrusted.', file=sys.stderr)
    return readback
