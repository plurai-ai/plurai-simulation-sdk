"""``plurai.seed_landing`` is :mod:`plurai.world.seed_landing` under a second import path.

The plugin's setup read-back (``verify_smoke.py``) imports this path. This module is that module object, so
every name (private ones included) resolves to the vendored copy.
"""

import sys

from plurai.world import seed_landing as _seed_landing

sys.modules[__name__] = _seed_landing
