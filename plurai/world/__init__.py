"""The scenario's world on the client: what the harness was asked to seed, and whether it read back as seeded.

* :mod:`plurai.world.seed_landing` — a copy of the backend's module, kept byte-identical and sha-pinned:
  the ONE seed-vs-read comparator. The backend's finalize voids a scenario on its verdict, so the
  runner and the plugin's setup read-back answer from the same bytes.
* :mod:`plurai.world.readback` — SDK-OWNED: the runner's read-back at each scenario's Init, which decides whether the
  scenario is driven (a seed that did not land is not).
* :mod:`plurai.world.seed_rows` — SDK-OWNED: ``plurai.store_record``, a seeded row as a record of the harness's own store.

Re-exports nothing: import the module you need. (``plurai.seed_landing`` is a compatibility shim of the comparator for
plugin builds that import it there.)
"""
