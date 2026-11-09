"""Gate authority on the client — the contract every SDK gate reports through.

A gate is a check that can stop, hold back or annotate work. Every gate reports one record shape,
and the catalogue, never the emitting site, says who may lift it:

* :mod:`plurai.gates.contract` and :mod:`plurai.gates.apply` — copies of the backend's gate modules
  kept byte-identical: the wire types and the pure functions
  (``make_finding``, ``fingerprint``, ``apply_overrules``). A copy, not a twin, so a finding minted
  here carries the fingerprint the backend and the plugin compute for the same evidence.
* :mod:`plurai.gates.catalogue` — SDK-OWNED: every gate id the SDK emits, with its authority,
  whether its advisory finding holds, its evidence keys and the commands whose ``--overrule`` reach it.
* :mod:`plurai.gates.cli` — the ``--overrule FILE`` flag and the loader for its file.
* :mod:`plurai.gates.invocation` — the per-invocation collector, the exit funnel, and the internal
  record ``gate_authority.json``.

Re-exports nothing: import the module you need.
"""
