"""Executable scan contract — scan conclusions compiled into checks that fail before generation.

A scan produces conclusions: which URLs the agent may emit, what the corpus does and does not
contain, which sources were read. Those conclusions land as PROSE and CONFIG, and prose first
meets reality inside the qualification loop — the most expensive stage there is. Every check in
this package exists to move one such conclusion earlier, by making it executable at scan time:

* :mod:`plurai.scan_contract.corpus` — the scanned corpus as typed units + a content hash, so
  "the corpus as of the scan" stops being a dated assertion and becomes a comparison.
* :mod:`plurai.scan_contract.urls` — URL derivation with placeholder TEMPLATES compiled to
  patterns, so a documented ``https://<your-org>.example.com`` licenses the hosts it stands for
  instead of licensing only the literal placeholder.
* :mod:`plurai.scan_contract.facts` — verification of the typed facts THEMSELVES against the
  corpus, upstream of the ledger gate that validates scenarios against those facts.
* :mod:`plurai.scan_contract.citations` — every fact's ``source`` resolved against the scanned
  tree, so a claim citing a line eleven off stops reading as evidence.
* :mod:`plurai.scan_contract.inputs` — the scan's outputs checked against EACH OTHER: whether
  ``agent_context`` is governed, and whether the typed complication pool can deliver the families
  the PRD's directives demand.
* :mod:`plurai.scan_contract.probes` — absence facts turned into adversarial probes, so a
  known gap is MEASURED rather than only suppressed.
* :mod:`plurai.scan_contract.smoke_probes` — the cheap live asks a scan could fire before
  generation. DERIVATION ONLY; execution is gated and unimplemented.
* :mod:`plurai.scan_contract.report` — the single pre-generation findings report, with an
  explicit proceed / fix-first signal.

Nothing here calls a model. Every check is deterministic, and every one that cannot decide says
so in its own record rather than passing quietly.

Deliberately re-exports NOTHING. ``urls`` reaches into the backend for the
matching that generated evals run, which pulls the dialog stack; a package ``__init__`` that
re-exported it would make ``import plurai.scan_contract`` -- and therefore the corpus hash, the
cheapest check here -- pay for the whole agent runtime. Import the module you need.
"""
