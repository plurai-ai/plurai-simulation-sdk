"""SDK-owned ingestion package.

Ingestion moves customer data to the backend under an env-scoped **anonymization
posture** (:mod:`plurai.ingest.posture`): ``raw`` (the default — data egresses
verbatim under the customer's DPA with Plurai) or ``anonymized`` (the opt-in
**anonymization** pass, :mod:`plurai.ingest.anonymize`, scrubs extractor output
fail-closed before it leaves the machine). The package also owns the object-store
storage client, the disclosure/audit layer (which runs loudly in BOTH postures),
and the ``plurai ingest-*`` CLI.
"""

from __future__ import annotations
