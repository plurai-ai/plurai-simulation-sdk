"""Check a scan fact claiming NO EXTERNAL SERVICE against the env/config the scan already read.

A scan fact can claim NO EXTERNAL SERVICE while a loaded ``.env`` enables tracing. Citation
resolution proves the ``.env`` EXISTS and carries the subject; it does not read the file for a
service that is switched ON. Left unchecked, the claim rides into generation and surfaces only
afterwards, at the price of a scenario regeneration.

WHY THIS IS DECIDABLE AT ALL, when a general "does this code call anything remote" is not. It
never asks that question. It asks a much smaller one with a closed answer set: does a loaded env
file ASSIGN a variable from :data:`SERVICE_INDICATORS` — a fixed table of well-known
service-enabling names — to a value that turns the service on? A credential, an endpoint or a
tracing flag is a DECLARATION by the operator that the service is configured, and that
declaration is what a "no external services" sentence contradicts. The table is closed, so an
unlisted service is silently absent from the answer — this check UNDER-reports by construction,
which is the safe direction for something that must not block a build.

UNDER-REPORTING MUST NOT RENDER AS COVERAGE, and that is the harder half. ``checked=True`` with
an empty ``contradicted`` list would read as "none names a service those files configure",
which a reader takes for "no external service is configured" — a sentence a closed table can
never say. A ``.env`` that carries ``BRAINTRUST_API_KEY`` or an in-house ``ACME_BILLING_URL``
matches no indicator, and the false premise would ride into generation with a clean verdict on
top of it. Three things answer that, in the order they matter:

* every checked :meth:`ExternalServiceAudit.summary` states the DENOMINATOR — how many
  services the table knows — and how many configured-looking variables it could not name, so the
  verdict can never be read as a sweep;
* :func:`load_service_indicators` merges a caller's own table into the built-in one (the
  ``--service-indicators`` flag on both CLI shells), so the blind spot is closable rather than
  only visible;
* :data:`_UNCLASSIFIED_SUFFIX` counts the assignments that LOOK configured and name nothing this
  table knows. ADVISORY, never a contradiction: "ends in ``_API_KEY``" names no service and
  refutes no sentence, and filing it as a finding would spend the false-positive discipline the
  rest of this module is built on. It is the size of the blind spot, not a finding inside it.

Three postures, taken verbatim from the siblings rather than re-decided:

* **ADVISORY, never blocking.** :func:`plurai.scan_contract.inputs.check_complication_coverage`
  is advisory because the link it draws is an authoring judgement. The same holds here twice
  over: a configured service may be genuinely unreachable at runtime, a ``.env`` in the tree may
  be a developer's and not the deployment's, and a PRD sentence may be scoped more narrowly than
  the checker reads it. A finding here is a question the author must answer, not a refusal.
* **NOT RUN is not CLEAN.** No root, no env file, or no such claim declared: each is
  ``checked=False`` with the reason, which the report files as a BLOCKED check and the headline
  counts in ``unexecuted_checks``. The one thing this must never do is render an unread tree as a
  verified absence.
* **Honest about what it could not read.** A file that exists and cannot be decoded is reported
  BY NAME beside the verdict, because "no indicator was found" over a file nobody read is the
  same sentence as "no indicator is there".

Deliberately NOT a scan for network calls in source. That is the "requires execution" class, which
this check leaves out: an ``http.get`` behind a feature flag says nothing about whether the
service is enabled, and reporting every import of a client library would file a finding against
every agent that has one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plurai.scan_contract.citations import read_source_file
from plurai.scan_contract.facts import KIND_ABSENCE, ScanFact

#: Well-known external services, and the environment variables whose ASSIGNMENT declares one
#: configured. Closed on purpose: a table of names is a claim this file can defend, whereas
#: "anything that looks like a key" matches ``APP_SECRET_KEY`` and files a finding about the
#: agent's own session cookie. An unlisted service is simply not answered for — this check
#: under-reports, and every checked summary says how many services it knows and how many
#: configured-looking variables it could not name. The BUILT-IN half of the table; a caller
#: extends it through :func:`load_service_indicators` rather than by editing this literal.
#:
#: Every entry is a variable whose presence with a real value means somebody configured the
#: service. ``DATABASE_URL`` and friends are deliberately ABSENT: a database URL is as likely to
#: be ``localhost`` as a vendor, and the host test that would separate them is exactly the
#: heuristic this check refuses to block or warn on.
SERVICE_INDICATORS: dict[str, tuple[str, ...]] = {
    'LangSmith': ('LANGCHAIN_TRACING_V2', 'LANGCHAIN_API_KEY', 'LANGCHAIN_ENDPOINT', 'LANGSMITH_TRACING', 'LANGSMITH_API_KEY', 'LANGSMITH_ENDPOINT'),
    'Langfuse': ('LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_HOST'),
    'OpenTelemetry': ('OTEL_EXPORTER_OTLP_ENDPOINT', 'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'),
    'Sentry': ('SENTRY_DSN',),
    'Datadog': ('DD_API_KEY', 'DATADOG_API_KEY'),
    'Weights & Biases': ('WANDB_API_KEY',),
    'Helicone': ('HELICONE_API_KEY',),
    'OpenAI': ('OPENAI_API_KEY', 'OPENAI_BASE_URL'),
    'Anthropic': ('ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL'),
    'Google AI': ('GOOGLE_API_KEY', 'GEMINI_API_KEY'),
    'Cohere': ('COHERE_API_KEY',),
    'Mistral': ('MISTRAL_API_KEY',),
    'Azure OpenAI': ('AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY'),
    'AWS': ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'),
    'Pinecone': ('PINECONE_API_KEY',),
    'Weaviate': ('WEAVIATE_URL', 'WEAVIATE_API_KEY'),
    'Qdrant': ('QDRANT_URL', 'QDRANT_API_KEY'),
    'Supabase': ('SUPABASE_URL', 'SUPABASE_KEY'),
    'Stripe': ('STRIPE_API_KEY', 'STRIPE_SECRET_KEY'),
    'Twilio': ('TWILIO_AUTH_TOKEN', 'TWILIO_ACCOUNT_SID'),
    'SendGrid': ('SENDGRID_API_KEY',),
    'Slack': ('SLACK_BOT_TOKEN', 'SLACK_WEBHOOK_URL'),
}

#: Variable-name suffixes that make an assignment LOOK like a configured service without naming
#: one the table knows. Anchored at the end of the name, because that is the whole difference
#: between this probe and the "anything that looks like a key" heuristic the table exists to
#: refuse: ``APP_SECRET_KEY`` ends in ``_KEY`` and is not here, so the agent's own session cookie
#: still files nothing. What IS here is the shape a vendor credential takes —
#: ``BRAINTRUST_API_KEY``, ``ACME_BILLING_WEBHOOK_URL`` — and the only thing it is ever reported
#: as is a COUNT of what this table could not speak for.
_UNCLASSIFIED_SUFFIX = re.compile(r'_(API_KEY|SECRET|TOKEN|DSN|ENDPOINT|BASE_URL|WEBHOOK_URL)$', re.IGNORECASE)

#: Env-file names this check reads. A ``.env`` is the file an operator LOADS, which is what makes
#: an assignment in it a declaration rather than a sample; ``.env.example`` and friends are
#: excluded below for exactly that reason.
_ENV_FILE = re.compile(r'^\.env(\..+)?$|^.+\.env$')

#: ...and the suffixes that mark one as a TEMPLATE rather than a loaded file. A committed
#: ``.env.example`` full of placeholder keys is documentation; treating it as configuration would
#: file a finding against every well-documented repo, which is how a check gets switched off.
_TEMPLATE_SUFFIXES: tuple[str, ...] = ('.example', '.sample', '.template', '.dist', '.tpl')

#: Directories never descended into. Vendor trees carry thousands of files and any ``.env`` in one
#: belongs to a dependency, not to this agent.
_SKIP_DIRS: frozenset[str] = frozenset(
    {'.git', '.hg', 'node_modules', '.venv', 'venv', 'env', '__pycache__', 'site-packages', 'dist', 'build', '.tox', '.mypy_cache'}
)

#: How deep under a root the walk goes, and how many env files it reads. Both are bounds on a
#: check that runs inside a scan, not judgements — a repo with more env files than this has a
#: configuration story no static table was going to summarise anyway.
_MAX_DEPTH = 3
_MAX_ENV_FILES = 12

#: ``KEY=value``, tolerating ``export ``, surrounding whitespace and quotes. Anchored per line, so
#: a key mentioned in a comment or in prose is not an assignment.
_ASSIGNMENT = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$')

#: Values that mean the variable is present and NOT configured. A flag explicitly turned off is
#: evidence FOR the claim, not against it, and reporting it would make the honest declaration the
#: expensive one.
_DISABLED_VALUES: frozenset[str] = frozenset({'', '0', 'false', 'no', 'off', 'none', 'null', 'nil', 'disabled'})

#: Placeholder shapes a template value takes. Matched case-insensitively over the whole value.
_PLACEHOLDER = re.compile(r'^(<.*>|\{\{.*\}\}|x{3,}|\.{3}|your[-_ ].*|change[-_]?me|replace[-_]?me|todo|tbd|sk-\.{3,})$', re.IGNORECASE)

#: Words that make an absence claim be ABOUT external services in general. A claim naming a
#: specific service is matched by the table instead, so this only has to catch the blanket form.
_BLANKET_CLAIM = re.compile(
    r'\b(external (service|api|dependenc|call|system)|third[-\s]?part|outbound (call|request|network)|'
    r'telemetry|tracing|observability|analytics|network access)\w*',
    re.IGNORECASE,
)


def load_service_indicators(extra: Mapping[str, Any] | None = None) -> dict[str, tuple[str, ...]]:
    """The built-in table MERGED with a caller's own — the extension point for the closed table.

    :data:`SERVICE_INDICATORS` is closed, and a closed table with no way to extend it leaves an
    operator two options: accept the blind spot, or patch the installed package. So a caller (the
    ``--service-indicators`` flag on both CLI shells) supplies ``{service: [VARIABLE, ...]}`` and
    it is MERGED, never substituted — an operator adding their in-house billing API must not
    silently drop LangSmith on the way in. A service already in the table gains the caller's
    variables beside its own.

    Variables are upper-cased because :func:`enabled_services` compares upper-cased; a table whose
    entries only matched when the customer happened to type them in caps would under-report
    exactly where the operator had asked it not to.

    Raises ``ValueError`` on a shape the operator can fix. Degrading to "the entries I could read"
    would hand back a table that knows fewer names than the file it came from, and every summary
    would then publish that smaller number as the denominator.
    """
    merged: dict[str, tuple[str, ...]] = dict(SERVICE_INDICATORS)
    if extra is None:
        return merged
    if not isinstance(extra, Mapping):
        raise ValueError(f'service indicators must be a mapping of service name -> variable list, not {type(extra).__name__}')
    for service, declared in extra.items():
        raw = [declared] if isinstance(declared, str) else declared
        if not isinstance(raw, (list, tuple)) or not all(isinstance(v, str) for v in raw):
            raise ValueError(f'service {service!r}: indicators must be a variable name or a list of them, not {type(declared).__name__}')
        names = tuple(v.strip().upper() for v in raw if v.strip())
        if not names:
            raise ValueError(f'service {service!r}: names no variable, so it could never match — remove it, or name the variable that configures it')
        merged[str(service)] = tuple(dict.fromkeys((*merged.get(str(service), ()), *names)))
    return merged


@dataclass(frozen=True)
class EnabledService:
    """One service an env file declares configured, and where it says so."""

    service: str
    variable: str
    #: The env file, relative to the root it was found under.
    source: str

    def as_dict(self) -> dict[str, str]:
        return {'service': self.service, 'variable': self.variable, 'source': self.source}


@dataclass(frozen=True)
class UnclassifiedVariable:
    """One assignment that LOOKS configured and names no service the table knows.

    Never a contradiction and never a finding against a claim — see :data:`_UNCLASSIFIED_SUFFIX`.
    It exists so a checked verdict can state the size of its own blind spot: ``BRAINTRUST_API_KEY``
    in a loaded ``.env`` refutes nothing this module can defend, but a summary that does not
    mention it reads as though the file held nothing at all.
    """

    variable: str
    #: The env file, relative to the root it was found under.
    source: str

    def as_dict(self) -> dict[str, str]:
        return {'variable': self.variable, 'source': self.source}


@dataclass(frozen=True)
class ServiceClaim:
    """One absence claim about external services, decided against the loaded env."""

    fact_id: str
    claim: str
    #: The services this claim is about — a single name for a specific claim, every known service
    #: for a blanket one. Published so a reader can see what was compared, not just the verdict.
    scope: tuple[str, ...]
    contradicted_by: tuple[EnabledService, ...] = ()

    @property
    def contradicted(self) -> bool:
        return bool(self.contradicted_by)

    def as_dict(self) -> dict[str, Any]:
        return {
            'fact_id': self.fact_id,
            'claim': self.claim,
            'scope': list(self.scope),
            'contradicted': self.contradicted,
            'contradicted_by': [e.as_dict() for e in self.contradicted_by],
        }


@dataclass(frozen=True)
class ExternalServiceAudit:
    """Whether the scan's no-external-service claims survive the env the scan can read.

    ``checked`` is separate from an empty ``claims`` list for the reason every check in this
    package keeps the distinction: "no claim was refuted" and "there was no env file to refute it
    against" are the same shape and license very different confidence.
    """

    checked: bool
    blocked_reason: str = ''
    claims: tuple[ServiceClaim, ...] = ()
    #: Env files that were read, relative to their root.
    files_read: tuple[str, ...] = ()
    #: ``path — why`` for each env file that exists and could not be read. A verdict computed
    #: without these is a verdict over part of the configuration, and it has to say so.
    unreadable: tuple[str, ...] = ()
    #: Configured-looking assignments the table could not name. ADVISORY — the size of the blind
    #: spot, never a contradiction. See :class:`UnclassifiedVariable`.
    unclassified: tuple[UnclassifiedVariable, ...] = ()
    #: The DENOMINATOR of this verdict: how many services, and how many variables, the table that
    #: decided it actually knew. Carried on the audit rather than read back off the module global,
    #: because a caller may have extended the table (:func:`load_service_indicators`) and a summary
    #: that names the built-in count would then describe a comparison that did not happen.
    services_known: int = 0
    indicators_known: int = 0

    @property
    def contradicted(self) -> tuple[ServiceClaim, ...]:
        return tuple(c for c in self.claims if c.contradicted)

    def coverage_note(self) -> str:
        """What this verdict could NOT speak for — appended to every checked summary, clean or not.

        ``checked=True`` with an empty contradiction list would read as "none names a service
        those files configure", and a reader takes that for "no external service is configured" —
        which a closed table can never say. A service the table does not list (an observability
        vendor, a hosted model API or an in-house billing API) matches no indicator, so the audit
        returns clean over a false premise.

        So the denominator is stated unconditionally and the unnameable count beside it. Zero
        unclassified variables is still a sentence about a CLOSED table, not a sweep.
        """
        closed = f'the table is CLOSED at {self.services_known} service(s) / {self.indicators_known} indicator variable(s)'
        if not self.unclassified:
            return f'; {closed}, so a vendor it does not list is ABSENT from this verdict rather than cleared by it'
        named = ', '.join(f'{u.variable} in {u.source}' for u in self.unclassified[:3])
        more = f', … (+{len(self.unclassified) - 3} more)' if len(self.unclassified) > 3 else ''
        return (
            f'; {closed}, and {len(self.unclassified)} configured-looking variable(s) in those files are ones it '
            f'CANNOT NAME ({named}{more}) — this verdict says nothing about them'
        )

    def summary(self) -> str:
        if not self.checked:
            return f'external-service claims NOT checked: {self.blocked_reason}'
        unread = f'; {len(self.unreadable)} env file(s) could NOT be read ({", ".join(self.unreadable[:3])})' if self.unreadable else ''
        if not self.contradicted:
            return (
                f'{len(self.claims)} no-external-service claim(s) checked against {len(self.files_read)} loaded env file(s) '
                f'({", ".join(self.files_read)}); none names a service those files configure{self.coverage_note()}{unread}'
            )
        rows = '; '.join(f'{c.fact_id} — {", ".join(f"{e.service} via {e.variable} in {e.source}" for e in c.contradicted_by)}' for c in self.contradicted)
        return (
            f'{len(self.contradicted)} of {len(self.claims)} no-external-service claim(s) are contradicted by the loaded env: '
            f'{rows}{self.coverage_note()}{unread}'
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            'checked': self.checked,
            'blocked_reason': self.blocked_reason,
            'files_read': list(self.files_read),
            'unreadable': list(self.unreadable),
            'services_known': self.services_known,
            'indicators_known': self.indicators_known,
            'unclassified': [u.as_dict() for u in self.unclassified],
            'summary': self.summary(),
            'claims': [c.as_dict() for c in self.claims],
        }


def _is_loaded_env(path: Path) -> bool:
    """A file an operator LOADS, as opposed to a committed template of the same shape."""
    return _ENV_FILE.match(path.name) is not None and not any(path.name.lower().endswith(s) for s in _TEMPLATE_SUFFIXES)


def _env_files(roots: Sequence[Path]) -> list[tuple[Path, str]]:
    """``(path, label)`` for every loaded env file under ``roots``, bounded and deterministic.

    Sorted rather than walk-ordered: the report names the files it read, and a list that reorders
    between two runs over the same tree reads as a changed configuration.
    """
    found: list[tuple[Path, str]] = []
    for root in roots:
        if not root.is_dir():
            continue
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            directory, depth = stack.pop()
            try:
                entries = sorted(directory.iterdir())
            except OSError:
                continue  # an unreadable DIRECTORY names no file to report; the reason belongs to files
            for entry in entries:
                if entry.is_dir():
                    if depth < _MAX_DEPTH and entry.name not in _SKIP_DIRS and not entry.is_symlink():
                        stack.append((entry, depth + 1))
                elif _is_loaded_env(entry):
                    found.append((entry, str(entry.relative_to(root))))
    found.sort(key=lambda pair: pair[1])
    return found[:_MAX_ENV_FILES]


def _configured_assignments(text: str) -> Iterator[str]:
    """Every variable in one env file assigned a value that means CONFIGURED.

    The single line-scanner both readers share. A variable is a declaration when it is assigned a
    value that is neither an explicit off-switch (:data:`_DISABLED_VALUES`) nor a placeholder
    (:data:`_PLACEHOLDER`). Those two exclusions are what keep a committed skeleton —
    ``LANGCHAIN_TRACING_V2=false``, ``OPENAI_API_KEY=your-key`` — from reading as a configured
    service, which is the false positive that would make this check fire on almost every repo.

    ONE scanner, because :func:`enabled_services` and :func:`unclassified_variables` partition the
    same lines: a variable the table names is a service, a variable it does not is a count of what
    the table could not speak for. Two copies of "is this value real?" would eventually disagree,
    and the disagreement would land as an unclassified count over placeholders — noise in the one
    channel whose whole value is that it is quiet unless something is really there.
    """
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            continue
        match = _ASSIGNMENT.match(line)
        if match is None:
            continue
        value = match.group(2).split(' #', 1)[0].strip().strip('\'"').strip()
        if value.lower() in _DISABLED_VALUES or _PLACEHOLDER.match(value):
            continue
        yield match.group(1)


def enabled_services(text: str, label: str, *, indicators: Mapping[str, Sequence[str]] | None = None) -> list[EnabledService]:
    """Every service one env file DECLARES configured, per ``indicators`` (default: the built-in table)."""
    table = SERVICE_INDICATORS if indicators is None else indicators
    by_variable = {variable.upper(): service for service, variables in table.items() for variable in variables}
    out: list[EnabledService] = []
    seen: set[tuple[str, str]] = set()
    for variable in _configured_assignments(text):
        service = by_variable.get(variable.upper())
        if service is not None and (service, variable) not in seen:
            seen.add((service, variable))
            out.append(EnabledService(service=service, variable=variable, source=label))
    return out


def unclassified_variables(text: str, label: str, *, indicators: Mapping[str, Sequence[str]] | None = None) -> list[UnclassifiedVariable]:
    """Configured-looking assignments in one env file that name NO service in ``indicators``.

    The suffix probe, and it runs AFTER the table rather than instead of it — order is the whole
    point. A name the table knows is a service, decided by name; only what is left over reaches
    the suffix rule, so ``BRAINTRUST_URL`` (no listed suffix, no table entry) stays missed and
    ``BRAINTRUST_API_KEY`` is counted.
    """
    table = SERVICE_INDICATORS if indicators is None else indicators
    known = {variable.upper() for variables in table.values() for variable in variables}
    out: list[UnclassifiedVariable] = []
    seen: set[str] = set()
    for variable in _configured_assignments(text):
        if variable.upper() in known or variable in seen or not _UNCLASSIFIED_SUFFIX.search(variable):
            continue
        seen.add(variable)
        out.append(UnclassifiedVariable(variable=variable, source=label))
    return out


def _claim_scope(fact: ScanFact, indicators: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """The services one absence fact is ABOUT, or ``()`` when it is about none of them.

    Two shapes, and the SPECIFIC one is tried first so a claim naming Langfuse is decided against
    Langfuse rather than against every service in the table:

    * a fact whose ``terms`` / ``subject`` name a service in :data:`SERVICE_INDICATORS` — either
      the service's own name or one of its indicator variables;
    * a BLANKET claim, whose text matches :data:`_BLANKET_CLAIM` ("no external services", "no
      telemetry"). Any configured service contradicts it.

    Only :data:`plurai.scan_contract.facts.KIND_ABSENCE` facts are considered: an existence claim
    about a service is the opposite statement, and an env file that lacks the variable does not
    refute it — the service may be configured somewhere this check cannot see.
    """
    if fact.kind != KIND_ABSENCE:
        return ()
    needles = {term.strip().lower() for term in (*fact.terms, fact.subject) if term and term.strip()}
    named = tuple(service for service, variables in indicators.items() if service.lower() in needles or any(v.lower() in needles for v in variables))
    if named:
        return named
    return tuple(indicators) if _BLANKET_CLAIM.search(fact.claim) else ()


def check_external_services(facts: Iterable[ScanFact], roots: Sequence[Path], *, indicators: Mapping[str, Sequence[str]] | None = None) -> ExternalServiceAudit:
    """Decide every no-external-service claim against the env files under ``roots``.

    ``indicators`` defaults to the built-in :data:`SERVICE_INDICATORS`; a caller that extended it
    through :func:`load_service_indicators` passes the merged table here, and the audit publishes
    ITS size as the denominator so the summary describes the comparison that actually ran.

    Never raises — the only I/O is :func:`plurai.scan_contract.citations.read_source_file`, which
    is where that promise is kept. Every way this could not look becomes ``checked=False`` with
    the reason, so a scan that read nothing can never render as a scan that found nothing.
    """
    table = SERVICE_INDICATORS if indicators is None else indicators
    declared = [fact for fact in facts if _claim_scope(fact, table)]
    if not roots:
        return ExternalServiceAudit(
            checked=False,
            blocked_reason='no source root was supplied, so no env or config file could be read — a "no external services" claim is unchecked, not clean',
        )
    if not declared:
        return ExternalServiceAudit(
            checked=False,
            blocked_reason='no typed ABSENCE fact claims that a service is absent (in general, or by name), so there was nothing for the loaded env to '
            'contradict. This is the absence of a CLAIM, not evidence that the agent calls nothing',
        )
    files = _env_files(roots)
    if not files:
        return ExternalServiceAudit(
            checked=False,
            blocked_reason=f'no loaded env file (.env / *.env, templates excluded) was found within {_MAX_DEPTH} level(s) of the scanned root(s), so the '
            'claim was compared against nothing. The configuration may live in a deployment secret store this scan cannot read',
        )
    configured: list[EnabledService] = []
    unclassified: list[UnclassifiedVariable] = []
    read: list[str] = []
    unreadable: list[str] = []
    for path, label in files:
        text, why = read_source_file(path)
        if why:
            unreadable.append(f'{label} — {why}')
            continue
        read.append(label)
        configured.extend(enabled_services(text, label, indicators=table))
        unclassified.extend(unclassified_variables(text, label, indicators=table))
    if not read:
        return ExternalServiceAudit(
            checked=False,
            blocked_reason=f'every env file found could be located and none could be read ({"; ".join(unreadable)}), so nothing was compared',
            unreadable=tuple(unreadable),
        )
    claims = tuple(
        ServiceClaim(
            fact_id=fact.fact_id,
            claim=fact.claim.strip(),
            scope=scope,
            contradicted_by=tuple(e for e in configured if e.service in set(scope)),
        )
        for fact in declared
        if (scope := _claim_scope(fact, table))
    )
    return ExternalServiceAudit(
        checked=True,
        claims=claims,
        files_read=tuple(read),
        unreadable=tuple(unreadable),
        unclassified=tuple(unclassified),
        services_known=len(table),
        indicators_known=sum(len(v) for v in table.values()),
    )
