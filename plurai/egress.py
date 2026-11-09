"""Can the agent under test still reach PRODUCTION while it is being simulated?

Every other check in this repo is about whether a measurement is sound. This one is the only
check whose failure mode lands OUTSIDE the test: isolation is BY CONFIG — the harness
redirects a base URL and the agent's tools resolve through the mock — so an endpoint the agent
holds in its own code, rather than reading from the config the harness controls, still points
at the customer's live system. A simulated user then drives a real refund, a real delete, a real
email.

Two channels, deliberately asymmetric, because only one of them is trusted as a measurement:

* **DECLARED** (``egress.production_hosts`` in ``plurai.config.yaml``) — the typed facts the
  scan writes: which hosts this agent talks to in production, and the code location each was
  extracted from. These are PROBED, and each verdict is a gate finding (:meth:`EgressPreflight.findings`).
  A reachable one HOLDS the run only when nothing enforces isolation for this agent
  (:func:`enforcement_layer` is ``none``): the probe is then the only thing between a simulated
  user and a real write, so the run waits for an overrule that says the endpoint is redirected.
  Everywhere else a reachable host ships with its finding.
* **DISCOVERED** (absolute URLs found in the agent's own source, in ANY language it is written
  in) — a supplement, and WARN-ONLY forever. A URL in a docstring, a documentation link, an
  example in a comment: none of those is egress, and none of them may be probed. Their value is
  that they tell an operator what to declare, with a ``file:line`` to look at. This channel
  reports what it READ as well as what it found: a walk that read zero files and a repo with
  nothing to declare print the same empty hint list, and only one of them is a finding.

What the probe actually sends, so it can be audited without reading the code:

* ``connect`` (default) — one TCP handshake per declared ``host:port``, closed immediately. No
  bytes are written, no TLS is negotiated, no credential of any kind is loaded or sent, and no
  HTTP request is made. This is the "minimal reachability check" and nothing more.
* ``dns`` — a name resolution only. Nothing is sent to the host itself.
* ``off`` — no network at all. The declaration is still recorded and reported.

``dns`` and ``off`` are a CONSENT setting (the operator may forbid TCP probes of production hosts),
and neither can ever produce ``breached``; a run that probed in either records an
``egress.probe_mode_reduced`` finding, so the reduced probe is never an unrecorded bypass.

Enforcement is separate from this measurement: :mod:`plurai.egress_guard` blocks the declared hosts
inside a Python agent's own process, armed by the runner from the declaration alone, before and
independently of anything here.

What this does NOT guarantee, in the deployment that matters most: the customer's agent runs on
the customer's machine, in a process this SDK starts but does not sandbox. A host reached by raw
IP, through a proxy, over a protocol we do not probe, or from a subprocess this check never sees
is invisible to it. This detects a DECLARED reachable host; it does not enforce isolation. Real
enforcement is deny-by-default egress in the execution environment, which only covers runs we
host ourselves.
"""

from __future__ import annotations

import errno
import ipaddress
import logging
import os
import re
import socket
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, get_args
from urllib.parse import urlsplit

import yaml

from plurai.egress_guard import GUARD_SCOPE
from plurai.gates.apply import make_finding
from plurai.gates.catalogue import (
    EGRESS_BREACHED,
    EGRESS_BREACHED_UNENFORCED,
    EGRESS_DECLARATION_ENTRY_UNUSABLE,
    EGRESS_DECLARATION_UNUSABLE,
    EGRESS_DENY_LIST_INCOMPLETE,
    EGRESS_LICENSED_ONLY_WITHHELD,
    EGRESS_PROBE_MODE_REDUCED,
    EGRESS_UNPROVABLE,
)
from plurai.gates.contract import GateFinding

# The ONE authority rule, imported rather than re-spelled here. See :data:`_URL_LITERAL`.
from plurai.scan_contract._dialog_urls import _URL_AUTHORITY_NON_ASCII

logger = logging.getLogger(__name__)

#: Workspace config carrying the declaration. Same file (and same discovery order) the
#: anonymization posture uses — see ``ingest/posture.py``; one config, one place to look.
CONFIG_FILENAME = 'plurai.config.yaml'
EGRESS_CONFIG_KEY = 'egress'

ProbeMode = Literal['connect', 'dns', 'off']

#: Statuses a preflight can reach. None of them refuses a run by itself: each becomes a gate finding
#: (:meth:`EgressPreflight.findings`). Three egress findings hold the run: ``egress.breached_unenforced``
#: (``breached`` with no enforcement layer), ``egress.deny_list_incomplete`` (the declaration could not be
#: read where it names production hosts) and ``egress.unguarded_unmeasured`` (the preflight raised with
#: nothing guarding or measuring the agent).
#:
#: ``not_declared`` is NOT a pass and is reported as loudly as a breach: an env that never
#: declared its production hosts has not been checked, and a check that silently reports
#: nothing reads exactly like a check that found nothing.
#:
#: ``licensed_only`` is a verdict BY DECLARATION, and it is named as such. ``isolated`` is a
#: claim about probes, and a tool-less LLM agent has nothing to probe: its only egress is the
#: model provider it must reach, which the operator allow-lists precisely because the run needs
#: it. Such an env could never leave ``unprovable``: an allow-listed host is "allow-listed and
#: therefore NOT probed", so the ``isolated`` clause's "every declared host was probed and
#: refused" cannot hold.
#: It says something WEAKER than ``isolated`` and says so: every host this agent declares is one
#: the operator licensed it to reach, and a scan of the files the declaration says were read
#: found no other host. It is not evidence that a packet cannot leave.
#: ``swept_clean`` and ``not_declared`` are near-opposites. ``not_declared`` means NOBODY LOOKED.
#: ``swept_clean`` means the scan looked — over a named roster of files it recorded — and found
#: no production egress to declare, so the empty ``production_hosts: []`` is a FINDING rather
#: than an omission. The plugin's ``record_egress.py`` writes the ``discovery`` block for exactly this reason
#: ("an absent key cannot say 'we looked'"). Reporting ``not_declared`` over a swept roster would
#: be a safety check printing "never checked" over "checked, nothing found".
#:
#: It is still NOT a probe and still not ``isolated``: a source sweep cannot see a host reached by
#: raw IP, through a proxy, or from a subprocess. It proceeds, like ``not_declared`` does.
#:
#: ``not_determinable`` is the scan saying it COULD NOT DECIDE — neither "nobody looked" nor "looked
#: and found none". It proceeds. ``guarded``: the declared hosts were NOT probed, because the Python
#: agent child blocks them in-process (:mod:`plurai.egress_guard`); weaker than ``isolated``, scoped
#: by :data:`GUARD_SCOPE`, and it proceeds.
EgressStatus = Literal['not_declared', 'swept_clean', 'not_determinable', 'guarded', 'isolated', 'licensed_only', 'breached', 'unprovable']

#: Hosts never treated as production, whatever a config says. The loopback family is the mock
#: itself; a config CANNOT add to this list, and the allow list a config CAN add to is separate,
#: so an operator cannot silence the check by allow-listing the world.
_ALWAYS_LOCAL = frozenset({'localhost', '127.0.0.1', '::1', '0.0.0.0'})  # noqa: S104 - a host to EXCLUDE from probing, never an address to bind

#: Default ports by scheme — a table whose two halves are read SEPARATELY, and never together.
#: The VALUES reach exactly one consumer, :attr:`HostRef.effective_port`'s fallback, which does
#: NOT look a scheme up: a ``HostRef`` carries no scheme, so that fallback always takes
#: ``'https'`` and the ``'http'`` value is unreachable there. The KEYS are the whole of
#: :data:`_WEB_SCHEMES`. Port SELECTION for a declared host is :func:`_entry_ports`.
_SCHEME_PORTS = {'http': 80, 'https': 443}

#: Probed when an entry declares no port of its own. A production API may answer on 80 (or
#: redirect from it) as well as 443, so the default is a SET and an entry that wants specific
#: ports says so — ``port: 8080`` or ``ports: [8080, 8443]``.
#: Two, not more: every port costs one handshake per declared host, spent before a run starts.
DEFAULT_PROBE_PORTS: Final[tuple[int, ...]] = (80, 443)

#: The schemes whose whole port family IS :data:`DEFAULT_PROBE_PORTS`, so an ``http``/``https``
#: declaration with no port is a DECLARATION of those two rather than a guess at them. The SET is
#: derived from :data:`_SCHEME_PORTS` rather than restated, so those two cannot disagree about
#: which schemes are web schemes.
#:
#: The PORTS are a different question, and derivation does not answer it: ``DEFAULT_PROBE_PORTS``
#: is its own literal, so adding ``'ftp': 21`` above would widen this set without widening the
#: probe ports and the sentence above would quietly stop being true. The two must stay the same
#: family.
_WEB_SCHEMES: Final[frozenset[str]] = frozenset(_SCHEME_PORTS)

#: What the plugin's scan concluded about production egress in the agent's source, read from
#: ``egress.discovery.status``. Three states because they are three different facts, and none may
#: collapse into "no declaration": ``declared`` (hosts were found and are listed),
#: ``none_discovered`` (the scan ran and found nothing, so "you never declared" is factually
#: wrong) and ``not_determinable`` (the scan could not decide, which is the check failing and
#: must never read as a clean sweep).
DiscoveryStatus = Literal['declared', 'none_discovered', 'not_determinable']

#: ``egress.agent_runtime`` — where the agent under test executes, as the scan typed it: inside the
#: Python child that arms :mod:`plurai.egress_guard` (``python_child``), or in a process the harness
#: starts (``subprocess``: a server, a CLI, a non-Python runtime), which the guard cannot see.
AgentRuntime = Literal['python_child', 'subprocess']
#: What the declaration says about it: one of :data:`AgentRuntime`, or ``undeclared`` (a config written
#: before the key). Undeclared never earns ``guarded``: it reads ``not_determinable`` and proceeds.
AgentRuntimeState = Literal['python_child', 'subprocess', 'undeclared']
#: Why a ``not_determinable`` verdict could not decide: the scan (``discovery``), or an undeclared agent runtime.
NotDeterminableCause = Literal['scan', 'agent_runtime']

#: The ONLY resolver failures that mean "this name does not exist here". Everything else a
#: resolver can return — ``EAI_AGAIN`` (DNS temporarily down), ``EAI_FAIL`` (resolver broken) —
#: is the PROBE failing, and must never be recorded as evidence of isolation.
_NXDOMAIN_ERRNOS: Final[frozenset[int]] = frozenset({getattr(socket, name) for name in ('EAI_NONAME', 'EAI_NODATA') if hasattr(socket, name)})

#: Connect failures that came from the REMOTE end and are therefore evidence. A local failure
#: (``ENETUNREACH`` after a VPN drop, ``EMFILE`` under concurrency, ``EACCES`` from a sandbox)
#: is the probe failing to run.
#:
#: ``EHOSTUNREACH`` is deliberately NOT here, despite naming the host: it is raised by the LOCAL
#: routing layer — no route to that address, or an ICMP unreachable from an intermediate hop —
#: exactly like the ``ENETUNREACH`` this module already classifies as local one line above. It
#: describes this machine's path to the host, never the host's answer, so counting it as a
#: refusal would let a broken route testify that the agent cannot reach production.
_REMOTE_REFUSAL_ERRNOS: Final[frozenset[int]] = frozenset({errno.ECONNREFUSED, errno.ECONNRESET})

#: One TCP handshake's budget. Short on purpose: a firewalled host hangs, and the answer
#: "did not answer within two seconds" is the answer we want — not a reason to wait.
PROBE_TIMEOUT_SECONDS = 2.0

#: Bounds on the source scan, so a monorepo cannot turn a preflight into a full-tree grep.
_SCAN_MAX_FILES = 400
_SCAN_MAX_BYTES = 512_000

#: Directories the scan never descends into. Without this the budget is spent entirely inside
#: ``.venv`` — dot-directories sort first, and a virtualenv holds thousands of ``.py`` files
#: containing every URL pytest and Pillow ever mention: scanning one spends the file budget on
#: site-packages ahead of the agent's own module, reports junk hosts, and may never read the real
#: hardcoded production host.
#: ``vendor`` (Go), ``target`` (Rust/Maven), ``.next`` (Next.js), ``Pods`` (iOS) and ``.gradle``
#: are skipped for the same reason: a Go agent would otherwise spend its entire 400-file budget
#: inside ``vendor/``.
_SCAN_SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {
        '.git',
        '.venv',
        'venv',
        'env',
        'site-packages',
        'node_modules',
        '__pycache__',
        'build',
        'dist',
        '.tox',
        '.mypy_cache',
        'vendor',
        'target',
        '.next',
        'Pods',
        '.gradle',
    }
)

#: The plugin's workdir, and the OUTPUT directories it writes beneath it — run artifacts, the
#: knowledge corpus, and on a trace-resumed suite the staged bundle's ``resume_pin.json`` with
#: verbatim customer messages (a recorded regression's pin and ``issue_turn.json`` alike). Read as
#: agent source, the customer's own conversations would surface as undeclared hosts and a
#: workspace with a few runs would spend the file budget before the agent's module was reached. Pruned ONLY under that workdir: the names are generic (``runs``,
#: ``artifacts``) and a customer's package by that name at the workspace root is agent source.
#: The harness itself (``plurai_sim/simulate.py``, or one under a named env) stays in the walk —
#: it is customer-authored, and a hardcoded production host there is as real as one in the agent.
PLUGIN_WORKDIR: Final = 'plurai_sim'
_PLUGIN_OUTPUT_DIRS: Final[frozenset[str]] = frozenset(
    {'artifacts', 'rag_corpus', 'traces', 'doc_examples', 'scenario_bundle', 'trace_scenario_bundle', 'grounded_scenario_bundle', 'runs', 'regressions'}
)

#: Files THIS PIPELINE writes. Never scanned for hosts, at any depth: a host in one of them is a
#: host this pipeline COPIED OUT OF the agent, so reporting it back as "undeclared production
#: egress you must declare" hands the operator their own generated output as a finding — a
#: security channel citing itself as evidence.
#:
#: BY FILENAME, not by directory. The output lands in `plurai_sim/` on most trees and beside it on
#: some (e.g. `artifacts/data_profile.md` at the AGENT root), and a
#: directory rule wide enough to catch the second swallows a customer package that merely shares a
#: name — `runs/` next to `plurai_sim/` is agent source and must stay scanned. These NAMES are
#: unmistakably this pipeline's, which is why the list is closed and deliberately excludes generic
#: ones (`evals.md`, `scenarios.md`) a customer might also write.
_GENERATED_WORKSPACE_FILES: Final[frozenset[str]] = frozenset(
    {
        CONFIG_FILENAME,
        'policies_config.yaml',
        'scenario_variables.yaml',
        'scenario_state_schema.yaml',
        'simulation_prd.md',
        'eval_intent.yaml',
        'generation_trace.md',
        'bench_manifest.json',
        'suite.json',
        'suite_diagnostics.json',
        'suite_state.json',
        'scan_contract.json',
        'scan_facts.yaml',
        'data_profile.md',
        'pre_generation_findings.md',
        'agent_findings.md',
        'agent_topology.json',
        'agent_topology.md',
    }
)

#: What the walk READS. Data, and deliberately wide, because the alternative is a check that is
#: inert on most of the agents this SDK targets: TypeScript is a first-class target
#: (``simulate.ts``, the simulation-setup-typescript skill) and Go / Java / Claude-skill agents
#: are too, and a walk that reads ZERO files of an agent prints the same clean line as a repo
#: with nothing to declare — the exact failure this module's own doctrine forbids.
#:
#: Config and prose are in the set on purpose: a hardcoded production base URL lives in a YAML or
#: a README at least as often as in code, and this channel is WARN-ONLY, so a noisy hint costs a
#: line and a missed one costs the declaration. A dotenv file is NOT in the set: see
#: :func:`_is_dotenv`.
SCAN_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        '.py',
        '.ts',
        '.tsx',
        '.js',
        '.jsx',
        '.mjs',
        '.cjs',
        '.go',
        '.java',
        '.kt',
        '.rb',
        '.cs',
        '.php',
        '.rs',
        '.sh',
        '.yaml',
        '.yml',
        '.json',
        '.toml',
        '.md',
    }
)


def _is_dotenv(filename: str) -> bool:
    """A dotenv credential file (``.env``, ``.env.<x>``, ``<x>.env``): read for variable NAMES only.

    Its values never reach a hint or the record; an endpoint set there is named by its variable,
    the declarable form ``{env: VAR, source: <file:line where the agent reads it>}``.
    """
    return filename.startswith('.env.') or filename.endswith('.env')


_DOTENV_ASSIGNMENT = re.compile(r'\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)')


def _names_an_undeclared_host(text: str, known: set[str]) -> bool:
    """The generic walk's two matchers over one dotenv line: a readable URL host, or the authority
    behind credentials :data:`_URL_LITERAL` declined — either one not local and not ``known``."""
    for u in _URL_LITERAL.finditer(text):
        host = _split_declaration(u.group(1))[0].rstrip('.')
        if host and host not in _ALWAYS_LOCAL and host not in known:
            return True
    for c in _CREDENTIALED_URL_LITERAL.finditer(text):
        behind = _split_declaration(c.group(1))[0]
        if not (behind and (behind in _ALWAYS_LOCAL or behind in known)):
            return True
    return False


def _dotenv_endpoint_vars(path: Path, *, cite: str, known: set[str], known_env: set[str]) -> list[str]:
    """`` `VAR` (cite:line) `` for each assignment whose value holds a host nobody declared.

    The value is parsed in memory and never returned: only the variable NAME leaves here. A line
    with such a host but no ``KEY=`` is cited by ``cite:line`` alone, and a file that could not be
    read by ``cite`` alone — never dropped uncounted.
    """
    try:
        if path.stat().st_size > _SCAN_MAX_BYTES:
            return [f'an unread dotenv file ({cite})']
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return [f'an unread dotenv file ({cite})']
    named: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith('#'):
            continue
        m = _DOTENV_ASSIGNMENT.match(line)
        if m is not None and m.group(1) in known_env:
            continue
        if _names_an_undeclared_host(m.group(2) if m is not None else line, known):
            named.append(f'`{m.group(1)}` ({cite}:{lineno})' if m is not None else f'a line with no `KEY=` ({cite}:{lineno})')
    return named


#: Where the full record lives, named in the truncation line so a capped list is not a lost one.
EGRESS_RECORD_NOTE: Final = 'egress_preflight.json'

#: How many undeclared-host hints to PRINT. The record keeps them all; the terminal does not,
#: because a first encounter that is sixty lines of noise is a feature nobody reads twice.
_MAX_HINT_LINES = 10

#: Any absolute URL, not only a web one: a production dependency spelled ``postgres://prod-db…``,
#: ``redis://`` or ``amqp://`` is the host most likely to be missing from a declaration,
#: precisely because nobody thinks of it as "a URL".
#:
#: The AUTHORITY class is not spelled here: it is composed from the dialog matcher's own
#: ``_URL_AUTHORITY_NON_ASCII``, the rule that already governs URL extraction everywhere else in
#: this stack. An ASCII-only class would miss an agent hardcoding ``https://例え.jp/api`` or
#: ``https://пример.рф/``, and a hardcoded IDN host is still a hardcoded production host. Two
#: spellings of one rule drift; one import cannot. What comes with it is the
#: upstream POSITION allowlist — a non-ASCII character is admitted only where a label can open or
#: continue — which is what keeps RTL and CJK prose from being glued onto an ASCII host
#: (``https://knowledge.example.com`` followed by Hebrew still ends at ``.com``). It costs a
#: Unicode-category derivation at import, once per process.
#:
#: The optional ``userinfo@`` group is what keeps ``postgres://svc:pw@db.prod/billing`` from
#: reporting the host as ``svc``; the negative lookahead is what stops it doing so by BACKTRACK.
#: When the host behind the credentials is one the authority class cannot read, the optional
#: group would give itself up, the match would restart at ``svc``, and the preflight would print
#: a USERNAME as a host to declare — a password one field away from the terminal. Refusing to match
#: at all is the honest answer there, and :func:`scan_source_for_hosts` counts what it refused.
_URL_LITERAL = re.compile(
    rf'\b[a-z][a-z0-9+.-]{{1,15}}://(?:[^\s/@"\']*@)?(?![^\s/@"\']*@)((?:[A-Za-z0-9._~-]|{_URL_AUTHORITY_NON_ASCII})+(?::\d+)?)',
    re.IGNORECASE,
)

#: The same literal up to and including its credentials — the one shape whose host :data:`_URL_LITERAL`
#: can decline to read. Counted against that matcher's own matches so a DSN host that went
#: unreported is a NAMED number in the scan's warnings, never a silent absence from the hints.
#:
#: Group 1 is the AUTHORITY behind the credentials, and nothing else: the userinfo run is greedy up
#: to the LAST ``@`` before a path separator, so ``https://u:p@ss@host`` (two ``@``, the shape that
#: makes :data:`_URL_LITERAL` give up) still surrenders ``host``. It exists so the count can ask
#: "is this a host nobody declared?" — a credentialed spelling of an ALREADY-DECLARED host is not a
#: host that went unlooked-for, and counting it as one would block ``licensed_only`` over the
#: declaration's own entry. Only the authority is ever read out; the credentials stay refused.
_CREDENTIALED_URL_LITERAL = re.compile(r'\b[a-z][a-z0-9+.-]{1,15}://[^\s/"\']*@([^\s/@"\']*)', re.IGNORECASE)

#: A URL literal whose authority ends AT A BACKSLASH is not a URL — it is a REGEX that spells one,
#: and the authority class stopped at the escape. ``'^https://docs\.example\.com/help/...'``
#: yields the host ``docs``: a label, not a name, and not a host anything could dial.
#:
#: `closed_class_detectors` in `plurai.config.yaml` is a mapping of exactly such patterns, written
#: BY this pipeline. The truncation is the tell and it is mechanical: a real authority is never
#: followed by a backslash.
#:
#: REFUSED, and COUNTED — like the credentialed literal one constant up. A refusal that leaves no
#: trace is indistinguishable from a host nobody wrote, which is the failure this whole module is
#: written against.
#:
#: IT IS A DISCRIMINATOR, NOT A PROOF, and the counting is what makes that acceptable. The same
#: shape occurs where a REAL authority is followed by a source-level escape rather than a regex
#: one — ``print("see https://api.example.com\n")`` in a Python literal, or a line-continuation
#: after the host — and that host is refused too. It is reported in the warning with its
#: ``file:line``, so an operator can see the name went unlooked-for; the alternative (parsing the
#: enclosing string's own escape conventions from a line of text) is not something this walk can
#: do in any language it reads.
_REGEX_ESCAPE_AFTER_AUTHORITY = '\\'

#: What makes a directory somebody's workspace ROOT. Language-agnostic on purpose — the walk up
#: stops at the first of these, so a Go agent stops at ``go.mod`` and a TS one at
#: ``package.json`` — and every one is a file a customer actually commits.
_WORKSPACE_MARKERS: Final[tuple[str, ...]] = ('.git', 'pyproject.toml', 'package.json', 'go.mod', 'Cargo.toml')

#: How far UP the workspace walk may climb. A bound, not a guess: without one, a harness under a
#: temp directory would resolve its "workspace" to the filesystem root and scan the machine.
_WORKSPACE_MAX_HOPS = 6


#: What is wrong with one part of an ``egress`` block. Entry-level codes name one ``production_hosts``
#: entry (or the ``agent_runtime`` value); block-level codes name a whole key or the file.
DeclarationDefectCode = Literal['missing_source', 'unparseable_host', 'bad_port', 'unknown_agent_runtime', 'not_a_list', 'not_a_mapping', 'yaml_error']
ENTRY_DEFECT_CODES: Final[frozenset[str]] = frozenset({'missing_source', 'unparseable_host', 'bad_port', 'unknown_agent_runtime'})

#: The ``source`` a host entry that declares none is read with. Kept, not dropped: the host still
#: parses, so it still goes into the guard's deny list.
UNDECLARED_SOURCE: Final = '<not declared>'


@dataclass(frozen=True, slots=True)
class DeclarationDefect:
    """One part of the ``egress`` block that could not be read as written.

    Reading FAILS CLOSED FOR ENFORCEMENT: a host that parses is kept (and so blocked by the guard)
    whatever else is wrong with its entry, a malformed ``allow_hosts`` is read as ``[]`` (no consent
    means MORE blocking), and only a host that does not parse at all is dropped. Each defect becomes an
    advisory finding the plugin can repair from; one that leaves the deny list missing declared hosts
    (:func:`_drops_declared_hosts`) HOLDS the run until it is fixed or overruled.
    """

    #: The config file it was read from.
    path: Path
    #: The key inside ``egress`` (``production_hosts``, ``allow_hosts``, ``agent_runtime``) or ``egress`` itself.
    key: str
    #: The entry's index under ``key``, ``None`` for a whole-key defect.
    index: int | None
    code: DeclarationDefectCode
    #: The value as written, as text.
    value: str

    @property
    def entry_level(self) -> bool:
        return self.code in ENTRY_DEFECT_CODES


#: Where isolation is ENFORCED for this agent, independently of the probe:
#:
#: * ``python_guard`` — the agent runs inside the Python child that arms :mod:`plurai.egress_guard`;
#: * ``skill_tools_channel`` — a claude-skill bench on the ``tools`` action channel, whose CLI has Bash
#:   blocked and reaches the world only through mocked tools;
#: * ``none`` — nothing does: a non-Python harness, ``agent_runtime: subprocess``, a mockless run, or a
#:   skill on the ``code`` channel (it runs its own code with Bash, outside any guard).
EnforcementLayer = Literal['python_guard', 'skill_tools_channel', 'none']

#: Why ``licensed_only`` was withheld (:func:`_licensed_only_blockers`).
WithheldCode = Literal['not_operator_allowed', 'scan_read_zero', 'scan_incomplete', 'roster_partial', 'root_derived']


@dataclass(frozen=True, slots=True)
class WithheldReason:
    """One reason the declaration may not have ``licensed_only``: a code, its typed detail, and its sentence."""

    code: WithheldCode
    detail: dict[str, Any]
    text: str


@dataclass(frozen=True, slots=True)
class DiscoveryRecord:
    """``egress.discovery`` as the scan wrote it — see :data:`DiscoveryStatus`.

    ``status`` is kept as the raw string even when it is none of the three known states: a
    producer newer than this reader must degrade to "could not be determined" and print the word
    it actually wrote, never to silence.
    """

    status: str
    detail: str | None = None
    #: The block existed and declared no ``status``. Read as ``not_determinable`` (below), and
    #: carried as its own fact so the preflight can WARN under every status rather than only
    #: under the one that prints the discovery line.
    status_absent: bool = False
    #: ``egress.discovery.scanned`` — the files/directories the SCAN actually read, which the
    #: plugin's ``record_egress.py`` requires and writes. It is the declaration's own account of
    #: WHERE the agent is. Without it the hint walk climbs to the first build-file marker above
    #: the harness, which for an agent directory holding only ``agent.py`` / ``README.md`` /
    #: ``requirements.txt`` inside a monorepo is the MONOREPO: the file budget goes to the parent
    #: repo's ``.env`` and workflows and the agent's own files are never reached.
    scanned: tuple[str, ...] = ()

    @property
    def is_known(self) -> bool:
        return self.status in get_args(DiscoveryStatus)

    @property
    def swept_clean(self) -> bool:
        """The scan RAN, over a NAMED roster, and found nothing.

        The one discovery state that is a finding rather than an absence — and still not a probe,
        so still not evidence of isolation.

        ``scanned`` IS PART OF THE CLAIM, not decoration. It defaults to ``()``, so reading the
        status word alone would make ``DiscoveryRecord(status='none_discovered', scanned=())`` a
        clean sweep that publishes ``scanned: []`` — the very "checked-and-clean versus
        never-checked" collapse the ``swept_clean`` STATUS exists to prevent. A status that means "we looked at these files" cannot be granted to
        a record that names none.

        This is the same condition :func:`_licensed_only_blockers` imposes on the other
        positive-sounding verdict: it refuses ``licensed_only`` when the roster that rooted the
        scan did not resolve.
        """
        return self.status == 'none_discovered' and bool(self.scanned)

    def as_record(self) -> dict[str, Any]:
        return {'status': self.status, 'detail': self.detail, 'status_absent': self.status_absent, 'scanned': list(self.scanned)}


@dataclass(frozen=True, slots=True)
class HostRef:
    """One host the agent may reach, and WHERE that was learned.

    ``source`` is the operator-facing half and is required: a preflight that says "api.example
    .com is reachable" without saying which line of which file put it there hands the operator a
    search, not a finding.

    ``port_declared`` says whether ANYBODY named this port or whether the probe is guessing at
    it, and it is what keeps ``isolated`` honest. A host declared as a bare name — a Postgres, a
    Kafka broker, an SMTP relay — is probed on 80 and 443, refused on both, and a refusal there
    says NOTHING about the 5432 the agent actually talks to. It defaults to ``False`` because
    that is the conservative direction: a ref nobody said anything about is a guess.
    """

    host: str
    port: int | None
    source: str
    port_declared: bool = False

    @property
    def effective_port(self) -> int:
        return self.port if self.port is not None else _SCHEME_PORTS['https']

    def as_record(self) -> dict[str, Any]:
        return {'host': self.host, 'port': self.port, 'source': self.source, 'port_declared': self.port_declared}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One host's verdict. ``reachable is None`` means the probe could not decide — never
    "fine": an undecided probe leaves isolation UNPROVEN, which is its own status.

    ``port_specific`` says WHICH question the verdict answered, and it exists so the port gate
    above does not narrow ``isolated`` out of existence. A TCP refusal answers "is this PORT
    shut?"; NXDOMAIN answers "does this NAME exist here?", and a name that does not resolve is
    unreachable on EVERY port — which is the normal shape of real isolation and the reason
    :func:`run_egress_preflight` keeps NXDOMAIN in the definitive set. It defaults to ``True``,
    the conservative direction: a verdict that does not say otherwise is treated as being about
    the one port it was taken on.
    """

    ref: HostRef
    reachable: bool | None
    detail: str
    port_specific: bool = True

    @property
    def reason_code(self) -> str:
        """Why an undecided probe did not decide, as one stable word: the NAME step never answered
        (a dns- or off-mode probe, an inconclusive resolver) or the CONNECT step never did (a timeout,
        a local failure). The prose stays in :attr:`detail`."""
        if self.reachable is not None:
            return 'decided'
        return 'connect_undecided' if self.port_specific else 'name_undecided'

    def as_record(self) -> dict[str, Any]:
        return {**self.ref.as_record(), 'reachable': self.reachable, 'detail': self.detail, 'port_specific': self.port_specific}


@dataclass(frozen=True, slots=True)
class Declaration:
    """What the workspace's ``plurai.config.yaml`` says about egress.

    A record rather than a tuple: a reader that unpacks positions silently keeps reading the old
    count when a fact is added.
    """

    hosts: tuple[HostRef, ...] = ()
    allowed: tuple[str, ...] = ()
    config_source: Path | None = None
    discovery: DiscoveryRecord | None = None
    #: ``egress.scan_root`` — where the operator says the AGENT is, relative to the config. The
    #: last word on the hint walk's root, ahead of the roster and ahead of the marker walk: an
    #: operator who has already watched the walk climb into a monorepo needs a way to say so that
    #: does not depend on which build files happen to sit where.
    scan_root: str | None = None
    #: ``production_hosts`` entries of the form ``{env: VAR, source: file:line}``: the customer's
    #: backend URL read from an env var. Resolved ONLY inside the agent child, from the environment
    #: the harness built (:mod:`plurai.egress_guard`) — never here, in the runner's environment.
    env_hosts: tuple[tuple[str, str], ...] = ()
    #: ``egress.agent_runtime`` (:data:`AgentRuntimeState`).
    agent_runtime: AgentRuntimeState = 'undeclared'
    #: Every part of the block that could not be read as written (:class:`DeclarationDefect`).
    defects: tuple[DeclarationDefect, ...] = ()

    def deny_hosts(self, extra_allowed: Sequence[str]) -> tuple[str, ...]:
        """What the child guard blocks: the declaration minus ``allow_hosts``, this run's own infrastructure
        (``extra_allowed``: the mock service, the backend) and loopback. Computed from the declaration ALONE,
        so the guard is armed before, and whatever happens to, the probe."""
        return _deny_hosts(self.hosts, (*self.allowed, *extra_allowed))

    def guard_allow(self, extra_allowed: Sequence[str]) -> tuple[str, ...]:
        """What the child guard lets through and records: ``allow_hosts`` plus this run's own infrastructure, minus loopback."""
        return tuple(sorted({h.lower() for h in (*self.allowed, *extra_allowed) if h} - _ALWAYS_LOCAL))

    def holding_findings(self) -> tuple[GateFinding, ...]:
        """The defect findings that hold the run, read off the declaration alone — so a measurement that
        crashes (and with it :meth:`EgressPreflight.findings`) cannot lose them."""
        return tuple(_defect_finding(d) for d in self.defects if _drops_declared_hosts(d))


def _deny_hosts(declared: Sequence[HostRef], allowed: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({h.host for h in declared} - {a.lower() for a in allowed} - _ALWAYS_LOCAL))


@dataclass(frozen=True, slots=True)
class EgressPreflight:
    """The whole verdict, as it rides the run artifacts."""

    status: EgressStatus
    probe_mode: ProbeMode
    #: Where isolation is enforced for this agent (:func:`enforcement_layer`). Decides whether a
    #: ``breached`` verdict holds the run.
    enforcement: EnforcementLayer
    declared: tuple[HostRef, ...] = ()
    allowed: tuple[str, ...] = ()
    results: tuple[ProbeResult, ...] = ()
    #: Absolute URLs found in the agent's source whose host is neither declared nor allowed.
    #: WARN-ONLY and never probed (see the module docstring).
    undeclared: tuple[HostRef, ...] = ()
    config_source: Path | None = None
    #: What the scan concluded, when it said (:data:`DiscoveryStatus`). ``None`` = the config
    #: carries no ``discovery`` block at all, which is a fourth state and not one of the three.
    discovery: DiscoveryRecord | None = None
    #: Where the DISCOVERED-host walk was rooted and how many files it actually read. Reported
    #: because an empty hint list means two opposite things — "this repo declares nothing" and
    #: "this walk read nothing" — and only one of them is a finding.
    scan_root: Path | None = None
    #: WHICH of the three sources produced that root (:data:`ScanRootSource`). Recorded because
    #: the root alone cannot say: a ``licensed_only`` resting on an operator's declaration and one
    #: resting on the derived marker walk — which can climb into an enclosing monorepo — would
    #: otherwise be the same two fields in the artifact.
    scan_root_source: ScanRootSource | None = None
    scan_files_read: int | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    #: Declared hosts the OPERATOR allow-listed (``egress.allow_hosts``): consent to reach them live.
    #: The child guard lets them through and records them; the report lists them, never as a breach.
    live_by_consent: tuple[str, ...] = ()
    #: :attr:`Declaration.env_hosts`, carried so the verdict and the child guard read one declaration.
    env_hosts: tuple[tuple[str, str], ...] = ()
    not_determinable_cause: NotDeterminableCause | None = None
    #: :attr:`Declaration.agent_runtime`, carried for the breach finding's evidence.
    agent_runtime: AgentRuntimeState = 'undeclared'
    #: :attr:`Declaration.defects`: each becomes its own finding.
    defects: tuple[DeclarationDefect, ...] = ()
    #: Why ``licensed_only`` was withheld, when this declaration had its shape (:func:`_licensed_only_blockers`).
    withheld: tuple[WithheldReason, ...] = ()

    @property
    def breaches(self) -> tuple[ProbeResult, ...]:
        return tuple(r for r in self.results if r.reachable is True)

    def as_record(self) -> dict[str, Any]:
        return {
            'status': self.status,
            'probe_mode': self.probe_mode,
            'enforcement': self.enforcement,
            'config_source': str(self.config_source) if self.config_source else None,
            'declared': [h.as_record() for h in self.declared],
            'allowed': list(self.allowed),
            'results': [r.as_record() for r in self.results],
            'undeclared': [h.as_record() for h in self.undeclared],
            'discovery': self.discovery.as_record() if self.discovery else None,
            'scan_root': str(self.scan_root) if self.scan_root else None,
            'scan_root_source': self.scan_root_source,
            'scan_files_read': self.scan_files_read,
            'warnings': list(self.warnings),
            'live_by_consent': list(self.live_by_consent),
            'env_hosts': [{'env': var, 'source': source} for var, source in self.env_hosts],
            'declaration_defects': [{'key': d.key, 'index': d.index, 'code': d.code, 'value': d.value, 'path': str(d.path)} for d in self.defects],
            'licensed_only_withheld': [{'code': r.code, 'detail': r.detail} for r in self.withheld],
            # `deny_hosts` rides `guarded` so a reader can tell a guard holding hosts by name from one
            # that holds none (`deny: []`) and only records: the second is reported as MONITORED.
            **({'guard_scope': GUARD_SCOPE, 'deny_hosts': list(self.deny_hosts)} if self.status == 'guarded' else {}),
            **({'not_determinable_cause': self.not_determinable_cause} if self.not_determinable_cause else {}),
        }

    @property
    def deny_hosts(self) -> tuple[str, ...]:
        """What the child guard blocks — :meth:`Declaration.deny_hosts` over this verdict's own allow list."""
        return _deny_hosts(self.declared, self.allowed)

    @property
    def unproven(self) -> tuple[ProbeResult, ...]:
        """The probes that DID NOT DECIDE. Named, because "isolation was not established" is a
        claim about these and about nothing else."""
        return tuple(r for r in self.results if r.reachable is None)

    @property
    def guessed_ports(self) -> tuple[ProbeResult, ...]:
        """The definitive refusals this verdict had to DISCOUNT, because nobody declared the port
        they came from.

        These are the other way isolation goes unestablished, and they are invisible to
        :attr:`unproven`: every one of them DECIDED — the far end refused — but it refused on a
        port the probe guessed. ``db.prod.example.com`` refused on 80 and 443 says nothing
        whatsoever about the Postgres on 5432 that the agent is holding a connection string for.
        """
        return tuple(r for r in self.results if r.reachable is False and r.port_specific and not r.ref.port_declared)

    def findings(self, *, harness_suffix: str, mock_mode: str) -> tuple[GateFinding, ...]:
        """This verdict as gate findings — the only way an egress verdict reaches the run's exit code.

        ``breached`` is ONE finding per reachable host, and which id it carries is the enforcement
        layer's call, never the caller's: with nothing enforcing isolation (``enforcement == 'none'``)
        it is :data:`~plurai.gates.catalogue.EGRESS_BREACHED_UNENFORCED`, which HOLDS the run until an
        overrule bound to its fingerprint lifts it; on the hermetic ``tools`` channel it is
        :data:`~plurai.gates.catalogue.EGRESS_BREACHED` and ships. Every other verdict ships with its
        finding. Evidence carries only facts that are stable across runs over the same declaration
        (no probe detail, timings or scan counts), so an overrule lapses when the host, its ports,
        its source, the runtime, the enforcement layer or the mock mode change, and only then.
        """
        out: list[GateFinding] = []
        for defect in self.defects:
            out.append(_defect_finding(defect))
        if self.probe_mode != 'connect' and self.results:
            hosts = sorted({r.ref.host for r in self.results})
            out.append(
                make_finding(
                    EGRESS_PROBE_MODE_REDUCED,
                    subject='egress',
                    evidence={'probe_mode': self.probe_mode, 'hosts': hosts},
                    observed={},
                    remedy=f'This run probed in {self.probe_mode!r} mode, which can never show a host reachable. Re-run with '
                    '`--egress-probe connect` (one TCP handshake per declared host:port, no bytes, no TLS, no credentials) to measure it.',
                    draft_ref=None,
                )
            )
        if self.status == 'breached':
            unenforced = self.enforcement == 'none'
            by_host: dict[str, list[ProbeResult]] = {}
            for result in self.breaches:
                by_host.setdefault(result.ref.host, []).append(result)
            for host, rows in sorted(by_host.items()):
                fields: dict[str, Any] = {
                    'subject': host,
                    'evidence': {
                        'host': host,
                        'reachable_ports': sorted({r.ref.effective_port for r in rows}),
                        'source': rows[0].ref.source,
                        'agent_runtime': self.agent_runtime,
                        'enforcement': self.enforcement,
                        'harness_suffix': harness_suffix,
                        'mock_mode': mock_mode,
                    },
                    'observed': {'detail': [r.detail for r in rows]},
                    'draft_ref': None,
                }
                if unenforced:
                    remedy = (
                        f'{host} is reachable from this machine (declared at {rows[0].ref.source}) and nothing enforces isolation for this '
                        'agent. Route it through the mock (the harness redirects by config; an endpoint the agent holds in its own code is '
                        'not redirected), or block it in this execution environment. If that source location already builds the endpoint '
                        'from configuration the harness redirects to the mock, overrule this finding by its fingerprint.'
                    )
                    out.append(make_finding(EGRESS_BREACHED_UNENFORCED, remedy=remedy, **fields))
                else:
                    remedy = (
                        f"{host} is reachable from this machine (declared at {rows[0].ref.source}); this agent's tool channel is "
                        'hermetic, so the run proceeds. Route the host through the mock to make the probe agree.'
                    )
                    out.append(make_finding(EGRESS_BREACHED, remedy=remedy, **fields))
        elif self.status == 'unprovable':
            out.append(
                make_finding(
                    EGRESS_UNPROVABLE,
                    subject='egress',
                    evidence={
                        'unproven': [{'host': r.ref.host, 'port': r.ref.effective_port, 'reason_code': r.reason_code} for r in self.unproven],
                        'guessed_ports': sorted({f'{r.ref.host}:{r.ref.effective_port}' for r in self.guessed_ports}),
                        'env_hosts': sorted(var for var, _ in self.env_hosts),
                        'cause': self._unprovable_cause(),
                    },
                    observed={'detail': [r.detail for r in self.unproven]},
                    remedy=self._unprovable_remedy(),
                    draft_ref=None,
                )
            )
        if self.withheld:
            out.append(
                make_finding(
                    EGRESS_LICENSED_ONLY_WITHHELD,
                    subject='egress',
                    evidence={'reasons': sorted(r.code for r in self.withheld)},
                    observed={'detail': {r.code: r.detail for r in self.withheld}},
                    remedy='The `licensed_only` verdict was withheld: ' + '; '.join(r.text for r in self.withheld) + '.',
                    draft_ref=None,
                )
            )
        return tuple(out)

    def _unprovable_cause(self) -> str:
        """Which of the ways isolation goes unestablished this ``unprovable`` is, as one stable word."""
        if self.unproven:
            return 'probes_undecided'
        if self.guessed_ports:
            return 'guessed_ports'
        if self.env_hosts and not self.results:
            return 'env_hosts'
        return 'nothing_probed'

    def _unprovable_remedy(self) -> str:
        if self.results and self.probe_mode != 'connect':
            return f'This run probed in {self.probe_mode!r} mode, which cannot establish reachability — re-run with `--egress-probe connect`.'
        if self.guessed_ports and not self.unproven:
            return (
                'Declare the port(s) those hosts actually answer on (`port:` / `ports:` under `egress.production_hosts`), or re-run with `--egress-probe dns`.'
            )
        if self.results:
            return 'Fix the path the probe could not measure; a probe that did not decide is not evidence that the agent cannot reach production.'
        if self.env_hosts:
            return 'The env-sourced hosts resolve only inside the agent child; run the agent under the Python guard to block them there.'
        return (
            'Remove the declared production host(s) from `egress.allow_hosts` so they are actually probed (or stop declaring them as '
            "production). This run's own infrastructure allow list — the mock service and the Plurai backend — also feeds it."
        )

    def _unproven_detail_lines(self) -> list[str]:
        """One line per probe that did not decide — or, when there are none, the reason there
        is nothing to list. Zero probes over a non-empty declaration is the allow-list case,
        and printing an empty list under an UNPROVEN heading reads as a clean result."""
        env = [f'       `{var}` (declared at {source}) — env-sourced, resolved only inside a guarded Python agent child' for var, source in self.env_hosts]
        if self.unproven:
            return env + [f'       {r.ref.host}:{r.ref.effective_port}  (declared at {r.ref.source}) — {r.detail}' for r in self.unproven]
        if self.guessed_ports:
            # Every probe DECIDED and the verdict is still unprovable. Without this branch the
            # line below would claim nothing was probed, which is a different (and false) finding.
            return [*env, '       every probe DECIDED — on ports nobody declared. See below.']
        return env or ['       no host was probed at all (every declared production host is allow-listed for this run).']

    def _guessed_port_lines(self) -> list[str]:
        """One line per HOST whose refusal this verdict could not use, with the remedy on it.

        Per host, not per probe row: a host guessed on 80 and 443 produces two rows carrying one
        finding, and printing the same remedy twice teaches an operator to skim it.
        """
        by_host: dict[str, list[ProbeResult]] = {}
        for result in self.guessed_ports:
            by_host.setdefault(result.ref.host, []).append(result)
        return [
            f'       {host}  (declared at {rows[0].ref.source}) — refused on '
            f'{", ".join(str(r.ref.effective_port) for r in rows)}, but nobody declared which port this host answers on, so '
            'a refusal there proves nothing. Add `port:` / `ports:` to it (Postgres 5432, Kafka 9092, SMTP 587, Redis 6379), '
            'or re-run with `--egress-probe dns` — a name that does not resolve is unreachable on EVERY port.'
            for host, rows in by_host.items()
        ]

    def _undeclared_hint_lines(self) -> list[str]:
        """The WARN-ONLY "your own code names this host" hints, capped for the terminal.

        Emitted under every status that reports, not only when nothing was declared: "you
        declared A, but your code also hardcodes B" is the most valuable form of this hint.
        """
        return [f'[egress]   undeclared host in agent source: {h.host}  ({h.source})' for h in self.undeclared[:_MAX_HINT_LINES]]

    def report_lines(self) -> list[str]:
        """What to print about this verdict.

        ``warnings`` are emitted under EVERY status, ``isolated`` included — which is why they
        open the list each status appends to, rather than being a branch of their own. A
        suppression that exists only in a JSON field nobody prints is not a record — and the
        suppression that matters most is the one that turned a probe into no probe at all.

        ``isolated`` DOES report: a run that established isolation and printed nothing would be
        indistinguishable, in the terminal, from an SDK with no isolation check in it.
        A positive claim that leaves no trace is not a claim. Whether a verdict holds the run is the
        gate funnel's to say, from :meth:`findings`; these lines only describe what was found.

        The scan line rides every status too, and says WHERE the hint walk was rooted and how
        much it read. The root is derived (a workspace marker above the harness), so a wrong one
        is a silent inert channel unless the derivation is printed.
        """
        lines = [f'[egress] {w}' for w in self.warnings]
        if self.scan_root is not None:
            # The SOURCE rides the line, not only the path. "rooted at <the enclosing monorepo>"
            # and "rooted at <the agent>" look equally plausible in a terminal; `derived` vs
            # `roster` is what tells an operator whether anybody actually said so.
            source = f' ({self.scan_root_source})' if self.scan_root_source else ''
            lines.append(f'[egress] agent-source scan rooted at {self.scan_root}{source} — {self.scan_files_read} file(s) read for undeclared-host hints.')
        # ONE branch for both, because the prose already splits them: `_nothing_declared_line`
        # prints the NONE DISCOVERED sentence, and the TYPED status the artifact carries is what
        # every downstream renderer reads.
        if self.not_determinable_cause == 'agent_runtime':
            hosts = len({h.host for h in self.declared}) + len(self.env_hosts)
            lines.append(
                f'[egress] NOT DETERMINABLE: plurai.config.yaml does not declare `{EGRESS_CONFIG_KEY}.agent_runtime`, so nothing says the agent '
                f'runs inside the Python process that blocks the {hosts} declared production host(s). They were NOT probed,'
                ' and network isolation could NOT be confirmed for this run; it proceeds. Re-run /simulation-scan-agent '
                f'(Step 3d writes `{EGRESS_CONFIG_KEY}.agent_runtime`).'
            )
            lines += self._undeclared_hint_lines()
        elif self.status in ('not_declared', 'swept_clean', 'not_determinable'):
            lines.append(f'[egress] {self._nothing_declared_line()}')
            lines += self._undeclared_hint_lines()
            if len(self.undeclared) > _MAX_HINT_LINES:
                # The hints ARE the report under this status, so a capped list says it was
                # capped and names the file that kept the rest.
                lines.append(f'[egress]   ...and {len(self.undeclared) - _MAX_HINT_LINES} more (all of them are in {EGRESS_RECORD_NOTE})')
        elif self.status == 'guarded' and not self.deny_hosts:
            # Armed with `deny: []`: the guard blocks no host by name, so GUARDED would claim a block
            # that does not exist. It still records every host the agent process reaches.
            lines.append(
                '[egress] MONITORED: no production host is declared to block by name, so this run blocks no host by name; the Python agent '
                f'process records every host it reaches (listed in the run summary, `egress.runtime.observed_hosts`). Coverage: {GUARD_SCOPE}.'
                + (f' Env-sourced, blocked only if set in the agent child: {", ".join(var for var, _ in self.env_hosts)}.' if self.env_hosts else '')
                + (f' Live by consent (`{EGRESS_CONFIG_KEY}.allow_hosts`), not blocked: {", ".join(self.live_by_consent)}.' if self.live_by_consent else '')
            )
            lines += self._undeclared_hint_lines()
        elif self.status == 'guarded':
            lines.append(
                f'[egress] GUARDED: {len(self.deny_hosts)} declared production host(s) ({", ".join(self.deny_hosts) or "none by name"}) were NOT probed; '
                f'the Python agent process blocks them, and the addresses they resolve to, for the whole run. Coverage: {GUARD_SCOPE}.'
                + (f' Env-sourced, resolved in the agent child: {", ".join(var for var, _ in self.env_hosts)}.' if self.env_hosts else '')
                + (f' Live by consent (`{EGRESS_CONFIG_KEY}.allow_hosts`), not blocked: {", ".join(self.live_by_consent)}.' if self.live_by_consent else '')
            )
            lines += self._undeclared_hint_lines()
        elif self.status == 'isolated':
            lines.append(
                f'[egress] ISOLATED: all {len(self.results)} probe(s) over {len({h.host for h in self.declared})} declared production '
                f'host(s) came back with a definitive answer from the far end (probe mode {self.probe_mode!r}); the agent cannot '
                'reach them from this machine by name or by TCP.'
            )
        elif self.status == 'licensed_only':
            licensed = ', '.join(sorted({h.host for h in self.declared})) or '(none)'
            lines.append(
                f"[egress] LICENSED-ONLY: every declared production host ({licensed}) is in this workspace's own "
                f'`{EGRESS_CONFIG_KEY}.allow_hosts`, so none was probed, and the agent-source scan read {self.scan_files_read} '
                f'file(s) under {self.scan_root} ({self.scan_root_source}) without stopping early and found no other host. This is '
                'a verdict BY DECLARATION, weaker than ISOLATED: it says the agent reaches only what it was licensed to reach, NOT '
                'that a packet cannot leave this machine.'
            )
            # The line says what the verdict rests on and what it does not cover: the allow list
            # has to be the operator's own (:func:`_licensed_only_blockers`).
            lines.append(
                '[egress]   The run PROCEEDS on that licence — isolation was not established, and this verdict is only as good as '
                f'the `{EGRESS_CONFIG_KEY}.allow_hosts` behind it. It says nothing about a host reached by raw IP, through a proxy, '
                'from a subprocess this check never sees, or over a protocol the source scan cannot read.'
            )
        elif self.status == 'breached':
            held = self.enforcement == 'none'
            lines.append(
                f'[egress] BREACHED: {len({r.ref.host for r in self.breaches})} declared production host(s) are reachable from this machine'
                + (', and nothing enforces isolation for this agent.' if held else f'; enforcement here is {self.enforcement!r}, so the run proceeds.')
            )
            lines += [f'[egress]   reachable: {r.ref.host}:{r.ref.effective_port}  (declared at {r.ref.source}) — {r.detail}' for r in self.breaches]
            lines += self._undeclared_hint_lines()
        elif self.status == 'unprovable':
            outcome = 'The run proceeds; isolation is not established.'
            # Distinct HOSTS across BOTH ways isolation goes unestablished — a probe that did not
            # decide, and a refusal on a port nobody declared. Counting ``unproven`` rows alone
            # would print "0 declared production host(s)" under an UNPROVEN heading in the second
            # case, and double the number for one host guessed on two ports in the first. Zero
            # of either means nothing was probed at all, where the honest count is the whole
            # declaration: not one of them was shown unreachable.
            blocked = len({r.ref.host for r in (*self.unproven, *self.guessed_ports)}) or len({h.host for h in self.declared})
            lines.append(f'[egress] UNPROVEN: {blocked} declared production host(s) could not be shown unreachable (probe mode {self.probe_mode!r}). {outcome}')
            lines += [line.replace('       ', '[egress]   ', 1) for line in self._unproven_detail_lines()]
            lines += [line.replace('       ', '[egress]   ', 1) for line in self._guessed_port_lines()]
            lines += self._undeclared_hint_lines()
        return lines

    def _nothing_declared_line(self) -> str:
        """Why nothing was probed — which is four different sentences, not one.

        The first sentence must not be printed unconditionally: an agent whose scan swept its
        source and found NO production egress would be told it had failed to declare something,
        and so would one whose scan COULD NOT DECIDE. The second is the one that matters: an
        undecided scan reported as a missing declaration is a check that did not run, dressed as
        a check that found nothing.
        """
        preamble = 'plurai.config.yaml declares no `egress.production_hosts`, so nothing was probed and this run proves nothing about isolation.'
        tail = ' Isolation here is BY CONFIG — an endpoint the agent holds in its own code is not redirected by the harness.'
        kind = _nothing_declared_kind(self.discovery)
        if self.discovery is None:
            return f'NOT DECLARED: {preamble}{tail}'
        detail = f' ({self.discovery.detail})' if self.discovery.detail else ''
        if kind == 'swept_clean':
            return (
                f"NONE DISCOVERED: the scan looked for production egress in the agent's source and found none{detail}, so there was nothing to "
                f'declare and nothing to probe. That is a scan finding, NOT a probe result: this run still establishes no isolation.{tail}'
            )
        if kind == 'sweep_unproven':
            # `none_discovered` WITHOUT a roster. The scan did decide — so "could not decide"
            # below would be false — but it named no file, so `swept_clean` is not earned either
            # (see :attr:`DiscoveryRecord.swept_clean`). Its own sentence, because the remedy is
            # specific: re-record the block WITH `scanned:`, which the plugin's `record_egress.py` requires.
            return (
                f'SWEEP UNPROVEN: the scan recorded `discovery.status: none_discovered`{detail} but listed NO files under '
                '`discovery.scanned`, so nothing shows what it looked at and this is NOT reported as a clean sweep. '
                f'Re-record the block with `scanned:` (`record_egress.py` requires it).{tail}'
            )
        if kind == 'declaration_incomplete':
            # A contradiction between the two halves of the same block. Reported rather than
            # resolved: whichever half is stale, the operator is the only one who can say.
            return (
                f'DECLARATION INCOMPLETE: the scan recorded `discovery.status: declared`{detail} but the config lists no '
                f'`egress.production_hosts`, so the hosts it found never reached the probe.{tail}'
            )
        # `not_determinable`, and anything a newer producer writes that this reader does not know.
        if self.discovery.status_absent:
            known = ' (the config carries an `egress.discovery` block that declares no `status`, so it decides nothing)'
        else:
            known = '' if self.discovery.is_known else f' (unrecognised discovery status {self.discovery.status!r}, read as not-determinable)'
        return (
            f'NOT DETERMINABLE: the scan could not decide whether this agent has production egress{detail}{known}. The check did not run — '
            f'this is NOT a clean sweep, and nothing here says the agent cannot reach production.{tail}'
        )


def _nothing_declared_kind(discovery: DiscoveryRecord | None) -> str:
    """Which of the five nothing-declared sentences applies. The typed status is read off this same key."""
    if discovery is None:
        return 'not_declared'
    if discovery.swept_clean:
        return 'swept_clean'
    if discovery.status == 'none_discovered':
        return 'sweep_unproven'
    if discovery.status == 'declared':
        return 'declaration_incomplete'
    return 'not_determinable'


def enforcement_layer(agent_module_path: Path, declaration: Declaration, *, mock_mode: str | None) -> EnforcementLayer:
    """Where isolation is enforced for this agent (:data:`EnforcementLayer`), from typed facts only.

    ``python_guard``: a ``.py`` harness, not declared ``egress.agent_runtime: subprocess``, not a
    claude-skill bench (``agent.action_channel`` declared) and not a mockless run — the agent runs
    inside the Python child that arms :mod:`plurai.egress_guard`. ``skill_tools_channel``: a skill on
    the ``tools`` action channel. Everything else is ``none``.
    """
    channel = _declared_action_channel(agent_module_path)
    if channel is not None:
        return 'skill_tools_channel' if channel == 'tools' else 'none'
    if Path(agent_module_path).suffix != '.py' or declaration.agent_runtime == 'subprocess' or mock_mode == 'mockless':
        return 'none'
    return 'python_guard'


def _declared_action_channel(agent_module_path: Path) -> str | None:
    """``agent.action_channel`` from the nearest readable config (any non-empty value), else ``None``."""
    for path in _config_candidates(agent_module_path):
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8')) if path.is_file() else None
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict):
            agent = data.get('agent')
            channel = agent.get('action_channel') if isinstance(agent, dict) else None
            return str(channel) if channel else None
    return None


def _drops_declared_hosts(defect: DeclarationDefect) -> bool:
    """True when ``defect`` leaves the guard's deny list without hosts the block declared: the whole block
    unread (``yaml_error``, ``not_a_mapping``), ``production_hosts`` read as empty, or one of its entries dropped."""
    return defect.key == EGRESS_CONFIG_KEY or (defect.key == 'production_hosts' and defect.code in ('not_a_list', 'unparseable_host'))


def _defect_finding(defect: DeclarationDefect) -> GateFinding:
    """One declaration defect as its finding: one that drops declared hosts (holding), else entry-level or whole-key."""
    where = f'`{EGRESS_CONFIG_KEY}.{defect.key}' + (f'[{defect.index}]`' if defect.index is not None else '`')
    remedy = {
        'missing_source': f'{where} declares no `source`; name the code location the host was extracted from (e.g. `svc/payments.py:42`).',
        'unparseable_host': f'{where} is not a hostname or an IP literal, so it was neither probed nor blocked; fix the entry.',
        'bad_port': f'{where} declares a port that is not 1-65535, so the default ports were probed instead; fix `port:` / `ports:`.',
        'unknown_agent_runtime': f'{where} is {defect.value!r}; it must be one of {", ".join(get_args(AgentRuntime))}. Read as undeclared.',
        'not_a_list': f'{where} must be a list; it was read as empty.',
        'not_a_mapping': f'{where} must be a mapping; nothing in it was read.',
        'yaml_error': f'{defect.path.name} declares `{EGRESS_CONFIG_KEY}` but is not parseable YAML; nothing in it was read.',
    }[defect.code]
    if _drops_declared_hosts(defect):
        return make_finding(
            EGRESS_DENY_LIST_INCOMPLETE,
            subject=f'{defect.key}[{defect.index}]' if defect.index is not None else defect.key,
            evidence={'key': defect.key, 'index': defect.index, 'code': defect.code, 'value': defect.value},
            observed={'config': str(defect.path)},
            remedy=remedy,
            draft_ref=None,
        )
    if defect.entry_level:
        return make_finding(
            EGRESS_DECLARATION_ENTRY_UNUSABLE,
            subject=f'{defect.key}[{defect.index}]' if defect.index is not None else defect.key,
            evidence={'key': defect.key, 'index': defect.index, 'code': defect.code, 'value': defect.value},
            observed={'config': str(defect.path)},
            remedy=remedy,
            draft_ref=None,
        )
    return make_finding(
        EGRESS_DECLARATION_UNUSABLE,
        subject=defect.key,
        evidence={'key': defect.key, 'code': defect.code},
        observed={'config': str(defect.path), 'value': defect.value},
        remedy=remedy,
        draft_ref=None,
    )


def _split_declaration(raw: str) -> tuple[str, int | None, str]:
    """``(host, port, scheme)`` for a bare host, ``host:port``, or a full URL.

    Accepting all three is not laxity: the scan extracts these from source, and what it finds
    there is whichever form the developer wrote.

    MIRRORED by the plugin's ``scripts/record_egress.py::split_host``, which produces the
    declarations this reads. The two splitters must agree on every input; a parity check runs both
    over one shared table of inputs.
    """
    text = raw.strip()
    if '://' in text:
        parsed = urlsplit(text)
        # ``parsed.port`` ONLY. A scheme is not a port declaration: ``https://api.example.com``
        # says which protocol the agent speaks there, not that 443 is the only door — a
        # production host may answer on 80 too. An entry that means one port says so; see
        # :func:`_entry_ports`.
        #
        # The scheme comes out BESIDE the port rather than being discarded, because it is a real
        # fact about the declaration and the only thing that lets ``http``/``https`` name the web
        # port FAMILY. Without it ``isolated`` would be unreachable for every URL-declared host —
        # the default form the scan skill teaches.
        return (parsed.hostname or '').lower(), parsed.port, parsed.scheme.lower()
    if text.startswith('['):
        # ``[2001:db8::1]:8443`` — brackets are the only way a colon-bearing IPv6 literal can
        # carry a port. Kept whole, they turn a real address into a guaranteed NXDOMAIN.
        head, _, tail = text.partition(']')
        port = tail.lstrip(':')
        return head[1:].lower(), int(port) if port.isdigit() else None, ''
    if text.count(':') == 1 and text.rsplit(':', 1)[1].isdigit():
        host, port = text.rsplit(':', 1)
        return host.lower(), int(port), ''
    # More than one colon and no brackets: a bare IPv6 literal, where every colon is part of the
    # address and none of them introduces a port.
    return text.lower(), None, ''


def _harness_dir(agent_module_path: Path) -> Path:
    """The directory the harness lives in. ``agent_module_path`` is normally the ``simulate.py``
    FILE, but a caller may hand over its folder — the suffix is what tells the two apart."""
    return agent_module_path.parent if agent_module_path.suffix else agent_module_path


def _config_candidates(agent_module_path: Path) -> list[Path]:
    """Where the declaration may live, nearest first.

    ``simulate.py`` normally sits beside its config in ``plurai_sim/``; a workspace-root layout
    is the documented alternative (``ingest/posture.py`` reads the same two rungs).
    """
    here = _harness_dir(agent_module_path)
    return [here / CONFIG_FILENAME, here.parent / CONFIG_FILENAME, here.parent / 'plurai_sim' / CONFIG_FILENAME]


def load_declaration(agent_module_path: Path) -> Declaration:
    """The workspace's egress :class:`Declaration`. Never raises on the config's CONTENT.

    ``config_source`` is the nearest config carrying an ``egress`` block, or — when none
    declares one — the nearest readable config regardless. That fallback is load-bearing rather
    than cosmetic: it is what ANCHORS THE WORKSPACE for the source scan when nothing is
    declared. Without it the scan would read ``plurai_sim/``, where the harness lives and the
    agent does not, and the whole DISCOVERED channel would be inert on exactly the runs that
    need its hint.

    A block that exists and is malformed FAILS CLOSED FOR ENFORCEMENT, part by part, and each part
    that could not be read is a :class:`DeclarationDefect` (see there). It never degrades to
    "nothing declared": an unparseable block on the file that carries it is a ``yaml_error``
    defect, not a skipped file.
    """
    anchor: Path | None = None
    for path in _config_candidates(agent_module_path):
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding='utf-8')
        except OSError:
            continue
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            # A defect ONLY when this file is the one carrying the block, which is what the raw
            # token test decides; an unrelated broken YAML on an outer rung is skipped.
            if re.search(rf'^\s*{EGRESS_CONFIG_KEY}\s*:', raw, re.MULTILINE):
                return Declaration(config_source=path, defects=(DeclarationDefect(path, EGRESS_CONFIG_KEY, None, 'yaml_error', type(exc).__name__),))
            logger.warning('egress: %s could not be parsed as YAML and declares no egress block; skipping it.', path)
            continue
        if anchor is None:
            anchor = path
        if not isinstance(data, dict) or EGRESS_CONFIG_KEY not in data:
            continue
        block = data[EGRESS_CONFIG_KEY]
        if block is None:
            return Declaration(config_source=path)
        if not isinstance(block, dict):
            return Declaration(config_source=path, defects=(DeclarationDefect(path, EGRESS_CONFIG_KEY, None, 'not_a_mapping', type(block).__name__),))
        defects: list[DeclarationDefect] = []
        scan_root = block.get('scan_root')
        raw_hosts = block.get('production_hosts')
        if raw_hosts is not None and not isinstance(raw_hosts, list):
            defects.append(DeclarationDefect(path, 'production_hosts', None, 'not_a_list', type(raw_hosts).__name__))
            raw_hosts = []
        hosts = _parse_hosts(raw_hosts or [], path, defects)
        return Declaration(
            hosts=tuple(hosts),
            env_hosts=_parse_env_hosts(raw_hosts, path, defects),
            allowed=tuple(_parse_allow(block.get('allow_hosts'), path, defects)),
            config_source=path,
            discovery=_parse_discovery(block.get('discovery')),
            # Read defensively, like `discovery`: this is a supplementary hint about WHERE to
            # look, written by another repo, and a one-token slip must not brick every run in the
            # workspace. A non-string is ignored and the resolution falls through to the roster.
            scan_root=scan_root.strip() if isinstance(scan_root, str) and scan_root.strip() else None,
            agent_runtime=_parse_agent_runtime(block.get('agent_runtime'), path, defects),
            defects=tuple(defects),
        )
    return Declaration(config_source=anchor)


#: A hostname or an IP literal. Deliberately permissive about labels and deliberately strict
#: about everything else: a declared host that is not a host at all (a trailing path, an
#: unbracketed IPv6, a typo) resolves to NXDOMAIN and would be recorded as POSITIVE evidence of
#: isolation — the one direction this module must never degrade in.
_HOSTNAME = re.compile(r'^(?=.{1,253}$)([A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?)(\.[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?)*$')


def _is_addressable(host: str) -> bool:
    """True when ``host`` is a hostname or an IP literal this can meaningfully probe."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return bool(_HOSTNAME.match(host))
    return True


_ENV_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _is_env_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and 'env' in entry and 'host' not in entry


def _parse_env_hosts(raw: Any, path: Path, defects: list[DeclarationDefect]) -> tuple[tuple[str, str], ...]:
    """``{env: VAR, source: file:line}`` rows. A row with no usable variable name is dropped with a
    defect; one with no ``source`` is kept (the guard still resolves and blocks it) with a defect."""
    rows: list[tuple[str, str]] = []
    for i, entry in enumerate(raw if isinstance(raw, list) else ()):
        if not _is_env_entry(entry):
            continue
        var, source = str(entry.get('env') or '').strip(), str(entry.get('source') or '').strip()
        if not _ENV_NAME.match(var):
            defects.append(DeclarationDefect(path, 'production_hosts', i, 'unparseable_host', var))
            continue
        if not source:
            defects.append(DeclarationDefect(path, 'production_hosts', i, 'missing_source', var))
            source = UNDECLARED_SOURCE
        rows.append((var, source))
    return tuple(rows)


def _parse_hosts(raw: list[Any], path: Path, defects: list[DeclarationDefect]) -> list[HostRef]:
    """Every ``production_hosts`` host entry whose host PARSES, whatever else is wrong with it.

    Fails closed for enforcement: a host that parses goes into the declaration (and so into the
    guard's deny list) even when its ``source`` is missing or its port is unusable; only an entry
    whose host is not a host at all is dropped, because there is nothing to block or probe. Each
    such part is recorded as a :class:`DeclarationDefect`.
    """
    hosts: list[HostRef] = []
    for i, written_entry in enumerate(raw):
        if _is_env_entry(written_entry):
            continue
        # A bare host (`- api.example.com`): the host is the whole entry. Read as `{host: <it>}`, so it is
        # still denied and probed; the missing `source` is recorded below like any other.
        entry = {'host': written_entry} if isinstance(written_entry, str) and written_entry.strip() else written_entry
        if not isinstance(entry, dict) or not str(entry.get('host') or '').strip():
            defects.append(DeclarationDefect(path, 'production_hosts', i, 'unparseable_host', str(entry)[:200]))
            continue
        written = str(entry['host'])
        try:
            host, host_port, host_scheme = _split_declaration(written)
        except ValueError:
            # ``urlsplit(...).port`` RAISES on a non-numeric port: the host cannot be read either.
            defects.append(DeclarationDefect(path, 'production_hosts', i, 'unparseable_host', written))
            continue
        if not _is_addressable(host):
            # Dropped rather than probed: an unresolvable typo comes back "does not resolve from
            # this machine", which this module would then record as POSITIVE evidence of isolation.
            defects.append(DeclarationDefect(path, 'production_hosts', i, 'unparseable_host', written))
            continue
        source = str(entry.get('source') or '').strip()
        if not source:
            defects.append(DeclarationDefect(path, 'production_hosts', i, 'missing_source', written))
            source = UNDECLARED_SOURCE
        # The scheme reaches this parser two ways: written INTO the host value (a full URL — the
        # form the scan skill teaches a developer to paste) or carried BESIDE it in `scheme:`,
        # which is what the producer writes once it has split that URL. The one in the host wins,
        # because it is the thing itself rather than a claim about it. A `scheme:` that is not a
        # string decides nothing, which costs a `port_declared` and therefore an isolation claim
        # — the safe direction for a field written by another repo.
        declared_scheme = entry.get('scheme')
        scheme = host_scheme or (declared_scheme.strip().lower() if isinstance(declared_scheme, str) else '')
        ports, port_declared = _entry_ports(entry, host_port, scheme)
        if ports is None:
            defects.append(DeclarationDefect(path, 'production_hosts', i, 'bad_port', written))
            ports, port_declared = DEFAULT_PROBE_PORTS, False
        # ONE HostRef per port: a host:port pair is what gets probed and reported.
        hosts.extend(HostRef(host=host, port=probe_port, source=source, port_declared=port_declared) for probe_port in ports)
    return hosts


def _entry_ports(entry: dict[str, Any], host_port: int | None, scheme: str) -> tuple[tuple[int, ...] | None, bool]:
    """Which ports THIS entry is probed on, and whether ANYBODY DECLARED them; ``(None, False)`` when the
    entry declares a port that is not one (the caller records a ``bad_port`` defect and guesses).

    ``ports: [...]`` > ``port: N`` > a port written into the host value itself
    (``api.example.com:8443``) > :data:`DEFAULT_PROBE_PORTS`. An out-of-range port is not probed:
    ``getaddrinfo`` rejects it, the probe comes back undecided, and a declaration typo must not be
    able to shape an isolation claim — so it falls back to the guessed default, which can never
    mint ``isolated``.

    The second value is what keeps ``isolated`` honest. A refusal on 80 and 443 says NOTHING
    about a Postgres on 5432, a Kafka on 9092 or an SMTP relay on 587, so a host whose port
    nobody declared is still probed on the defaults — the hint is worth having — but its refusal
    cannot mint the one positive safety claim this module makes.

    An ``http``/``https`` scheme DOES declare a port: the whole web family of one, which is
    exactly :data:`DEFAULT_PROBE_PORTS`. A URL is the form the scan skill teaches, so treating a
    portless ``https://…`` as a guess would make every such preflight ``unprovable``. What a
    scheme must NOT do is collapse to ONE port: a production host may answer on 80 as well as 443.
    """

    def _valid(port: Any) -> bool:
        # ``bool`` is an ``int`` in Python and ``ports: [true]`` is a typo, not port 1.
        return isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535

    declared_list = entry.get('ports')
    if declared_list is not None:
        if not isinstance(declared_list, list) or not declared_list or not all(_valid(p) for p in declared_list):
            return None, False
        return tuple(dict.fromkeys(declared_list)), True
    declared_one = entry.get('port')
    if declared_one is not None:
        return ((declared_one,), True) if _valid(declared_one) else (None, False)
    if host_port is not None:
        return ((host_port,), True) if _valid(host_port) else (None, False)
    # Nobody wrote a port. A web scheme still DECLARES this set (see the docstring); anything
    # else — a bare host, `postgres://`, `redis://`, `amqp://`, a gRPC endpoint — leaves the
    # probe guessing, and a guess may inform but may not prove.
    return DEFAULT_PROBE_PORTS, scheme in _WEB_SCHEMES


def _parse_discovery(raw: Any) -> DiscoveryRecord | None:
    """The scan's ``egress.discovery`` verdict, or ``None`` when the block is absent.

    ``None`` is the ABSENT block and nothing else. A present-but-malformed one — not a mapping,
    or carrying no ``status`` — degrades to ``not_determinable``, on the same rule the rest of
    this module follows: the one degradation that must never happen is a broken statement reading
    as no statement. An unrecognised ``status`` VALUE lands there too, carried through verbatim,
    because a producer newer than this reader is a version skew, not a reason to refuse every run
    in the tree.

    Unlike its sibling parsers this one never RAISES, which is why it needs no ``path`` to name in
    an error — the two branches below say why.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        # NOT a raise either (see the status branch below). This whole channel is documented
        # WARN-ONLY forever, it is written by another repo, and a raise here becomes
        # `unusable_declaration` — an unconditional refusal in EVERY phase. A one-token YAML slip
        # (`discovery: declared`) would hard-stop every run in the workspace, while an
        # unrecognised status VALUE is tolerated as ordinary version skew.
        return DiscoveryRecord(status='not_determinable', detail=f'the `{EGRESS_CONFIG_KEY}.discovery` block is a {type(raw).__name__}, not a mapping')
    # `why` is the key the plugin's record_egress.py writes; `detail` is also accepted.
    detail_value = raw.get('detail') or raw.get('why')
    detail = str(detail_value).strip() if isinstance(detail_value, str) and detail_value.strip() else None
    raw_scanned = raw.get('scanned')
    # Same never-raise posture as the rest of this parser: a malformed roster degrades to "no
    # roster", which puts the scan root back on the marker walk.
    scanned = tuple(str(s).strip() for s in raw_scanned if str(s).strip()) if isinstance(raw_scanned, list) else ()
    status = raw.get('status')
    if not isinstance(status, str) or not status.strip():
        # NOT a raise. This block is written by another repo, and refusing to construct here
        # makes every run in the workspace fail closed on `unusable_declaration` — a hard stop in
        # EVERY phase over a supplementary hint, and stricter than the skew tolerance two lines
        # below, which reads an UNRECOGNISED status as not-determinable rather than fatal. A
        # producer that writes `discovery: {}` must not brick the workspace.
        return DiscoveryRecord(status='not_determinable', detail=detail, status_absent=True, scanned=scanned)
    return DiscoveryRecord(status=status.strip(), detail=detail, scanned=scanned)


def _parse_agent_runtime(raw: Any, path: Path, defects: list[DeclarationDefect]) -> AgentRuntimeState:
    """Absent is ``undeclared``. Anything outside :data:`AgentRuntime` is read as ``undeclared`` with a
    defect — which still arms the guard for a ``.py`` harness and never earns ``guarded``."""
    if raw is None:
        return 'undeclared'
    if raw in get_args(AgentRuntime):
        return raw
    defects.append(DeclarationDefect(path, 'agent_runtime', None, 'unknown_agent_runtime', str(raw)))
    return 'undeclared'


def _parse_allow(raw: Any, path: Path, defects: list[DeclarationDefect]) -> list[str]:
    """``allow_hosts``. A malformed list is read as ``[]``: no consent means MORE blocking, never less."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        defects.append(DeclarationDefect(path, 'allow_hosts', None, 'not_a_list', type(raw).__name__))
        return []
    allowed: set[str] = set()
    for i, entry in enumerate(raw):
        if not str(entry).strip():
            continue
        try:
            allowed.add(_split_declaration(str(entry))[0])
        except ValueError:
            defects.append(DeclarationDefect(path, 'allow_hosts', i, 'unparseable_host', str(entry)))
    return sorted(allowed)


def probe_host(ref: HostRef, mode: ProbeMode, *, timeout: float = PROBE_TIMEOUT_SECONDS) -> ProbeResult:
    """Reachability of ONE host. Never sends bytes, never negotiates TLS, never authenticates.

    ``reachable`` is three-valued on purpose. Under ``dns`` a name that RESOLVES says nothing
    about whether a packet would arrive, so it stays ``None`` (unproven) rather than becoming a
    breach — and a name that does not resolve is real evidence of isolation.

    Every verdict also says whether it is about the PORT or about the NAME
    (:attr:`ProbeResult.port_specific`). Only the connect step's answers are about a port; the
    resolver's are about the name, and are therefore true of every port on it.
    """
    if mode == 'off':
        return ProbeResult(ref, None, 'not probed (probe mode is off)', port_specific=False)
    try:
        socket.getaddrinfo(ref.host, ref.effective_port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        # NXDOMAIN is real evidence: the name does not exist here, so nothing can reach it by
        # name. Every OTHER resolver error is the PROBE failing — a DNS server that is down or
        # a resolver that is broken says nothing whatsoever about the host — and recording those
        # as ``False`` writes the check's own failure into the artifact as proof of isolation.
        if exc.errno in _NXDOMAIN_ERRNOS:
            return ProbeResult(ref, False, f'does not resolve from this machine ({exc.strerror or exc})', port_specific=False)
        return ProbeResult(ref, None, f'name resolution failed INCONCLUSIVELY ({exc.strerror or exc}) — the probe did not run', port_specific=False)
    except OSError as exc:
        return ProbeResult(ref, None, f'name resolution failed INCONCLUSIVELY ({type(exc).__name__}: {exc}) — the probe did not run', port_specific=False)
    if mode == 'dns':
        return ProbeResult(ref, None, 'resolves; reachability not tested (probe mode is dns)', port_specific=False)
    try:
        with socket.create_connection((ref.host, ref.effective_port), timeout=timeout):
            pass
    except TimeoutError:
        # NOT evidence, and the reason this whole module exists in three-valued form. A firewall
        # that DROPs the SYN, a resolver that stalls, a host under load: each produces silence,
        # and silence is indistinguishable from an isolated host. Recorded as ``False`` it would
        # make the STRONGEST safety claim this system makes — "the agent cannot reach production" —
        # out of a measurement that failed. Same rule as the local-``OSError`` branch below.
        return ProbeResult(ref, None, f'no answer within {timeout:g}s — the probe did not decide; this is NOT evidence of isolation')
    except OSError as exc:
        # Same split, at the connect step. A REMOTE refusal is evidence; a LOCAL failure — no
        # route, fd exhaustion, a sandbox denying connect(), no ephemeral ports — is the probe
        # failing to run, and the strongest safety claim this system makes must never be
        # produced by the check breaking.
        if exc.errno in _REMOTE_REFUSAL_ERRNOS:
            return ProbeResult(ref, False, f'refused by the remote end ({errno.errorcode.get(exc.errno or 0, exc.errno)})')
        code = errno.errorcode.get(exc.errno or 0, exc.errno)
        return ProbeResult(ref, None, f'the probe could not run ({type(exc).__name__}: {code}: {exc}) — this is NOT evidence of isolation')
    return ProbeResult(ref, True, 'TCP connection established')


@dataclass(frozen=True, slots=True)
class SourceScan:
    """What the DISCOVERED-host walk actually DID, not only what it found.

    A record rather than a tuple for the reason :class:`Declaration` is one: a caller that
    unpacks positions breaks when a field is added.

    ``files_read`` is the field that matters. Zero found hosts means two opposite things — "this
    repo declares nothing" and "this walk read nothing" — and printing them identically is the
    same silent-pass shape this module refuses everywhere else.

    The counters below are the SAME rule applied one notch down: a walk that read 400 of
    several thousand files, skipped an oversize one, could not open one, could not descend into a whole
    subtree, or declined a credentialed literal did not finish looking either. They are counters,
    not only warning TEXT, so the one caller that must not read an unfinished sweep as evidence
    of absence (``licensed_only``) can see them — see :attr:`incomplete`.
    """

    hosts: tuple[HostRef, ...] = ()
    warnings: tuple[str, ...] = ()
    files_read: int = 0
    #: Candidate files the budget cap left unread.
    truncated: int = 0
    #: Files skipped for size, and files that could not be opened at all.
    skipped_big: int = 0
    skipped_unreadable: int = 0
    #: Directories :func:`os.walk` could not LIST — a whole subtree, of unknown size, that this
    #: walk never entered. ``os.walk`` swallows those by default, so they are counted: otherwise a
    #: permission-blocked tree leaves no warning and a ``SourceScan`` reporting a complete sweep.
    skipped_dirs: int = 0
    #: ``scheme://user:pw@host`` literals whose host :data:`_URL_LITERAL` declined to read AND
    #: which name no host this run already knows about. Their own warning says "their absence
    #: from the hints below is not evidence about them"; this is that sentence in a form a
    #: verdict can act on. A literal naming an ALREADY-DECLARED host is not counted: nothing
    #: about it went unlooked-for, and counting it would block ``licensed_only`` over a host the
    #: declaration itself names.
    unread_credentialed: int = 0
    #: `` `VAR` (file:line) `` for each dotenv variable set to an endpoint the declaration does not
    #: name. A finding, like :attr:`hosts`, not a gap in the walk: it withholds ``licensed_only``.
    env_endpoints: tuple[str, ...] = ()

    @property
    def incomplete(self) -> tuple[str, ...]:
        """The reasons this walk DID NOT FINISH LOOKING, empty when it did.

        Named reasons rather than a bool because the caller has to print them: "the scan did not
        finish" is not actionable, "400 files past the budget" is. A walk that read ZERO files is
        deliberately NOT in here — that is a walk with no opinion at all, a stronger and separately
        reported condition (:attr:`files_read`), and collapsing the two would make one message
        answer two different questions.
        """
        reasons: list[str] = []
        if self.truncated:
            reasons.append(f'{self.truncated} candidate file(s) went unread past the {_SCAN_MAX_FILES}-file budget')
        if self.skipped_big:
            reasons.append(f'{self.skipped_big} file(s) over {_SCAN_MAX_BYTES // 1000}KB were skipped')
        if self.skipped_unreadable:
            reasons.append(f'{self.skipped_unreadable} file(s) could not be read')
        if self.skipped_dirs:
            reasons.append(f'{self.skipped_dirs} directory(ies) could not be listed, so their whole subtree went unread')
        if self.unread_credentialed:
            reasons.append(f'{self.unread_credentialed} credentialed URL literal(s) carry a host this scan could not read')
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class RosterRoot:
    """Where the DECLARATION's roster puts the agent, and HOW MUCH of the roster agreed.

    ``resolved``/``total`` are the point of the record. A root is not a yes/no fact: eleven
    entries agreeing on one directory and ONE entry agreeing on another are the same ``Path``
    return value and completely different evidence (see :func:`_resolve_roster`).
    """

    path: Path | None
    resolved: int = 0
    total: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def partial(self) -> bool:
        """The roster answered, but not all of it did — so the root covers less than the
        declaration says was read, and a scan over it cannot stand in for the whole roster."""
        return self.path is not None and self.resolved < self.total


def _resolve_roster(scanned: Sequence[str], *, anchor: Path) -> RosterRoot:
    """The directory the DECLARATION's roster sits in, plus how completely the roster resolved.

    ``egress.discovery.scanned`` is the list of files the SCAN read to write the declaration —
    the one first-hand account anybody has of where this agent's source actually is. Its common
    parent is therefore a better scan root than any walk: it was observed, not derived.

    Entries are resolved against ``anchor`` (the config's own directory) AND against its PARENT,
    and the anchor that resolves the MOST entries wins. The scan writes the roster after
    ``cd "<path-to-agent-module>"`` (`simulation-scan-agent/SKILL.md`), so a real declaration's
    paths are relative to the AGENT directory while the config it lands in sits one level down in
    ``plurai_sim/``: typically none of the entries resolve against the config directory and all of
    them against its parent.

    QUORUM, not first-past-the-post. Taking the first anchor that resolves ANYTHING is wrong: a
    roster typically names both a ``README.md`` and a ``.env``, either of which plausibly exists
    in ``plurai_sim/`` beside the harness, so ONE incidental collision there would beat every
    other entry resolving at the agent directory and the scan root would become the harness
    folder: precisely the inert scan :func:`_resolve_workspace`'s hop exists to prevent. A tie
    keeps the config directory, which is tried first.

    PARTIAL resolution is reported rather than repaired. A roster where ten of eleven files are
    gone still names a real directory, and dropping it would hand the scan back to the marker
    walk — the worse root. But it narrows: if the one surviving entry is ``assets/docs/…``, the
    root becomes ``assets/docs`` and the agent's own module is never read. So the shortfall is a
    WARNING, and :func:`run_egress_preflight` refuses to mint a positive verdict over it. The
    per-entry ``except OSError`` below is the same shortfall by another route and counts the same
    way — an entry lost to a permission error is an entry that did not agree.

    CONTAINMENT, per ENTRY. ``commonpath`` over an absolute entry, or one that climbs with
    ``../``, can hoist the root arbitrarily — to the enclosing monorepo, or to ``/``. The root
    must stay within ``anchor.parent``, which is exactly the agent directory in the layout above.
    The bound is tested per entry rather than on the COMMON parent, which would be
    all-or-nothing: ONE escaping entry would defeat the quorum the paragraph above protects and
    hand the scan back to the marker walk — "the worse root", reached by the one input a stale
    declaration is most likely to carry. So the escaping entries are DISCARDED and the root is
    recomputed over the survivors, the same treatment an entry whose file is gone gets: it is
    counted as not agreeing, and the shortfall lands in ``partial`` where a verdict can act on
    it.

    A roster that resolves NOWHERE is still dropped rather than guessed at: it degrades to "no
    roster" and lets the marker walk answer, never roots the scan somewhere nothing was read —
    and it SAYS SO. Staying silent would make the run quieter as its declaration got worse: a
    PARTIALLY stale roster prints reasons, an entirely stale one would print none.
    """

    def _resolve(base: Path) -> list[Path]:
        found: list[Path] = []
        for entry in scanned:
            candidate = Path(entry)
            candidate = candidate if candidate.is_absolute() else (base / candidate)
            try:
                if candidate.exists():
                    found.append((candidate if candidate.is_dir() else candidate.parent).resolve())
            except OSError:
                continue
        return found

    total = len(scanned)
    if not total:
        return RosterRoot(path=None)
    # ``max`` keeps the FIRST maximal element, so a tie still prefers the config directory.
    resolved = max((_resolve(anchor), _resolve(anchor.parent)), key=len)
    bound = anchor.parent.resolve()
    contained = [p for p in resolved if p == bound or bound in p.parents]
    warnings: list[str] = []
    if escaped := len(resolved) - len(contained):
        warnings.append(
            f'{escaped} of the {total} entries in the declared `{EGRESS_CONFIG_KEY}.discovery.scanned` roster resolve OUTSIDE '
            f'{bound} — an absolute or `../` entry, which can hoist the agent-source scan root anywhere on this machine. They '
            'were DISCARDED and the root recomputed over the entries that stayed inside the agent directory.'
        )
    if not contained:
        warnings.append(
            f'the declared `{EGRESS_CONFIG_KEY}.discovery.scanned` roster contributed NONE of its {total} entries '
            f'({"all of them resolve outside the agent directory" if resolved else "not one of them exists on this machine"}), so '
            "the SCAN's own first-hand account of where this agent lives was DISCARDED; the agent-source scan root falls back to "
            'the DERIVED workspace walk, which can climb into an enclosing monorepo.'
        )
        return RosterRoot(path=None, total=total, warnings=tuple(warnings))
    # The common parent of everything the scan read. One file's directory when they share one;
    # the enclosing directory when the roster spans `agent.py` and `config/settings.py`.
    common = Path(os.path.commonpath([str(p) for p in contained]))
    root = common if common.is_dir() else common.parent
    if len(contained) < total:
        warnings.append(
            f'the declared `{EGRESS_CONFIG_KEY}.discovery.scanned` roster resolved only {len(contained)} of its {total} entries; the '
            f'agent-source scan is rooted at {root}, which covers LESS than the declaration says was read, so an empty hint list '
            'from it is not evidence about the rest.'
        )
    return RosterRoot(path=root, resolved=len(contained), total=total, warnings=tuple(warnings))


#: Which of the three sources answered :func:`_resolve_scan_root`. Recorded beside the root
#: itself because they are not interchangeable evidence: an operator's ``scan_root`` and the
#: scan's own roster are STATEMENTS about where the agent is, while the marker walk is a
#: derivation that can climb into an enclosing monorepo — and a verdict read out of
#: ``egress_preflight.json`` must be able to tell which one it rested on.
ScanRootSource = Literal['declared', 'roster', 'derived']


@dataclass(frozen=True, slots=True)
class ScanRootResolution:
    """The scan root, WHO said so, and everything the resolution had to report on the way."""

    path: Path
    source: ScanRootSource
    warnings: tuple[str, ...] = ()
    #: The root came from a roster that did not fully resolve — see :attr:`RosterRoot.partial`.
    roster_partial: bool = False


def _resolve_scan_root(declaration: Declaration, *, fallback: Path) -> ScanRootResolution:
    """Where the DISCOVERED-host hint walk is rooted, in declaration-first order.

    THE ordering, and why it is this way round: the marker walk is a DERIVATION and the other two
    are STATEMENTS. Deriving first can put the root at an enclosing monorepo — an agent directory
    holding ``agent.py``, ``README.md`` and ``requirements.txt`` has none of
    ``_WORKSPACE_MARKERS``, so the walk climbs past it — where the file budget goes to the parent
    repo's workflows and ``.env``, reporting "undeclared" hosts that have nothing to do with the
    agent while the agent's own files go unread.

    1. ``egress.scan_root`` — the operator said it outright. Relative to the config.
    2. the roster's parent — the SCAN said it, first-hand, by listing what it read.
    3. the marker walk from ``fallback`` — the derivation, for a declaration that states
       neither.

    A statement beats a better guess: growing the marker list fixes one repo layout, not the
    class.

    Every degradation on that ladder comes back in ``warnings`` and not only in the log. This
    module's own rule is that a suppression which exists in no printed channel is not a record
    (:meth:`EgressPreflight.report_lines`), so an unusable ``scan_root`` is reported in
    ``egress_preflight.json`` and on the terminal, never discarded in silence.
    """
    anchor = declaration.config_source.parent if declaration.config_source is not None else fallback
    unusable: Path | None = None
    if declaration.scan_root:
        declared = Path(declaration.scan_root)
        declared = declared if declared.is_absolute() else (anchor / declared)
        if declared.is_dir():
            return ScanRootResolution(path=declared.resolve(), source='declared')
        unusable = declared
    roster = _resolve_roster(declaration.discovery.scanned if declaration.discovery is not None else (), anchor=anchor)
    warnings = list(roster.warnings)
    if unusable is not None:
        # Named for what ACTUALLY happens next: the ladder's next rung is the marker walk
        # whenever there is no usable roster.
        following = 'the declared roster' if roster.path is not None else 'the DERIVED workspace walk'
        logger.warning('egress: `%s.scan_root` names %s, which is not a directory; falling back to %s.', EGRESS_CONFIG_KEY, unusable, following)
        warnings.insert(
            0,
            f"`{EGRESS_CONFIG_KEY}.scan_root` names {unusable}, which is not a directory, so the operator's own statement about "
            f'where this agent lives was DISCARDED; the agent-source scan root falls back to {following}.',
        )
    if roster.path is not None:
        return ScanRootResolution(path=roster.path, source='roster', warnings=tuple(warnings), roster_partial=roster.partial)
    return ScanRootResolution(path=_resolve_workspace(fallback), source='derived', warnings=tuple(warnings))


def _resolve_workspace(start: Path) -> Path:
    """The directory the DISCOVERED-host scan reads, DERIVED — the last resort of
    :func:`_resolve_scan_root`, used only when the declaration states no root of its own.

    ``simulate.py`` normally lives in ``plurai_sim/`` and the agent does not, so scanning beside
    the harness reads the harness and reports nothing. The hop must not be a literal directory
    NAME, which would make the name the ONLY escape from that layout: a customer whose harness
    lives in ``sim/``, ``tests/plurai/`` or ``tools/simulation/`` — normal in a
    monorepo — would get exactly the inert scan the hop exists to prevent.

    So the name stays as the FAST PATH (it is right on every env the setup skills produce) and a
    bounded walk up to a workspace MARKER covers the rest. It stops at the FIRST marker,
    ``start`` itself included, so a package inside a monorepo does not drag the whole monorepo
    into the walk — and it gives up rather than climbing forever, because a "workspace" that
    resolved to the filesystem root would scan the machine.
    """
    if start.name == 'plurai_sim':
        start = start.parent
    here = start
    for _ in range(_WORKSPACE_MAX_HOPS):
        if any((here / marker).exists() for marker in _WORKSPACE_MARKERS):
            return here
        if here.parent == here:
            break
        here = here.parent
    return start


def scan_source_for_hosts(root: Path, *, known: set[str], known_env: set[str], workdir: Path | None = None) -> SourceScan:
    """Absolute-URL hosts under ``root`` that are neither declared nor allowed, plus what the
    walk read and what it had to SUPPRESS to stay bounded.

    WARN-ONLY and never probed. A URL in a docstring is not egress, and a preflight that
    refused a run over one would be untrustworthy within a week. What this buys is the thing an
    operator needs to fill the declaration in: a host and a ``file:line``.

    ``known_env`` is the variables the declaration already carries as ``{env: VAR}``; a dotenv
    assignment to one of them is not named again.

    ``root`` is the WORKSPACE, resolved by :func:`_resolve_workspace` from the config that won —
    not ``simulate.py``'s own directory. The harness normally lives in ``plurai_sim/`` and the
    agent does not, so scanning beside the harness would read the harness and report nothing.

    Bounded by pruning vendored trees DURING the walk rather than after sorting them: the cap
    is a cap on files READ, and a ``sorted(rglob())`` materializes the whole tree before it
    applies. Every suppression — the cap, an oversize file, an unreadable one, a subtree that
    could not be listed, a walk that read nothing at all — is returned so the caller can say so;
    a hint producer that quietly truncates tells an operator there is nothing to declare on
    exactly the repos where there is.

    ``onerror`` is what makes that claim true of DIRECTORIES too. ``os.walk`` swallows a failing
    ``scandir`` by default, so without it a permission-blocked subtree — of unknown size, possibly
    holding the agent's whole source — would drop with no count and no warning, and the
    ``SourceScan`` would report a sweep that finished. That is the same silent-truncation shape
    the paragraph above refuses one level down, and ``licensed_only`` reads it.
    """
    suffixes = tuple(SCAN_SUFFIXES)
    found: dict[str, HostRef] = {}
    read = skipped_big = skipped_unreadable = unscanned = 0
    unreadable: list[str] = []
    blocked: list[str] = []
    regex_literals: list[str] = []
    duplicate_workdirs: list[str] = []
    generated_skipped: list[str] = []
    workdir_dotfiles_skipped: list[str] = []
    dotenv_files: list[str] = []
    env_endpoints: list[str] = []
    live_workdir = workdir.resolve() if workdir is not None else None
    # The live WORKSPACE: the config's directory, or `plurai_sim/` when the config sits in the
    # manage-envs layout `plurai_sim/envs/<name>/`.
    live_workspace = live_workdir
    if live_workdir is not None and live_workdir.parent.name == 'envs' and live_workdir.parent.parent.name == PLUGIN_WORKDIR:
        live_workspace = live_workdir.parent.parent

    def _unlistable(error: OSError) -> None:
        blocked.append(str(getattr(error, 'filename', None) or root))

    def _is_duplicate_workdir(here: Path) -> bool:
        """A second simulation workspace under the scan root that is NOT the one this run came from.

        A tree can carry a full second copy of the workspace at `<agent>/data/<agent>/plurai_sim/`
        (the plugin's `resolve-workdir.sh` documents the layout). A stale copy of
        this pipeline's own output is not the agent, and a finding sourced to it sends the
        operator to a path they will not recognise. Skipped, and NAMED — a walk that quietly
        declined a whole subtree is the silent truncation the sweep refuses everywhere else.

        Known by CONTENT, not by name: a workspace is a directory named `plurai_sim` OR holding
        its own `plurai.config.yaml`, so `plurai_sim.bak/` or `sim_old/` is a copy too. Never the
        live workdir or a directory above it, and never a config-holding directory inside the
        live workspace (`plurai_sim/envs/<other>/` is a customer-authored harness).
        """
        if live_workdir is None or live_workspace is None:
            return False
        named = here.name == PLUGIN_WORKDIR
        try:
            if not named and not (here / CONFIG_FILENAME).is_file():
                return False
            resolved = here.resolve()
        except OSError:
            # Not a verdict on the directory: the walk's `onerror` counts a subtree it cannot list.
            return False
        if live_workdir.is_relative_to(resolved):
            return False
        return named or not resolved.is_relative_to(live_workspace)

    for dirpath, dirnames, filenames in os.walk(root, onerror=_unlistable):
        at = Path(dirpath)
        rel_parts = at.relative_to(root).parts
        # Vendored trees by NAME at any depth; the plugin's own output only UNDER its workdir.
        # NOT beside one: a customer package named `runs/` sitting next to `plurai_sim/` is agent
        # source, and that distinction is load-bearing. This
        # pipeline's output OUTSIDE the workdir is excluded BY FILENAME below instead, which
        # cannot swallow a directory that merely shares a name.
        under_workdir = root.name == PLUGIN_WORKDIR or PLUGIN_WORKDIR in rel_parts
        keep: list[str] = []
        for d in sorted(dirnames):
            if d in _SCAN_SKIP_DIRS:
                continue
            if under_workdir and d in _PLUGIN_OUTPUT_DIRS:
                continue
            if _is_duplicate_workdir(at / d):
                duplicate_workdirs.append(str(at / d))
                continue
            keep.append(d)
        dirnames[:] = keep
        for filename in sorted(f for f in filenames if f.endswith(suffixes) or _is_dotenv(f)):
            if filename in _GENERATED_WORKSPACE_FILES:
                # COUNTED, like `duplicate_workdirs` and `regex_literals` below: "no hint" and
                # "nothing read" must not print alike, and the names are not all unmistakably
                # ours — a customer package can hold `suite.json`, `suite_state.json` or
                # `agent_topology.json` with a real endpoint in it.
                generated_skipped.append(str(Path(dirpath) / filename))
                continue
            # DOT-PREFIXED files strictly INSIDE the workdir are this pipeline's own state and
            # backups — `.eval_set_id`, `.policies_config.baseline.yaml`,
            # `.generation_trace.pre_extend.md`. Nothing a customer writes lives there under a
            # leading dot, and a host cited to `.policies_config.baseline.yaml` would be a
            # superseded copy of a file this pipeline generated, two removes from the agent.
            if under_workdir and filename.startswith('.'):
                workdir_dotfiles_skipped.append(str(Path(dirpath) / filename))
                continue
            # A dotenv file is a credential file: its VALUES never reach a hint or the record. An
            # endpoint set there is named by its variable, the form the declaration takes for it.
            if _is_dotenv(filename):
                dotenv = Path(dirpath) / filename
                dotenv_files.append(str(dotenv))
                env_endpoints.extend(_dotenv_endpoint_vars(dotenv, cite=str(dotenv.relative_to(root)), known=known, known_env=known_env))
                continue
            if read >= _SCAN_MAX_FILES:
                # Keep walking once the budget is gone, but ONLY to count. "Stopped at 400" with
                # no second number cannot tell a nearly-complete sweep from a token one, and
                # counting a directory entry costs no read — which is what the cap bounds.
                unscanned += 1
                continue
            candidate = Path(dirpath) / filename
            try:
                if candidate.stat().st_size > _SCAN_MAX_BYTES:
                    skipped_big += 1
                    continue
                text = candidate.read_text(encoding='utf-8', errors='replace')
            except OSError:
                skipped_unreadable += 1
                continue
            read += 1
            for lineno, line in enumerate(text.splitlines(), start=1):
                matched_at: set[int] = set()
                for match in _URL_LITERAL.finditer(line):
                    matched_at.add(match.start())
                    if line[match.end() : match.end() + 1] == _REGEX_ESCAPE_AFTER_AUTHORITY:
                        # The authority stopped at a regex escape, so what group 1 holds is the
                        # first LABEL of a pattern, not a host. See _REGEX_ESCAPE_AFTER_AUTHORITY.
                        regex_literals.append(f'{candidate}:{lineno}')
                        continue
                    host, _port, _scheme = _split_declaration(match.group(1))
                    # The DNS ROOT LABEL, dropped. `example.com.` and `example.com` are the same
                    # name, so a trailing dot can only ever split one host into two entries — and
                    # in prose it is not a root label at all but the END OF THE SENTENCE:
                    # "…see the Help Center at https://help.example.com."  # noqa: ERA001 - prose, not code
                    # would report `help.example.com.` as a SECOND undeclared host beside the
                    # real one, a name the operator could not declare away.
                    host = host.rstrip('.')
                    if not host or host in _ALWAYS_LOCAL or host in known or host in found:
                        continue
                    found[host] = HostRef(host=host, port=None, source=f'{candidate}:{lineno}')
                # A credentialed literal the matcher above declined is a host NOT looked for, and
                # the whole point of this channel is that "no hint" and "nothing read" must not
                # print the same way. Compared by START OFFSET, not by count: two URLs on one
                # line, one of them readable, is the case a count alone gets wrong.
                #
                # Filtered by ``known``, which is the difference between "a host went unlooked-for"
                # and "a host the declaration already names is spelled with credentials here". Only
                # the first is a gap, and only the AUTHORITY is read out of the literal — never the
                # credentials, which is why this channel declines the literal in the first place.
                for m in _CREDENTIALED_URL_LITERAL.finditer(line):
                    if m.start() in matched_at:
                        continue
                    behind, _port, _scheme = _split_declaration(m.group(1))
                    if behind and (behind in _ALWAYS_LOCAL or behind in known):
                        continue
                    unreadable.append(f'{candidate}:{lineno}')
    warnings: list[str] = []
    if unscanned:
        warnings.append(
            f'the agent-source scan budget was exhausted at {_SCAN_MAX_FILES} files; {unscanned} further candidate file(s) '
            'were NOT read, and undeclared hosts in them were not looked for'
        )
    if skipped_big:
        warnings.append(f'{skipped_big} source file(s) over {_SCAN_MAX_BYTES // 1000}KB were not scanned')
    if skipped_unreadable:
        warnings.append(f'{skipped_unreadable} source file(s) could not be read and were not scanned')
    if blocked:
        warnings.append(
            f'{len(blocked)} directory(ies) under {root} could NOT be listed — first {blocked[0]}. Their whole subtree went '
            'unread and its files were never counted, so undeclared hosts in them were not looked for'
        )
    if duplicate_workdirs:
        warnings.append(
            f'{len(duplicate_workdirs)} simulation workspace director(ies) under {root} (named `{PLUGIN_WORKDIR}/` or holding '
            f'their own `{CONFIG_FILENAME}`) are NOT the workspace this run came from — first {duplicate_workdirs[0]}. They are '
            "a stale copy of this pipeline's own output, so they were NOT walked and hosts inside them are not reported"
        )
    if generated_skipped:
        warnings.append(
            f'{len(generated_skipped)} file(s) this PIPELINE generates were NOT scanned — first {generated_skipped[0]}. '
            'A host in one of them is a host this pipeline copied out of the agent, so reporting it back would cite our '
            'own output as a finding; but the exclusion is by FILENAME, so a file of yours sharing one of those names '
            'went unread and undeclared hosts in it were not looked for'
        )
    if workdir_dotfiles_skipped:
        warnings.append(
            f'{len(workdir_dotfiles_skipped)} dot-prefixed file(s) inside the `{PLUGIN_WORKDIR}/` workspace were NOT '
            f"scanned — first {workdir_dotfiles_skipped[0]}. Those are this pipeline's own state and backups"
        )
    if dotenv_files:
        warnings.append(
            f'{len(dotenv_files)} dotenv file(s) are credential files, so only variable NAMES are reported from them — first {dotenv_files[0]}'
            + (
                f'. {len(env_endpoints)} endpoint(s) there that this declaration does not name: '
                f'{", ".join(env_endpoints)} — declare each variable the agent reads to reach your own system as '
                '`{env: VAR, source: <file:line where the agent reads it>}`; the model provider the agent thinks with is not one'
                if env_endpoints
                else ''
            )
        )
    if regex_literals:
        warnings.append(
            f'{len(regex_literals)} URL literal(s) whose authority is a REGEX PATTERN, not a host — first at '
            f'{regex_literals[0]}. Their authority stops at a `\\` escape (`^https://host\\.example\\.com/...`), so the name is '
            'truncated and was REFUSED rather than reported as a host to declare'
        )
    if unreadable:
        warnings.append(
            f'{len(unreadable)} credentialed URL literal(s) (`scheme://user:pw@host`) carry a host this scan could NOT read '
            f'— first at {unreadable[0]}. The credentials were refused rather than reported as a host, so those hosts were '
            'not looked for: their absence from the hints below is not evidence about them'
        )
    if read == 0:
        # The whole point of :attr:`SourceScan.files_read`. A walk that read nothing has no
        # opinion, and letting it print the same clean line as a repo with nothing to declare is
        # the failure this module's own docstring calls out: "a check that silently reports
        # nothing reads exactly like a check that found nothing".
        warnings.append(
            f'the agent-source scan read ZERO files under {root} — the DISCOVERED-host channel is INERT for this agent, so '
            'the absence of undeclared-host hints below means "nothing was looked at", NOT "nothing is there". Check the '
            'workspace root, and that the agent is written in a language this walk reads.'
        )
    return SourceScan(
        hosts=tuple(sorted(found.values(), key=lambda h: h.host)),
        warnings=tuple(warnings),
        files_read=read,
        truncated=unscanned,
        skipped_big=skipped_big,
        skipped_unreadable=skipped_unreadable,
        skipped_dirs=len(blocked),
        unread_credentialed=len(unreadable),
        env_endpoints=tuple(env_endpoints),
    )


def _licensed_only_blockers(
    declared: Sequence[HostRef],
    operator_allowed: Sequence[str],
    scan: SourceScan,
    resolution: ScanRootResolution,
) -> tuple[WithheldReason, ...]:
    """Why this declaration may NOT have ``licensed_only``, or empty when it may — typed, with the sentence.

    ``licensed_only`` says isolation was not established and still names what the agent may
    reach. It is defensible only as "``unprovable`` plus a written per-host operator waiver,
    backed by a scan that finished", so each clause here defends one half of that sentence:

    * **the waiver is the OPERATOR's.** ``probed`` empties against the UNION of the config's
      ``allow_hosts`` and the caller's ``extra_allowed`` — the mock service and the Plurai
      backend, hosts this frame supplies, not the operator. A mock served on a declared
      production hostname would otherwise satisfy "every host the operator licensed" with an
      allow-listing the operator never wrote, and it belongs in ``unprovable``.
    * **the scan FINISHED.** ``scan_source_for_hosts`` has four ways of saying "I did not finish
      looking" — the file-budget cap, oversize skips, unreadable skips, and credentialed literals
      whose own warning reads *"their absence from the hints below is not evidence about them"*.
      An unfinished sweep cannot testify to absence, which is the entire second half of what
      this status claims.
    * **the root covers the declaration.** A roster that resolves 1 of 11 entries still names a
      real directory, and a hint-free scan of it says nothing about the ten files that went
      missing.
    * **somebody SAID where the agent is.** ``roster_partial`` alone is not enough: a roster
      resolving 0 of 11, or no roster at all, leaves it False and falls through to the marker
      walk. So the source itself is the clause: ``declared`` and ``roster`` are STATEMENTS about
      where this agent lives (:data:`ScanRootSource`), ``derived`` is a walk that can climb into
      an enclosing monorepo and spend its whole budget there. A verdict whose second half is
      "and a scan of the agent found nothing else" may not rest on a root nobody vouched for.
    * **the walk read something.** A walk with no opinion is not a walk that found nothing.
    """
    allow_set = {h.lower() for h in operator_allowed if h}
    blockers: list[WithheldReason] = []
    if unlicensed := sorted({h.host for h in declared if h.host not in allow_set}):
        blockers.append(
            WithheldReason(
                'not_operator_allowed',
                {'hosts': unlicensed},
                f"{', '.join(unlicensed)} is not in this workspace's `{EGRESS_CONFIG_KEY}.allow_hosts` — it went unprobed because "
                "THIS RUN's own infrastructure allow list (the mock service, the Plurai backend) covers it, which is not a licence "
                'the operator wrote',
            )
        )
    if scan.files_read == 0:
        blockers.append(WithheldReason('scan_read_zero', {}, 'the agent-source scan read ZERO files, so it has no opinion about other hosts'))
    elif reasons := scan.incomplete:
        counts = {
            'truncated': scan.truncated,
            'skipped_big': scan.skipped_big,
            'skipped_unreadable': scan.skipped_unreadable,
            'skipped_dirs': scan.skipped_dirs,
            'unread_credentialed': scan.unread_credentialed,
        }
        blockers.append(
            WithheldReason(
                'scan_incomplete',
                {k: v for k, v in counts.items() if v},
                'the agent-source scan did not finish looking (' + '; '.join(reasons) + '), so it is not evidence of absence',
            )
        )
    if resolution.roster_partial:
        blockers.append(
            WithheldReason('roster_partial', {}, f'the `{EGRESS_CONFIG_KEY}.discovery.scanned` roster that rooted the scan resolved only partially')
        )
    if resolution.source == 'derived':
        blockers.append(
            WithheldReason(
                'root_derived',
                {},
                f'the agent-source scan root was DERIVED by the marker walk — this declaration states no usable '
                f'`{EGRESS_CONFIG_KEY}.scan_root` and no usable `{EGRESS_CONFIG_KEY}.discovery.scanned` roster, so nothing vouches '
                'for the directory the "no other host" half of this verdict was measured over',
            )
        )
    return tuple(blockers)


def run_egress_preflight(
    agent_module_path: Path,
    *,
    mode: ProbeMode,
    enforcement: EnforcementLayer,
    extra_allowed: tuple[str, ...] = (),
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> EgressPreflight:
    """Resolve the declaration, probe what it declares, and report.

    ``enforcement`` (the caller's :func:`enforcement_layer`): under ``python_guard`` the declared
    hosts are blocked inside the agent process, so they are NOT probed and the verdict is
    ``guarded``. It also decides which finding a ``breached`` verdict is (:meth:`EgressPreflight.findings`).

    ``extra_allowed`` carries the hosts THIS run is deliberately allowed to reach — the mock
    service and the Plurai backend — resolved by the caller, which is the only frame that
    knows them. They are excluded from both the probe and the source-scan hints; a run that
    refused itself for talking to its own mock would be useless.
    """
    declaration = load_declaration(agent_module_path)
    declared, config_source = declaration.hosts, declaration.config_source
    allow_set = {h.lower() for h in (*declaration.allowed, *extra_allowed) if h} | set(_ALWAYS_LOCAL)
    probed = [h for h in declared if h.host not in allow_set]
    warnings: list[str] = []
    if len(probed) != len(declared):
        # A guard that suppressed something says so. An allow-listed production host is an
        # operator decision, and it must be visible in the artifact that claims isolation.
        suppressed = sorted({h.host for h in declared if h.host in allow_set})
        warnings.append(f'allow-listed and therefore NOT probed: {", ".join(suppressed)}')
    in_process_guard = enforcement == 'python_guard'
    results = () if in_process_guard else tuple(probe_host(ref, mode, timeout=timeout) for ref in probed)
    # The WORKSPACE, not the harness's own folder: ``simulate.py`` lives in ``plurai_sim/`` and
    # the agent does not, so scanning beside the harness would read only the harness. The
    # config that won names the workspace; without one, fall back to the harness's parent.
    # :func:`_resolve_workspace` then climbs to a repo marker, so a harness in ``sim/`` is
    # covered too.
    # DECLARATION FIRST (:func:`_resolve_scan_root`): an explicit ``egress.scan_root``, then the
    # roster of files the scan says it read, and only then the marker walk. The walk is a
    # derivation and the other two are statements about where this agent actually is.
    resolution = _resolve_scan_root(declaration, fallback=config_source.parent if config_source is not None else _harness_dir(agent_module_path))
    workspace = resolution.path
    warnings.extend(resolution.warnings)
    # The workspace this run's config came from — so a NESTED second copy of it under the scan
    # root can be told apart from the live one and skipped. See `_is_duplicate_workdir`.
    scan = scan_source_for_hosts(
        workspace,
        known=allow_set | {h.host for h in declared},
        known_env={var for var, _ in declaration.env_hosts},
        workdir=config_source.parent if config_source is not None else None,
    )
    warnings.extend(scan.warnings)
    # The shape ``licensed_only`` is FOR, and every reason this instance of it is not one. Kept
    # apart from the verdict chain because the reasons have to be PRINTED: silently degrading to
    # ``unprovable`` tells an operator their declaration was wrong without saying which part.
    licensed_shape = bool(declared) and not probed and not scan.hosts and not scan.env_endpoints
    withheld = _licensed_only_blockers(declared, declaration.allowed, scan, resolution) if licensed_shape else ()
    env_hosts = declaration.env_hosts
    cause: NotDeterminableCause | None = None
    if not declared and not env_hosts:
        # THE DISCOVERY BLOCK IS A BRANCH, not only a warning source. An empty declaration beside
        # a `discovery.status: none_discovered` is the scan saying it swept and found nothing; an
        # empty declaration with no discovery block at all is nobody having looked. Collapsing
        # them would print the safe-looking wrong word over the safe case — see :data:`EgressStatus`.
        # The status comes off the same key `_nothing_declared_line` picks its sentence by;
        # `sweep_unproven` and the `declared`-contradiction stay `not_declared`.
        kind = _nothing_declared_kind(declaration.discovery)
        status: EgressStatus = 'swept_clean' if kind == 'swept_clean' else 'not_determinable' if kind == 'not_determinable' else 'not_declared'
        cause = 'scan' if status == 'not_determinable' else None
    elif in_process_guard and (probed or env_hosts):
        # `guarded` claims the agent runs in the guarded process; an undeclared runtime cannot say so.
        status, cause = ('not_determinable', 'agent_runtime') if declaration.agent_runtime == 'undeclared' else ('guarded', None)
    elif env_hosts and not any(r.reachable is True for r in results):
        # Nothing in the runner can resolve them (their value lives in the child's environment), so
        # nothing here establishes their isolation.
        status = 'unprovable'
    elif any(r.reachable is True for r in results):
        status = 'breached'
    elif (
        len(results) == len(declared)
        and results
        and all(r.reachable is False for r in results)
        and all(r.ref.port_declared or not r.port_specific for r in results)
    ):
        # ``isolated`` is the one POSITIVE safety claim here, so it is spelled as "every probe
        # came back with DEFINITIVE evidence" — and ``reachable is False`` is set at exactly two
        # places (NXDOMAIN, a remote refusal), both of which are the host answering. Everything
        # a probe can fail on — a timeout, a broken resolver, a local connect error — is ``None``
        # and lands below, in ``unprovable``.
        #
        # NXDOMAIN belongs in the definitive set and must not be "tightened" out of it later: a
        # resolver saying the name does not exist here is an ANSWER, not silence, and nothing can
        # reach a name that does not resolve. Narrowing this to a TCP refusal alone would delete
        # the `isolated` verdict for every correctly isolated env — the ones whose declared
        # production hosts are not resolvable from the test machine at all, which is the normal
        # shape of real isolation.
        #
        # ``len(results) == len(declared)`` is what makes the claim cover the DECLARATION rather
        # than the subset that happened to be probed. ``results and all(...)`` alone would close
        # only TOTAL suppression: allow-list the ONE reachable host out of two and the remaining
        # probe comes back refused, so `isolated` would be written over a production host nobody
        # looked at. An operator cannot silence this check by allow-listing the world, but could
        # by allow-listing the host that mattered — and ``extra_allowed`` (the mock service, the
        # Plurai backend) reaches the same set.
        #
        # ``results and`` still guards the vacuous case: ``all(())`` is True, so zero probes over
        # a non-empty declaration would otherwise be a POSITIVE claim over nothing.
        #
        # The PORT clause is the last of these. A refusal is definitive about the port it was
        # taken on, and nothing more: ``db.prod.example.com`` declared bare is probed on 80
        # and 443, refused on both, and would ship ``isolated`` over a Postgres the agent can
        # reach RIGHT NOW on 5432 — the same shape for a Kafka, an SMTP relay, a
        # Redis or a gRPC endpoint. So a definitive refusal counts only where somebody DECLARED
        # the port (explicitly, or by naming an http/https scheme, whose port family is exactly
        # ``DEFAULT_PROBE_PORTS``); otherwise the run is ``unprovable`` and
        # :meth:`EgressPreflight.report_lines` names each guessed-port host with the remedy.
        #
        # ``not r.port_specific`` is what stops that clause from eating the verdict it protects.
        # NXDOMAIN is an answer about the NAME, so it is true of every port on it — and it is the
        # normal shape of real isolation, per the paragraph above. Gating it on a declared port
        # would delete ``isolated`` for exactly the correctly-isolated envs the paragraph names:
        # every one of them would read ``unprovable``.
        status = 'isolated'
    elif licensed_shape and not withheld:
        # LICENSED-ONLY: every declared production host is one the OPERATOR allow-listed, so
        # there was nothing left to probe — and a scan that actually read the agent's source, over
        # a root the declaration vouches for, and WITHOUT stopping early, found no other host. That
        # is the whole shape of a tool-less LLM agent, whose only egress is the model provider the
        # run cannot work without, and it could never leave ``unprovable``: an allow-listed host is
        # "allow-listed and therefore NOT probed", so ``isolated``'s "every declared host was
        # probed and refused" is unreachable by construction.
        #
        # This status says isolation was not established and still names what the agent may reach, so what
        # it rests on is enumerated in :func:`_licensed_only_blockers` rather than spelled inline
        # here — every clause there is load-bearing and none may be dropped for convenience.
        # ``files_read > 0`` alone does not prevent a mis-rooted scan from upgrading itself into a
        # positive verdict: a mis-rooted scan can read hundreds of files.
        #
        # It is WEAKER than ``isolated`` and the report says so: a licence is not a wall.
        status = 'licensed_only'
    else:
        status = 'unprovable'
    if withheld and status == 'unprovable':
        # The near-miss, named. This shape is one edit away from the only status that proceeds
        # without a probe, so the operator gets the list rather than a bare `unprovable`.
        warnings.append('the `licensed_only` verdict was WITHHELD and this run is `unprovable` instead: ' + '; '.join(r.text for r in withheld) + '.')
    if declaration.discovery is not None and declaration.discovery.swept_clean and declared:
        # The two halves of one block disagreeing. Neither half can be believed over the other
        # from here, so it is reported, not resolved.
        warnings.append(f'the scan recorded `discovery.status: none_discovered` but {len(declared)} production host(s) are declared; one of the two is stale.')
    return EgressPreflight(
        status=status,
        probe_mode=mode,
        enforcement=enforcement,
        declared=tuple(declared),
        allowed=tuple(sorted(allow_set - set(_ALWAYS_LOCAL))),
        results=results,
        undeclared=scan.hosts,
        config_source=config_source,
        discovery=declaration.discovery,
        scan_root=workspace,
        scan_root_source=resolution.source,
        scan_files_read=scan.files_read,
        warnings=tuple(warnings),
        live_by_consent=tuple(sorted({h.host for h in declared} & {a.lower() for a in declaration.allowed})),
        env_hosts=env_hosts,
        not_determinable_cause=cause,
        agent_runtime=declaration.agent_runtime,
        defects=declaration.defects,
        withheld=withheld if status == 'unprovable' else (),
    )
