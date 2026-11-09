"""Low-level HTTP client — one method per REST endpoint."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal

import httpx
from pydantic import Field

from plurai import _http, command_run
from plurai._endpoints import bearer_for_origin, resolve_document_reference, resolve_server_relative
from plurai._protocol import QUARANTINE_REFUSAL_PREFIX, OrchestratorAdapterError, ScenarioQuarantined
from plurai.content_parts import document_part_values
from plurai.customer_safe import safe_text
from plurai.gates.invocation import current as current_gate_authority
from plurai.host_effects import register_backend_token
from plurai.orchestrator._retry import RetryConfig, request_with_retry
from plurai.orchestrator.models import (
    AgentAction,
    CreateThreadRequest,
    CreateThreadResponse,
    EvalSetPoliciesResponse,
    FailResponse,
    FinalizeJobStatusResponse,
    FinalizeJobSubmitResponse,
    FinalizeRequest,
    FinalizeResponse,
    GetThreadResponse,
    Overrule as WireOverrule,
    RescoreJobStatusResponse,
    RescoreJobSubmitResponse,
    RescoreRequest,
    RescoreResult,
    ScenarioSetFetchResponse,
    SerializedMessage,
    TurnWorld,
    UserActionResult,
    UserTurnRequest,
    UserTurnResponse,
)
from plurai.orchestrator.state_wire import DESCRIPTOR_PREFIX, map_string_leaves

if TYPE_CHECKING:
    from plurai.gates.contract import Overrule
    from plurai.settings import Settings

logger = logging.getLogger(__name__)


def _wire_overrules(overrules: Sequence[Overrule]) -> list[WireOverrule]:
    return [WireOverrule(gate_id=o.gate_id, fingerprint=o.fingerprint, reason=o.reason) for o in overrules]


async def submit_carrying_overrules[T](submit: Callable[[Sequence[Overrule]], Awaitable[T]], *, endpoint: str) -> T:
    """Call ``submit`` with the open invocation's backend overrules (SDK ids apply locally, others are forwarded).

    FEATURE DETECTION BY THE ANSWER, not by a version probe. A backend that predates the ``overrules`` field ignores the
    key (every request model is ``extra='ignore'``) and its result carries no ``gate_authority``, so nothing is
    acknowledged. One that refuses the field — a build that applies no overrule, or an overrule it cannot validate —
    answers **422** naming ``overrules``: the collector records why (:meth:`GateAuthority.backend_declined`), stops
    forwarding for the rest of the invocation, and the request is re-sent once without them. Either way the run goes on
    and the ids stay ``unacknowledged``. A 422 that is not about ``overrules`` propagates unchanged. A current backend
    refuses only a MALFORMED overrule (which :func:`~plurai.gates.cli.load_overrules` already refuses here); an id it does
    not declare (a gate renamed or removed since the ledger line was written) is answered ``not_evaluated`` beside the
    others, so one stale id never costs the rest.

    SCOPE: a backend score finding's fingerprint covers the thread it was judged on, and this run mints that thread.
    So an overrule written before the run answers ``stale`` on its ``/finalize``; REST forwarding answers overrules for
    findings the backend already reported for that thread. Restoring a finished run's score verdicts is the MCP
    ``score_run`` path, not this one.
    """
    authority = current_gate_authority()
    overrules = authority.forwarded_overrules()
    if not overrules:
        return await submit(())
    try:
        return await submit(overrules)
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.text[:_REFUSAL_SCAN_CHARS]
        except Exception:  # noqa: BLE001 - `.text` RAISES on a streamed, unread response: then it is no answer about overrules
            body = ''
        if exc.response.status_code != 422 or 'overrules' not in body:
            raise
        authority.backend_declined(endpoint, f'HTTP 422 {safe_text(body)}')
    return await submit(())


#: The key the backend stamps on a cell an applied overrule restored: the fingerprints it lifted.
#: Overrules are INTERNAL — the applied outcome, with its finding, is recorded in
#: ``gate_authority.json`` — and a run artifact carries the effective verdict only.
OVERRULED_FINDINGS_KEY = 'overruled_findings'


def _effective_cell(cell: dict[str, Any]) -> tuple[dict[str, Any], list[Any], list[Any]]:
    """``cell`` as its effective verdict alone: no :data:`OVERRULED_FINDINGS_KEY`, and none of the ``gate_findings``
    rows it lifted (a finding left beside the verdict it no longer describes is the overrule by another name) — with
    the marker's fingerprints and the lifted rows that came off it. Pure."""
    lifted = cell.get(OVERRULED_FINDINGS_KEY)
    out = {k: v for k, v in cell.items() if k != OVERRULED_FINDINGS_KEY}
    rows = out.get('gate_findings')
    removed: list[Any] = []
    if isinstance(lifted, list) and isinstance(rows, list):
        fingerprints = {fp for fp in lifted if isinstance(fp, str)}
        kept = [row for row in rows if not (isinstance(row, dict) and row.get('fingerprint') in fingerprints)]
        removed = [row for row in rows if isinstance(row, dict) and row.get('fingerprint') in fingerprints]
        if kept:
            out['gate_findings'] = kept
        else:
            del out['gate_findings']
    marker = list(lifted) if isinstance(lifted, list) else ([] if lifted is None else [lifted])
    return out, marker, removed


def _effective_cells(cells: Any, *, accounted: frozenset[str], unaccounted: list[dict[str, Any]]) -> Any:
    """``{policy: cell}`` through :func:`_effective_cell`; a non-mapping (cell or map) passes through. Every marker
    fingerprint no recorded ``applied`` outcome ``accounted`` for is appended to ``unaccounted`` with the rows it lifted
    — kept for the internal record, never left on the cell."""
    if not isinstance(cells, dict):
        return cells
    out: dict[str, Any] = {}
    for policy, cell in cells.items():
        if not isinstance(cell, dict):
            out[policy] = cell
            continue
        out[policy], marker, removed = _effective_cell(cell)
        loose = [fp for fp in marker if fp not in accounted]
        if loose:
            unaccounted.append(
                {'policy': policy, OVERRULED_FINDINGS_KEY: loose, 'gate_findings': [row for row in removed if row.get('fingerprint') in set(loose)]}
            )
    return out


def take_gate_authority[R: (FinalizeResponse, RescoreResult)](result: R) -> R:
    """The ONE place a backend score result crosses into this run: its ``gate_authority`` answer is merged into the
    open invocation's collector (``gate_authority.json``), and the result comes back without it and without any
    overrule marker on its cells. Everything persisted or uploaded from the result (``transcript.json``, the run
    summary, the run upload, the sinks) reads the copy returned here.

    A cell marker no recorded ``applied`` outcome accounts for (the block, or its row, did not validate, or no block
    came) is not discarded: it moves, with the rows it lifted, to the internal record
    (``backend_unaccounted_markers``), and the cell keeps the effective verdict alone."""
    authority = current_gate_authority()
    authority.answer_backend(result.gate_authority)
    accounted = authority.applied_backend_fingerprints()
    unaccounted: list[dict[str, Any]] = []
    if isinstance(result, RescoreResult):
        cells = _effective_cells(result.evaluation_results, accounted=accounted, unaccounted=unaccounted)
        authority.backend_unaccounted_markers.extend(unaccounted)
        return result.model_copy(update={'gate_authority': None, 'evaluation_results': cells})
    per = result.policy_evaluation_results
    if isinstance(per, dict) and 'evaluation_results' in per:
        per = {**per, 'evaluation_results': _effective_cells(per['evaluation_results'], accounted=accounted, unaccounted=unaccounted)}
    authority.backend_unaccounted_markers.extend(unaccounted)
    return result.model_copy(update={'gate_authority': None, 'policy_evaluation_results': per})


class UserTurn(UserTurnResponse):
    """:class:`UserTurnResponse` plus ``ended_early_cause``, which the generated model may not declare.

    A subclass and not an edit to ``models.py``, because that module is GENERATED from the
    backend's API spec. Every generated model is ``extra='ignore'``, so a field the spec does not
    declare is dropped on read, and a give-up would then route the operator to the agent for a
    harness fault. Declared here, the field is read whenever a backend sends it and reads ``None``
    from one that does not; a same-named field on the parent makes the override a no-op.

    ``ended_early_cause`` is the simulated user's own stated CAUSE for a ``user_gave_up``
    (``orchestrator.GIVE_UP_CAUSE_*`` names the values this build routes on). A plain ``str``
    for the same additive-skew reason as the reason beside it; ``None`` on every turn that
    carries no give-up and from every backend that predates it.
    """

    ended_early_cause: Annotated[str | None, Field(title='Ended Early Cause')] = None


#: The request body size a fronting proxy accepts: 10 MiB.
PROXY_BODY_LIMIT_BYTES = 10 * 1024 * 1024


def _json_body(endpoint: str, body: Mapping[str, Any]) -> bytes:
    """``body`` as the exact UTF-8 JSON bytes the request carries.

    WARNs first, naming the largest top-level fields, when those bytes exceed :data:`PROXY_BODY_LIMIT_BYTES`.
    The serialization is httpx's own (compact separators, non-ASCII kept as UTF-8), so the size measured
    is the size sent. The request is still sent when it is over: a box reached without that proxy
    accepts it. A 413 is not retryable and its body says nothing about which field grew, so this
    line is what makes one diagnosable.
    """
    payload = _dump_json(body)
    if len(payload) > PROXY_BODY_LIMIT_BYTES:
        sizes = {key: len(_dump_json(value)) for key, value in body.items()}
        largest = ', '.join(f'{key}={size:,} B' for key, size in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:3])
        logger.warning(
            '%s request body is ~%s B, over the %s MiB a fronting proxy accepts; a 413 here is not retryable. Largest fields: %s.',
            endpoint,
            f'{len(payload):,}',
            PROXY_BODY_LIMIT_BYTES // 1_048_576,
            largest,
        )
    return payload


def _dump_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


_JSON_HEADERS = {'content-type': 'application/json'}


#: How much of a refusal body to look at. The prefix is the FIRST thing in the ``detail``, so a
#: bounded read cannot miss it — and a huge HTML error page from a proxy cannot make this scan.
_REFUSAL_SCAN_CHARS = 4096


def _quarantine_detail(response: httpx.Response | None) -> str | None:
    """The refusal message when ``response`` is a quarantine refusal, else ``None``.

    Matches the PREFIX the backend freezes for exactly this purpose, never the sentence after it —
    an English sentence is not a contract, and the two facts it separates ("the suite refused to
    measure this" / "the agent errored") must never merge into one number because a message was
    reworded.

    The FastAPI ``detail`` is preferred and the raw body is the fallback: a proxy that rewrites
    the envelope but forwards the text must not turn a typed refusal back into a generic 400.
    """
    if response is None or response.status_code != 400:
        return None
    try:
        body = response.text[:_REFUSAL_SCAN_CHARS]
    except Exception:  # noqa: BLE001 - `.text` RAISES on a streamed, unread response
        return None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    detail = payload.get('detail') if isinstance(payload, dict) else None
    if isinstance(detail, str) and detail.lstrip().startswith(QUARANTINE_REFUSAL_PREFIX):
        return safe_text(detail)
    return safe_text(body[body.index(QUARANTINE_REFUSAL_PREFIX) :]) if QUARANTINE_REFUSAL_PREFIX in body else None


def eval_set_policy_names(answer: EvalSetPoliciesResponse, eval_set_id: str) -> list[str]:
    """The policy names off an eval-set answer, with the ONE degradation rule for a missing field.

    Module-level rather than inline in the fetch so the two readers of that response — the names
    projection and the pre-flight, which needs the provenance beside them — cannot come to
    disagree about what an omitted ``policies`` field means. It degrades to ``[]``, loudly: the
    server is expected to return a list (possibly empty), and the pre-flight refuses a run over
    an empty one for the same reason it refuses a genuinely empty set — a run that cannot
    establish that it has policies must not be allowed to report that it graded any.
    """
    if answer.policies is None:
        logger.warning(
            'server omitted the policies field for eval set %s (contract violation); degrading to []',
            eval_set_id,
        )
        return []
    return answer.policies


class OrchestratorClient:
    """Async wrapper around the backend orchestration REST API."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 1200.0,
        api_key: str | None = None,
        retry: RetryConfig | None = None,
        command_run_id: str | None = None,
    ) -> None:
        headers: dict[str, str] = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        if command_run_id:
            # On every request, not only the ones that create a thread or a job: the label is the client's,
            # and one client serves one command.
            headers[command_run.HEADER] = command_run_id
        # One READ timeout for every request. It is set generously (see
        # NetworkSettings.http_timeout) because POST /user-turn legitimately sits near it —
        # an LLM user-simulator turn plus the per-turn judges. NOT because of /finalize:
        # that is submit + poll (finalize_async_submit / finalize_job_status), so each call
        # is fast and the multi-minute wait is bounded by
        # NETWORK__FINALIZE_POLL_MAX_WALLCLOCK_SECONDS instead — that, not this value, is
        # the knob for a slow finalize. CONNECT is split off at 10s: a bare float sets
        # connect=read alike, which makes a DEAD host indistinguishable from a
        # slow request for the whole ceiling per call — down-vs-slow must resolve in
        # seconds (a healthy connect says the box is up; only the response may be
        # slow).
        self._base_url = base_url.rstrip('/')
        self._api_key = api_key or None
        self._client = _http.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            headers=headers,
        )
        self._retry = retry if retry is not None else RetryConfig()
        # thread id -> (served ``<scheme>:///documents/…`` reference -> the ``data:`` URI folded from it).
        self._served: dict[str, dict[str, str]] = {}

    def served_documents(self, thread_id: str) -> Mapping[str, str]:
        """``<scheme>:///documents/…`` reference → folded ``data:`` URI for every served document this client fetched for the thread.

        A live view: a document a ``user_turn`` hands over is in it when that call returns. The agent
        holds the bytes and the backend holds the document, so the orchestrator hands the agent the fold
        and sends the backend the reference wherever the agent echoes it. Keyed by reference, so two
        references to byte-identical documents are two entries. Empty for a thread with none, and for one
        released by :meth:`forget_thread`.
        """
        return MappingProxyType(self._served.get(thread_id, {}))

    def forget_thread(self, thread_id: str) -> None:
        """Drop what the client holds for the thread: its served documents, which are up to several MB each.

        The caller releases a thread once nothing reads its served documents any more; one client serves
        every scenario of a run, so an unreleased thread holds its documents until the run ends.
        """
        self._served.pop(thread_id, None)

    async def __aenter__(self) -> OrchestratorClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request_with_retry(
        self,
        call: Callable[[], Awaitable[httpx.Response]],
        *,
        idempotent: bool,
        endpoint: str = '',
        retry_unforwarded: bool = False,
    ) -> httpx.Response:
        """Execute *call* under the risk-scoped retry policy (see :mod:`._retry`)."""
        return await request_with_retry(call, config=self._retry, idempotent=idempotent, endpoint=endpoint, retry_unforwarded=retry_unforwarded)

    async def create_thread(
        self,
        scenario_set_id: str,
        scenario_id: str,
        *,
        eval_set_id: str | None = None,
        agent_name: str = 'plurai-agent',
        agent_version: str = 'sim-1',
        max_turns: int = 5,
        experiment_name: str | None = None,
        mock_service_url: str | None = None,
        agent_opens: bool = False,
        shape: Literal['dialog', 'pipeline'] | None = None,
    ) -> CreateThreadResponse:
        body = CreateThreadRequest(
            scenario_set_id=scenario_set_id,
            scenario_id=scenario_id,
            eval_set_id=eval_set_id,
            agent_name=agent_name,
            agent_opens=agent_opens,
            shape=shape,
            agent_version=agent_version,
            max_turns=max_turns,
            experiment_name=experiment_name,
            mock_service_url=mock_service_url,
        ).model_dump(exclude_none=False)
        # Same proxy-provenance widening as user_turn below: a 503 on /threads/* is only the
        # gateway refusing to forward (no app 503 exit on these routes), so the
        # request was never applied and a retry cannot double-create a thread.
        try:
            response = await self._request_with_retry(
                lambda: self._client.post('/threads', json=body), idempotent=False, endpoint='/threads', retry_unforwarded=True
            )
        except httpx.HTTPStatusError as exc:
            # The suite REFUSING to measure a quarantined scenario is not a bad request and not
            # an agent failure, and it arrives on the same 400 as both. Classified HERE, at the
            # one wire-receive seam, so every downstream frame — error.json's ``kind``, the
            # runner's failure line, a report counting agent failures — sees a typed refusal
            # rather than re-deriving one from prose.
            if (detail := _quarantine_detail(exc.response)) is not None:
                raise ScenarioQuarantined(detail, scenario_id=scenario_id) from exc
            raise
        thread = CreateThreadResponse.model_validate(response.json())
        try:
            return await self._resolve_server_paths(thread)
        except BaseException:
            # The caller never learns the id of a thread this call failed to return, so it cannot release it.
            self.forget_thread(thread.thread_id)
            raise

    async def _resolve_server_paths(self, thread: CreateThreadResponse) -> CreateThreadResponse:
        """Resolve server-relative paths in the thread payload against THIS client's base URL.

        The server advertises ``mock_service_url`` as a server-RELATIVE path
        (``/mock/<env_id>/`` — a dedicated URL field, so a leading slash is unambiguous)
        and any over-cap document folded into ``initial_agent_context`` as a
        self-describing ``<scheme>:///documents/...`` reference (scheme-tagged, because
        that dict is customer-schema data — resolution dispatches on the scheme, never
        on a value's shape). Only the client knows the base URL that actually reaches
        the box. Resolving here, at the single wire-receive seam, means every downstream
        consumer sees absolute URLs and fetched documents. A served document stays a
        reference in the thread's ``initial_agent_context`` and its bytes are held in
        :meth:`served_documents`: the orchestrator folds them in where it hands the context to
        the agent, and every recorded or re-posted copy carries the reference.
        An absolute ``mock_service_url`` passes through because it is the client's OWN
        value echoed back (the request field is client-authoritative); absolute
        agent-context values pass through because they are customer data. Neither is a
        compatibility affordance — a backend that advertises these two forms is REQUIRED
        (version skew fails loudly at the resolve seams, never silently works).
        """
        base = str(self._client.base_url)
        if thread.mock_service_url and thread.mock_service_url.startswith('/'):
            thread.mock_service_url = resolve_server_relative(base, thread.mock_service_url)
        served = self._served.setdefault(thread.thread_id, {})
        if thread.initial_agent_context:
            thread.initial_agent_context = await self._resolve_agent_context(base, thread.initial_agent_context, served=served, thread_id=thread.thread_id)
        await self._fold_message_documents(base, thread.initial_messages, served=served, where='initial_messages', thread_id=thread.thread_id)
        seed = thread.resume_seed
        if seed is not None and isinstance(seed.get('agent_state'), dict):
            await self._fold_seed_state(base, seed['agent_state'], served=served, thread_id=thread.thread_id)
        return thread

    async def _resolve_agent_context(self, base: str, context: dict[str, Any], *, served: dict[str, str], thread_id: str) -> dict[str, Any]:
        """``context`` with every ``<scheme>:///`` string leaf, at any depth, resolved against ``base``.

        A served document is fetched into ``served`` and its leaf stays the reference the backend sent;
        any other ``<scheme>:///`` leaf becomes its absolute URL, at a nested position as at the top level.
        """
        leaves: list[tuple[str, str]] = []

        def collect(path: str, leaf: str) -> str:
            leaves.append((path, leaf))
            return leaf

        map_string_leaves(context, collect, 'agent_context')
        for path, leaf in leaves:
            await self._resolve_served_value(base, path, leaf, served=served, thread_id=thread_id)
        return map_string_leaves(context, lambda _, leaf: leaf if leaf in served else resolve_document_reference(base, leaf))

    async def _fold_seed_state(self, base: str, state: dict[str, Any], *, served: dict[str, str], thread_id: str) -> None:
        """Fetch every served-document reference among ``state``'s string leaves, at any depth, into ``served``.

        A resume pin carries the state a trace ended in, and a trace records a served document as
        its reference. The pin stays as the backend sent it, like a user turn's handed-over document;
        the orchestrator swaps the fold in only where the seed is handed to the child, which
        restores it into an agent that cannot open the reference.

        A leaf that is a descriptor (the wire cap replaced the value when the source run was
        recorded) holds no value; it is restored as it is and one WARNING names where.
        """
        leaves: list[tuple[str, str]] = []

        def collect(path: str, leaf: str) -> str:
            leaves.append((path, leaf))
            return leaf

        map_string_leaves(state, collect, 'resume_seed.agent_state')
        descriptors = [path for path, leaf in leaves if leaf.startswith(DESCRIPTOR_PREFIX)]
        if descriptors:
            logger.warning(
                'Thread %s: %s hold a descriptor ("agent_state value not sent"), not a value: the wire cap replaced it in the source run. '
                'The agent is seeded with the descriptor text.',
                thread_id,
                ', '.join(descriptors),
            )
        for path, leaf in leaves:
            await self._resolve_served_value(base, path, leaf, served=served, thread_id=thread_id)

    async def _resolve_served_value(self, base: str, label: str, value: str, *, served: dict[str, str], thread_id: str) -> str:
        """One server-authored value, resolved — and, for a DOCUMENT, folded to its bytes (returned, and kept in ``served``). ``label`` names where it sat.

        The backend folds a document under the inline cap as ``data:<type>;base64,…`` and an OVER-cap
        one as a ``<scheme>:///documents/…`` reference. Rewriting that reference to a URL would hand
        the customer's agent a bare link to a BEARER-GUARDED route, and the bearer never reaches
        the agent. This client already holds the bearer and already owns this seam, so it
        resolves the reference ITSELF and folds the bytes the way the inline arm does: the agent
        sees ONE form for a binary document regardless of size.

        THE ASSUMPTION THIS RESTS ON: today only the BINARY arm produces a reference. The backend
        folds a text document inline as plain text and skips an over-cap one with a warning — it has no text reference arm — so a
        ``<scheme>:///documents/…`` value can only ever be a binary. If that ever changes, this
        fold would emit ``data:text/plain;base64,…`` against the text arm's bare string, i.e. two
        forms for one document kind, which is the exact divergence this seam exists to remove.
        The fix then is on the OTHER side of the ``if``, not a text arm here.

        A fold is recorded in ``served`` as ``reference → folded``: the agent holds the bytes, but
        the backend holds the document, so wherever the agent echoes the fold back the orchestrator
        sends this reference instead. A reference already in ``served`` is not fetched again.

        A failed fetch is LABELLED with the thread it belongs to: it runs inside the call that returns
        the thread or the turn, so a bare ``httpx`` error would reach the runner with no thread id and
        stamp ``thread_id: "unknown"`` into error.json. ``HTTPError`` and not only ``HTTPStatusError``: a
        SLOW box raises ``ConnectError`` / ``ReadTimeout`` / ``RemoteProtocolError``, so the status is
        rendered only when there is one and a transport error names its class instead. A credit refusal is not a failed fetch:
        it propagates as :class:`CreditRefused`, which stops the run.
        """
        if value in served:
            return served[value]
        resolved = resolve_document_reference(base, value)
        if resolved == value or '/documents/' not in resolved:
            return resolved
        # THROUGH THE RETRY POLICY, like every other scenario-path request in this client
        # (``/threads``, ``/user-turn``, ``/finalize``, the job poll, ``/fail``, the two set
        # fetches), so a transient 503 on the document store is retried as the same blip on
        # ``/user-turn`` is. ``idempotent=True`` is a statement of fact about the route: it is a content-
        # addressed GET, so a repeat cannot double-apply anything.
        try:
            response = await self._request_with_retry(
                lambda: self._client.get(resolved),
                idempotent=True,
                endpoint='/documents/{scenario_set}/{scenario}/{path}',
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            detail = f'{exc.response.status_code} from {exc.request.url}' if isinstance(exc, httpx.HTTPStatusError) else f'{type(exc).__name__}: {exc}'
            raise OrchestratorAdapterError(f'Resolving the document for {label} on thread {thread_id} failed: {detail}', thread_id=thread_id) from exc
        media_type = (response.headers.get('content-type') or 'application/octet-stream').split(';')[0].strip()
        logger.info('resolved document for %s: %d bytes, %s', label, len(response.content), media_type)
        folded = f'data:{media_type};base64,' + base64.b64encode(response.content).decode('ascii')
        served[value] = folded
        return folded

    async def _fold_message_documents(
        self, base: str, messages: Sequence[SerializedMessage] | None, *, served: dict[str, str], where: str, thread_id: str
    ) -> None:
        """Fetch every served document a user turn's ``file`` / ``image_url`` parts reference, into ``served``.

        The messages themselves are left as the backend sent them: the transcript and the backend's judges
        key a handed-over document by that value. The orchestrator swaps the fold in only where the
        turn is handed to the agent, which has no bearer for the ``/documents`` route.
        """
        for index, message in enumerate(messages or ()):
            for value in document_part_values(message.content):
                await self._resolve_served_value(base, f'{where}[{index}]', value, served=served, thread_id=thread_id)

    async def user_turn(
        self,
        thread_id: str,
        *,
        agent_response: str,
        agent_state: dict[str, Any] | None = None,
        agent_actions: list[dict[str, Any]] | None = None,
        reply_source: str | None = None,
        turn_messages: list[dict[str, Any]] | None = None,
        turn_idx: int | None = None,
        turn_world: dict[str, Any] | None = None,
        user_action_result: dict[str, Any] | None = None,
    ) -> UserTurn:
        """POST /threads/{id}/user-turn — ship the agent's reply, get the next user message.

        ``agent_state`` is the agent's NON-message return state for THIS turn; the
        server files it under this turn's 0-based index and accumulates it, so a
        ``turn_based`` judge sees only its own turn's state. The
        field is OPTIONAL on both ends and every wire model is ``extra='ignore'``, so an
        older backend simply drops it and a newer backend tolerates an older SDK — no
        version gate is needed here.

        ``agent_actions`` (``[{name, args, status}]``) is what the agent DID this turn, and
        ``reply_source`` is where ``agent_response`` came from. Together they make the turn
        TYPED: ``agent_response`` may be empty on a turn that is complete (an agent whose
        output is a render verb list acts without speaking), and empty on a turn this runner
        emptied — and the server cannot tell those apart from the string. ``agent_actions``
        is deliberately tri-state, ``None`` meaning "this client cannot observe its agent's
        actions" as against ``[]``'s "it can, and there were none": collapsing the two would
        let an unobservable turn read as a turn in which nothing happened. Same
        optional-both-ends posture as the fields above.

        ``turn_idx`` is this turn's 0-based loop index (the orchestrator's
        ``range(max_turns)`` counter), which the server treats as an IDEMPOTENCY KEY:
        it equals the server's pre-increment ``turn_count``, so a re-delivered
        POST is safe — the server replays that turn's stored reply (``turn_idx ==
        turn_count - 1``) instead of re-driving the dialog, and 409s a real divergence.
        Sending it is what lets this POST be marked ``idempotent`` below, so a read-timeout
        or 502/503/504 is retried instead of one slow turn killing the whole scenario.

        ``turn_world`` is the world at the END of this agent turn as the child captured it
        (:mod:`plurai.world_delta`): what the simulated user can see. ``user_action_result`` is how
        the act the PREVIOUS response asked for was applied. Both are sent as the child produced
        them, and both are absent from a child that predates them.

        Dumped with ``exclude_none=True`` so a stateless agent's body carries only what it
        actually has: ``agent_response`` is required, so nothing load-bearing can be
        dropped by that flag.
        """
        # ``turn_idx`` present → the request is IDEMPOTENT: the server keys on it to replay
        # a lost turn's stored reply rather than re-drive the dialog, so a retried POST
        # cannot double-apply. That upgrades the whole transient class — read/write
        # timeouts AND 502/503/504 — to retryable (``idempotent=True``), which is the point:
        # without the key, one slow turn kills the scenario because this POST cannot be
        # retried.
        #
        # ``turn_idx`` absent (a caller that sends none) → non-idempotent, with the ONE
        # provenance-justified widening ``retry_unforwarded``. A 503 on this route is only the
        # gateway refusing to forward (the backend has no 503 exit here), so it was provably
        # never applied and retrying cannot double-apply. Other status errors stay fatal
        # without the key — a mid-flight 502/504 or timeout may have already driven the turn.
        #
        # ``turn_messages`` / ``agent_state`` are additive optional channels, optional both
        # ends, extra='ignore'. ``exclude_none=True`` keeps the body of a caller that has none
        # free of them — an absent turn_idx is a MISSING key, not a null.
        body = UserTurnRequest(
            agent_response=agent_response,
            agent_state=agent_state,
            agent_actions=None if agent_actions is None else [AgentAction.model_validate(a) for a in agent_actions],
            reply_source=reply_source,
            turn_messages=None if turn_messages is None else [SerializedMessage.model_validate(m) for m in turn_messages],
            turn_idx=turn_idx,
            turn_world=None if turn_world is None else TurnWorld.model_validate(turn_world),
            user_action_result=None if user_action_result is None else UserActionResult.model_validate(user_action_result),
        ).model_dump(exclude_none=True)
        payload = _json_body('/threads/{id}/user-turn', body)
        response = await self._request_with_retry(
            lambda: self._client.post(f'/threads/{thread_id}/user-turn', content=payload, headers=_JSON_HEADERS),
            idempotent=turn_idx is not None,
            endpoint='/threads/{id}/user-turn',
            retry_unforwarded=turn_idx is None,
        )
        turn = UserTurn.model_validate(response.json())
        await self._fold_message_documents(
            str(self._client.base_url), turn.user_messages, served=self._served.setdefault(thread_id, {}), where='user_messages', thread_id=thread_id
        )
        return turn

    async def finalize_async_submit(
        self,
        thread_id: str,
        *,
        user_messages: list[dict[str, str]] | None = None,
        chatbot_messages: list[dict[str, str]] | None = None,
        final_world_state: dict[str, Any] | None = None,
        tool_events: dict[str, Any] | None = None,
        closing_agent_response: str | None = None,
        closing_agent_state: dict[str, Any] | None = None,
        closing_turn_messages: list[dict[str, Any]] | None = None,
        closing_agent_actions: list[dict[str, Any]] | None = None,
        closing_reply_source: str | None = None,
        ended_early: str | None = None,
        resume_fidelity: dict[str, Any] | None = None,
        pipeline_inputs: dict[str, Any] | None = None,
        pipeline_outputs: dict[str, Any] | None = None,
        pipeline_error: dict[str, Any] | None = None,
        user_action_result: dict[str, Any] | None = None,
        unread_inputs: list[str] | None = None,
        post_seed_world_state: dict[str, Any] | None = None,
        overrules: Sequence[Overrule],
    ) -> FinalizeJobSubmitResponse:
        """POST /threads/{id}/finalize (async submit) → typed ``FinalizeJobSubmitResponse``.

        Marked ``idempotent=True``: the server's submit-dedup guarantee (same thread →
        same ``job_id`` for any live job) makes a retried POST safe — it returns the
        existing job rather than spawning a second one.

        ``closing_agent_state`` is the agent's non-message return state for the CLOSING
        reply — the one agent turn that never passes through ``/user-turn``, which
        is exactly why it needs its own field here rather than riding the per-turn wire.
        Like ``agent_state`` on ``/user-turn`` it is
        OPTIONAL on both ends and every wire model is ``extra='ignore'``, so an older
        backend drops it and a newer backend tolerates an older SDK — no version gate.
        This dump uses ``exclude_none=False``: the finalize path sends explicit nulls for
        its optional fields.

        ``ended_early`` reports a reason only WE know first-hand — the per-scenario time budget
        is our own dialog clock, invisible to the server. Everything else in that vocabulary is
        recorded server-side (that is where the judges are), and the server fills this field
        only when it recorded nothing itself, so reporting it can never overwrite what actually
        ended the dialog. Same optional-both-ends posture as the fields above: an older backend
        drops it, a newer one tolerates an SDK that never sends it.

        ``overrules`` are the invocation's BACKEND gate overrules (internal), REQUIRED so a caller
        states "none" as ``()``; route them through :func:`submit_carrying_overrules`, which owns
        what an older backend does with the field. The answer rides the job result's
        ``gate_authority``.

        Raises ``pydantic.ValidationError`` when the server returns a body that does
        not match ``FinalizeJobSubmitResponse`` (e.g. missing ``job_id``, unknown
        ``status``). The caller (``_finalize_async``) catches and converts to
        ``OrchestratorAdapterError``.
        """
        body = FinalizeRequest(
            user_messages=[SerializedMessage.model_validate(m) for m in (user_messages or [])],
            chatbot_messages=[SerializedMessage.model_validate(m) for m in (chatbot_messages or [])],
            # NOT ``or {}``. ``None`` is a THIRD state — "this run's world was never observed" —
            # and collapsing it into the second ("observed, and empty") is the one reading that
            # gets TRUSTED downstream: an empty snapshot from a scenario that seeded nothing is
            # read as a world with nothing to mutate, so every world-reading judge passes
            # vacuously on a world nobody ever looked at. The field is ``dict | None`` on BOTH
            # ends of the wire already (the generated models here and the backend's there) and this
            # dump is ``exclude_none=False``, so the null crosses as a null with no version gate:
            # an older backend reads it exactly as it reads today's ``{}``.
            final_world_state=final_world_state,
            # NOT ``or {}``: ``FinalizeRequest.tool_events`` is nullable on both sides of this
            # wire, and ``None`` is the one value that tells the server no tool channel was read
            # at all. Coercing it to ``{}`` here would make an unread channel indistinguishable from a
            # channel that recorded nothing, which is what a tool-trail judge grades the agent on.
            # The dump below is ``exclude_none=False``, so the ``null`` rides.
            tool_events=tool_events,
            closing_agent_response=closing_agent_response,
            closing_agent_state=closing_agent_state,
            closing_turn_messages=(None if closing_turn_messages is None else [SerializedMessage.model_validate(m) for m in closing_turn_messages]),
            # The closing turn's two TYPED channels. ``None`` is NOT ``[]`` on the actions half:
            # ``[]`` says "this client can observe its agent's actions and there were none", which
            # is what makes a render-verb turn gradable at all, so a producer that cannot observe
            # must say so — the same tri-state ``UserTurnRequest.agent_actions`` keeps one route
            # over. The dump below is ``exclude_none=False``, so the null rides as a null.
            closing_agent_actions=(None if closing_agent_actions is None else [AgentAction.model_validate(a) for a in closing_agent_actions]),
            closing_reply_source=closing_reply_source,
            ended_early=ended_early,
            # The ONE-SHOT lane's (inputs, outputs) pair, on the SAME body and TYPED — the
            # generated model carries them. ``None`` on every dialog scenario, which is what the server
            # reads as "this run had no hop" and what leaves its end-to-end state unchanged.
            pipeline_inputs=pipeline_inputs,
            pipeline_outputs=pipeline_outputs,
            pipeline_error=pipeline_error,
            resume_fidelity=resume_fidelity,
            # The act the dialog ended on: applied by the closing turn, which has no /user-turn.
            user_action_result=None if user_action_result is None else UserActionResult.model_validate(user_action_result),
            # The child's read record (``None`` = not recorded); the backend judges it against the declared channels.
            unread_inputs=unread_inputs,
            # The world right after ``seed_state`` (``None`` = not read); the backend takes origins against it.
            post_seed_world_state=post_seed_world_state,
            overrules=_wire_overrules(overrules),
        ).model_dump(exclude_none=False)
        payload = _json_body('/threads/{id}/finalize', body)
        response = await self._request_with_retry(
            lambda: self._client.post(f'/threads/{thread_id}/finalize', content=payload, headers=_JSON_HEADERS),
            idempotent=True,
            endpoint='/threads/{id}/finalize',
        )
        return FinalizeJobSubmitResponse.model_validate(response.json())

    async def finalize_job_status(self, thread_id: str, job_id: str) -> FinalizeJobStatusResponse:
        """GET /threads/{id}/finalize/jobs/{job_id} → typed ``FinalizeJobStatusResponse``.

        A 404 (job evicted / unknown) raises ``httpx.HTTPStatusError`` and propagates
        to the caller. Marked idempotent so ``ReadTimeout`` on the poll GET is retried.

        Raises ``pydantic.ValidationError`` when the server returns an unrecognized
        ``status`` value (schema drift). The caller (``_finalize_async``) catches and
        converts to ``OrchestratorAdapterError``.
        """
        response = await self._request_with_retry(
            lambda: self._client.get(f'/threads/{thread_id}/finalize/jobs/{job_id}'),
            idempotent=True,
            endpoint='/threads/{id}/finalize/jobs/{job_id}',
        )
        return FinalizeJobStatusResponse.model_validate(response.json())

    async def cancel_finalize_job(self, thread_id: str, job_id: str) -> None:
        """DELETE /threads/{id}/finalize/jobs/{job_id} — best-effort cancel, never raises.

        Called from the per-scenario error/abandon path to stop an in-flight async
        finalize task from billing LLM calls after the SDK has given up on the run.
        """
        try:
            await self._request_with_retry(
                lambda: self._client.delete(f'/threads/{thread_id}/finalize/jobs/{job_id}'),
                idempotent=True,
                endpoint='/threads/{id}/finalize/jobs/{job_id}',
            )
        except Exception:  # noqa: BLE001 — best-effort; caller cannot act on a failed cancel
            logger.debug(
                'Best-effort cancel of finalize job %s on thread %s failed',
                job_id,
                thread_id,
                exc_info=True,
            )

    async def rescore_async_submit(
        self,
        thread_id: str,
        *,
        transcript: dict[str, Any],
        final_world_state: dict[str, Any] | None,
        tool_events: dict[str, Any] | None,
        overrules: Sequence[Overrule],
    ) -> RescoreJobSubmitResponse:
        """POST /threads/{id}/rescore (async submit) → typed ``RescoreJobSubmitResponse``.

        A JUDGE REPLAY of a finished thread: the first attempt's saved artifacts, verbatim, re-judged
        by the thread's OWN eval set — the body carries the input only, and the server reads which
        eval set and which scenario from the thread. Marked ``idempotent=True`` for the reason the
        finalize submit is: the server returns the live job for a re-submit rather than spawning a
        second paid fan-out. A 404 (unknown thread, or a backend with no such route) raises
        ``httpx.HTTPStatusError`` for the caller to classify. ``overrules`` as on
        :meth:`finalize_async_submit`; the answer rides the replay result's ``gate_authority``.
        """
        body = RescoreRequest(
            transcript=transcript, final_world_state=final_world_state, tool_events=tool_events, overrules=_wire_overrules(overrules)
        ).model_dump(exclude_none=False)
        response = await self._request_with_retry(
            lambda: self._client.post(f'/threads/{thread_id}/rescore', json=body),
            idempotent=True,
            endpoint='/threads/{id}/rescore',
        )
        return RescoreJobSubmitResponse.model_validate(response.json())

    async def rescore_job_status(self, thread_id: str, job_id: str) -> RescoreJobStatusResponse:
        """GET /threads/{id}/rescore/jobs/{job_id} → typed ``RescoreJobStatusResponse``. Idempotent,
        so a read timeout on the poll is retried; a 404 (job evicted, or another thread's) raises."""
        response = await self._request_with_retry(
            lambda: self._client.get(f'/threads/{thread_id}/rescore/jobs/{job_id}'),
            idempotent=True,
            endpoint='/threads/{id}/rescore/jobs/{job_id}',
        )
        return RescoreJobStatusResponse.model_validate(response.json())

    async def health(self) -> bool:
        """``GET /health`` — is the box serving? Never raises.

        The first phase of the re-drive readiness gate. Deliberately does NOT go
        through ``_request_with_retry``: this is a probe, and every outcome other
        than 200 (including the 404 an older box without the native ``/health``
        route answers, and the connection error a box mid-boot answers) is
        information, not an exception. ``/health`` is pre-auth on the box, so the
        probe works with or without a bearer token.
        """
        try:
            response = await self._client.get('/health')
        except Exception:  # noqa: BLE001 — a probe reports, it does not raise
            return False
        return response.status_code == 200

    async def probe_url(self, url: str) -> tuple[int | None, str | None]:
        """Probe an ABSOLUTE url; return ``(status_code, body_code)``. Never raises.

        Used by the readiness gate on the scenario's ``mock_service_url``.
        ``body_code`` is the backend's machine-readable discriminator (``restarting``
        / ``awaiting_edit_finalize`` / ``needs_redeploy``) when the body carries one —
        the gate branches on THAT, never on the human prose beside it, because two
        503s here differ only in whether waiting can ever help.

        ``(None, None)`` means the probe never got an answer (transport failure).
        The client's bearer goes with the request only when ``url`` is under this client's
        origin, exactly: ``/mock`` sits behind the box's auth middleware, and a mock
        url the run was pointed at elsewhere, a loopback port included, never receives the token.
        """
        try:
            request = self._client.build_request('GET', url)
            if bearer_for_origin(self._base_url, self._api_key, url) is None:
                request.headers.pop('Authorization', None)
            response = await self._client.send(request)
        except Exception:  # noqa: BLE001 — a probe reports, it does not raise
            return None, None
        code: str | None = None
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001 — a non-JSON body simply carries no code
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get('code'), str):
            code = payload['code']
        return response.status_code, code

    async def get_thread(self, thread_id: str) -> GetThreadResponse:
        response = await self._request_with_retry(lambda: self._client.get(f'/threads/{thread_id}'), idempotent=True, endpoint='/threads/{id}')
        return GetThreadResponse.model_validate(response.json())

    async def fail(self, thread_id: str, *, exception: str, error: str) -> FailResponse:
        """Tell the server a run crashed before /finalize (marks the thread failed).

        The client still writes error.json locally; this is server-side bookkeeping
        so /threads/{id} reflects the failure.
        """
        body = {'exception': exception, 'error': error}
        response = await self._request_with_retry(
            lambda: self._client.post(f'/threads/{thread_id}/fail', json=body),
            idempotent=False,
            endpoint='/threads/{id}/fail',
        )
        return FailResponse.model_validate(response.json())

    async def fetch_eval_set(self, eval_set_id: str) -> EvalSetPoliciesResponse:
        """The WHOLE ``GET /eval-sets/{eval_set_id}/policies`` answer: names AND provenance.

        The additive read endpoint the server exposes so the client can resolve policy names
        before launching an experiment run. Raises ``httpx.HTTPStatusError`` on a non-2xx
        (e.g. 404 for an unknown eval set id).

        The response also carries ``state`` (``draft`` | ``promoted`` | ``unknown``) and, on a
        draft, ``edit_session_id`` — which bytes answered. Only this method can see
        them; :meth:`fetch_eval_set_policies` projects the names and drops the rest, and a reader
        of the names alone cannot tell that a run launched during an open edit session scores the
        session's mid-edit draft. ``state is None`` means the SERVER predates the
        field, which is a fourth thing and never ``promoted``.
        """
        response = await self._request_with_retry(
            lambda: self._client.get(f'/eval-sets/{eval_set_id}/policies'),
            idempotent=True,
            endpoint='/eval-sets/{id}/policies',
        )
        return EvalSetPoliciesResponse.model_validate(response.json())

    async def fetch_eval_set_policies(self, eval_set_id: str) -> list[str]:
        """Fetch the policy names registered in an eval set — the names half of
        :meth:`fetch_eval_set`, for the callers that only resolve evaluator columns.
        """
        return eval_set_policy_names(await self.fetch_eval_set(eval_set_id), eval_set_id)

    async def fetch_scenario_set(self, scenario_set_id: str) -> ScenarioSetFetchResponse:
        """Fetch a scenario set's per-scenario inputs + metadata for dataset upload.

        Calls ``GET /scenario-sets/{scenario_set_id}`` — the KEY-FREE endpoint the
        server exposes so the client can gather scenario content and upload it to
        whichever observability sink it holds keys for. Raises
        ``httpx.HTTPStatusError`` on a non-2xx (e.g. 404 for an unknown set id) so
        the caller can surface a clear error.
        """
        response = await self._request_with_retry(
            lambda: self._client.get(f'/scenario-sets/{scenario_set_id}'),
            idempotent=True,
            endpoint='/scenario-sets/{id}',
        )
        return ScenarioSetFetchResponse.model_validate(response.json())


def build_default_client(settings: Settings) -> OrchestratorClient:
    """Build the default :class:`OrchestratorClient` when no client is injected.

    The DI seam the Orchestrator uses to construct a real network client. The backend and its bearer
    are the run's resolved :attr:`~plurai.settings.Settings.endpoint`; the timeout and the retry knobs
    (``retries`` / ``backoff_base`` / ``backoff_cap``, via :meth:`RetryConfig.from_settings`) come from
    the :class:`~plurai.settings.NetworkSettings` component.
    """
    endpoint = settings.endpoint
    register_backend_token(endpoint.token)
    return OrchestratorClient(
        endpoint.url,
        timeout=settings.network.http_timeout,
        api_key=endpoint.token,
        retry=RetryConfig.from_settings(settings.network),
        command_run_id=settings.command_run_id,
    )
