"""Local (in-process) mock-service helpers.

Owns the loopback-guard, free-port picker, and the in-process ``MockService``
launcher. The caller (``plurai._mocking.session.configure``) retains the service
in ``_local_service`` and registers the atexit hook.
"""

from __future__ import annotations

import inspect
import socket
import sys

from plurai_mockenv import MockService, build_service
from plurai_mockenv import _registry

from plurai._endpoints import is_loopback_url
from plurai._mocking.client import DEFAULT_MOCK_SERVICE_URL, MockServiceClient
from plurai._protocol import StubMode


def _free_loopback_port() -> int:
    """Reserve a free loopback TCP port (bind-0) for the in-process mock.

    A small TOCTOU window exists between releasing this socket and ``MockService``
    re-binding the port; for an isolated per-run loopback mock that's an acceptable
    trade (the integration harness's free-port helper makes the same one). ``MockService``
    binds a fixed port and does not report an OS-assigned one back, so we pick here.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _start_without_bind_banner(service: MockService) -> None:
    """Start ``service``, suppressing the engine's own "[plurai] Mock service at …" line.

    That banner is printed from inside ``plurai_mockenv.MockService.start`` and announces a
    MOCK — under ``mockless``, the one posture whose entire claim is that nothing is mocked,
    it contradicts the very next line this module prints. The posture-correct sentence is
    written HERE, where both halves (the declaration and the handler registry) are visible.

    The engine is a separately versioned, tag-pinned package, so the kwarg is passed only
    when the installed one accepts it and the banner simply stays when it does not: a
    version skew must never fail a run over a log line.
    """
    if 'banner' in inspect.signature(service.start).parameters:
        # Spread a named mapping rather than writing the kwarg out: against an engine whose
        # signature has no ``banner``, a static call reads as a type error, even though the
        # line above is what decides whether it runs at all.
        quiet: dict[str, object] = {'banner': False}
        service.start(**quiet)
        return
    service.start()


def _requested_url_suffix(service_url: str | None, *, posture: StubMode) -> str:
    """The " (requested …)" clause, or '' when there is nothing a human actually requested.

    A url that was GIVEN is said out loud beside the one actually bound: ignoring it silently
    makes "I configured :8199 and the agent dialed something else" look like a bug in the
    harness rather than the documented behaviour of these postures.

    Two urls are NOT that, and announcing them would manufacture the confusion the clause
    exists to prevent: ``None`` (nothing was given) and the package default, which
    the backend hands every thread that has no deployed mock — announcing either fires the
    clause on a url nobody chose, in the postures where no url is honored anyway.

    The posture is interpolated rather than hardcoded to ``local``: ``mockless`` shares this
    launcher, and naming the wrong posture in the one line that explains the behaviour is how
    a reader concludes the run was something it was not.
    """
    if service_url is None or service_url.rstrip('/') == DEFAULT_MOCK_SERVICE_URL:
        return ''
    return f' (requested {service_url} — host/port NOT honored in {posture} mode)'


def _serve_local_mock(service_url: str | None = None, *, posture: StubMode = 'local') -> MockService:
    """Spin up the in-process :class:`MockService` for ``mode='local'`` / ``'mockless'``.

    ``posture`` is the DECLARATION the caller made, and it changes only what is said, never
    what is served: both modes bind the same idle-capable loopback stub. ``'mockless'`` means
    the caller declared that nothing is mocked, so a registered handler under that posture is
    a contradiction worth saying out loud (below) rather than a mock to announce.

    ``service_url`` is OPTIONAL and is a loopback TOKEN, never an address: this function
    always selects its own ephemeral port so parallel scenarios cannot collide, so the
    host and port of a given url are not honored. What a given url buys is the loopback
    CHECK — local mode must never shadow a remote mock — and with none given there is
    nothing to shadow, so there is nothing to check.

    Constructs the service via ``build_service`` so the staged ``@plurai.mock`` handlers /
    observers / descriptors **and** the typed-RPC schemas + decorator opt-outs are threaded
    in explicitly (the engine reads no module-global registries), starts the server, and
    points ``_registry._client`` at the bound port.

    Returns the started :class:`MockService` so the caller (``plurai.configure``)
    can store it in ``_local_service`` and register the ``atexit`` cleanup hook.
    """
    # The strict check the bearer posture uses: a literal loopback address or `localhost`. A name that merely begins
    # `127.`, and the bind-all `0.0.0.0`, are not loopback.
    if service_url is not None and not is_loopback_url(service_url):
        raise RuntimeError(
            f'plurai.configure(mode={posture!r}) requires a loopback URL, got {service_url!r}. '
            "Refusing to shadow a remote mock with an in-process one — use mode='connect' "
            'to talk to a deployed mock.'
        )
    port = _free_loopback_port()
    base_url = f'http://127.0.0.1:{port}'
    # build_service collects the staging registry (handlers, observers,
    # descriptors, typed-RPC schemas, decorator opt-outs) and threads it into the
    # engine; passing only host/port is all the local path needs.
    service = build_service(host='127.0.0.1', port=port)
    _start_without_bind_banner(service)
    _registry._client = MockServiceClient(base_url=base_url)
    # Say what is actually being served: this boot is unconditional per scenario child, so a
    # tool-less agent (`tools: {}`, run_mode=local) would otherwise get a confident "serving"
    # line per scenario and an analyst would reason past a mock service that mocks nothing.
    # The stub still starts — harness templates call plurai.mock_service_url() in seed_state —
    # but an empty handler registry is labelled as the idle stub it is.
    # The "requested" clause is compared as a whole STRING (see _requested_url_suffix): parsing
    # the authority first would read `urlparse(...).port`, which RAISES on a malformed url —
    # after the server has already started and rebound the registry client, leaking the port
    # and skipping the atexit stop.
    requested = _requested_url_suffix(service_url, posture=posture)
    if posture == 'mockless':
        # The DECLARED posture is "no surface is held constant", so the stub is scaffolding for
        # plurai.mock_service_url(), not a mock. Handlers registered under it are a real
        # contradiction — the harness mocks something the topology says runs for real — and it
        # is said loudly here because nothing downstream can see both halves at once.
        if service.mock_handlers:
            # NOT followed by the "no mocked surfaces" line below: that line would be false in
            # exactly the branch this warning exists to flag, and printing both would let a reader
            # skim past the contradiction to the reassuring half.
            print(
                f'[plurai] WARNING: mode=mockless declares nothing is mocked, but '
                f'{len(service.mock_handlers)} @plurai.mock handler(s) are registered and WILL '
                f'intercept. The declared posture and the harness disagree — fix plurai.config.yaml '
                f'or the run mode before trusting this run. Loopback stub on {base_url}{requested}.',
                file=sys.stderr,
            )
        else:
            print(f'[plurai] No mocked surfaces — every tool runs for real. Loopback stub on {base_url}{requested}.', file=sys.stderr)
    elif service.mock_handlers:
        print(f'[plurai] Local mock serving on {base_url}{requested}', file=sys.stderr)
    else:
        print(f'[plurai] Local mock stub on {base_url}{requested} (no @plurai.mock tools registered — nothing to serve)', file=sys.stderr)
    return service
