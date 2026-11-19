"""How the scenario's date reached the agent: ``plurai.clock_delivered`` and ``plurai.pin_now``.

Every scenario declares its "now" in ``initial_agent_context['current_date']`` (else ``'today'``), and the mock answers
every time-relative question against it. Whether the AGENT ran on that date is a fact only the harness knows, so the
harness states it, by route:

- ``pinned``: the harness rebound the agent's clock reads to the scenario date (:func:`pin_now` records this itself);
- ``store``: the harness wrote the date into the agent's own world or store, which the agent reads as its clock;
- ``told``: the harness put the date into the agent's own instructions or input.

The child carries the strongest route recorded for its session to ``/finalize`` as ``clock_delivery``
(``pinned`` > ``store`` > ``told``); nothing recorded is ``None``. A READ of ``current_date`` off the context is not a
delivery and records nothing: only these two calls do.
"""

from __future__ import annotations

import datetime as _dt
import email.utils
import re
import sys
import time as _time
import types
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, Literal, get_args

from plurai_mockenv._registry import _active_session_id

ClockRoute = Literal['pinned', 'store', 'told']

#: Strongest first: the route carried when a session recorded several.
_PRECEDENCE: tuple[ClockRoute, ...] = get_args(ClockRoute)
#: The agent-context keys a scenario declares its clock under, in the order they are read (the backend's run clock keys).
_CLOCK_KEYS: tuple[str, ...] = ('current_date', 'today')
#: The attribute a pinned shim carries: the real object it stands in for, so a second pin re-anchors it.
_REAL_ATTR = '_plurai_real'
_UTC_SUFFIX = re.compile(r'\s*(?:UTC|GMT|Z)$', flags=re.IGNORECASE)
#: Where a date-only scenario's clock starts: mid-day, so a run's ticking never leaves the declared date.
_DATE_ONLY_TIME = _dt.time(12, 0, 0)

# Per-session routes recorded, keyed by session id. Finalize only READS them (a retried finalize must carry the same
# route); the child's shutdown takes (and clears) them.
_delivered: dict[str, set[str]] = {}


def clock_delivered(route: ClockRoute) -> None:
    """Record that the scenario's date reached the agent by ``route`` (``'pinned'``, ``'store'`` or ``'told'``).

    Call it where the harness delivers the date: after writing it into the agent's store (``'store'``), after putting
    it in the agent's instructions or input (``'told'``), or after shimming the agent's own clock helper with the class
    :func:`pin_now` returned (``'pinned'``). Records for the session active in this context; with none active it warns
    and records nothing. Raises ``ValueError`` on any other route.
    """
    if route not in _PRECEDENCE:
        raise ValueError(f'plurai.clock_delivered: route must be one of {", ".join(map(repr, _PRECEDENCE))}, got {route!r}')
    session_id = _active_session_id()
    if not session_id:
        print(f'[plurai] WARNING: plurai.clock_delivered({route!r}) was called with no active session; nothing was recorded.', file=sys.stderr)
        return
    _delivered.setdefault(session_id, set()).add(route)


def _strongest(recorded: set[str]) -> ClockRoute | None:
    return next((route for route in _PRECEDENCE if route in recorded), None)


def recorded_clock_delivery(session_id: str) -> ClockRoute | None:
    """The strongest route recorded for ``session_id`` (``None`` when none was). Reads only: a second finalize of the
    same session (the parent's retry after a failed snapshot read) carries the same route."""
    return _strongest(_delivered.get(session_id, set()))


def take_clock_delivery(session_id: str) -> ClockRoute | None:
    """:func:`recorded_clock_delivery`, and forget the record (the child's shutdown, so a re-used interpreter doesn't leak)."""
    return _strongest(_delivered.pop(session_id, set()))


def _declared(ctx: Any) -> Any:
    if not isinstance(ctx, Mapping):
        return None
    for key in _CLOCK_KEYS:
        value = ctx.get(key)
        if value is not None:
            return value
    return None


def _scenario_instant(value: Any) -> _dt.datetime | None:
    """The instant a declared clock starts the pinned clock at: a time with no offset, or a trailing UTC / GMT / Z, is
    UTC; a date ALONE is that day's noon UTC, so the clock can tick through the run and stay on the scenario's date
    (the backend reads a date alone as that day, all day). ``None`` when unreadable."""
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.UTC)
    if isinstance(value, _dt.date):
        return _dt.datetime.combine(value, _DATE_ONLY_TIME, tzinfo=_dt.UTC)
    try:
        text = _UTC_SUFFIX.sub('', str(value).strip())
        with suppress(ValueError):
            return _dt.datetime.combine(_dt.date.fromisoformat(text), _DATE_ONLY_TIME, tzinfo=_dt.UTC)
        try:
            stamp = _dt.datetime.fromisoformat(text)
        except ValueError:
            stamp = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=_dt.UTC)


class _Real(type):
    """A real ``datetime`` / ``date`` still passes ``isinstance`` against the pinned name."""

    def __instancecheck__(cls, obj: Any) -> bool:
        return isinstance(obj, cls.__mro__[1])


def _real_of(value: Any) -> Any:
    """The real object a value stands for: itself, or what a shim from an earlier pin stands in for."""
    return vars(value).get(_REAL_ATTR, value) if isinstance(value, type | types.SimpleNamespace) else value


def pin_now(target: Any, ctx: Mapping[str, Any]) -> type[_dt.datetime]:
    """Rebind the clock names ``target`` (a module or object) holds to the scenario's clock from ``ctx``.

    Reads ``ctx['current_date']``, else ``ctx['today']``. Rebinds only the names ``target`` itself holds that are
    ``datetime.datetime``, ``datetime.date``, the ``datetime`` module (any alias) or the ``time`` module, never the
    process's clock, which the SDK, provider auth and traces read. The pinned clock starts at the scenario's instant
    (a date alone starts at that day's noon UTC) and ticks on in real time, so a deadline loop or a backoff still
    advances; a naive read answers in the declared offset, an aware one converts. Through a rebound ``time`` module,
    every wall-clock read is pinned (``time``, ``time_ns``, and ``localtime`` / ``gmtime`` / ``ctime`` / ``asctime`` /
    ``strftime`` called without an explicit time); ``monotonic``, ``perf_counter`` and ``sleep`` stay real. Dates the
    agent builds through the pinned class (``fromisoformat``, ``strptime``, the constructor, ...) are real
    ``datetime`` / ``date`` objects, so they pickle and serialize as usual. Pinning an already pinned target re-anchors it.

    Records ``plurai.clock_delivered('pinned')`` only when at least one name was rebound (a name that cannot be
    set, e.g. on a frozen object, is skipped). Returns the pinned ``datetime`` class, for shimming the agent's own
    clock helper (then call ``plurai.clock_delivered('pinned')`` yourself). Never raises: with no readable date in
    ``ctx`` it warns, rebinds and records nothing, and returns the real ``datetime.datetime``.
    """
    declared = _declared(ctx)
    anchor = None if declared is None else _scenario_instant(declared)
    if anchor is None:
        what = 'declares no current_date / today' if declared is None else f'declares a clock that cannot be read ({declared!r})'
        print(f'[plurai] WARNING: plurai.pin_now: the scenario {what}; nothing was pinned and the agent reads the host clock.', file=sys.stderr)
        return _dt.datetime
    t0 = _time.monotonic()

    class PinnedDatetime(_dt.datetime, metaclass=_Real):
        # Every datetime built THROUGH the pinned class is a real one: a local subclass instance would not pickle.
        def __new__(cls, *args: Any, **kwargs: Any) -> _dt.datetime:  # type: ignore[misc]
            return _dt.datetime(*args, **kwargs)  # noqa: DTZ001 - the caller's own arguments, tzinfo included

        @classmethod
        def now(cls, tz: _dt.tzinfo | None = None) -> _dt.datetime:  # type: ignore[override]
            moment = anchor + _dt.timedelta(seconds=_time.monotonic() - t0)
            return moment.astimezone(tz) if tz else moment.replace(tzinfo=None)

        @classmethod
        def today(cls) -> _dt.datetime:  # type: ignore[override]
            return cls.now()

        @classmethod
        def utcnow(cls) -> _dt.datetime:  # type: ignore[override]
            return cls.now(_dt.UTC).replace(tzinfo=None)

    class PinnedDate(_dt.date, metaclass=_Real):
        def __new__(cls, *args: Any, **kwargs: Any) -> _dt.date:  # type: ignore[misc]
            return _dt.date(*args, **kwargs)

        @classmethod
        def today(cls) -> _dt.date:  # type: ignore[override]
            return PinnedDatetime.now().date()

    def _seconds() -> float:
        return PinnedDatetime.now(_dt.UTC).timestamp()

    def _or_now(secs: float | None) -> float:
        return _seconds() if secs is None else secs

    pinned_time = {
        'time': _seconds,
        'time_ns': lambda: int(_seconds() * 1_000_000_000),
        'localtime': lambda secs=None: _time.localtime(_or_now(secs)),
        'gmtime': lambda secs=None: _time.gmtime(_or_now(secs)),
        'ctime': lambda secs=None: _time.ctime(_or_now(secs)),
        'asctime': lambda t=None: _time.asctime(_time.localtime(_seconds()) if t is None else t),
        'strftime': lambda fmt, t=None: _time.strftime(fmt, _time.localtime(_seconds()) if t is None else t),
    }
    setattr(PinnedDatetime, _REAL_ATTR, _dt.datetime)
    setattr(PinnedDate, _REAL_ATTR, _dt.date)
    shims: dict[int, Any] = {
        id(_dt.datetime): PinnedDatetime,
        id(_dt.date): PinnedDate,
        id(_dt): types.SimpleNamespace(**{**vars(_dt), 'datetime': PinnedDatetime, 'date': PinnedDate, _REAL_ATTR: _dt}),
        id(_time): types.SimpleNamespace(**{**vars(_time), **pinned_time, _REAL_ATTR: _time}),
    }
    try:
        names = list(vars(target).items())
    except TypeError:  # an object with no attribute dict holds no name to rebind
        names = []
    rebound = 0
    for name, value in names:
        shim = shims.get(id(_real_of(value)))
        if shim is None:
            continue
        try:
            setattr(target, name, shim)
        except (AttributeError, TypeError, ValueError):  # a frozen dataclass / pydantic model, a read-only property
            continue
        rebound += 1
    if rebound:
        clock_delivered('pinned')
    else:
        label = getattr(target, '__name__', type(target).__name__)
        print(f'[plurai] WARNING: plurai.pin_now: {label} holds no datetime / date / time name it could pin; nothing was recorded.', file=sys.stderr)
    return PinnedDatetime
