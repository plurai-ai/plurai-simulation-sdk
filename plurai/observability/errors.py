"""Shared best-effort helpers for plurai observability sinks.

Best-effort posture: swallow sink/HTTP errors so observability failures never
abort a simulation run. :func:`swallow` and :func:`offload` are the load-bearing
backstop: they catch bare ``Exception`` (including programmer bugs such as
TypeError or KeyError) and log them via ``logger.exception`` — they do NOT
re-raise. :func:`swallowable_error_types` builds a narrower typed-except tuple
via dynamic lookup so the SDK imports without any sink SDK installed; at
typed-tuple call sites, programmer bugs fall outside that tuple and propagate,
but :func:`swallow`/:func:`offload` are the outer guard that prevents any
unhandled exception from reaching the simulation.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


async def offload[T](fn: Callable[..., T], *args: Any, sink: str, what: str = 'operation', **kwargs: Any) -> T | None:
    """Run a blocking best-effort sink op OFF the event loop: ``swallow(fn)`` via ``to_thread``.

    The async counterpart to :func:`swallow`, and the single de-boilerplated form of the
    sinks' ``return await asyncio.to_thread(swallow, self._x, ..., sink=, what=)`` offload —
    each such site collapses to ``return await offload(self._x, ..., sink=, what=)``. Native
    async is NOT uniformly available across the sink SDKs, so the thread hop is KEPT (a
    blocking SDK called directly in an ``async def`` would stall the loop under
    ``asyncio.gather``); this only removes the repeated ``asyncio.to_thread(swallow, ...)``
    scaffolding. ``*args`` / ``**kwargs`` forward to ``fn``; ``sink`` / ``what`` thread into
    :func:`swallow` for its log line. Returns ``fn``'s ``T`` on success or ``None`` on a
    swallowed failure (same best-effort contract as :func:`swallow`).
    """
    return await asyncio.to_thread(swallow, fn, *args, sink=sink, what=what, **kwargs)


def swallow[T](fn: Callable[..., T], *args: Any, sink: str, what: str = 'operation', **kwargs: Any) -> T | None:
    """Run ``fn(*args, **kwargs)``; log + swallow ANY exception, returning None.

    The load-bearing best-effort backstop: a sink throw must never abort a
    simulation that already produced a valid result. Concrete sinks call this
    (or a narrower typed-except) so an outer ``try`` is never forgotten. ``sink``
    names the sink in the log line; ``what`` names the failed operation.

    Generic in ``fn``'s return type (PEP 695): a successful call returns ``fn``'s
    own ``T``, and a swallowed failure returns ``None`` — so callers keep their
    return type (``T | None``) instead of the type-erasing ``Any``.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception('Observability %s failed for sink %s; simulation is unaffected.', what, sink)
        return None


def swallowable_error_types(
    *candidates: tuple[str, str],
    fallback: tuple[type[BaseException], ...] = (OSError,),
) -> tuple[type[BaseException], ...]:
    """Build a tuple of exception classes safe to swallow for a sink.

    For each ``(module_path, attr_name)`` pair: import the module and
    getattr the class, skipping on ``ImportError`` or ``AttributeError``.
    ``AttributeError`` must be caught because ``getattr`` differs from
    ``from X import Y`` — an SDK that renames the attribute raises
    ``AttributeError`` at runtime, not ``ImportError``, and must degrade
    gracefully rather than crash the run.

    Returns ``tuple(found_types)`` or ``fallback`` when nothing resolves.
    ``fallback`` defaults to ``(OSError,)`` which covers raw socket errors
    as a last resort.
    """
    types: list[type[BaseException]] = []
    for module_path, attr_name in candidates:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, attr_name)
            types.append(cls)
        except (ImportError, AttributeError):
            pass
    return tuple(types) or fallback
