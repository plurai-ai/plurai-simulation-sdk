"""Arize experiments-path surface — REQUIRES the ``arize`` extra at import time.

Unlike :mod:`plurai.observability.arize` (eagerly imported via ``plurai.settings``,
so it must stay importable without the vendor SDK), importing THIS module means
the SDK is present and every symbol is defined at module level — no lazy class
factories; ``sys.modules`` is the cache. :meth:`ArizeSink.run_experiment_set`
imports it lazily, exactly where the experiments path begins (next to its
``from arize import ArizeClient``), so ``import plurai``
without ``[arize]`` is unaffected.
"""

from __future__ import annotations

import math
from typing import Any

from arize.experiments.evaluators.base import CodeEvaluator
from arize.experiments.evaluators.types import EvaluationResult

from plurai.observability.arize import _safe_eval_name
from plurai.observability.rollup import ALWAYS_PRESENT_ROLLUP_NAMES, VERDICT_ROLLUP_NAMES


class PassthroughEvaluator(CodeEvaluator):
    """Surface one precomputed eval from the task output as an AX evaluator column.

    See the inline comment on :meth:`evaluate` for the Flight-NULL / OTLP-NaN
    null-handling rationale.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def evaluate(self, *, output: dict[str, Any] | None = None, **kwargs: Any) -> EvaluationResult:
        evals = output.get('evals') if isinstance(output, dict) else None
        entry = evals.get(self._name) if isinstance(evals, dict) else None
        entry = entry if isinstance(entry, dict) else {}
        score = entry.get('score')
        label = entry.get('label')
        explanation = entry.get('explanation')
        # experiments.run materializes eval.<name>.{score,label,explanation} columns from
        # every EvaluationResult onto TWO sinks with conflicting null rules: the Flight
        # results upload rejects a pyarrow NULL-type column ("unsupported cast from null to
        # <type>"), while the OTLP evaluation-SPAN export rejects NaN ("unsupported value:
        # NaN", which can poison the whole span batch). So a missing/absent metric (an
        # evaluator whose name is not in THIS run's output — the rare residual of a policy
        # pruned for all run scenarios) scores 0.0: a FINITE float that BOTH accept (typed,
        # non-null, non-NaN). The non-empty 'absent' label disambiguates it from a real
        # zero in the grid. label/explanation are non-empty strings (EvaluationResult.
        # __post_init__ coerces '' back to None, so empty is not enough).
        #
        # Hardening: only a numeric, FINITE value is accepted as a score. ``bool`` is an
        # int subtype in Python but not a meaningful score — treat it as missing so it
        # doesn't accidentally become 0.0 (False) or 1.0 (True) without being explicit.
        # Non-numeric types (str, list, …) and non-finite floats (NaN, ±inf) are treated
        # as missing; all map to score=0.0 with label='absent'.
        score_missing = not (isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score))
        if score_missing:
            # Force the 'absent' label even when the producer supplied one: a 0.0
            # substituted for an unusable score must not masquerade as a real value.
            score = 0.0
            label_str = 'absent'
        else:
            label_str = str(label).strip()[:100] if label is not None else ''
            if not label_str:
                label_str = str(score)[:100]
        explanation_str = str(explanation).strip()[:10_000] if explanation is not None else ''
        if not explanation_str:
            explanation_str = label_str
        return EvaluationResult(score=score, label=label_str, explanation=explanation_str)


def build_passthrough_evaluators(policy_names: list[str]) -> list[CodeEvaluator]:
    """Build one passthrough evaluator per (sanitized) policy name ∪ every non-conditional rollup.

    Dedups sanitized collisions (so two names that sanitize the same share one column),
    and returns one instance each — the evaluator set the experiment runs against.

    The set is the always-present rollups PLUS the two verdict axes. The axes are declared even
    though they are emitted only when the backend published them: an undeclared column simply does
    not exist in the grid, so leaving them out would delete "did it pass" from every experiment,
    including fully-graded ones. A row without them renders ``label='absent'`` (see
    :meth:`PassthroughEvaluator.evaluate`), which is distinguishable from a real verdict. The
    CONDITIONAL dimension rollups stay out: they are all-absent on any run without that
    dimension, which is a column of noise rather than a column with holes.
    """
    names: list[str] = []
    seen: set[str] = set()
    for raw in (*policy_names, *ALWAYS_PRESENT_ROLLUP_NAMES, *VERDICT_ROLLUP_NAMES):
        n = _safe_eval_name(raw)
        if n not in seen:
            seen.add(n)
            names.append(n)
    return [PassthroughEvaluator(n) for n in names]
