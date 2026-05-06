"""Cost evaluator.

Reports what your pipeline costs to *serve* a query in production: total spend
across the run, mean per-query cost, p95 cost, and token totals.

Cost values come from the :class:`GenerationResult` fields the user populates
in their ``Generator``. The harness does not infer cost — your generator
knows the model and pricing it's using.

Judge cost (the cost of running the evaluators themselves) is not included.
That's an evaluation-time expense, not a per-query one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..core.types import AggregateScore, EvaluatorReport, ItemScore, RunResult


class CostEvaluator:
    name = "cost"

    async def evaluate(self, results: Sequence[RunResult]) -> EvaluatorReport:
        per_item: list[ItemScore] = [
            ItemScore(
                item_id=r.item_id,
                value=r.generation.cost_usd,
                details={
                    "input_tokens": r.generation.input_tokens,
                    "output_tokens": r.generation.output_tokens,
                },
            )
            for r in results
        ]

        costs = [r.generation.cost_usd for r in results if not math.isnan(r.generation.cost_usd)]
        inputs = [r.generation.input_tokens for r in results]
        outputs = [r.generation.output_tokens for r in results]

        aggregate = [
            AggregateScore(metric="cost_usd_total", value=sum(costs)),
            AggregateScore(metric="cost_usd_mean", value=_mean(costs)),
            AggregateScore(metric="cost_usd_p95", value=_percentile(costs, 0.95)),
            AggregateScore(metric="input_tokens_total", value=float(sum(inputs))),
            AggregateScore(metric="output_tokens_total", value=float(sum(outputs))),
        ]
        return EvaluatorReport(evaluator=self.name, per_item=per_item, aggregate=aggregate)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(p * len(sorted_vals)) - 1))
    return sorted_vals[idx]
