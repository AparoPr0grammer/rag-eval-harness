"""Latency evaluator.

Operates on the timings already captured by :class:`PipelineRunner` — no LLM
calls, no extra cost. Reports p50/p95/p99 separately for retrieval,
generation, and end-to-end so you can attribute slowdowns correctly.

Note: percentiles on small datasets are noisy. ``p99`` of 50 items is
effectively "the worst single observation". Treat absolute numbers with
appropriate sample-size scepticism; deltas across runs are still meaningful.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..core.types import AggregateScore, EvaluatorReport, ItemScore, RunResult


class LatencyEvaluator:
    name = "latency"

    async def evaluate(self, results: Sequence[RunResult]) -> EvaluatorReport:
        per_item: list[ItemScore] = [
            ItemScore(
                item_id=r.item_id,
                value=r.total_latency_ms,
                details={
                    "retrieval_latency_ms": r.retrieval_latency_ms,
                    "generation_latency_ms": r.generation.latency_ms,
                },
            )
            for r in results
        ]

        totals = [r.total_latency_ms for r in results if not math.isnan(r.total_latency_ms)]
        retrieval = [
            r.retrieval_latency_ms for r in results if not math.isnan(r.retrieval_latency_ms)
        ]
        generation = [
            r.generation.latency_ms for r in results if not math.isnan(r.generation.latency_ms)
        ]

        aggregate = [
            AggregateScore(metric="latency_p50_ms", value=_percentile(totals, 0.5)),
            AggregateScore(metric="latency_p95_ms", value=_percentile(totals, 0.95)),
            AggregateScore(metric="latency_p99_ms", value=_percentile(totals, 0.99)),
            AggregateScore(metric="latency_mean_ms", value=_mean(totals)),
            AggregateScore(metric="retrieval_p50_ms", value=_percentile(retrieval, 0.5)),
            AggregateScore(metric="generation_p50_ms", value=_percentile(generation, 0.5)),
        ]
        return EvaluatorReport(evaluator=self.name, per_item=per_item, aggregate=aggregate)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(p * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
