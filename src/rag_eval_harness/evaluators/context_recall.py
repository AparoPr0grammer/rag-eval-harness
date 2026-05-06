"""Context recall evaluator.

Of the gold (known-relevant) contexts, how many did the retriever actually
surface? Requires ``expected_context_ids`` on each dataset item — items
without gold contexts are excluded from the mean (NaN per-item score).

This is the cheapest and most decisive retrieval metric: no LLM judge, no
embedding similarity, just set membership. If recall is low, you have a
retrieval problem; no amount of generator tuning will fix it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..core.types import AggregateScore, EvaluatorReport, ItemScore, RunResult


class ContextRecallEvaluator:
    name = "context_recall"

    async def evaluate(self, results: Sequence[RunResult]) -> EvaluatorReport:
        per_item: list[ItemScore] = [self._score_one(r) for r in results]

        valid = [s for s in per_item if not math.isnan(s.value)]
        mean_value = sum(s.value for s in valid) / len(valid) if valid else 0.0
        return EvaluatorReport(
            evaluator=self.name,
            per_item=per_item,
            aggregate=[
                AggregateScore(
                    metric="context_recall_mean",
                    value=mean_value,
                    details={"sampled": len(valid), "total": len(results)},
                )
            ],
        )

    @staticmethod
    def _score_one(result: RunResult) -> ItemScore:
        gold = result.expected_context_ids
        if not gold:
            return ItemScore(
                item_id=result.item_id,
                value=float("nan"),
                details={"reason": "no_gold_contexts"},
            )
        retrieved_ids = {r.document.id for r in result.retrieved}
        found = sum(1 for cid in gold if cid in retrieved_ids)
        recall = found / len(gold)
        return ItemScore(
            item_id=result.item_id,
            value=recall,
            details={
                "gold_count": len(gold),
                "found_count": found,
                "retrieved_count": len(result.retrieved),
            },
        )
