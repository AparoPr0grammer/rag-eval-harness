"""Context precision evaluator.

Measures the *ranking quality* of the retriever, not just the binary "did it
find anything relevant". A retriever that returns 1 relevant doc at position 1
should score higher than one that returns the same doc at position 5 with
four irrelevant docs above it.

Implementation follows RAGAS:

1. For each retrieved passage, ask the judge whether it's relevant.
2. Compute precision@k for every position k.
3. Sum precision@k * relevance_at_k, divide by total relevant.

The result is mean-average-precision-style: 1.0 means every retrieved doc is
relevant; 0.0 means none are; intermediate values reflect both how many were
relevant *and* whether they appeared near the top.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.types import AggregateScore, EvaluatorReport, ItemScore, RunResult
from ..judges.claude_judge import ClaudeJudge

SYSTEM_PROMPT = """You judge whether a single retrieved passage is useful for answering a question.

Definitions:
- A passage is RELEVANT if it contains information that helps answer the question (even if partial).
- A passage is NOT RELEVANT if the information is unrelated, off-topic, or only superficially mentions the topic.

Output strict JSON only (no markdown, no commentary):
{ "relevant": true|false, "reason": "..." }"""


@dataclass(frozen=True, slots=True)
class ContextPrecisionOptions:
    judge: ClaudeJudge | None = None
    concurrency: int = 4


class ContextPrecisionEvaluator:
    name = "context_precision"

    def __init__(self, options: ContextPrecisionOptions | None = None) -> None:
        opts = options or ContextPrecisionOptions()
        self._judge = opts.judge or ClaudeJudge()
        self._concurrency = max(1, opts.concurrency)

    async def evaluate(self, results: Sequence[RunResult]) -> EvaluatorReport:
        per_item: list[ItemScore] = []
        for result in results:
            per_item.append(await self._score_one(result))

        valid = [s for s in per_item if not math.isnan(s.value)]
        mean_value = sum(s.value for s in valid) / len(valid) if valid else 0.0
        return EvaluatorReport(
            evaluator=self.name,
            per_item=per_item,
            aggregate=[
                AggregateScore(
                    metric="context_precision_mean",
                    value=mean_value,
                    details={"sampled": len(valid), "total": len(results)},
                )
            ],
        )

    async def _score_one(self, result: RunResult) -> ItemScore:
        if not result.retrieved:
            return ItemScore(
                item_id=result.item_id,
                value=0.0,
                details={"reason": "no_retrieved"},
            )

        sem = asyncio.Semaphore(self._concurrency)

        async def judge_one(content: str) -> bool:
            async with sem:
                user_prompt = (
                    f"Question: {result.question}\n\n"
                    f"Passage:\n{content}\n\n"
                    "Return JSON only."
                )
                judged = await self._judge.judge(SYSTEM_PROMPT, user_prompt)
                return _parse_relevance(judged.text)

        judgements = await asyncio.gather(
            *(judge_one(r.document.content) for r in result.retrieved)
        )

        relevant_so_far = 0
        weighted_sum = 0.0
        for idx, rel in enumerate(judgements):
            if rel:
                relevant_so_far += 1
                precision_at_k = relevant_so_far / (idx + 1)
                weighted_sum += precision_at_k

        total_relevant = sum(1 for r in judgements if r)
        score = weighted_sum / total_relevant if total_relevant > 0 else 0.0

        return ItemScore(
            item_id=result.item_id,
            value=score,
            details={
                "judgements": list(judgements),
                "total_retrieved": len(result.retrieved),
                "total_relevant": total_relevant,
            },
        )


def _parse_relevance(raw: str) -> bool:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return False
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False
    return bool(obj.get("relevant")) if isinstance(obj, dict) else False
