"""Faithfulness evaluator.

Faithfulness measures whether a generated answer is grounded in the retrieved
context. The standard naive approach — "ask an LLM if the answer is faithful"
— is too coarse: a long answer with one wrong claim gets the same binary score
as a fully fabricated one.

Instead we use **claim decomposition** (RAGAS-style):

1. Break the answer into atomic factual claims.
2. For each claim, ask the judge whether the context entails it.
3. Score = supported / total.

This gives a fractional score that reflects partial faithfulness, surfaces
exactly which claims are unsupported (in ``per_item.details.claims``), and is
less sensitive to answer length than a single yes/no.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.types import AggregateScore, EvaluatorReport, ItemScore, RunResult
from ..judges.claude_judge import ClaudeJudge

SYSTEM_PROMPT = """You are an expert evaluator assessing the faithfulness of an answer with respect to provided context.

Your task: decompose the answer into atomic factual claims, then judge whether each claim is directly supported by the context.

Definitions:
- An "atomic claim" is a single, indivisible factual statement.
- A claim is SUPPORTED only if the context directly entails it. Inference beyond the context, speculation, or general knowledge does NOT count as supported.
- Ignore stylistic phrasing; focus on factual content.

Output strict JSON in exactly this format (no markdown, no commentary):
{
  "claims": [
    { "claim": "...", "supported": true|false, "reason": "..." }
  ]
}

If the answer contains no factual claims (e.g. only a refusal or clarifying question), return {"claims": []}."""


@dataclass(frozen=True, slots=True)
class FaithfulnessOptions:
    judge: ClaudeJudge | None = None


class FaithfulnessEvaluator:
    name = "faithfulness"

    def __init__(self, options: FaithfulnessOptions | None = None) -> None:
        opts = options or FaithfulnessOptions()
        self._judge = opts.judge or ClaudeJudge()

    async def evaluate(self, results: Sequence[RunResult]) -> EvaluatorReport:
        per_item: list[ItemScore] = []
        for result in results:
            per_item.append(await self._score_one(result))

        valid = [s for s in per_item if not math.isnan(s.value)]
        mean_value = sum(s.value for s in valid) / len(valid) if valid else 0.0
        aggregate = [
            AggregateScore(
                metric="faithfulness_mean",
                value=mean_value,
                details={"sampled": len(valid), "total": len(results)},
            )
        ]
        return EvaluatorReport(evaluator=self.name, per_item=per_item, aggregate=aggregate)

    async def _score_one(self, result: RunResult) -> ItemScore:
        if result.error or not result.generation.text.strip():
            return ItemScore(
                item_id=result.item_id,
                value=0.0,
                details={"reason": result.error or "empty_answer"},
            )

        context_text = (
            "\n\n".join(
                f"[Context {i + 1}]\n{r.document.content}"
                for i, r in enumerate(result.retrieved)
            )
            or "(no context retrieved)"
        )

        user_prompt = (
            f"Question: {result.question}\n\n"
            f"Answer: {result.generation.text}\n\n"
            f"Context:\n{context_text}\n\n"
            "Return JSON only."
        )

        judged = await self._judge.judge(SYSTEM_PROMPT, user_prompt)
        parsed = _parse_judge_output(judged.text)

        if parsed is None:
            return ItemScore(
                item_id=result.item_id,
                value=float("nan"),
                details={"reason": "parse_failed", "raw": judged.text},
            )

        if not parsed["claims"]:
            # No factual claims to verify (refusal, clarifying question, etc.).
            # Following RAGAS convention: score as 1 (no unsupported claims).
            return ItemScore(
                item_id=result.item_id,
                value=1.0,
                details={"reason": "no_factual_claims"},
            )

        supported = sum(1 for c in parsed["claims"] if c.get("supported"))
        score = supported / len(parsed["claims"])
        return ItemScore(
            item_id=result.item_id,
            value=score,
            details={
                "total_claims": len(parsed["claims"]),
                "supported_claims": supported,
                "claims": parsed["claims"],
            },
        )


def _parse_judge_output(raw: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "claims" not in obj:
        return None
    claims = obj["claims"]
    if not isinstance(claims, list):
        return None
    valid_claims = [
        c
        for c in claims
        if isinstance(c, dict)
        and isinstance(c.get("claim"), str)
        and isinstance(c.get("supported"), bool)
    ]
    return {"claims": valid_claims}
