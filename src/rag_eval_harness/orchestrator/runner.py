"""Pipeline orchestrator.

Drives a (Retriever, Generator) pair across a dataset with bounded
concurrency, measures retrieval and generation latency separately, and
catches per-item exceptions so one bad query can't tank a 1000-item run.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..core.types import (
    DatasetItem,
    GenerationResult,
    Generator,
    Retriever,
    RunResult,
)


@dataclass(frozen=True, slots=True)
class RunnerOptions:
    top_k: int
    concurrency: int = 4
    on_progress: Callable[[int, int], None] | None = None


class PipelineRunner:
    """Run a (Retriever, Generator) pipeline across a dataset.

    Concurrency is bounded by an :class:`asyncio.Semaphore`. Retrieval and
    generation are timed independently so the latency evaluator can attribute
    slowdowns to the right side of the pipeline.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        options: RunnerOptions,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._options = options

    async def run(self, dataset: Sequence[DatasetItem]) -> list[RunResult]:
        sem = asyncio.Semaphore(max(1, self._options.concurrency))
        completed = 0
        total = len(dataset)
        on_progress = self._options.on_progress

        async def run_with_progress(item: DatasetItem) -> RunResult:
            nonlocal completed
            async with sem:
                result = await self._run_one(item)
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total)
                return result

        return list(await asyncio.gather(*(run_with_progress(item) for item in dataset)))

    async def _run_one(self, item: DatasetItem) -> RunResult:
        start = time.perf_counter()
        try:
            retrieval_start = time.perf_counter()
            retrieved = await self._retriever.retrieve(item.question, self._options.top_k)
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

            generation = await self._generator.generate(
                item.question, [r.document for r in retrieved]
            )

            return RunResult(
                item_id=item.id,
                question=item.question,
                expected_answer=item.expected_answer,
                expected_context_ids=item.expected_context_ids,
                retrieved=list(retrieved),
                generation=generation,
                retrieval_latency_ms=retrieval_latency_ms,
                total_latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 — broad catch is intentional here
            return RunResult(
                item_id=item.id,
                question=item.question,
                expected_answer=item.expected_answer,
                expected_context_ids=item.expected_context_ids,
                retrieved=[],
                generation=GenerationResult(
                    text="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0,
                    cost_usd=0,
                ),
                retrieval_latency_ms=0,
                total_latency_ms=(time.perf_counter() - start) * 1000,
                error=str(exc),
            )
