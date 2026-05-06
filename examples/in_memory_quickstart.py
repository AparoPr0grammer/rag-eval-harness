"""In-memory quickstart: a tiny RAG pipeline with no external dependencies.

Run with:

    ANTHROPIC_API_KEY=sk-... python examples/in_memory_quickstart.py

This script runs a 4-document, 4-question pipeline end-to-end and writes
``run.json`` and ``report.html`` to the current directory.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from anthropic import AsyncAnthropic

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_eval_harness import (  # noqa: E402
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    CostEvaluator,
    DatasetItem,
    Document,
    FaithfulnessEvaluator,
    GenerationResult,
    InMemoryRetriever,
    LatencyEvaluator,
    PipelineMetadata,
    PipelineRunner,
    RunnerOptions,
    RunSummary,
    render_html_report,
)

CORPUS = [
    Document(
        id="doc-1",
        content=(
            "The Sydney Harbour Bridge opened on 19 March 1932. It is a steel "
            "through-arch bridge across Sydney Harbour, carrying rail, vehicular, "
            "bicycle and pedestrian traffic between the Sydney central business "
            "district and the North Shore."
        ),
    ),
    Document(
        id="doc-2",
        content=(
            "The Great Barrier Reef is the world's largest coral reef system, "
            "composed of over 2,900 individual reefs and 900 islands stretching "
            "for over 2,300 kilometres along the coast of Queensland, Australia."
        ),
    ),
    Document(
        id="doc-3",
        content=(
            "Uluru, also known as Ayers Rock, is a large sandstone monolith in "
            "central Australia. It rises 348 metres above the surrounding plain "
            "and has a circumference of 9.4 kilometres. It is sacred to the "
            "Anangu people."
        ),
    ),
    Document(
        id="doc-4",
        content=(
            "The koala (Phascolarctos cinereus) is an arboreal herbivorous "
            "marsupial native to Australia. Koalas typically inhabit open eucalypt "
            "woodlands, and the leaves of these trees make up most of their diet."
        ),
    ),
]


DATASET = [
    DatasetItem(
        id="q1",
        question="When did the Sydney Harbour Bridge open?",
        expected_answer="19 March 1932",
        expected_context_ids=["doc-1"],
    ),
    DatasetItem(
        id="q2",
        question="How tall is Uluru?",
        expected_answer="348 metres",
        expected_context_ids=["doc-3"],
    ),
    DatasetItem(
        id="q3",
        question="What do koalas eat?",
        expected_answer="eucalypt leaves",
        expected_context_ids=["doc-4"],
    ),
    DatasetItem(
        id="q4",
        question="How long is the Great Barrier Reef?",
        expected_answer="over 2,300 kilometres",
        expected_context_ids=["doc-2"],
    ),
]


class ClaudeGenerator:
    """A minimal Generator that calls Claude with a context-grounded prompt."""

    name = "claude-opus-4-7"

    def __init__(self) -> None:
        self._client = AsyncAnthropic()

    async def generate(
        self, query: str, contexts: Sequence[Document]
    ) -> GenerationResult:
        block = "\n\n".join(f"[{i + 1}] {c.content}" for i, c in enumerate(contexts))
        prompt = (
            "You are answering questions using the provided context. If the "
            "context does not contain the answer, say so. Be concise — a single "
            "sentence is ideal.\n\n"
            f"Context:\n{block}\n\nQuestion: {query}"
        )

        start = time.perf_counter()
        response = await self._client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(b.text for b in response.content if b.type == "text")
        # Opus 4.7 published pricing: $5 / 1M input, $25 / 1M output.
        cost_usd = (
            response.usage.input_tokens * 5 / 1_000_000
            + response.usage.output_tokens * 25 / 1_000_000
        )

        return GenerationResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this example.", file=sys.stderr)
        sys.exit(1)

    retriever = InMemoryRetriever(CORPUS)
    generator = ClaudeGenerator()

    def on_progress(done: int, total: int) -> None:
        print(f"  Progress: {done}/{total}")

    runner = PipelineRunner(
        retriever,
        generator,
        RunnerOptions(top_k=2, concurrency=2, on_progress=on_progress),
    )

    print("Running pipeline...")
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    results = await runner.run(DATASET)
    print("Pipeline complete.\n")

    print("Running evaluators...")
    evaluators = [
        FaithfulnessEvaluator(),
        ContextPrecisionEvaluator(),
        ContextRecallEvaluator(),
        LatencyEvaluator(),
        CostEvaluator(),
    ]
    reports = []
    for evaluator in evaluators:
        print(f"  {evaluator.name}...")
        reports.append(await evaluator.evaluate(results))

    summary = RunSummary(
        pipeline=PipelineMetadata(
            name="in-memory-bm25 + claude-opus-4-7",
            description="BM25 retrieval over a tiny corpus of Australian landmarks",
        ),
        started_at=started_at,
        completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        results=list(results),
        reports=reports,
    )

    print("\n=== Aggregate scores ===")
    for report in reports:
        print(f"\n[{report.evaluator}]")
        for score in report.aggregate:
            print(f"  {score.metric}: {_fmt(score.value)}")

    summary_dict = asdict(summary)
    Path("run.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    Path("report.html").write_text(
        render_html_report([summary_dict], title="In-Memory Quickstart"),
        encoding="utf-8",
    )
    print("\nWrote run.json and report.html in the current directory.")


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


if __name__ == "__main__":
    asyncio.run(main())
