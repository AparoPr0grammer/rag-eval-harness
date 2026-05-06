"""Smoke tests for core building blocks (no Claude API calls)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from rag_eval_harness import (
    ContextRecallEvaluator,
    DatasetItem,
    Document,
    GenerationResult,
    InMemoryRetriever,
    LatencyEvaluator,
    PipelineRunner,
    RetrievalResult,
    RunnerOptions,
    RunResult,
    load_jsonl_dataset,
    write_jsonl_dataset,
)

DOCS = [
    Document(id="a", content="The Sydney Harbour Bridge opened in 1932."),
    Document(id="b", content="The Eiffel Tower is in Paris and is 330 metres tall."),
    Document(id="c", content="Mount Everest is the tallest mountain on Earth."),
]


class StubGenerator:
    name = "stub"

    async def generate(self, query: str, contexts: Sequence[Document]) -> GenerationResult:
        return GenerationResult(
            text=f"Answer to: {query} (using {len(contexts)} contexts)",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            cost_usd=0.0001,
        )


@pytest.mark.asyncio
async def test_in_memory_retriever_returns_relevant_documents() -> None:
    retriever = InMemoryRetriever(DOCS)
    results = await retriever.retrieve("Sydney Harbour Bridge", 2)
    assert len(results) > 0
    assert results[0].document.id == "a"


@pytest.mark.asyncio
async def test_in_memory_retriever_respects_top_k() -> None:
    retriever = InMemoryRetriever(DOCS)
    results = await retriever.retrieve("Sydney Harbour Bridge", 1)
    assert len(results) <= 1


@pytest.mark.asyncio
async def test_pipeline_runner_produces_one_result_per_item() -> None:
    dataset = [
        DatasetItem(id="1", question="When did the Sydney Harbour Bridge open?"),
        DatasetItem(id="2", question="How tall is the Eiffel Tower?"),
    ]
    runner = PipelineRunner(
        InMemoryRetriever(DOCS),
        StubGenerator(),
        RunnerOptions(top_k=2, concurrency=1),
    )
    results = await runner.run(dataset)
    assert len(results) == 2
    assert {r.item_id for r in results} == {"1", "2"}


@pytest.mark.asyncio
async def test_pipeline_runner_records_per_item_errors() -> None:
    class ExplodingRetriever:
        name = "exploding"

        async def retrieve(self, query: str, top_k: int) -> Sequence[RetrievalResult]:
            raise RuntimeError("kaboom")

    runner = PipelineRunner(
        ExplodingRetriever(),
        StubGenerator(),
        RunnerOptions(top_k=2, concurrency=1),
    )
    [result] = await runner.run([DatasetItem(id="1", question="Anything?")])
    assert result.error == "kaboom"
    assert result.retrieved == []


@pytest.mark.asyncio
async def test_latency_evaluator_computes_percentiles() -> None:
    evaluator = LatencyEvaluator()
    results = [
        RunResult(
            item_id="1",
            question="q",
            retrieved=[],
            generation=GenerationResult(
                text="", input_tokens=0, output_tokens=0, latency_ms=100, cost_usd=0
            ),
            retrieval_latency_ms=50,
            total_latency_ms=150,
        ),
        RunResult(
            item_id="2",
            question="q",
            retrieved=[],
            generation=GenerationResult(
                text="", input_tokens=0, output_tokens=0, latency_ms=200, cost_usd=0
            ),
            retrieval_latency_ms=100,
            total_latency_ms=300,
        ),
    ]
    report = await evaluator.evaluate(results)
    p50 = next(a for a in report.aggregate if a.metric == "latency_p50_ms")
    assert p50.value > 0


@pytest.mark.asyncio
async def test_context_recall_evaluator_measures_gold_coverage() -> None:
    evaluator = ContextRecallEvaluator()
    results = [
        RunResult(
            item_id="1",
            question="q",
            expected_context_ids=["a", "b"],
            retrieved=[
                RetrievalResult(document=DOCS[0], score=1),
                RetrievalResult(document=DOCS[2], score=0.5),
            ],
            generation=GenerationResult(
                text="", input_tokens=0, output_tokens=0, latency_ms=0, cost_usd=0
            ),
            retrieval_latency_ms=0,
            total_latency_ms=0,
        ),
    ]
    report = await evaluator.evaluate(results)
    assert math.isclose(report.per_item[0].value, 0.5)


def test_jsonl_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "ds.jsonl"
    items = [
        DatasetItem(id="1", question="Q?", expected_answer="A", expected_context_ids=["x"]),
        DatasetItem(id="2", question="Q2?"),
    ]
    write_jsonl_dataset(items, path)
    loaded = load_jsonl_dataset(path)
    assert [i.id for i in loaded] == ["1", "2"]
    assert loaded[0].expected_answer == "A"
    assert list(loaded[0].expected_context_ids or []) == ["x"]
