"""Core type definitions for the RAG evaluation harness.

The five protocols and dataclasses below are the entire surface a pipeline
author needs to understand. Implement :class:`Retriever` and
:class:`Generator`, build a sequence of :class:`DatasetItem`, and the harness
handles orchestration and scoring.

Why ``Protocol`` instead of an abstract base class: structural typing keeps
adapters thin. Any class with the right shape works — no need to inherit from
or import a base. This matters because in production RAG codebases the
retriever is usually a thin wrapper around a vendor SDK that already exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    content: str
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    document: Document
    score: float


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


@dataclass(frozen=True, slots=True)
class DatasetItem:
    id: str
    question: str
    expected_answer: str | None = None
    expected_context_ids: Sequence[str] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    """One row in a pipeline run.

    A failed item is recorded with ``error`` set and empty retrieval/generation
    fields, rather than aborting the run. This keeps a single bad query from
    invalidating the whole evaluation.
    """

    item_id: str
    question: str
    retrieved: Sequence[RetrievalResult]
    generation: GenerationResult
    retrieval_latency_ms: float
    total_latency_ms: float
    expected_answer: str | None = None
    expected_context_ids: Sequence[str] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ItemScore:
    item_id: str
    value: float
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AggregateScore:
    metric: str
    value: float
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvaluatorReport:
    evaluator: str
    per_item: Sequence[ItemScore]
    aggregate: Sequence[AggregateScore]


@dataclass(frozen=True, slots=True)
class PipelineMetadata:
    name: str
    description: str | None = None
    tags: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Self-contained record of one evaluation run.

    Serialised to JSON via ``dataclasses.asdict`` and consumed by the HTML
    report renderer or the dashboard.
    """

    pipeline: PipelineMetadata
    started_at: str
    completed_at: str
    results: Sequence[RunResult]
    reports: Sequence[EvaluatorReport]


class Retriever(Protocol):
    """Anything that turns a query string into ranked documents."""

    name: str

    async def retrieve(self, query: str, top_k: int) -> Sequence[RetrievalResult]: ...


class Generator(Protocol):
    """Anything that produces an answer given a question and retrieved context."""

    name: str

    async def generate(
        self, query: str, contexts: Sequence[Document]
    ) -> GenerationResult: ...


class Evaluator(Protocol):
    """Scores a set of run results.

    Each evaluator returns per-item scores plus aggregate metrics. Per-item
    scores let you drill into individual failures; aggregates let you compare
    pipelines.
    """

    name: str

    async def evaluate(self, results: Sequence[RunResult]) -> EvaluatorReport: ...
