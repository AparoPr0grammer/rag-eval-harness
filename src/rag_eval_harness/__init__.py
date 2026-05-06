"""rag-eval-harness — evaluate RAG pipelines like any other production system."""

from .core.types import (
    AggregateScore,
    DatasetItem,
    Document,
    Evaluator,
    EvaluatorReport,
    GenerationResult,
    Generator,
    ItemScore,
    PipelineMetadata,
    RetrievalResult,
    Retriever,
    RunResult,
    RunSummary,
)
from .dataset.jsonl import load_jsonl_dataset, write_jsonl_dataset
from .evaluators import (
    ContextPrecisionEvaluator,
    ContextPrecisionOptions,
    ContextRecallEvaluator,
    CostEvaluator,
    FaithfulnessEvaluator,
    FaithfulnessOptions,
    LatencyEvaluator,
)
from .judges.claude_judge import ClaudeJudge, JudgeResult
from .orchestrator.runner import PipelineRunner, RunnerOptions
from .report.html_report import render_html_report
from .retrievers.in_memory import InMemoryRetriever
from .synthetic.generator import (
    SyntheticGeneratorOptions,
    SyntheticItem,
    SyntheticTestSetGenerator,
)

__all__ = [
    "AggregateScore",
    "ClaudeJudge",
    "ContextPrecisionEvaluator",
    "ContextPrecisionOptions",
    "ContextRecallEvaluator",
    "CostEvaluator",
    "DatasetItem",
    "Document",
    "Evaluator",
    "EvaluatorReport",
    "FaithfulnessEvaluator",
    "FaithfulnessOptions",
    "GenerationResult",
    "Generator",
    "InMemoryRetriever",
    "ItemScore",
    "JudgeResult",
    "LatencyEvaluator",
    "PipelineMetadata",
    "PipelineRunner",
    "RetrievalResult",
    "Retriever",
    "RunnerOptions",
    "RunResult",
    "RunSummary",
    "SyntheticGeneratorOptions",
    "SyntheticItem",
    "SyntheticTestSetGenerator",
    "load_jsonl_dataset",
    "render_html_report",
    "write_jsonl_dataset",
]

__version__ = "0.1.0"
