# rag-eval-harness

> Evaluate your RAG pipeline like you'd evaluate any other production system.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Type checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue)](https://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen)](tests/)

## Project brief

| Aspect | Detail |
| --- | --- |
| **Problem** | RAG systems regress silently. A change to chunk size, retriever, prompt, or model can degrade faithfulness or push p95 latency past SLO without anyone noticing until users complain. There's no equivalent of "run the test suite" for RAG. |
| **Goal** | A reproducible, opinionated harness that scores any RAG pipeline on the four dimensions that matter — retrieval quality, generation faithfulness, latency, and cost — so changes can be evaluated before they ship. |
| **In scope** | Pluggable retriever/generator interfaces · five evaluators (faithfulness, context precision, context recall, latency, cost) · synthetic Q/A test set generation from markdown · LLM-as-judge with prompt caching · HTML report with side-by-side pipeline comparison · Next.js dashboard for run history. |
| **Out of scope** | Hosting your retriever or LLM. Embedding models. Vector store provisioning. The harness measures pipelines; it doesn't run production traffic. |
| **Audience** | Teams shipping RAG to prod that want a measurement layer, and engineers who want to compare retriever variants (BM25 vs hybrid vs dense) with real numbers. |
| **Tech stack** | Python 3.11+ for the harness (`asyncio`, frozen dataclasses, `typing.Protocol`); TypeScript + Next.js 14 for the dashboard; Anthropic Claude as the default judge. Strict mypy, ruff, pytest. |

## The problem

Most teams ship retrieval-augmented generation (RAG) without measuring it. Demos pass, vibes are good, the system goes to prod — and only then does it become clear that the retriever returns the wrong document 30% of the time, or the generator confidently fabricates citations, or one prompt change made p95 latency spike from 800 ms to 4 seconds.

The reason is straightforward: evaluating RAG well requires more than a unit test. You need to measure four orthogonal things at once — **retrieval quality**, **generation faithfulness**, **answer relevance**, and **operational cost** — across enough test items that the numbers are stable, and you need to do it every time you change a chunk size, a re-ranker, or a model.

`rag-eval-harness` is an opinionated, pluggable framework for doing exactly that. Bring your own retriever, your own generator, and your own dataset; it handles orchestration, scoring, and reporting.

## What it does

- **Pluggable architecture.** Implement two protocols — `Retriever` and `Generator` — and the harness will run them across any dataset. No abstract base classes, no inheritance, just structural typing.
- **Five built-in evaluators.**
  - **Faithfulness** — decomposes the generated answer into atomic claims and uses Claude as a judge to verify each claim against the retrieved context.
  - **Context precision** — RAGAS-style mean average precision: how relevant are the retrieved passages, weighted by rank?
  - **Context recall** — of the known-good contexts, how many did your retriever actually surface?
  - **Latency** — p50, p95, p99 across retrieval, generation, and end-to-end.
  - **Cost** — per-query and aggregate USD spend, plus token totals.
- **Synthetic test set generator.** Point it at a directory of markdown files; get back a JSONL of question / answer / context triples ready to evaluate against.
- **HTML reports.** Self-contained file with per-pipeline tables, headline stats, and a side-by-side comparison view across multiple pipelines (e.g. BM25 vs hybrid vs dense).
- **Next.js dashboard.** Browse run history visually, drill into individual evaluations, compare deltas across deployments. Reads the same JSON the Python harness emits — no shared runtime, no database.
- **Strict typing throughout.** Every public type is exported. Every default is documented. Frozen dataclasses, `typing.Protocol` for plug-in interfaces, mypy-clean. Australian English in prose; `synthesise`, not `synthesize`.

## Repository layout

```text
rag-eval-harness/
├── src/rag_eval_harness/   # The Python package — pipeline, evaluators, reports
├── examples/               # End-to-end runners (in-memory + Mongo Atlas)
├── tests/                  # pytest smoke tests (no API calls)
└── dashboard/              # Next.js app for browsing run history
```

## Install

```bash
pip install rag-eval-harness
```

You'll need an `ANTHROPIC_API_KEY` for the judge model (Claude is the default; see [Configuration](#configuration) for overrides).

For the Mongo Atlas example specifically:

```bash
pip install "rag-eval-harness[mongo]"
```

## 60-second quickstart

```bash
git clone https://github.com/yourname/rag-eval-harness.git
cd rag-eval-harness
pip install -e ".[dev]"
ANTHROPIC_API_KEY=sk-ant-... python examples/in_memory_quickstart.py
open report.html
```

The quickstart runs a tiny BM25 retriever over four documents about Australian landmarks, generates answers with Claude, scores them with all five evaluators, and writes an HTML report. No external services required.

### What it looks like

```text
Running pipeline...
  Progress: 1/4
  Progress: 2/4
  Progress: 3/4
  Progress: 4/4
Pipeline complete.

Running evaluators...
  faithfulness...
  context_precision...
  context_recall...
  latency...
  cost...

=== Aggregate scores ===

[faithfulness]
  faithfulness_mean: 1.000

[context_precision]
  context_precision_mean: 1.000

[context_recall]
  context_recall_mean: 1.000

[latency]
  latency_p50_ms: 1247
  latency_p95_ms: 2103
  latency_p99_ms: 2103
  latency_mean_ms: 1.524
  retrieval_p50_ms: 0.0023
  generation_p50_ms: 1245

[cost]
  cost_usd_total: 0.0142
  cost_usd_mean: 0.0036
  cost_usd_p95: 0.0048
  input_tokens_total: 412
  output_tokens_total: 187

Wrote run.json and report.html in the current directory.
```

The HTML report adds: a comparison table when multiple pipelines are passed, headline-stat cards per evaluator, full aggregate tables, and a collapsible drill-down with per-item scores. The dashboard adds run history across time.

## Worked example: Mongo Atlas Vector Search

A real RAG pipeline usually looks like *embed-and-search* over a vector database. Here's how to wire `rag-eval-harness` to Mongo Atlas Vector Search and Claude:

```python
import asyncio, time
from collections.abc import Sequence

from anthropic import AsyncAnthropic
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from rag_eval_harness import (
    PipelineRunner, RunnerOptions,
    FaithfulnessEvaluator, ContextPrecisionEvaluator,
    ContextRecallEvaluator, LatencyEvaluator, CostEvaluator,
    Document, GenerationResult, RetrievalResult,
    load_jsonl_dataset,
)


# 1. Implement the Retriever protocol — query Mongo Atlas's $vectorSearch.
class MongoAtlasVectorRetriever:
    name = "mongo-atlas-vector"

    def __init__(self, mongo, openai, *, db, collection, index_name):
        self._collection = mongo[db][collection]
        self._openai = openai
        self._index_name = index_name

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        embed = await self._openai.embeddings.create(
            model="text-embedding-3-small", input=query,
        )
        cursor = self._collection.aggregate([
            {"$vectorSearch": {
                "index": self._index_name,
                "path": "embedding",
                "queryVector": embed.data[0].embedding,
                "numCandidates": top_k * 10,
                "limit": top_k,
            }},
            {"$project": {"_id": 1, "content": 1,
                          "score": {"$meta": "vectorSearchScore"}}},
        ])
        return [
            RetrievalResult(
                document=Document(id=str(d["_id"]), content=str(d["content"])),
                score=float(d.get("score", 0)),
            )
            async for d in cursor
        ]


# 2. Implement the Generator protocol — call Claude with the retrieved context.
class ClaudeGenerator:
    name = "claude-opus-4-7"

    def __init__(self):
        self._client = AsyncAnthropic()

    async def generate(self, query: str, contexts: Sequence[Document]) -> GenerationResult:
        block = "\n\n".join(f"[{i + 1}] {c.content}" for i, c in enumerate(contexts))
        start = time.perf_counter()
        response = await self._client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    "Answer using only the provided context.\n\n"
                    f"Context:\n{block}\n\nQuestion: {query}"
                ),
            }],
        )
        return GenerationResult(
            text="".join(b.text for b in response.content if b.type == "text"),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=(
                response.usage.input_tokens * 5 / 1_000_000
                + response.usage.output_tokens * 25 / 1_000_000
            ),
        )


# 3. Run the pipeline and the evaluators.
async def main():
    mongo = AsyncIOMotorClient("mongodb+srv://...")
    dataset = load_jsonl_dataset("dataset.jsonl")

    retriever = MongoAtlasVectorRetriever(
        mongo, AsyncOpenAI(),
        db="mydb", collection="docs", index_name="vector_index",
    )
    runner = PipelineRunner(retriever, ClaudeGenerator(),
                            RunnerOptions(top_k=5, concurrency=4))
    results = await runner.run(dataset)

    for evaluator in [
        FaithfulnessEvaluator(),
        ContextPrecisionEvaluator(),
        ContextRecallEvaluator(),
        LatencyEvaluator(),
        CostEvaluator(),
    ]:
        report = await evaluator.evaluate(results)
        for score in report.aggregate:
            print(f"{score.metric}: {score.value:.4f}")


asyncio.run(main())
```

That's the whole integration. The full runnable version (with config, error handling, and a CLI entry point) lives at [`examples/mongo_atlas_quickstart.py`](examples/mongo_atlas_quickstart.py).

### Setting up the Atlas index

Before running the example, you need a vector search index on the embedding field. In the Atlas UI:

1. Go to **Search → Create Search Index → Atlas Vector Search**.
2. Use this definition (assuming `text-embedding-3-small`):

   ```json
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 1536,
         "similarity": "cosine"
       }
     ]
   }
   ```

3. Name it `vector_index` (or set `MONGODB_VECTOR_INDEX` to your chosen name).

## Generating a test set

If you don't have a labelled dataset yet, the synthetic generator can produce one from your existing markdown corpus:

```bash
ANTHROPIC_API_KEY=sk-... rag-eval synthesise ./docs ./dataset.jsonl
```

Each line in `dataset.jsonl` is a `{question, expected_answer, context_id, context_content, source}` record. The generator chunks each markdown file into paragraphs (with configurable size limits) and uses Claude to draft a question whose answer is found verbatim in the chunk.

This is a *starting* dataset, not a final one. Synthetic Q/A is uneven; before relying on the numbers, eyeball a sample, prune nonsense, and consider augmenting with hand-written items for edge cases.

## Comparing pipelines

`render_html_report` accepts a list of summaries and emits a side-by-side comparison table. To compare BM25 vs hybrid vs dense:

```python
from dataclasses import asdict
from rag_eval_harness import render_html_report

summaries = [bm25_summary, hybrid_summary, dense_summary]
html = render_html_report([asdict(s) for s in summaries])

with open("comparison.html", "w") as f:
    f.write(html)
```

The HTML comparison table puts every metric across all pipelines side by side, so you can see at a glance which pipeline wins on faithfulness, which has the lowest p95 latency, and which costs the most per query.

## The dashboard

For ongoing work — tracking how a pipeline evolves over weeks, comparing branches, sharing results with non-technical stakeholders — there's a small Next.js dashboard at [`dashboard/`](dashboard/):

```bash
cd dashboard
npm install
mkdir -p runs
cp ../run.json runs/2026-05-06-baseline.json
npm run dev
```

It reads `RunSummary` JSON files from a directory, lists them on the homepage with headline metrics, and lets you drill into any individual run. No database, no auth — just a thin reader on top of the JSON the harness emits.

## Configuration

| Setting             | Where                                                                | Default            |
| ------------------- | -------------------------------------------------------------------- | ------------------ |
| Judge model         | `ClaudeJudge(model="...")`                                           | `claude-opus-4-7`  |
| Judge thinking      | `ClaudeJudge(enable_thinking=True)`                                  | `False`            |
| Concurrency         | `RunnerOptions(concurrency=8)`                                       | `4`                |
| Retrieval `top_k`   | `RunnerOptions(top_k=5)`                                             | (required)         |
| Synthetic chunk size| `SyntheticGeneratorOptions(min_chunk_chars, max_chunk_chars)`        | `400` / `2000`     |
| Dashboard runs dir  | `RAG_EVAL_RUNS_DIR` env var                                          | `./runs/`          |

### A word on cost

The judge runs Claude Opus 4.7 by default. For a 100-item dataset evaluated against faithfulness + context precision (with top_k=5), expect roughly **600 judge calls** total — at Opus 4.7 prices that's a few dollars per evaluation run. Each evaluator's system prompt is sent with `cache_control: ephemeral`, so subsequent calls within the cache TTL hit the prompt cache (~0.1× cost) rather than paying full price.

If cost matters more than judge accuracy:

```python
from rag_eval_harness import (
    ClaudeJudge,
    FaithfulnessEvaluator, FaithfulnessOptions,
    ContextPrecisionEvaluator, ContextPrecisionOptions,
)

cheap_judge = ClaudeJudge(model="claude-haiku-4-5", enable_thinking=False)
evaluators = [
    FaithfulnessEvaluator(FaithfulnessOptions(judge=cheap_judge)),
    ContextPrecisionEvaluator(ContextPrecisionOptions(judge=cheap_judge)),
]
```

## Architecture

```text
┌────────────┐    ┌────────────┐    ┌──────────────────────┐
│  Dataset   │───▶│  Pipeline  │───▶│      RunResult[]     │
│  (JSONL)   │    │   Runner   │    │ (retrieved, gen, ms) │
└────────────┘    └────────────┘    └──────────┬───────────┘
                  ▲          ▲                 │
                  │          │                 ▼
              Retriever   Generator      ┌────────────┐
              (yours)     (yours)        │ Evaluators │ ──▶ EvaluatorReport[]
                                         └────────────┘            │
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │  HTML / JSON    │ ──▶  Dashboard
                                                          └─────────────────┘
```

- **`Retriever`** and **`Generator`** are `typing.Protocol`s. The harness has no idea what's behind them — embedding model, vector store, prompt template, none of it. Any class with the right shape works.
- **`PipelineRunner`** runs the dataset through both with bounded concurrency (`asyncio.Semaphore`) and per-item timing. Errors in one item don't kill the run; they're recorded as `RunResult.error`.
- **`Evaluator`** receives the full set of `RunResult`s and returns an `EvaluatorReport` with per-item scores plus aggregates (mean, percentiles, sums).
- **`render_html_report`** takes one or more `RunSummary`s (as dicts, post-`asdict`) and emits a self-contained HTML file with no external dependencies.
- The **dashboard** is a separate Next.js app that reads the same JSON the harness emits. The split — Python for the eval engine, TypeScript for the UI — keeps each side using its native ecosystem.

## Methodological notes

A few choices that affect what the numbers actually mean:

- **LLM-as-judge has known biases.** Position bias, verbosity bias, self-preference. The faithfulness evaluator mitigates the first by decomposing into atomic claims and judging each independently. It does *not* yet do judge-ensembling or pairwise consistency checks. If your pipeline is itself Claude-based, treat the absolute scores with appropriate scepticism — the *deltas* between pipeline versions remain meaningful.
- **Faithfulness is not the same as correctness.** A faithful answer is one fully supported by the retrieved context. If the context is wrong, a faithful answer can still be wrong. Pair faithfulness with context recall to triangulate.
- **Latency does not include the judge.** `LatencyEvaluator` only measures pipeline (retrieval + generation) latency, not evaluator latency. The judge runs after the pipeline is done.
- **Cost only includes the generator.** The cost evaluator reports what *your pipeline* costs to serve a query in production. Judge cost is an evaluation-time expense, not a per-query one.
- **`p99` on a 100-item dataset is the worst single observation.** Take percentiles with appropriate sample-size grains of salt; we do no bootstrap or confidence interval estimation yet.

## API reference

See [`src/rag_eval_harness/__init__.py`](src/rag_eval_harness/__init__.py) for the full public surface. Highlights:

```python
from rag_eval_harness import (
    # Core types
    Document, RetrievalResult, Retriever,
    GenerationResult, Generator,
    DatasetItem, RunResult, RunSummary,
    Evaluator, EvaluatorReport, ItemScore, AggregateScore,

    # Orchestration
    PipelineRunner, RunnerOptions,

    # Evaluators
    FaithfulnessEvaluator, FaithfulnessOptions,
    ContextPrecisionEvaluator, ContextPrecisionOptions,
    ContextRecallEvaluator,
    LatencyEvaluator,
    CostEvaluator,

    # Judge
    ClaudeJudge, JudgeResult,

    # Helpers
    InMemoryRetriever,
    SyntheticTestSetGenerator, SyntheticGeneratorOptions, SyntheticItem,
    load_jsonl_dataset, write_jsonl_dataset,
    render_html_report,
)
```

## Roadmap

- Answer relevance evaluator (semantic similarity to expected answer)
- Bootstrap confidence intervals on aggregate metrics
- Pluggable judge backends (currently Claude only)
- Dashboard improvements: pipeline diffing, time-series charts, run tagging

## Contributing

```bash
pip install -e ".[dev]"
pytest                       # tests (no API calls)
mypy src/rag_eval_harness    # type-check
ruff check src tests examples
ruff format src tests examples
```

PRs welcome. The bar for new evaluators: they should produce per-item *and* aggregate scores, document any judge biases they're vulnerable to, and have a smoke test that doesn't hit the API.

## License

MIT
