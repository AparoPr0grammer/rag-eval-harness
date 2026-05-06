"""Mongo Atlas Vector Search quickstart.

Wires the harness to a Mongo Atlas vector index for retrieval, OpenAI for
query embeddings, and Claude for generation. This is the canonical "real
production RAG pipeline" shape: dense retrieval over a vector store + an
LLM generator.

Prerequisites
-------------

* A Mongo Atlas cluster with a vector search index on the ``embedding``
  field. Index definition (assuming ``text-embedding-3-small``)::

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

* Documents in your collection with shape::

      {"_id": "...", "content": "...", "embedding": [float, ...]}

* ``pip install rag-eval-harness[mongo]``

* Environment variables:
    - MONGODB_URI               Mongo Atlas connection string
    - MONGODB_DATABASE          Database name
    - MONGODB_COLLECTION        Collection name
    - MONGODB_VECTOR_INDEX      Vector index name (e.g. "vector_index")
    - OPENAI_API_KEY            For query embeddings
    - ANTHROPIC_API_KEY         For the judge and the generator

Usage::

    python examples/mongo_atlas_quickstart.py path/to/dataset.jsonl
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
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_eval_harness import (  # noqa: E402
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    CostEvaluator,
    Document,
    FaithfulnessEvaluator,
    GenerationResult,
    LatencyEvaluator,
    PipelineMetadata,
    PipelineRunner,
    RetrievalResult,
    RunnerOptions,
    RunSummary,
    load_jsonl_dataset,
    render_html_report,
)


class MongoAtlasVectorRetriever:
    """Retriever that issues a $vectorSearch against a Mongo Atlas collection."""

    name = "mongo-atlas-vector"

    def __init__(
        self,
        client: AsyncIOMotorClient,
        openai: AsyncOpenAI,
        *,
        database: str,
        collection: str,
        index_name: str,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._collection = client[database][collection]
        self._openai = openai
        self._index_name = index_name
        self._embedding_model = embedding_model

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        embed = await self._openai.embeddings.create(
            model=self._embedding_model, input=query
        )
        query_vector = embed.data[0].embedding

        cursor = self._collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": self._index_name,
                        "path": "embedding",
                        "queryVector": query_vector,
                        # numCandidates controls recall vs latency: higher = better
                        # recall, slower. 10x topK is a reasonable default.
                        "numCandidates": max(top_k * 10, 100),
                        "limit": top_k,
                    },
                },
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    },
                },
            ]
        )

        results: list[RetrievalResult] = []
        async for doc in cursor:
            results.append(
                RetrievalResult(
                    document=Document(id=str(doc["_id"]), content=str(doc["content"])),
                    score=float(doc.get("score", 0)),
                )
            )
        return results


class ClaudeGenerator:
    name = "claude-opus-4-7"

    def __init__(self) -> None:
        self._client = AsyncAnthropic()

    async def generate(
        self, query: str, contexts: Sequence[Document]
    ) -> GenerationResult:
        block = "\n\n".join(f"[{i + 1}] {c.content}" for i, c in enumerate(contexts))
        prompt = (
            "Answer the question using only the provided context. If the context "
            "does not contain enough information, say so honestly.\n\n"
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
    required = [
        "MONGODB_URI",
        "MONGODB_DATABASE",
        "MONGODB_COLLECTION",
        "MONGODB_VECTOR_INDEX",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.jsonl"
    dataset = load_jsonl_dataset(dataset_path)
    print(f"Loaded {len(dataset)} items from {dataset_path}")

    mongo_client = AsyncIOMotorClient(os.environ["MONGODB_URI"])
    try:
        retriever = MongoAtlasVectorRetriever(
            mongo_client,
            AsyncOpenAI(),
            database=os.environ["MONGODB_DATABASE"],
            collection=os.environ["MONGODB_COLLECTION"],
            index_name=os.environ["MONGODB_VECTOR_INDEX"],
        )
        generator = ClaudeGenerator()

        def on_progress(done: int, total: int) -> None:
            print(f"  Progress: {done}/{total}")

        runner = PipelineRunner(
            retriever,
            generator,
            RunnerOptions(top_k=5, concurrency=4, on_progress=on_progress),
        )

        print("Running pipeline...")
        started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        results = await runner.run(dataset)
        print("Pipeline complete.\n")

        evaluators = [
            FaithfulnessEvaluator(),
            ContextPrecisionEvaluator(),
            ContextRecallEvaluator(),
            LatencyEvaluator(),
            CostEvaluator(),
        ]
        print("Running evaluators...")
        reports = []
        for evaluator in evaluators:
            print(f"  {evaluator.name}...")
            reports.append(await evaluator.evaluate(results))

        summary = RunSummary(
            pipeline=PipelineMetadata(
                name="mongo-atlas-vector + claude-opus-4-7",
                description="Dense retrieval via Mongo Atlas Vector Search and OpenAI embeddings",
            ),
            started_at=started_at,
            completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            results=list(results),
            reports=reports,
        )

        summary_dict = asdict(summary)
        Path("run.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
        Path("report.html").write_text(render_html_report([summary_dict]), encoding="utf-8")
        print("\nWrote run.json and report.html.")

        for report in reports:
            print(f"\n[{report.evaluator}]")
            for score in report.aggregate:
                print(f"  {score.metric}: {score.value:.4f}")
    finally:
        mongo_client.close()


if __name__ == "__main__":
    asyncio.run(main())
