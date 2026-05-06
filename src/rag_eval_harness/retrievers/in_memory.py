"""In-memory BM25 retriever.

A self-contained reference retriever for examples, smoke tests, and as a
keyword baseline to compare against vector retrievers. Implements Okapi BM25
with the standard k1=1.5, b=0.75 defaults.

The point of having BM25 in here isn't to be a great retriever — it isn't —
it's to give the harness a working baseline that requires no API key, no
embedding model, and no external service. If your dense retriever can't beat
this, you have a problem.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from ..core.types import Document, RetrievalResult


class InMemoryRetriever:
    name = "in-memory-bm25"

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._documents = list(documents)
        self._tokenised = [_tokenise(d.content) for d in self._documents]
        self._avg_doc_len = (
            sum(len(t) for t in self._tokenised) / len(self._tokenised)
            if self._tokenised
            else 0.0
        )

        df: dict[str, int] = {}
        for tokens in self._tokenised:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._df = df
        self._k1 = k1
        self._b = b

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_terms = _tokenise(query)
        n = len(self._documents)

        scored: list[RetrievalResult] = []
        for doc, tokens in zip(self._documents, self._tokenised, strict=True):
            tf: dict[str, int] = {}
            for term in tokens:
                tf[term] = tf.get(term, 0) + 1

            score = 0.0
            for term in query_terms:
                term_freq = tf.get(term, 0)
                if not term_freq:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                numerator = term_freq * (self._k1 + 1)
                length_norm = (
                    1 - self._b + self._b * (len(tokens) / self._avg_doc_len)
                    if self._avg_doc_len > 0
                    else 1.0
                )
                denominator = term_freq + self._k1 * length_norm
                score += idf * (numerator / denominator)
            scored.append(RetrievalResult(document=doc, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return [r for r in scored[:top_k] if r.score > 0]


def _tokenise(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", re.sub(r"[^\w\s]", " ", text.lower())) if len(t) > 1]
