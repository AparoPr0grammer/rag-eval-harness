"""JSONL dataset loader and writer.

JSONL is the format the synthetic generator emits and the format pipeline
runners consume. One item per line, each line a JSON object with at minimum:

    {"id": "...", "question": "...", "expected_answer": "...", "expected_context_ids": [...]}

The loader is forgiving about field names (``answer`` works as well as
``expected_answer``; ``context_id`` as a single string is auto-wrapped) so
synthetic and hand-written datasets can share the same loader.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from ..core.types import DatasetItem


def load_jsonl_dataset(path: str | Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i} of {p}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} of {p} is not a JSON object")

            question = obj.get("question")
            if not isinstance(question, str):
                raise ValueError(f"Line {i} of {p} missing string 'question'")

            expected_answer = obj.get("expected_answer") or obj.get("answer")
            if expected_answer is not None and not isinstance(expected_answer, str):
                expected_answer = None

            ctx_ids: list[str] | None = None
            raw_ctx = obj.get("expected_context_ids")
            if isinstance(raw_ctx, list):
                ctx_ids = [c for c in raw_ctx if isinstance(c, str)]
            elif isinstance(obj.get("context_id"), str):
                ctx_ids = [obj["context_id"]]

            metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else None

            items.append(
                DatasetItem(
                    id=str(obj.get("id", f"item-{i - 1}")),
                    question=question,
                    expected_answer=expected_answer,
                    expected_context_ids=ctx_ids,
                    metadata=metadata,
                )
            )
    return items


def write_jsonl_dataset(items: Iterable[DatasetItem], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")
