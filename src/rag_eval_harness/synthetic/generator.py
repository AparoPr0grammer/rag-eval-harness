"""Synthetic test set generator.

Bootstraps a Q/A/context dataset from an existing markdown corpus. Walks the
directory, splits each file into paragraph-aligned chunks within configured
size bounds, and asks Claude to draft a question whose answer is found in the
chunk.

This is a *starting* dataset, not a final one. Synthetic Q/A is uneven —
some questions are too easy (the answer is paraphrased verbatim from one
sentence), some are off-topic, some have ambiguous answers. Eyeball a sample
and prune before you trust the numbers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..judges.claude_judge import ClaudeJudge

SYSTEM_PROMPT = """You generate question/answer pairs for evaluating retrieval-augmented generation systems.

Given a passage of text, produce one self-contained question whose answer is found in the passage. The question must be answerable using only the passage.

Quality guidelines:
- Specific and factual; avoid yes/no questions.
- Avoid questions whose answer is the title of the document or a section header.
- Use natural, search-realistic phrasing — questions a real user might type.
- The answer must be a verbatim or near-verbatim quote from the passage.

Output strict JSON only (no markdown, no commentary):
{
  "question": "...",
  "answer": "..."
}"""


@dataclass(frozen=True, slots=True)
class SyntheticItem:
    id: str
    question: str
    expected_answer: str
    context_id: str
    context_content: str
    source: str


@dataclass(frozen=True, slots=True)
class SyntheticGeneratorOptions:
    judge: ClaudeJudge | None = None
    min_chunk_chars: int = 400
    max_chunk_chars: int = 2000
    skip_short_chunks: bool = True


class SyntheticTestSetGenerator:
    """Walks a markdown directory and synthesises a Q/A/context dataset."""

    def __init__(self, options: SyntheticGeneratorOptions | None = None) -> None:
        opts = options or SyntheticGeneratorOptions()
        self._judge = opts.judge or ClaudeJudge()
        self._min_chunk_chars = opts.min_chunk_chars
        self._max_chunk_chars = opts.max_chunk_chars
        self._skip_short_chunks = opts.skip_short_chunks

    async def from_directory(self, dir_path: str | Path) -> list[SyntheticItem]:
        root = Path(dir_path)
        items: list[SyntheticItem] = []
        counter = 0

        for file in _collect_markdown_files(root):
            content = file.read_text(encoding="utf-8")
            chunks = list(_chunk_markdown(content, self._min_chunk_chars, self._max_chunk_chars))
            rel_path = str(file.relative_to(root))

            for idx, chunk in enumerate(chunks):
                if self._skip_short_chunks and len(chunk) < self._min_chunk_chars:
                    continue
                generated = await self._generate_one(chunk)
                if generated is None:
                    continue
                items.append(
                    SyntheticItem(
                        id=f"synth-{counter}",
                        question=generated["question"],
                        expected_answer=generated["answer"],
                        context_id=f"{rel_path}#{idx}",
                        context_content=chunk,
                        source=rel_path,
                    )
                )
                counter += 1
        return items

    @staticmethod
    def write_jsonl(items: Iterable[SyntheticItem], output_path: str | Path) -> None:
        path = Path(output_path)
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(asdict(item)) + "\n")

    async def _generate_one(self, chunk: str) -> dict[str, str] | None:
        user_prompt = f"Passage:\n{chunk}\n\nReturn JSON only."
        judged = await self._judge.judge(SYSTEM_PROMPT, user_prompt)
        match = re.search(r"\{[\s\S]*\}", judged.text)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("question"), str)
            and isinstance(obj.get("answer"), str)
        ):
            return {"question": obj["question"], "answer": obj["answer"]}
        return None


def _collect_markdown_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".mdx")
    )


def _chunk_markdown(content: str, min_chars: int, max_chars: int) -> Iterable[str]:
    """Greedily concatenate paragraphs into chunks within size bounds.

    A paragraph longer than ``max_chars`` is kept intact rather than split
    mid-sentence — better to feed the judge an oversized chunk than a
    truncated one.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) > max_chars and len(buffer) >= min_chars:
            yield buffer
            buffer = para
        else:
            buffer = candidate
    if buffer:
        yield buffer
