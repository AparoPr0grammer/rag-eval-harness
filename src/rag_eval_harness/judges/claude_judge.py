"""Claude-backed LLM judge.

The judge is the most expensive part of the harness — a 100-item dataset
evaluated on faithfulness + context precision (top-k=5) means roughly 600
judge calls. Two design choices keep cost manageable:

1. **Prompt caching.** Each evaluator has a fixed system prompt. We mark it
   with ``cache_control: ephemeral`` so the first call writes the cache (~1.25x
   cost) and every subsequent call reads it (~0.1x cost).

2. **Adaptive thinking off by default for low-stakes judgments.** Callers can
   opt in via ``enable_thinking=True`` for evaluators where extra reasoning
   improves accuracy enough to justify the latency.
"""

from __future__ import annotations

from dataclasses import dataclass

from anthropic import AsyncAnthropic

DEFAULT_MODEL = "claude-opus-4-7"


@dataclass(frozen=True, slots=True)
class JudgeResult:
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int


class ClaudeJudge:
    """Wraps the Anthropic SDK with prompt caching for evaluator system prompts."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 2048,
        client: AsyncAnthropic | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self._client = client or AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking

    async def judge(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            thinking=(
                {"type": "adaptive"} if self._enable_thinking else {"type": "disabled"}
            ),
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return JudgeResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            cache_creation_tokens=response.usage.cache_creation_input_tokens or 0,
        )
