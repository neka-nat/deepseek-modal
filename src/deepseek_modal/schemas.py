"""Request / response models shared by the CLI (local) and the Modal container."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

EFFORT_LEVELS = ("low", "high", "xhigh", "max")
Role = Literal["system", "user", "assistant"]
# Named level, integer budget 1-100 (chat_template_kwargs.reasoning_effort), or None = thinking off.
Effort = str | int | None


def parse_effort(text: str | None) -> Effort:
    """Turn a CLI value into an effort: `low|high|xhigh|max`, an integer 1-100, or `none`."""
    if text is None:
        return None
    value = text.strip().lower()
    if value in {"none", "off", "no", "false"}:
        return None
    if value in EFFORT_LEVELS:
        return value
    if value.isdigit():
        budget = int(value)
        if 1 <= budget <= 100:
            return budget
        raise ValueError("an integer reasoning effort must be between 1 and 100")
    raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}, an integer 1-100, or none")


class Message(BaseModel):
    role: Role
    content: str | list[dict[str, Any]]  # plain text, or OpenAI content parts (future image input)


class GenerationRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    effort: Effort = "high"
    max_tokens: int = Field(default=32768, ge=1)  # covers reasoning + answer
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)  # model card: 1.0 / 0.95
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)

    @field_validator("effort", mode="before")
    @classmethod
    def _check_effort(cls, value: object) -> Effort:
        # "before" mode: pydantic would otherwise coerce True -> 1 before an "after" validator runs.
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("effort must be a level name, an integer 1-100, or null")
        if isinstance(value, int):
            if not 1 <= value <= 100:
                raise ValueError("an integer reasoning effort must be between 1 and 100")
            return value
        if isinstance(value, str) and value in EFFORT_LEVELS:
            return value
        raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}, an integer 1-100, or null")

    @property
    def thinking(self) -> bool:
        return self.effort is not None

    def to_openai_payload(self, model: str, *, stream: bool) -> dict[str, Any]:
        """Body for SGLang's /v1/chat/completions, with DeepSeek-V4.1 reasoning controls."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.stop:
            payload["stop"] = self.stop
        template_kwargs: dict[str, Any] = {"thinking": self.thinking}
        if isinstance(self.effort, int):
            template_kwargs["reasoning_effort"] = self.effort  # integer budgets go through the template
        elif isinstance(self.effort, str):
            payload["reasoning_effort"] = self.effort  # named levels are a top-level field
        payload["chat_template_kwargs"] = template_kwargs
        return payload


class StreamChunk(BaseModel):
    """One streamed delta. `done` carries the finish reason, token usage and container timings."""

    kind: Literal["reasoning", "content", "done"]
    text: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None


class GenerationResult(BaseModel):
    content: str
    reasoning: str
    finish_reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    model: str = ""

    @classmethod
    def from_chunks(cls, chunks: Iterable[StreamChunk], model: str = "") -> GenerationResult:
        content: list[str] = []
        reasoning: list[str] = []
        finish_reason = None
        usage: dict[str, Any] = {}
        stats: dict[str, Any] = {}
        for chunk in chunks:
            if chunk.kind == "content":
                content.append(chunk.text)
            elif chunk.kind == "reasoning":
                reasoning.append(chunk.text)
            else:
                finish_reason = chunk.finish_reason
                usage = chunk.usage or {}
                stats = chunk.stats or {}
        return cls(
            content="".join(content),
            reasoning="".join(reasoning),
            finish_reason=finish_reason,
            usage=usage,
            stats=stats,
            model=model,
        )
