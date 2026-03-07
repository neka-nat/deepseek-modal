import pytest

from deepseek_modal.schemas import GenerationRequest, GenerationResult, StreamChunk, parse_effort


@pytest.mark.parametrize(
    ("text", "expected"),
    [("high", "high"), ("MAX", "max"), ("none", None), ("off", None), ("42", 42), (None, None)],
)
def test_parse_effort(text, expected):
    assert parse_effort(text) == expected


@pytest.mark.parametrize("text", ["medium", "0", "101", "lots"])
def test_parse_effort_rejects(text):
    with pytest.raises(ValueError):
        parse_effort(text)


def _req(**kwargs):
    return GenerationRequest(messages=[{"role": "user", "content": "hi"}], **kwargs)


def test_named_effort_goes_top_level():
    payload = _req(effort="xhigh").to_openai_payload("m", stream=True)
    assert payload["reasoning_effort"] == "xhigh"
    assert payload["chat_template_kwargs"] == {"thinking": True}
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 1.0 and payload["top_p"] == 0.95 and payload["max_tokens"] == 32768


def test_integer_effort_goes_through_template_kwargs():
    payload = _req(effort=85).to_openai_payload("m", stream=False)
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 85}
    assert "stream_options" not in payload


def test_thinking_off():
    payload = _req(effort=None, seed=7, stop=["</end>"]).to_openai_payload("m", stream=True)
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"] == {"thinking": False}
    assert payload["seed"] == 7 and payload["stop"] == ["</end>"]


@pytest.mark.parametrize("effort", ["medium", 0, 101, True])
def test_request_rejects_bad_effort(effort):
    with pytest.raises(ValueError):
        _req(effort=effort)


def test_request_roundtrips_through_dict():
    req = _req(effort=30, max_tokens=10)
    assert GenerationRequest.model_validate(req.model_dump()) == req


def test_result_from_chunks():
    chunks = [
        StreamChunk(kind="reasoning", text="think "),
        StreamChunk(kind="reasoning", text="more"),
        StreamChunk(kind="content", text="answer"),
        StreamChunk(kind="done", finish_reason="stop", usage={"completion_tokens": 3}, stats={"seconds": 1.5}),
    ]
    result = GenerationResult.from_chunks(chunks, model="m")
    assert result.reasoning == "think more"
    assert result.content == "answer"
    assert result.finish_reason == "stop"
    assert result.usage == {"completion_tokens": 3}
    assert result.stats == {"seconds": 1.5}
    assert result.model == "m"
