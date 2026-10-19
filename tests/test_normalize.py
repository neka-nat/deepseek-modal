import pytest

from deepseek_modal.engine import normalize_request


def test_chat_injects_named_effort_and_model():
    out = normalize_request("/v1/chat/completions", {"model": "x", "messages": []}, "high", "served")
    assert out["model"] == "served"
    assert out["reasoning_effort"] == "high"


def test_chat_injects_integer_effort_via_template_kwargs():
    out = normalize_request("/v1/chat/completions?x=1", {"model": "x"}, 42, None)
    assert out["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 42}
    assert "reasoning_effort" not in out


@pytest.mark.parametrize(
    "body",
    [
        {"reasoning_effort": "low"},
        {"chat_template_kwargs": {"thinking": False}},
        {"chat_template_kwargs": {"reasoning_effort": 10}},
    ],
)
def test_chat_leaves_explicit_thinking_alone(body):
    out = normalize_request("/v1/chat/completions", body, "high")
    assert out == body


def test_messages_injects_thinking_and_effort():
    out = normalize_request("/v1/messages", {"model": "claude-x", "max_tokens": 10}, "max", "served")
    assert out["thinking"]["type"] == "enabled"
    assert out["output_config"] == {"effort": "max"}
    assert out["model"] == "served"


def test_messages_respects_client_thinking():
    body = {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
    out = normalize_request("/v1/messages", body, "high")
    assert out["thinking"] == {"type": "disabled"}
    assert out["output_config"] == {"effort": "low"}


def test_responses_injects_reasoning():
    out = normalize_request("/v1/responses", {"input": "hi"}, "high")
    assert out["reasoning"] == {"effort": "high"}
    out = normalize_request("/v1/responses", {"input": "hi", "reasoning": {"effort": "low"}}, "high")
    assert out["reasoning"] == {"effort": "low"}


def test_no_default_effort_only_rewrites_model():
    out = normalize_request("/v1/messages", {"model": "a"}, None, "served")
    assert out == {"model": "served"}


def test_other_routes_untouched():
    out = normalize_request("/v1/models", {}, "high", "served")
    assert out == {}
