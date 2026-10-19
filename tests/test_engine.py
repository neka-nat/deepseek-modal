import json

import pytest

from deepseek_modal.engine import (
    EngineSettings,
    build_command,
    build_env,
    chunks_from_events,
    coalesce,
    iter_sse_data,
    resolve_engine_python,
    seed_cache_dir,
)
from deepseek_modal.schemas import StreamChunk


def _settings(**kwargs):
    return EngineSettings(model_dir="/models/x", **kwargs)


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_build_command_defaults():
    cmd = build_command(_settings(tp=4))
    assert cmd[:3] == ["python3", "-m", "sglang.launch_server"]
    assert _flag(cmd, "--model-path") == "/models/x"
    assert _flag(cmd, "--tp-size") == "4" and _flag(cmd, "--ep-size") == "4"
    assert _flag(cmd, "--reasoning-parser") == "deepseek-v41"
    assert _flag(cmd, "--tool-call-parser") == "deepseekv41"
    assert _flag(cmd, "--mem-fraction-static") == "0.85"
    assert "--trust-remote-code" in cmd
    assert "--weight-loader-prefetch-checkpoints" in cmd
    assert json.loads(_flag(cmd, "--model-loader-extra-config")) == {"enable_multithread_load": True}
    assert "--speculative-algorithm" not in cmd


def test_build_command_options():
    cmd = build_command(
        _settings(python="/opt/sglang/bin/python", tp=8, ep=2, fast_load=False, speculative="dspark", extra_args="--foo 'a b'")
    )
    assert cmd[0] == "/opt/sglang/bin/python"
    assert _flag(cmd, "--tp-size") == "8" and _flag(cmd, "--ep-size") == "2"
    assert "--weight-loader-prefetch-checkpoints" not in cmd
    assert _flag(cmd, "--speculative-algorithm") == "DSPARK"
    assert _flag(cmd, "--speculative-dspark-block-size") == "5"
    assert cmd[-2:] == ["--foo", "a b"]


def test_build_command_autotune_toggle():
    assert "--disable-flashinfer-autotune" not in build_command(_settings())
    assert "--disable-flashinfer-autotune" in build_command(_settings(flashinfer_autotune=False))


def test_build_command_rejects_unknown_speculative():
    with pytest.raises(ValueError):
        build_command(_settings(speculative="eagle"))


def test_build_env_engram_toggle():
    off = build_env(_settings(), base={"PATH": "/bin"})
    assert off["PATH"] == "/bin"
    assert off["SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"] == "0"
    assert "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT" not in off
    assert off["SGLANG_DEFAULT_THINKING"] == "0" and off["SGLANG_DSV41_REASONING_EFFORT"] == "high"
    assert off["PYTHONUNBUFFERED"] == "1"
    on = build_env(_settings(engram_host_table=True, default_effort="max", default_thinking=True), base={})
    assert on["SGLANG_DEFAULT_THINKING"] == "1"
    assert on["SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"] == "1"
    assert on["SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT"] == "per_rank"
    assert on["SGLANG_DSV41_REASONING_EFFORT"] == "max"


def _sse(*objs, done=True):
    lines = []
    for obj in objs:
        lines += ["data: " + json.dumps(obj), ""]
    if done:
        lines.append("data: [DONE]")
    return lines


def test_sse_parsing_and_chunks():
    events = list(
        iter_sse_data(
            _sse(
                {"choices": [{"delta": {"reasoning_content": "hmm"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 5}},
            )
        )
    )
    chunks = list(chunks_from_events(events))
    assert [c.kind for c in chunks] == ["reasoning", "content", "done"]
    assert chunks[0].text == "hmm" and chunks[1].text == "Hi"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage == {"completion_tokens": 2, "prompt_tokens": 5}


def test_sse_stops_at_done_and_skips_comments():
    lines = [": keepalive", "data: {\"choices\": []}", "data: [DONE]", "data: {\"never\": 1}"]
    assert list(iter_sse_data(lines)) == [{"choices": []}]


def test_server_error_event_raises():
    with pytest.raises(RuntimeError):
        list(chunks_from_events([{"error": {"message": "boom"}}]))


def test_coalesce_merges_until_kind_change_or_interval():
    now = [0.0]
    chunks = [
        StreamChunk(kind="reasoning", text="a"),
        StreamChunk(kind="reasoning", text="b"),
        StreamChunk(kind="content", text="c"),
        StreamChunk(kind="content", text="d"),
        StreamChunk(kind="content", text="e"),
        StreamChunk(kind="done", finish_reason="stop"),
    ]

    def feed():
        for chunk in chunks:
            yield chunk
            if chunk.text == "c":
                now[0] += 1.0  # the interval elapses while "c" is buffered

    out = list(coalesce(feed(), interval_s=0.5, clock=lambda: now[0]))
    # "ab": flushed by the kind change; "cd": flushed because the interval elapsed when "d" arrived;
    # "e": flushed by the final `done`.
    assert [(c.kind, c.text) for c in out[:-1]] == [("reasoning", "ab"), ("content", "cd"), ("content", "e")]
    assert out[-1].kind == "done" and out[-1].finish_reason == "stop"


def test_coalesce_flushes_on_size():
    chunks = [StreamChunk(kind="content", text="x" * 300), StreamChunk(kind="content", text="y" * 300)]
    out = list(coalesce(chunks, interval_s=100, max_chars=512, clock=lambda: 0.0))
    assert [len(c.text) for c in out] == [600]
    assert "".join(c.text for c in out) == "x" * 300 + "y" * 300


def test_resolve_engine_python(tmp_path):
    exe = tmp_path / "python"
    exe.write_text("")
    assert resolve_engine_python(str(exe)) == str(exe)
    fallback = resolve_engine_python(str(tmp_path / "missing"))
    assert fallback.endswith(("python3", "python"))


def test_seed_cache_dir_copies_only_missing(tmp_path):
    seed = tmp_path / "seed"
    (seed / "flashinfer").mkdir(parents=True)
    (seed / "flashinfer" / "k.so").write_text("kernel")
    (seed / "note.txt").write_text("seed")
    target = tmp_path / "cache"
    target.mkdir()
    (target / "note.txt").write_text("volume")

    copied = seed_cache_dir(seed, target, log=lambda _msg: None)

    assert copied == ["flashinfer"]
    assert (target / "flashinfer" / "k.so").read_text() == "kernel"
    assert (target / "note.txt").read_text() == "volume"
    assert seed_cache_dir(seed, target, log=lambda _msg: None) == []
    assert seed_cache_dir(tmp_path / "missing", target) == []
