"""SGLang process management and OpenAI-compatible streaming, used inside the Modal container.

Nothing here imports `modal`; the command builder, the SSE parser and the chunk coalescing are
pure functions so they can be unit-tested locally without a GPU.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .schemas import GenerationRequest, GenerationResult, StreamChunk

Logger = Callable[[str], None]


@dataclass
class EngineSettings:
    model_dir: str
    python: str = "python3"  # interpreter that has sglang installed
    served_model_name: str = "deepseek-v4.1-flash"
    port: int = 30000
    tp: int = 4
    ep: int | None = None  # defaults to tp; `--ep-size` is mandatory at TP4 for this model shape
    context_len: int = 262144
    mem_fraction_static: float = 0.85
    max_running_requests: int = 8
    cuda_graph_max_bs: int = 8
    chunked_prefill: int = 4096
    reasoning_parser: str = "deepseek-v41"
    tool_call_parser: str = "deepseekv41"
    default_effort: str = "high"
    speculative: str = ""  # "" or "dspark"
    dspark_block_size: int = 5
    engram_host_table: bool = False
    engram_layout: str = "per_rank"
    fast_load: bool = True
    extra_args: str = ""
    watchdog_timeout: int = 3600

    @property
    def ep_size(self) -> int:
        return self.ep or self.tp

    def summary(self) -> dict[str, object]:
        return asdict(self)


def build_command(settings: EngineSettings) -> list[str]:
    """The `sglang.launch_server` command line for these settings."""
    s = settings
    cmd = [
        s.python, "-m", "sglang.launch_server",
        "--model-path", s.model_dir,
        "--served-model-name", s.served_model_name,
        "--host", "0.0.0.0",
        "--port", str(s.port),
        "--tp-size", str(s.tp),
        "--ep-size", str(s.ep_size),
        "--context-length", str(s.context_len),
        "--mem-fraction-static", f"{s.mem_fraction_static:g}",
        "--max-running-requests", str(s.max_running_requests),
        "--cuda-graph-max-bs-decode", str(s.cuda_graph_max_bs),
        "--chunked-prefill-size", str(s.chunked_prefill),
        "--reasoning-parser", s.reasoning_parser,
        "--tool-call-parser", s.tool_call_parser,
        "--watchdog-timeout", str(s.watchdog_timeout),
        "--trust-remote-code",
    ]  # fmt: skip
    if s.fast_load:
        cmd += [
            "--model-loader-extra-config", json.dumps({"enable_multithread_load": True}),
            "--weight-loader-prefetch-checkpoints",
        ]  # fmt: skip
    speculative = s.speculative.strip().lower()
    if speculative == "dspark":
        cmd += ["--speculative-algorithm", "DSPARK", "--speculative-dspark-block-size", str(s.dspark_block_size)]
    elif speculative:
        raise ValueError(f"unsupported speculative algorithm {s.speculative!r} (use '' or 'dspark')")
    cmd += shlex.split(s.extra_args)
    return cmd


def build_env(settings: EngineSettings, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for the server process: DeepSeek-V4.1 defaults from the SGLang cookbook."""
    env = dict(os.environ if base is None else base)
    env.update(
        {
            "SGLANG_DEFAULT_THINKING": "1",
            "SGLANG_DSV41_REASONING_EFFORT": settings.default_effort,
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1" if settings.engram_host_table else "0",
            "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_CUMEM_ENABLE": "1",
        }
    )
    if settings.engram_host_table:
        env["SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT"] = settings.engram_layout
    return env


def resolve_engine_python(preferred: str) -> str:
    """The interpreter with sglang: `preferred` if it exists, else python3 from PATH."""
    if Path(preferred).exists():
        return preferred
    return shutil.which("python3") or shutil.which("python") or "python3"


def visible_gpu_count(engine_python: str | None = None) -> int:
    """How many GPUs this container can see (drives the tensor-parallel size)."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is not None and devices.strip():
        return len([d for d in devices.split(",") if d.strip()])
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=60, check=False)
            count = sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
            if count:
                return count
        except (OSError, subprocess.SubprocessError):
            pass
    if engine_python:
        try:
            out = subprocess.run(
                [engine_python, "-c", "import torch; print(torch.cuda.device_count())"],
                capture_output=True, text=True, timeout=300, check=False,
            )  # fmt: skip
            count = int(out.stdout.strip() or 0)
            if count:
                return count
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return 1


def seed_cache_dir(seed: Path | str, target: Path | str, log: Logger = print) -> list[str]:
    """Copy top-level entries of the image's cache snapshot into the (Volume-backed) cache dir.

    The Volume mounted on /root/.cache hides whatever JIT artifacts the image shipped there; this
    puts them back once, without overwriting anything the Volume already holds.
    """
    seed_path, target_path = Path(seed), Path(target)
    if not seed_path.is_dir():
        return []
    target_path.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for entry in sorted(seed_path.iterdir()):
        destination = target_path / entry.name
        if destination.exists():
            continue
        if entry.is_dir():
            shutil.copytree(entry, destination, symlinks=True)
        else:
            shutil.copy2(entry, destination, follow_symlinks=False)
        copied.append(entry.name)
    if copied:
        log(f"seeded {target_path} from image: {', '.join(copied)}")
    return copied


# ------------------------------------------------------------------ SSE parsing (pure)
def iter_sse_data(lines: Iterable[str]) -> Iterator[dict]:
    """JSON objects from an OpenAI-style SSE stream; stops at `[DONE]`."""
    for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if data:
            yield json.loads(data)


def chunks_from_events(events: Iterable[dict]) -> Iterator[StreamChunk]:
    """Reasoning / content deltas plus a final `done` chunk with finish reason and usage."""
    finish_reason: str | None = None
    usage: dict | None = None
    for event in events:
        if event.get("error"):
            raise RuntimeError(f"server error: {event['error']}")
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            if reasoning:
                yield StreamChunk(kind="reasoning", text=reasoning)
            if content:
                yield StreamChunk(kind="content", text=content)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    yield StreamChunk(kind="done", finish_reason=finish_reason, usage=usage)


def coalesce(
    chunks: Iterable[StreamChunk],
    *,
    interval_s: float = 0.1,
    max_chars: int = 512,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[StreamChunk]:
    """Merge consecutive deltas of one kind so a Modal generator ships ~10 messages/s, not one per token."""
    kind: str | None = None
    parts: list[str] = []
    size = 0
    last_flush = clock()

    def flush() -> Iterator[StreamChunk]:
        nonlocal kind, parts, size, last_flush
        if parts:
            yield StreamChunk(kind=kind, text="".join(parts))  # type: ignore[arg-type]
        kind, parts, size = None, [], 0
        last_flush = clock()

    for chunk in chunks:
        if chunk.kind == "done":
            yield from flush()
            yield chunk
            continue
        if kind is not None and kind != chunk.kind:
            yield from flush()
        kind = chunk.kind
        parts.append(chunk.text)
        size += len(chunk.text)
        if size >= max_chars or clock() - last_flush >= interval_s:
            yield from flush()
    yield from flush()


# ------------------------------------------------------------------ process + HTTP client
class SGLangServer:
    """Owns the `sglang.launch_server` subprocess and talks to it over localhost."""

    def __init__(self, settings: EngineSettings, log: Logger = print) -> None:
        self.settings = settings
        self.log = log
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.settings.port}"

    def start(self) -> None:
        cmd = build_command(self.settings)
        self.log("launching: " + " ".join(shlex.quote(c) for c in cmd))
        # Logs are inherited so they show up in the Modal container logs.
        self.process = subprocess.Popen(cmd, env=build_env(self.settings), start_new_session=True)

    def wait_ready(self, timeout_s: float, poll_s: float = 5.0) -> float:
        """Block until /health answers 200; raise if the server exits or the budget runs out."""
        import httpx

        assert self.process is not None, "call start() first"
        t0 = time.monotonic()
        next_report = 60.0
        while True:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f"sglang server exited with code {code} during startup (see container logs)")
            try:
                if httpx.get(f"{self.base_url}/health", timeout=5.0).status_code == 200:
                    elapsed = time.monotonic() - t0
                    self.log(f"server ready after {elapsed:.0f}s")
                    return elapsed
            except httpx.HTTPError:
                pass
            elapsed = time.monotonic() - t0
            if elapsed >= timeout_s:
                self.stop()
                raise TimeoutError(f"sglang server not ready after {timeout_s:.0f}s")
            if elapsed >= next_report:
                self.log(f"still loading ({elapsed:.0f}s)")
                next_report += 60.0
            time.sleep(poll_s)

    def warmup(self) -> float | None:
        """One tiny generation so CUDA graphs and caches are hot before the first real request."""
        request = GenerationRequest(messages=[{"role": "user", "content": "Say OK."}], effort=None, max_tokens=8)
        t0 = time.monotonic()
        try:
            result = self.complete(request)
        except Exception as exc:  # noqa: BLE001 - warmup is best effort
            self.log(f"warmup failed (continuing): {exc}")
            return None
        elapsed = time.monotonic() - t0
        self.log(f"warmup ok in {elapsed:.1f}s: {result.content!r}")
        return elapsed

    def stop(self, grace_s: float = 30.0) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    def stream_chat(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Stream one chat completion as coalesced chunks; the final chunk carries timings."""
        import httpx

        payload = request.to_openai_payload(self.settings.served_model_name, stream=True)
        t0 = time.monotonic()
        first_token_at: float | None = None
        timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    body = response.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"server returned HTTP {response.status_code}: {body[:2000]}")
                for chunk in coalesce(chunks_from_events(iter_sse_data(response.iter_lines()))):
                    if chunk.kind != "done":
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        yield chunk
                        continue
                    now = time.monotonic()
                    usage = chunk.usage or {}
                    completion_tokens = usage.get("completion_tokens")
                    details = usage.get("completion_tokens_details") or {}
                    decode_s = now - (first_token_at or now)
                    # tok/s is only meaningful once there is something to measure
                    measurable = bool(completion_tokens) and completion_tokens >= 16 and decode_s >= 0.2
                    chunk.stats = {
                        "seconds": round(now - t0, 2),
                        "ttft_s": round((first_token_at or now) - t0, 2),
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": usage.get("reasoning_tokens", details.get("reasoning_tokens")),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "tok_per_s": round(completion_tokens / decode_s, 1) if measurable else None,
                        "model": self.settings.served_model_name,
                    }
                    yield chunk

    def complete(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult.from_chunks(self.stream_chat(request), model=self.settings.served_model_name)

    def server_info(self) -> dict:
        """SGLang's /get_server_info (resolved backends, speculative settings, KV capacity); {} on failure."""
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/get_server_info", timeout=30.0)
            response.raise_for_status()
            info = response.json()
        except (httpx.HTTPError, ValueError):
            return {}
        keys = (
            "version", "tp_size", "ep_size", "attention_backend", "prefill_attention_backend",
            "decode_attention_backend", "moe_runner_backend", "moe_a2a_backend", "fp8_gemm_backend",
            "kv_cache_dtype", "speculative_algorithm", "speculative_dspark_block_size", "mem_fraction_static",
            "context_length", "max_total_num_tokens", "max_running_requests", "reasoning_parser", "tool_call_parser",
        )  # fmt: skip
        return {k: info[k] for k in keys if k in info}
