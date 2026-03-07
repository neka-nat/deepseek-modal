"""Modal app: weight download, the SGLang server class and a dev entrypoint.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run modal deploy -m deepseek_modal.app   # build image, create functions
    uv run dsv4 env-check                                  # image sanity check (CPU container)
    uv run dsv4 download                                   # once: 510 GB into the weights Volume
    uv run dsv4 generate "..."                             # first call pays the cold start (10-25 min)
    uv run modal run -m deepseek_modal.app --prompt "..."  # dev: ephemeral app, always a cold start
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import modal

from . import config as cfg

app = modal.App(cfg.APP_NAME)

weights_volume = modal.Volume.from_name(cfg.WEIGHTS_VOLUME, create_if_missing=True, version=2)
cache_volume = modal.Volume.from_name(cfg.CACHE_VOLUME, create_if_missing=True, version=2)

# The repo is not gated; a token only helps with rate limits. Picked up from the deploying shell.
_hf_secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []

download_image = (
    modal.Image.debian_slim(python_version=cfg.DOWNLOAD_PYTHON)
    .uv_pip_install("huggingface_hub[hf_xet]>=1.0,<2.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_XET_CACHE": "/tmp/hf-xet-cache",  # keep the xet chunk cache off the Volume
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
    .add_local_python_source("deepseek_modal")
)

# `add_python` gives Modal its own interpreter for our code, so Modal's dependency pins are never
# installed into the SGLang venv (/opt/sglang). Needs the 2025.06 image builder (see docstring).
server_image = (
    modal.Image.from_registry(cfg.IMAGE, add_python=cfg.IMAGE_PYTHON)
    .entrypoint([])
    .pip_install("httpx>=0.27,<1", "pydantic>=2,<3")  # for our code on Modal's interpreter
    .run_commands(
        # The sgl-kernel JIT build shells out to ninja; the image normally has it, this makes sure.
        f"command -v ninja >/dev/null || {cfg.ENGINE_PYTHON} -m pip install -q ninja || true",
        # Modal refuses to mount a Volume on a non-empty directory, so the image's /root/.cache is
        # moved aside and replaced by a symlink to the (initially absent) Volume mount point. This is
        # the last build step so nothing refills /root/.cache; DeepSeekServer.start() seeds the Volume
        # from the copy on first boot.
        f"mkdir -p {cfg.IMAGE_CACHE_SEED}"
        f" && (cp -a /root/.cache/. {cfg.IMAGE_CACHE_SEED}/ 2>/dev/null || true)"
        f" && rm -rf /root/.cache && ln -s {cfg.CACHE_DIR} /root/.cache",
    )
    .env(
        {
            **cfg.exported_env(),
            "HF_HOME": f"{cfg.CACHE_DIR}/huggingface",
            "TRITON_CACHE_DIR": f"{cfg.CACHE_DIR}/triton",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("deepseek_modal")
)


# ------------------------------------------------------------------ weights
def _scan_model_dir() -> dict[str, int]:
    """{relative path: size} of the downloaded files (skips huggingface_hub's .cache bookkeeping)."""
    root = Path(cfg.MODEL_DIR)
    if not root.is_dir():
        return {}
    return {
        str(p.relative_to(root)): p.stat().st_size
        for p in root.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(root).parts
    }


def _expected_sizes(revision: str) -> dict[str, int]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(cfg.MODEL_ID, revision=revision, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings or [] if s.size is not None}


def _weights_report(revision: str) -> dict:
    have = _scan_model_dir()
    report: dict = {
        "path": str(cfg.MODEL_DIR),
        "files": len(have),
        "gib": round(sum(have.values()) / 2**30, 1),
        "revision": revision,
    }
    try:
        expected = _expected_sizes(revision)
    except Exception as exc:  # noqa: BLE001 - offline or rate limited: report what is on disk
        report.update(complete=None, note=f"could not fetch the file list from the Hub: {exc}")
        return report
    missing = sorted(f for f in expected if f not in have)
    mismatched = sorted(f for f in expected if f in have and have[f] != expected[f])
    report.update(
        expected_files=len(expected),
        expected_gib=round(sum(expected.values()) / 2**30, 1),
        missing_count=len(missing),
        missing=missing[:10],
        mismatched=mismatched[:10],
        complete=not missing and not mismatched,
    )
    return report


@app.function(
    image=download_image,
    volumes={str(cfg.MODELS_DIR): weights_volume},
    cpu=8,
    memory=32 * 1024,
    timeout=24 * 3600,
    secrets=_hf_secrets,
)
def download(revision: str = cfg.MODEL_REVISION) -> dict:
    """Populate the weights Volume (~510 GB, resumable: re-run if it stops early)."""
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    snapshot_download(cfg.MODEL_ID, revision=revision, local_dir=str(cfg.MODEL_DIR), max_workers=8)
    weights_volume.commit()
    report = _weights_report(revision)
    report["seconds"] = round(time.perf_counter() - t0)
    print(report)
    return report


@app.function(image=download_image, volumes={str(cfg.MODELS_DIR): weights_volume}, cpu=2, memory=4 * 1024, timeout=900)
def weights_status(revision: str = cfg.MODEL_REVISION) -> dict:
    """What is in the weights Volume, compared with the Hub file list."""
    weights_volume.reload()
    return _weights_report(revision)


# ------------------------------------------------------------------ image sanity check (no GPU)
@app.function(image=server_image, cpu=4, memory=16 * 1024, timeout=1800)
def env_check() -> dict:
    """Versions, flags and files inside the SGLang image (`dsv4 env-check`)."""
    import importlib
    import json
    import platform
    import shutil
    import subprocess

    from .engine import resolve_engine_python

    python = resolve_engine_python(cfg.ENGINE_PYTHON)
    out: dict = {
        "modal_python": platform.python_version(),
        "engine_python": python,
        "image": cfg.IMAGE,
        "ninja": shutil.which("ninja"),
    }
    for name in ("httpx", "pydantic"):  # what our own code needs on Modal's interpreter
        try:
            out[f"modal_{name}"] = str(importlib.import_module(name).__version__)
        except Exception as exc:  # noqa: BLE001
            out[f"modal_{name}"] = f"MISSING ({type(exc).__name__}: {exc})"
    probe = r"""
import importlib, importlib.util, json, platform
out = {"python": platform.python_version()}
for name in ("sglang", "torch", "flashinfer", "sgl_kernel", "transformers", "triton", "httpx", "pydantic"):
    try:
        out[name] = str(importlib.import_module(name).__version__)
    except Exception as exc:
        out[name] = f"MISSING ({type(exc).__name__}: {exc})"
try:
    import torch
    out["torch_cuda"] = torch.version.cuda
except Exception:
    pass
out["model_modules"] = {}
for name in ("sglang.srt.models.deepseek_v41", "sglang.srt.models.deepseek_v4"):
    try:
        out["model_modules"][name] = importlib.util.find_spec(name) is not None
    except Exception as exc:
        out["model_modules"][name] = f"ERROR ({type(exc).__name__}: {exc})"
print(json.dumps(out))
"""
    probe_run = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=900, check=False)
    try:
        out["engine"] = json.loads(probe_run.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        out["engine"] = f"probe failed: {probe_run.stderr[-2000:]}"
    help_run = subprocess.run(
        [python, "-m", "sglang.launch_server", "--help"], capture_output=True, text=True, timeout=900, check=False
    )
    help_text = help_run.stdout + help_run.stderr
    out["launch_server_help_ok"] = help_run.returncode == 0
    out["flags"] = {
        flag: flag in help_text
        for flag in (
            "--reasoning-parser",
            "--tool-call-parser",
            "--ep-size",
            "--cuda-graph-max-bs-decode",
            "--model-loader-extra-config",
            "--weight-loader-prefetch-checkpoints",
            "--speculative-dspark-block-size",
        )
    }
    pkg_run = subprocess.run(
        [python, "-c", "import os, sglang; print(os.path.dirname(sglang.__file__))"],
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    pkg = pkg_run.stdout.strip()
    if pkg:  # does this build know the model's architecture and the V4.1 parsers?
        srt = Path(pkg) / "srt"
        # V4.1 is served by DeepseekV4ForCausalLM; DeepseekV41Config is what maps the checkpoint's architecture onto it.
        for needle in ("DeepseekV41Config", cfg.REASONING_PARSER, cfg.TOOL_CALL_PARSER, "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"):
            hits = subprocess.run(["grep", "-rl", needle, str(srt)], capture_output=True, text=True, check=False).stdout
            out[f"source_has:{needle}"] = len(hits.split())
    seed = Path(cfg.IMAGE_CACHE_SEED)
    out["image_cache_seed"] = sorted(p.name for p in seed.iterdir()) if seed.is_dir() else []
    return out


# ------------------------------------------------------------------ the server
@app.cls(
    image=server_image,
    gpu=cfg.GPU,
    cpu=cfg.CPU_CORES,
    memory=cfg.MEMORY_MIB,
    timeout=cfg.TIMEOUT_S,
    startup_timeout=cfg.STARTUP_TIMEOUT_S,
    scaledown_window=cfg.SCALEDOWN_S,
    max_containers=cfg.MAX_CONTAINERS,
    region=cfg.REGION,
    volumes={str(cfg.MODELS_DIR): weights_volume, str(cfg.CACHE_DIR): cache_volume},
)
@modal.concurrent(max_inputs=cfg.MAX_INPUTS)
class DeepSeekServer:
    @modal.enter()
    def start(self) -> None:
        from .engine import EngineSettings, SGLangServer, resolve_engine_python, seed_cache_dir, visible_gpu_count

        t0 = time.perf_counter()
        if not Path(cfg.MODEL_DIR, "config.json").exists():
            raise RuntimeError(f"no weights at {cfg.MODEL_DIR}; run `dsv4 download` first")
        seed_cache_dir(cfg.IMAGE_CACHE_SEED, cfg.CACHE_DIR)
        python = resolve_engine_python(cfg.ENGINE_PYTHON)
        gpus = visible_gpu_count(python)
        self.settings = EngineSettings(
            model_dir=str(cfg.MODEL_DIR),
            python=python,
            served_model_name=cfg.SERVED_MODEL_NAME,
            port=cfg.PORT,
            tp=cfg.TP or gpus,
            context_len=cfg.CONTEXT_LEN,
            mem_fraction_static=cfg.MEM_FRACTION_STATIC,
            max_running_requests=cfg.MAX_RUNNING_REQUESTS,
            cuda_graph_max_bs=cfg.CUDA_GRAPH_MAX_BS,
            chunked_prefill=cfg.CHUNKED_PREFILL,
            reasoning_parser=cfg.REASONING_PARSER,
            tool_call_parser=cfg.TOOL_CALL_PARSER,
            default_effort=cfg.DEFAULT_EFFORT,
            speculative=cfg.SPECULATIVE,
            dspark_block_size=cfg.DSPARK_BLOCK_SIZE,
            engram_host_table=cfg.ENGRAM_HOST_TABLE,
            engram_layout=cfg.ENGRAM_LAYOUT,
            fast_load=cfg.FAST_LOAD,
            extra_args=cfg.EXTRA_ARGS,
        )
        print(f"{gpus} GPU(s) visible -> tp={self.settings.tp} ep={self.settings.ep_size}; gpu spec {cfg.GPU_SPEC}")
        self.server = SGLangServer(self.settings)
        self.server.start()
        self.ready_s = self.server.wait_ready(max(cfg.STARTUP_TIMEOUT_S - 120, 300))
        self.warmup_s = self.server.warmup()
        try:
            cache_volume.commit()  # persist JIT artifacts produced during this boot
        except Exception as exc:  # noqa: BLE001
            print(f"cache volume commit skipped: {exc}")
        self.load_s = round(time.perf_counter() - t0)
        print(f"DeepSeekServer up in {self.load_s}s (server ready {self.ready_s:.0f}s, warmup {self.warmup_s})")

    @modal.exit()
    def stop(self) -> None:
        server = getattr(self, "server", None)
        if server is not None:
            server.stop()

    @modal.method()
    def generate(self, request: dict) -> Iterator[dict]:
        """Stream one chat completion as StreamChunk dicts (reasoning / content / done)."""
        from .schemas import GenerationRequest

        req = GenerationRequest.model_validate(request)
        for chunk in self.server.stream_chat(req):
            yield chunk.model_dump(exclude_none=True)

    @modal.method()
    def complete(self, request: dict) -> dict:
        """Non-streaming variant, meant for `.spawn()` so a dropped connection wastes nothing."""
        from .schemas import GenerationRequest

        return self.server.complete(GenerationRequest.model_validate(request)).model_dump()

    @modal.method()
    def info(self) -> dict:
        return {
            "gpu": cfg.GPU_SPEC,
            "image": cfg.IMAGE,
            "load_s": self.load_s,
            "ready_s": round(self.ready_s),
            "warmup_s": self.warmup_s,
            "settings": self.settings.summary(),
            "server_info": self.server.server_info(),
        }

    if cfg.WEB_SERVER:

        @modal.web_server(cfg.PORT, startup_timeout=cfg.STARTUP_TIMEOUT_S, requires_proxy_auth=True)
        def api(self) -> None:
            """OpenAI-compatible API: the SGLang server started in `start()` already listens on PORT."""


# ------------------------------------------------------------------ keep-warm helper
@app.function(image=download_image, cpu=0.125, memory=256, timeout=24 * 3600)
def auto_down(delay_s: int) -> dict:
    """Sleep, then let the server scale to zero again (`dsv4 up --hours N` spawns this)."""
    time.sleep(max(0, min(delay_s, 24 * 3600 - 600)))
    handle = modal.Cls.from_name(cfg.APP_NAME, "DeepSeekServer")().generate
    settings = handle.update_autoscaler(min_containers=0)
    return {"min_containers": settings.min_containers, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# ------------------------------------------------------------------ dev entrypoint
@app.local_entrypoint()
def main(
    prompt: str = "Explain, in three sentences, why mixture-of-experts models are cheap to run per token.",
    system: str = "",
    effort: str = cfg.DEFAULT_EFFORT,
    max_tokens: int = 4096,
) -> None:
    """Dev entrypoint: one streamed generation (ephemeral app, so it always pays a cold start)."""
    import sys

    from .schemas import GenerationRequest, GenerationResult, StreamChunk, parse_effort

    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    req = GenerationRequest(messages=messages, effort=parse_effort(effort), max_tokens=max_tokens)
    chunks: list[StreamChunk] = []
    for raw in DeepSeekServer().generate.remote_gen(req.model_dump()):
        chunk = StreamChunk.model_validate(raw)
        chunks.append(chunk)
        stream = sys.stderr if chunk.kind == "reasoning" else sys.stdout
        stream.write(chunk.text)
        stream.flush()
    result = GenerationResult.from_chunks(chunks)
    print(f"\n--- finish={result.finish_reason} usage={result.usage} stats={result.stats}", file=sys.stderr)
