"""Deploy-time configuration.

Every knob can be overridden with a `DSV4_*` environment variable when the app is deployed,
e.g. `DSV4_GPU=H200:8 uv run modal deploy -m deepseek_modal.app`. The resolved values are
also baked into the server image environment (`exported_env()`), so the code that runs inside
the container sees exactly the settings of the machine that deployed it.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath

APP_NAME = "deepseek-v41"
MODEL_ID = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8"
MODEL_REVISION = "d61c59ea5e514e25d305b5850e8a432f7a9969f2"  # main as of 2026-09-11, pinned

# Modal Volumes (v2): one for the 510 GB checkpoint, one for JIT / compile caches.
WEIGHTS_VOLUME = "deepseek-v41-weights"
CACHE_VOLUME = "deepseek-v41-cache"
MODELS_DIR = PurePosixPath("/models")
MODEL_DIR = MODELS_DIR / MODEL_ID
# The cache Volume is mounted on an empty path (Modal refuses non-empty mount points) and the image's
# /root/.cache becomes a symlink to it, so every library's default cache lands on the Volume.
CACHE_DIR = PurePosixPath("/cache")
IMAGE_CACHE_SEED = PurePosixPath("/opt/image-cache")  # the image's original /root/.cache, see app.py

_EXPORTED: dict[str, str] = {}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    _EXPORTED[name] = value
    return value


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(float(_env(name, str(default))))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def exported_env() -> dict[str, str]:
    """All DSV4_* settings as resolved on the deploying machine, for `Image.env`."""
    return dict(_EXPORTED)


# ------------------------------------------------------------------ images
# SGLang preview build with DeepSeek-V4.1 support. This digest is the image the SGLang cookbook
# lists for its verified 4xB200 cells (2026-10-16). The model authors validated the older
# `lmsysorg/sglang:dev-dsv41` (2026-09-11), which works as a fallback (set DSV4_FAST_LOAD=0 there).
IMAGE = _env(
    "DSV4_IMAGE",
    "lmsysorg/sglang:nightly-dev-cu13-20261014-684fea8a"
    "@sha256:ddae0e399d3d94f7827f16183301cf8dac4cf0eae0ef25590b7dac5881d713d8",
)
DOWNLOAD_PYTHON = "3.12"  # debian_slim for the download / helper functions
# Modal runs our code on its own interpreter (`add_python`), so its dependency pins never touch the
# SGLang venv; the server itself is launched with the image's interpreter (ENGINE_PYTHON).
IMAGE_PYTHON = "3.12"
ENGINE_PYTHON = _env("DSV4_ENGINE_PYTHON", "/opt/sglang/bin/python")  # falls back to python3 on PATH

# ------------------------------------------------------------------ container sizing
# The checkpoint needs ~614 GB of GPU memory unless the 183 GiB Engram tables are moved to host
# RAM. 4xB200 (720 GB) fits without that; 4xH200 (564 GB) needs DSV4_ENGRAM_HOST_TABLE=1 and
# ~200 GiB of host RAM; 8xH200 fits without it. A comma-separated list is a Modal fallback list
# (first available wins); tensor parallelism follows the number of GPUs the container gets.
# 4xB200 capacity was scarce on 2026-10-19 (30 min without a worker), hence the 8xH200 fallback.
GPU_SPEC = _env("DSV4_GPU", "B200:4,H200:8")
GPU: str | list[str] = [g.strip() for g in GPU_SPEC.split(",")] if "," in GPU_SPEC else GPU_SPEC
CPU_CORES = _env_float("DSV4_CPU", 16)
ENGRAM_HOST_TABLE = _env_bool("DSV4_ENGRAM_HOST_TABLE", False)
MEMORY_MIB = int(_env_float("DSV4_MEMORY_GIB", 256 if ENGRAM_HOST_TABLE else 128) * 1024)
TIMEOUT_S = int(_env_float("DSV4_TIMEOUT_MIN", 60) * 60)  # per generation call
STARTUP_TIMEOUT_S = int(_env_float("DSV4_STARTUP_TIMEOUT_MIN", 60) * 60)  # weight load + JIT + CUDA graphs
SCALEDOWN_S = min(int(_env_float("DSV4_SCALEDOWN_MIN", 20) * 60), 20 * 60)  # Modal caps this at 20 min
MAX_CONTAINERS = _env_int("DSV4_MAX_CONTAINERS", 1)  # spend cap: never two 4-GPU replicas
MAX_INPUTS = _env_int("DSV4_MAX_INPUTS", 8)  # concurrent requests per container
REGION = _env("DSV4_REGION", "") or None  # e.g. "us-east"; None lets Modal choose
WEB_SERVER = _env_bool("DSV4_WEB", False)  # also expose the OpenAI-compatible API (proxy auth)

# ------------------------------------------------------------------ engine (SGLang)
PORT = 30000
SERVED_MODEL_NAME = _env("DSV4_SERVED_MODEL_NAME", "deepseek-v4.1-flash")
TP = _env_int("DSV4_TP", 0)  # 0 = one rank per visible GPU; expert parallelism matches it
CONTEXT_LEN = _env_int("DSV4_CONTEXT_LEN", 262144)
# Weights + KV pool share of GPU memory. Without the host table the weights alone are ~128 GB
# per GPU at TP4 on B200 (180 GB); with it they drop to ~76 GB (cookbook uses 0.70-0.78 then).
MEM_FRACTION_STATIC = _env_float("DSV4_MEM_FRACTION", 0.78 if ENGRAM_HOST_TABLE else 0.85)
MAX_RUNNING_REQUESTS = _env_int("DSV4_MAX_RUNNING_REQUESTS", 8)
CUDA_GRAPH_MAX_BS = _env_int("DSV4_CUDA_GRAPH_MAX_BS", 8)
CHUNKED_PREFILL = _env_int("DSV4_CHUNKED_PREFILL", 4096)
REASONING_PARSER = _env("DSV4_REASONING_PARSER", "deepseek-v41")  # explicit: `auto` leaks <think> on older builds
TOOL_CALL_PARSER = _env("DSV4_TOOL_CALL_PARSER", "deepseekv41")
DEFAULT_EFFORT = _env("DSV4_DEFAULT_EFFORT", "high")  # server default; the CLI sends one per request anyway
SPECULATIVE = _env("DSV4_SPECULATIVE", "")  # "" or "dspark" (draft head quality is doubtful on this build)
DSPARK_BLOCK_SIZE = _env_int("DSV4_DSPARK_BLOCK_SIZE", 5)
FAST_LOAD = _env_bool("DSV4_FAST_LOAD", True)  # multithreaded safetensors load + prefetch (newer builds only)
ENGRAM_LAYOUT = _env("DSV4_ENGRAM_LAYOUT", "per_rank")
EXTRA_ARGS = _env("DSV4_EXTRA_ARGS", "")  # appended verbatim to the launch command (shlex split)

# ------------------------------------------------------------------ cost hints (USD per GPU hour, modal.com/pricing 2026-10-19)
GPU_HOURLY_USD = {"B300": 7.10, "B200": 6.25, "H200": 4.54, "H100": 3.95, "A100": 2.50}


def hourly_cost_usd(spec: str) -> float | None:
    """Rough GPU cost per hour for one Modal gpu spec such as `B200:4`."""
    name, _, count = spec.partition(":")
    rate = GPU_HOURLY_USD.get(name.rstrip("!+").upper())
    if rate is None:
        return None
    return rate * (int(count) if count else 1)


def hourly_cost_text(spec: str | list[str] | None = None) -> str:
    """Human-readable cost of the configured GPU spec(s), e.g. `$25/h (B200:4) or $36/h (H200:8)`."""
    specs = GPU if spec is None else spec
    specs = specs if isinstance(specs, list) else [specs]
    parts = []
    for s in specs:
        cost = hourly_cost_usd(s)
        parts.append(f"${cost:.0f}/h ({s})" if cost is not None else f"?/h ({s})")
    return " or ".join(parts)
