# deepseek-modal

Run [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8)
(a 552B-parameter MoE, 510 GB of FP8/FP4 weights, 1M context) on [Modal](https://modal.com) with
[SGLang](https://github.com/sgl-project/sglang), and drive it from the command line.

* Serving: SGLang's DeepSeek-V4.1 preview build on one multi-GPU container (`B200:4`, falling
  back to `H200:8` when no B200 node is free; tensor + expert parallel), weights streamed from a
  Modal Volume.
* Client: `dsv4`, a small typer CLI that streams reasoning and answer over Modal's RPC
  (`remote_gen`). No public endpoint, no extra credentials beyond your Modal token.
* Cost control: `dsv4 up` / `dsv4 down` toggle a warm container; nothing else keeps GPUs busy.

The model is an "abliterated" build whose authors state that users are responsible for what they
generate. The base model is MIT licensed. Design notes and the research behind the defaults are in
[docs/design.md](docs/design.md) (Japanese).

## Setup

```bash
uv sync                                        # local deps: modal, typer, pydantic
uv run modal token new                         # once, if ~/.modal.toml does not exist yet
MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run modal deploy -m deepseek_modal.app   # image build: ~15 GB pull
uv run dsv4 env-check                          # CPU container: versions and launch flags in the image
uv run dsv4 download                           # one-off: ~510 GB into the `deepseek-v41-weights` Volume
```

`dsv4 download` runs on a CPU container and is resumable; `dsv4 weights` compares the Volume with
the Hub file list afterwards.

The SGLang image needs Modal's `2025.06` image builder: the legacy `2023.12` builder tries to pin
2022-era packages (aiohttp 3.8, pydantic 1) into the image's Python 3.12 and fails. Either pass
`MODAL_IMAGE_BUILDER_VERSION=2025.06` as above or make it the workspace default under
Settings > Image Config. Modal runs this project's code on its own interpreter (`add_python`), so
the SGLang venv at `/opt/sglang` is never modified.

## Generate

```bash
uv run dsv4 generate "Explain speculative decoding in two paragraphs."
uv run dsv4 generate -s "Answer in Japanese." -e max --max-tokens 65536 "..."   # deeper reasoning
uv run dsv4 generate -e none "..."                                               # thinking off
uv run dsv4 generate -f prompt.md --hide-reasoning > answer.md                   # answer only on stdout
cat notes.txt | uv run dsv4 generate - --json                                    # full result as JSON
uv run dsv4 generate --no-stream "..."          # spawn a job; `dsv4 jobs fetch <id>` picks it up later
uv run dsv4 chat -e high                        # REPL: /effort, /system, /reset, /save, /exit
```

Reasoning is streamed to stderr (dimmed), the answer to stdout, followed by one summary line
(`[12.3s  ttft 0.4s  512 tokens  98.1 tok/s  effort=high  finish=stop]`). `--effort` takes
`low|high|xhigh|max`, an integer budget 1-100, or `none`. Sampling defaults follow the model card
(temperature 1.0, top_p 0.95); `--max-tokens` covers reasoning and answer together.

### Cold starts and keeping the server warm

The first call after an idle period boots a fresh container: 510 GB of weights are read from the
Volume, FlashInfer autotunes its kernels, JIT caches are filled (persisted in the
`deepseek-v41-cache` Volume) and CUDA graphs are captured. Expect about 15 minutes, at about $25/h
for 4xB200 or $36/h for 8xH200 (the app logs say which worker type Modal is waiting for). Idle
containers are released after `DSV4_SCALEDOWN_MIN` minutes (20, Modal's maximum). For a working
session:

```bash
uv run dsv4 up --wait --hours 2   # min_containers=1, block until ready; auto_down resets it after 2 h
uv run dsv4 status                # autoscaler settings, running containers
uv run dsv4 logs                  # SGLang log of the latest boot (stored on the cache Volume)
uv run dsv4 down                  # back to scale-to-zero; the container stops after the idle window
uv run dsv4 down --now            # ... and stop the idle container right away
```

`down --now` works by shrinking the idle window to two seconds for a moment; `modal container stop`
is not a substitute, because the autoscaler replaces a stopped container while it still wants one.
About 7 of the 15 minutes are FlashInfer tuning its MoE/GEMM kernels. `DSV4_FLASHINFER_AUTOTUNE=0`
removes the explicit autotune pass, but on this build the same tuning then happens lazily during
CUDA graph capture (488 s instead of 87 s), so the boot is no shorter; the on-disk autotune cache
did not shorten it either. Keeping a container warm is the only effective lever.

## Coding agents

The model serves tool calls and reasoning through all three API flavours SGLang offers
(OpenAI chat completions, Anthropic Messages, OpenAI Responses). `dsv4 serve` exposes them on
localhost and forwards over Modal's RPC, so no public endpoint or extra credential is needed:

```bash
uv run dsv4 serve --port 8000       # http://127.0.0.1:8000/v1 (OpenAI), http://127.0.0.1:8000 (Anthropic)
uv run dsv4 agent claude            # Claude Code against this model (proxy + env set up for you)
uv run dsv4 agent codex             # Codex CLI (provider overrides passed with -c)
uv run dsv4 agent openai -- aider --model openai/deepseek-v4.1-flash
```

`dsv4 agent` starts the proxy on a free port, keeps one container warm while the tool runs
(`--no-warm` to skip), launches the tool with its environment pointed at the proxy, and resets
`min_containers` when it exits. The proxy rewrites the model name to the served one and injects a
reasoning effort (`--effort`, default `DSV4_PROXY_EFFORT=high`) into requests that carry no
reasoning control, so answers come back with reasoning separated instead of plain.

Manual configuration, for tools without a preset (all through `dsv4 serve`):

| Tool | Setting |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`, `ANTHROPIC_AUTH_TOKEN=any`, `ANTHROPIC_MODEL=deepseek-v4.1-flash` (also the `ANTHROPIC_DEFAULT_*_MODEL` variables), `CLAUDE_CODE_EFFORT_LEVEL=high`, `CLAUDE_CODE_ATTRIBUTION_HEADER=0`, `API_TIMEOUT_MS=3000000` |
| Codex CLI | `~/.codex/config.toml`: `model_provider = "dsv4"`, `model = "deepseek-v4.1-flash"`, `[model_providers.dsv4] base_url = "http://127.0.0.1:8000/v1"`, `wire_api = "responses"`, `model_reasoning_effort = "high"` |
| OpenCode | `opencode.json` provider with `npm: "@ai-sdk/openai-compatible"`, `options.baseURL: "http://127.0.0.1:8000/v1"`, model `deepseek-v4.1-flash` with `limit.context` 262144 |
| aider | `OPENAI_API_BASE=http://127.0.0.1:8000/v1 OPENAI_API_KEY=any aider --model openai/deepseek-v4.1-flash` |
| Cline / Roo / Kilo / Continue | "OpenAI Compatible" provider, base URL `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash` |

Notes from testing against this build of SGLang:

* Requests that say nothing about thinking get a plain answer (`DSV4_DEFAULT_THINKING=0`). With
  the server default set to thinking on, such requests come back with the reasoning leaked into
  `content` and a stray `</think>`, so the proxy's injection is the safer way to get reasoning by default.
* Anthropic clients: `thinking.type` `enabled` or `adaptive` both turn reasoning on,
  `output_config.effort` maps to the model's effort, and `cache_control`, `strict` tools,
  `metadata`, `context_management` are accepted. `/v1/messages/count_tokens` works.
* Responses API: `reasoning.effort`, `instructions`, function tools and streaming work; Codex
  uses this API.
* Prefix caching makes multi-turn agent loops cheap; `CLAUDE_CODE_ATTRIBUTION_HEADER=0` keeps
  the prompt prefix stable for it.
* Requests are capped at 8 concurrent per container (`DSV4_MAX_INPUTS`), beyond that they queue.

## Configuration

Deploy-time knobs are environment variables read by `src/deepseek_modal/config.py`
(`DSV4_GPU=H200:8 uv run modal deploy -m deepseek_modal.app`). They are also baked into the image
environment, so the container runs with the values of the machine that deployed it.

| Variable | Default | Meaning |
|---|---|---|
| `DSV4_GPU` | `B200:4,H200:8` | Modal GPU spec; a comma-separated list is a fallback list (first available wins). TP/EP follow the GPU count |
| `DSV4_ENGRAM_HOST_TABLE` | `0` | `1` moves the 183 GiB Engram tables to host RAM (needed for `H200:4`; raises memory to 256 GiB) |
| `DSV4_MEMORY_GIB` | `128` (`256` with host table) | Host RAM request |
| `DSV4_CPU` | `16` | CPU cores |
| `DSV4_CONTEXT_LEN` | `262144` | `--context-length` (up to 1048576) |
| `DSV4_MEM_FRACTION` | `0.85` (`0.78` with host table) | `--mem-fraction-static` |
| `DSV4_MAX_RUNNING_REQUESTS` / `DSV4_CUDA_GRAPH_MAX_BS` | `8` / `8` | Concurrency inside the server |
| `DSV4_MAX_INPUTS` | `8` | Concurrent calls per container on the Modal side |
| `DSV4_SPECULATIVE` | `` | `dspark` enables DSpark speculative decoding (block size `DSV4_DSPARK_BLOCK_SIZE`, 5) |
| `DSV4_FAST_LOAD` | `1` | Multithreaded weight loading + prefetch flags (set `0` on the older `dev-dsv41` image) |
| `DSV4_IMAGE` | SGLang nightly `20261014`, digest pinned | Server image. `lmsysorg/sglang:dev-dsv41` is the build the model authors validated |
| `DSV4_TIMEOUT_MIN` / `DSV4_STARTUP_TIMEOUT_MIN` | `60` / `60` | Per-call timeout; time allowed for the boot |
| `DSV4_SCALEDOWN_MIN` | `20` | Idle time before a warm container is released (Modal caps it at 20) |
| `DSV4_MAX_CONTAINERS` | `1` | Spend cap |
| `DSV4_REGION` | unset | Modal region |
| `DSV4_WEB` | `0` | `1` also exposes SGLang's OpenAI-compatible API behind Modal proxy auth (experimental) |
| `DSV4_DEFAULT_THINKING` | `0` | Thinking for requests without any reasoning control (`1` leaks reasoning into `content` on this build) |
| `DSV4_PROXY_EFFORT` | `high` | Effort `dsv4 serve` / `dsv4 agent` inject into requests without one |
| `DSV4_FLASHINFER_AUTOTUNE` | `1` | `0` adds `--disable-flashinfer-autotune` (no faster on this build: the tuning moves into graph capture) |
| `DSV4_EXTRA_ARGS` | `` | Extra `sglang.launch_server` flags, appended verbatim |
| `DSV4_ENGINE_PYTHON` | `/opt/sglang/bin/python` | Interpreter inside the image that has sglang (falls back to `python3`) |

Sizing background: the checkpoint needs roughly 614 GB of GPU memory unless the Engram tables are
offloaded. 4xB200 (720 GB) fits as is; 4xH200 (564 GB) needs `DSV4_ENGRAM_HOST_TABLE=1`; 8xH200
fits without it. Hourly Modal rates (2026-10): B200 $6.25, H200 $4.54, H100 $3.95 per GPU.

## How it works

```
src/deepseek_modal/
  config.py    deploy-time settings (DSV4_* env overrides), exported into the image env
  schemas.py   GenerationRequest -> OpenAI payload with DeepSeek-V4.1 reasoning controls; StreamChunk; GenerationResult
  engine.py    sglang.launch_server command/env, health wait, SSE parsing, chunk coalescing (no modal import)
  app.py       modal.App: images, Volumes, download(), weights_status(), env_check(), DeepSeekServer, auto_down()
  cli.py       `dsv4` (typer): talks to the deployed app via modal.Cls.from_name; `serve`/`agent` run a local aiohttp proxy
```

`DeepSeekServer.start()` seeds the cache Volume (mounted at `/cache`, with the image's
`/root/.cache` symlinked to it) from the image's original cache contents, launches
`python3 -m sglang.launch_server` with `--tp-size N --ep-size N` (N = visible GPUs),
`--reasoning-parser deepseek-v41 --tool-call-parser deepseekv41`, waits for `/health`, and runs a
warmup request. `generate()` posts to the local OpenAI-compatible endpoint with `stream=True`,
splits `reasoning_content` from `content`, coalesces deltas (~10 messages/s) and yields them; the
final chunk carries usage and timings. `complete()` is the non-streaming twin for `spawn()`.

## Measured performance (4xB200, 2026-10-19)

Three boots with `DSV4_CONTEXT_LEN=262144`, no speculative decoding. The second and third boots
had the JIT caches and the FlashInfer autotune files on the cache Volume already:

| Phase | Boot 1 (empty caches) | Boot 2 | Boot 3 |
|---|---|---|---|
| Python init + model registry | ~150 s | ~150 s | ~70 s |
| NCCL / parallel init | ~75 s | 74 s | 76 s |
| Weight load from the Volume (510 GB) | 276 s | 138 s | 134 s |
| FlashInfer autotune | ~500 s | 393 s | 434 s |
| CUDA graph capture (decode, bs 1-8) | 49 s | 93 s | 87 s |
| `/health` ready | 871 s | 887 s | 910 s |

The autotune pass dominates and its on-disk cache did not shorten it on this build. A fourth boot
with `DSV4_FLASHINFER_AUTOTUNE=0` took 876 s: no autotune phase, but CUDA graph capture grew to
488 s because the kernels were tuned there instead, and generation speed was unchanged
(295 tok/s). Each boot costs about $6.5 of GPU time; `dsv4 up --wait` reports the numbers for
the boot it triggered and `dsv4 logs` has the full SGLang log.

After capture the server reported 25 GB free per GPU and a KV pool of 18.6M tokens. Warm requests
(single stream, numbers from the CLI's summary line):

| Request | Output | Time | Speed |
|---|---|---|---|
| effort=low, three-sentence answer | 191 tokens | 2.1 s (ttft 1.5 s) | 304 tok/s |
| effort=high, code + tests, `--max-tokens 4096` | 4096 tokens (budget hit) | 14.4 s (ttft 0.3 s) | 289 tok/s |
| effort=low, 700-word explainer | 2394 tokens | 8.4 s (ttft 0.2 s) | 292 tok/s |
| effort=low, Japanese one-liner | 91 tokens (76 reasoning) | 0.4 s | n/a |

Coding agents through `dsv4 agent` (warm container): Claude Code `-p` reading a file and
answering took 55 s wall including its own startup; Codex `exec` doing the same took 11-15 s.

One request with thinking off took 15.7 s to its first token while three other requests were in
flight; the logs show SGLang compiling a Triton kernel at serving time, which stalls the engine
once per new shape.

The weight download took 1470 s for 475.3 GiB on one CPU container.

## Development

```bash
uv run pytest                                            # schemas + engine, no GPU needed
uv run modal run -m deepseek_modal.app --prompt "..."    # ephemeral app: one streamed generation
```
