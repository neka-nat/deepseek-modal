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
Volume, kernels are JIT-compiled (cached in the `deepseek-v41-cache` Volume after the first boot)
and CUDA graphs are captured. Expect 10-25 minutes, at about $25/h for 4xB200 or $36/h for 8xH200
(the app logs say which worker type Modal is waiting for). Idle containers are
released after `DSV4_SCALEDOWN_MIN` minutes (20, Modal's maximum). For a working session:

```bash
uv run dsv4 up --hours 2     # min_containers=1; an auto_down job resets it after 2 h
uv run dsv4 status           # autoscaler settings, running containers
uv run dsv4 down             # back to scale-to-zero right away
```

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
  cli.py       `dsv4` (typer): talks to the deployed app via modal.Cls.from_name
```

`DeepSeekServer.start()` seeds the cache Volume (mounted at `/cache`, with the image's
`/root/.cache` symlinked to it) from the image's original cache contents, launches
`python3 -m sglang.launch_server` with `--tp-size N --ep-size N` (N = visible GPUs),
`--reasoning-parser deepseek-v41 --tool-call-parser deepseekv41`, waits for `/health`, and runs a
warmup request. `generate()` posts to the local OpenAI-compatible endpoint with `stream=True`,
splits `reasoning_content` from `content`, coalesces deltas (~10 messages/s) and yields them; the
final chunk carries usage and timings. `complete()` is the non-streaming twin for `spawn()`.

## Measured performance (4xB200, 2026-10-19)

First boot, empty JIT cache, `DSV4_CONTEXT_LEN=262144`, no speculative decoding:

| Phase | Time |
|---|---|
| Scheduling (B200:4 worker found) | ~30 s (an earlier attempt waited 30 min without one) |
| Weight load from the Volume (510 GB) | 276 s (about 1.85 GB/s) |
| Scheduler start incl. JIT compilation | 792 s end to end |
| CUDA graph capture (decode, bs 1-8) | 49 s |
| `/health` ready, as seen by the container | 871 s |
| `dsv4 generate` wall time incl. cold start | 15 min 25 s (about $6.5 of GPU time) |

After capture the server reported 25 GB free per GPU and a KV pool of 18.6M tokens. Warm requests
(single stream, numbers from the CLI's summary line):

| Request | Output | Time | Speed |
|---|---|---|---|
| effort=low, three-sentence answer | 191 tokens | 2.1 s (ttft 1.5 s) | 304 tok/s |
| effort=high, code + tests, `--max-tokens 4096` | 4096 tokens (budget hit) | 14.4 s (ttft 0.3 s) | 289 tok/s |
| effort=low, 700-word explainer | 2394 tokens | 8.4 s (ttft 0.2 s) | 292 tok/s |
| effort=low, Japanese one-liner | 91 tokens (76 reasoning) | 0.4 s | n/a |

One request with thinking off took 15.7 s to its first token while three other requests were in
flight; the logs show SGLang compiling a Triton kernel at serving time, which stalls the engine
once per new shape. The second boot should be faster, since the JIT cache now lives in the Volume;
not measured yet.

The weight download took 1470 s for 475.3 GiB on one CPU container.

## Development

```bash
uv run pytest                                            # schemas + engine, no GPU needed
uv run modal run -m deepseek_modal.app --prompt "..."    # ephemeral app: one streamed generation
```
