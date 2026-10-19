"""`dsv4`: command line client for the deployed Modal app.

Only needs `modal`, `typer` and `pydantic` locally (aiohttp comes with modal); SGLang and the
weights live on Modal. Everything goes over Modal's own RPC, so the only credential involved is
your Modal token, and `dsv4 serve` gives local tools an OpenAI / Anthropic / Responses-compatible
endpoint without exposing anything publicly.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer

from . import config as cfg
from .schemas import EFFORT_LEVELS, GenerationRequest, GenerationResult, StreamChunk, parse_effort

app = typer.Typer(
    help="Generate text with DeepSeek-V4.1-Flash on Modal. Deploy first: uv run modal deploy -m deepseek_modal.app",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
jobs_app = typer.Typer(help="Inspect and fetch spawned (non-streaming) generations.", no_args_is_help=True)
app.add_typer(jobs_app, name="jobs")

STATE_DIR = Path.home() / ".cache" / "deepseek-v41"
JOBS_FILE = STATE_DIR / "jobs.jsonl"
AUTO_DOWN_FILE = STATE_DIR / "auto_down.json"
DEPLOY_HINT = "run: uv run modal deploy -m deepseek_modal.app"
EFFORT_HELP = f"Reasoning effort: {', '.join(EFFORT_LEVELS)}, an integer 1-100, or none (thinking off)"
AGENT_PRESETS = ("claude", "codex", "openai", "anthropic")


# ----------------------------------------------------------------- helpers
def _fail(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _note(message: str) -> None:
    typer.secho(message, fg=typer.colors.BLUE, err=True)


def _server() -> Any:
    import modal

    try:
        return modal.Cls.from_name(cfg.APP_NAME, "DeepSeekServer")()
    except modal.exception.NotFoundError:
        _fail(f"app {cfg.APP_NAME!r} is not deployed; {DEPLOY_HINT}")


def _function(name: str) -> Any:
    import modal

    try:
        return modal.Function.from_name(cfg.APP_NAME, name)
    except modal.exception.NotFoundError:
        _fail(f"app {cfg.APP_NAME!r} is not deployed; {DEPLOY_HINT}")


def _wait(function_call: Any, poll: float, what: str) -> Any:
    import modal

    t0 = time.monotonic()
    while True:
        try:
            result = function_call.get(timeout=poll)
            typer.echo("", err=True)
            return result
        except TimeoutError:
            typer.echo(f"\r  {what} ... {int(time.monotonic() - t0)}s", nl=False, err=True)
        except modal.exception.FunctionTimeoutError:
            typer.echo("", err=True)
            _fail("the remote call hit its timeout")
        except modal.exception.RemoteError as exc:
            typer.echo("", err=True)
            _fail(f"remote failure: {exc}")


def _effort(text: str) -> Any:
    try:
        return parse_effort(text)
    except ValueError as exc:
        _fail(str(exc))


def _prompt_text(prompt: str | None, file: Path | None) -> str:
    if file is not None:
        try:
            return file.read_text(encoding="utf-8")
        except OSError as exc:
            _fail(f"cannot read {file}: {exc}")
    if prompt == "-" or (prompt is None and not sys.stdin.isatty()):
        text = sys.stdin.read()
        if text.strip():
            return text
        _fail("empty prompt on stdin")
    if prompt is None:
        _fail("give a prompt argument, --file, or pipe text on stdin")
    return prompt


def _request(
    messages: list[dict[str, str]],
    effort: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
) -> GenerationRequest:
    try:
        return GenerationRequest(
            messages=messages, effort=effort, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
        )
    except ValueError as exc:
        _fail(str(exc))


def _containers() -> list[dict]:
    """Running containers of this app, straight from `modal container list --json` (the autoscaler
    stats only count containers of the current deployment version)."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "modal", "container", "list", "--json"], capture_output=True, text=True, timeout=120
        )
        rows = json.loads(out.stdout or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    return [row for row in rows if row.get("App Name") == cfg.APP_NAME or row.get("app_name") == cfg.APP_NAME]


def _autoscaler(server: Any) -> Any:
    """Autoscaler namespace of the server's container pool (shared by all its methods)."""
    return server.generate.autoscaler


def _cold_start_hint(server: Any) -> None:
    if not _containers():
        _note(f"no warm container: cold start ahead (loading 510 GB of weights, 10-15 min, {cfg.hourly_cost_text()})")


class _Printer:
    """Streams reasoning to stderr (dim) and the answer to stdout, then a one-line summary."""

    def __init__(self, *, show_reasoning: bool, quiet: bool) -> None:
        self.show_reasoning = show_reasoning
        self.quiet = quiet
        self.dim = sys.stderr.isatty()
        self.reasoning_started = False
        self.content_started = False
        self.last_content = ""

    def _err(self, text: str) -> None:
        typer.echo(typer.style(text, dim=True) if self.dim else text, nl=False, err=True)
        sys.stderr.flush()

    def feed(self, chunk: StreamChunk) -> None:
        if self.quiet:
            return
        if chunk.kind == "reasoning":
            if not self.show_reasoning:
                return
            if not self.reasoning_started:
                self._err("[reasoning]\n")
                self.reasoning_started = True
            self._err(chunk.text)
        elif chunk.kind == "content":
            if self.reasoning_started and not self.content_started:
                self._err("\n[answer]\n")
            self.content_started = True
            self.last_content = chunk.text
            typer.echo(chunk.text, nl=False)
            sys.stdout.flush()

    def finish(self, result: GenerationResult, effort: Any) -> None:
        if not self.quiet and self.content_started and not self.last_content.endswith("\n"):
            typer.echo("")
        stats = result.stats
        tokens = f"{stats.get('completion_tokens', '?')} tokens"
        if stats.get("reasoning_tokens"):
            tokens += f" ({stats['reasoning_tokens']} reasoning)"
        parts = [
            f"{stats.get('seconds', '?')}s",
            f"ttft {stats.get('ttft_s', '?')}s",
            tokens,
            f"{stats['tok_per_s']} tok/s" if stats.get("tok_per_s") else "tok/s n/a",
            f"effort={effort if effort is not None else 'none'}",
            f"finish={result.finish_reason}",
        ]
        typer.secho("[" + "  ".join(parts) + "]", fg=typer.colors.BLUE, err=True)


def _stream(server: Any, req: GenerationRequest, *, show_reasoning: bool, quiet: bool) -> GenerationResult:
    import modal

    printer = _Printer(show_reasoning=show_reasoning, quiet=quiet)
    chunks: list[StreamChunk] = []
    try:
        for raw in server.generate.remote_gen(req.model_dump()):
            chunk = StreamChunk.model_validate(raw)
            chunks.append(chunk)
            printer.feed(chunk)
    except KeyboardInterrupt:
        typer.echo("\n(interrupted)", err=True)
        raise typer.Exit(code=130) from None
    except modal.exception.FunctionTimeoutError:
        _fail("the generation hit the per-call timeout")
    except modal.exception.RemoteError as exc:
        _fail(f"remote failure: {exc}")
    result = GenerationResult.from_chunks(chunks, model=cfg.SERVED_MODEL_NAME)
    printer.finish(result, req.effort)
    return result


def _record_job(job_id: str, req: GenerationRequest, out: Path | None) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    entry = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": (last_user if isinstance(last_user, str) else "<parts>")[:120],
        "out": str(out) if out else None,
    }
    with JOBS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _emit_result(result: GenerationResult, *, out: Path | None, as_json: bool, printed: bool) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.content, encoding="utf-8")
        _note(f"saved answer to {out}")
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    elif not printed:
        typer.echo(result.content)


# ---------------------------------------------------------------- local proxy
def _make_proxy_app(server: Any, effort: Any) -> Any:
    """aiohttp app that forwards every request to the container over Modal RPC and streams back."""
    from aiohttp import web

    async def handle(request: Any) -> Any:
        import modal

        body = await request.read()
        headers = {k: v for k, v in request.headers.items()}
        gen = server.proxy.remote_gen.aio(request.method, request.path_qs, headers, body or None, effort)
        try:
            head = await gen.__anext__()
        except modal.exception.RemoteError as exc:
            return web.json_response({"error": {"message": f"remote failure: {exc}", "type": "proxy_error"}}, status=502)
        except StopAsyncIteration:
            return web.json_response({"error": {"message": "empty response from the container"}}, status=502)
        response = web.StreamResponse(status=head["status"], headers=head.get("headers") or {})
        await response.prepare(request)
        try:
            async for item in gen:
                data = item.get("data")
                if data:
                    await response.write(data)
            await response.write_eof()
        except (ConnectionResetError, ConnectionError, modal.exception.RemoteError):
            pass  # the client went away (or the container failed mid-stream); nothing left to say
        return response

    proxy = web.Application(client_max_size=256 * 1024 * 1024)
    proxy.router.add_route("*", "/{tail:.*}", handle)
    return proxy


class _ProxyThread:
    """Runs the aiohttp proxy on a background thread so `dsv4 agent` can spawn a tool in front of it."""

    def __init__(self, server: Any, effort: Any, host: str, port: int) -> None:
        self.host, self.port = host, port
        self._server, self._effort = server, effort
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()
        self._loop: Any = None
        self._stop_event: Any = None

    def _run(self) -> None:
        import asyncio

        from aiohttp import web

        async def main() -> None:
            self._stop_event = asyncio.Event()
            runner = web.AppRunner(_make_proxy_app(self._server, self._effort))
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            self._ready.set()
            try:
                await self._stop_event.wait()
            finally:
                await runner.cleanup()

        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(main())
        finally:
            self._loop.close()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=30):
            _fail("the local proxy did not start")

    def stop(self) -> None:
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=15)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _agent_env(preset: str, base: str, effort: Any) -> dict[str, str]:
    """Environment that points a known tool at the local proxy."""
    level = effort if isinstance(effort, str) else cfg.PROXY_EFFORT if effort is not None else "none"
    model = cfg.SERVED_MODEL_NAME
    env = dict(os.environ)
    if preset in {"claude", "anthropic"}:
        env.update(
            {
                "ANTHROPIC_BASE_URL": base,
                "ANTHROPIC_AUTH_TOKEN": "dsv4-local",
                "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "CLAUDE_CODE_SUBAGENT_MODEL": model,
                "CLAUDE_CODE_EFFORT_LEVEL": level if level in EFFORT_LEVELS else "high",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",  # keeps the prompt prefix stable for SGLang's prefix cache
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(min(cfg.CONTEXT_LEN, 1_000_000)),
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
                "API_TIMEOUT_MS": "3000000",
            }
        )
    if preset in {"codex", "openai"}:
        env.update({"OPENAI_BASE_URL": f"{base}/v1", "OPENAI_API_KEY": "dsv4-local"})
    return env


def _agent_command(preset: str, args: list[str], base: str, effort: Any) -> list[str]:
    if preset == "codex":
        level = effort if isinstance(effort, str) else "high"
        overrides = {
            "model_provider": "dsv4",
            "model": cfg.SERVED_MODEL_NAME,
            "model_providers.dsv4.name": "DeepSeek-V4.1-Flash on Modal",
            "model_providers.dsv4.base_url": f"{base}/v1",
            "model_providers.dsv4.wire_api": "responses",
            "model_reasoning_effort": level,
        }
        cmd = ["codex"]
        for key, value in overrides.items():
            cmd += ["-c", f'{key}="{value}"']
        cmd += ["-c", f"model_context_window={cfg.CONTEXT_LEN}"]
        return cmd + args
    if preset == "claude":
        return ["claude", *args]
    return args


# ---------------------------------------------------------------- commands
@app.command()
def generate(
    prompt: Annotated[str | None, typer.Argument(help="The user prompt ('-' reads stdin)")] = None,
    system: Annotated[str | None, typer.Option("--system", "-s", help="System prompt")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Read the prompt from a file")] = None,
    effort: Annotated[str, typer.Option("--effort", "-e", help=EFFORT_HELP)] = cfg.DEFAULT_EFFORT,
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1, help="Budget for reasoning + answer")] = 32768,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0, max=2.0)] = 1.0,
    top_p: Annotated[float, typer.Option("--top-p", min=0.01, max=1.0)] = 0.95,
    seed: Annotated[int | None, typer.Option("--seed", min=0)] = None,
    hide_reasoning: Annotated[bool, typer.Option("--hide-reasoning", help="Do not show the reasoning stream")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON instead of streaming")] = False,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Also save the answer to this file")] = None,
    stream: Annotated[bool, typer.Option("--stream/--no-stream", help="--no-stream spawns a job you can fetch later")] = True,
    poll: Annotated[float, typer.Option("--poll", help="Seconds between polls with --no-stream")] = 15.0,
) -> None:
    """Generate one answer. Reasoning goes to stderr, the answer to stdout (pipe-friendly)."""
    text = _prompt_text(prompt, file)
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    req = _request(messages, _effort(effort), max_tokens, temperature, top_p, seed)
    server = _server()
    _cold_start_hint(server)
    if stream:
        result = _stream(server, req, show_reasoning=not hide_reasoning, quiet=as_json)
        _emit_result(result, out=out, as_json=as_json, printed=True)
        return
    call = server.complete.spawn(req.model_dump())
    _record_job(call.object_id, req, out)
    _note(f"job {call.object_id} spawned; fetch later with: dsv4 jobs fetch {call.object_id}")
    result = GenerationResult.model_validate(_wait(call, poll, "generating"))
    _Printer(show_reasoning=False, quiet=True).finish(result, req.effort)
    _emit_result(result, out=out, as_json=as_json, printed=False)


@app.command()
def chat(
    system: Annotated[str | None, typer.Option("--system", "-s")] = None,
    effort: Annotated[str, typer.Option("--effort", "-e", help=EFFORT_HELP)] = cfg.DEFAULT_EFFORT,
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 32768,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0, max=2.0)] = 1.0,
    top_p: Annotated[float, typer.Option("--top-p", min=0.01, max=1.0)] = 0.95,
    hide_reasoning: Annotated[bool, typer.Option("--hide-reasoning")] = False,
) -> None:
    """Multi-turn chat in the terminal. Commands: /effort X, /system TEXT, /reset, /save FILE, /exit."""
    current_effort = _effort(effort)
    system_prompt = system
    history: list[dict[str, str]] = []
    server = _server()
    _cold_start_hint(server)
    _note("type a message; /exit to quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("", err=True)
            break
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            break
        if line == "/reset":
            history.clear()
            _note("history cleared")
            continue
        if line.startswith("/effort"):
            current_effort = _effort(line.partition(" ")[2] or "high")
            _note(f"effort = {current_effort if current_effort is not None else 'none'}")
            continue
        if line.startswith("/system"):
            system_prompt = line.partition(" ")[2] or None
            _note("system prompt " + ("set" if system_prompt else "cleared"))
            continue
        if line.startswith("/save"):
            target = Path(line.partition(" ")[2] or "chat.json")
            target.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            _note(f"saved {len(history)} messages to {target}")
            continue
        history.append({"role": "user", "content": line})
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + history
        req = _request(messages, current_effort, max_tokens, temperature, top_p)
        try:
            result = _stream(server, req, show_reasoning=not hide_reasoning, quiet=False)
        except typer.Exit:
            history.pop()  # interrupted or failed turn: keep the history consistent
            continue
        history.append({"role": "assistant", "content": result.content})  # reasoning is never fed back


@app.command()
def serve(
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    effort: Annotated[str, typer.Option("--effort", "-e", help="Injected when a client sends no reasoning control; none = leave requests alone")] = cfg.PROXY_EFFORT,
) -> None:
    """Local OpenAI-, Anthropic- and Responses-compatible endpoint forwarding to the Modal server.

    Point tools at http://127.0.0.1:PORT (Anthropic clients) or http://127.0.0.1:PORT/v1 (OpenAI
    clients); any API key works. Requests travel over Modal's RPC, so nothing is exposed publicly.
    """
    from aiohttp import web

    server = _server()
    _cold_start_hint(server)
    base = f"http://{host}:{port}"
    _note(f"OpenAI-compatible: {base}/v1   Anthropic-compatible: {base}   model: {cfg.SERVED_MODEL_NAME}")
    _note(f"default reasoning effort injected: {effort}   (Ctrl-C to stop)")
    web.run_app(_make_proxy_app(server, _effort(effort)), host=host, port=port, print=None)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def agent(
    ctx: typer.Context,
    preset: Annotated[str, typer.Argument(help="claude | codex | anthropic | openai (env only) ; then the tool's own arguments")],
    effort: Annotated[str, typer.Option("--effort", "-e", help=EFFORT_HELP)] = cfg.PROXY_EFFORT,
    warm: Annotated[bool, typer.Option("--warm/--no-warm", help="Keep one container running while the tool runs (and a bit after)")] = True,
) -> None:
    """Run a coding agent against this model: `dsv4 agent claude`, `dsv4 agent codex`, `dsv4 agent openai -- aider ...`.

    Starts a local proxy on a free port, sets the tool's environment (and config overrides for
    Codex), runs it, then stops the proxy. The GPU container stays warm for the idle window after.
    """
    if preset not in AGENT_PRESETS:
        _fail(f"preset must be one of {', '.join(AGENT_PRESETS)}")
    args = [a for a in ctx.args if a != "--"]
    if preset in {"anthropic", "openai"} and not args:
        _fail("give the command to run after the preset, e.g. `dsv4 agent openai -- aider --model openai/deepseek-v4.1-flash`")
    tool = (_agent_command(preset, args, "http://127.0.0.1:0", None) or [preset])[0]
    if shutil.which(tool) is None:
        _fail(f"{tool!r} is not on PATH")
    server = _server()
    level = _effort(effort)
    if warm:
        _autoscaler(server).update(min_containers=1)
    _cold_start_hint(server)
    port = _free_port()
    proxy = _ProxyThread(server, level, "127.0.0.1", port)
    proxy.start()
    base = f"http://127.0.0.1:{port}"
    cmd = _agent_command(preset, args, base, level)
    _note(f"proxy on {base}; running: {' '.join(cmd[:1] + [a for a in cmd[1:] if not a.startswith('model_providers')][:6])} ...")
    try:
        code = subprocess.call(cmd, env=_agent_env(preset, base, level))
    except KeyboardInterrupt:
        code = 130
    finally:
        proxy.stop()
        if warm:
            _autoscaler(server).update(min_containers=0)
            _note(f"min_containers back to 0; the container stops after {cfg.SCALEDOWN_S // 60} min idle (`dsv4 down --now` stops it sooner)")
    raise typer.Exit(code=code)


@app.command()
def download(
    wait: Annotated[bool, typer.Option("--wait/--no-wait")] = True,
    revision: Annotated[str, typer.Option("--revision")] = cfg.MODEL_REVISION,
    poll: Annotated[float, typer.Option("--poll")] = 30.0,
) -> None:
    """Download the checkpoint (~510 GB) into the weights Volume. One-off, resumable."""
    call = _function("download").spawn(revision)
    typer.echo(f"download spawned as job {call.object_id}")
    if not wait:
        return
    report = _wait(call, poll, "downloading")
    typer.echo(json.dumps(report, indent=2))
    if report.get("complete") is False:
        _fail("the download is incomplete; run `dsv4 download` again to resume")


@app.command()
def weights() -> None:
    """Show what the weights Volume holds versus the Hub file list."""
    typer.echo(json.dumps(_function("weights_status").remote(), indent=2))


@app.command("env-check")
def env_check() -> None:
    """Report versions and launch flags available inside the SGLang image (CPU container)."""
    typer.echo(json.dumps(_function("env_check").remote(), indent=2))


@app.command()
def up(
    hours: Annotated[float, typer.Option("--hours", min=0.1, max=24.0, help="Auto scale-down after this long")] = 2.0,
    auto_down: Annotated[bool, typer.Option("--auto-down/--no-auto-down")] = True,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Block until the server is ready")] = False,
) -> None:
    """Keep one server container running (min_containers=1), so calls skip the cold start."""
    server = _server()
    settings = _autoscaler(server).update(min_containers=1)
    typer.echo(f"min_containers={settings.min_containers} ({cfg.hourly_cost_text()})")
    if auto_down:
        call = _function("auto_down").spawn(int(hours * 3600))
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        AUTO_DOWN_FILE.write_text(json.dumps({"call_id": call.object_id, "hours": hours}))
        typer.echo(f"auto_down job {call.object_id} resets min_containers to 0 in {hours:g} h; `dsv4 down` does it now")
    if wait:
        info = _wait(server.info.spawn(), 15.0, "booting")
        typer.echo(f"ready: boot {info['load_s']}s (server ready {info['ready_s']}s), tp={info['settings']['tp']}, log {info.get('log_path')}")


@app.command()
def down(
    now: Annotated[bool, typer.Option("--now", help="Also stop the idle container right away instead of after the idle window")] = False,
) -> None:
    """Let the server scale to zero again (and cancel a pending auto_down)."""
    import modal

    if AUTO_DOWN_FILE.exists():
        try:
            call_id = json.loads(AUTO_DOWN_FILE.read_text())["call_id"]
            modal.FunctionCall.from_id(call_id).cancel()
            typer.echo(f"cancelled auto_down job {call_id}")
        except Exception as exc:  # noqa: BLE001 - already finished or gone
            typer.echo(f"auto_down job not cancelled ({exc})", err=True)
        AUTO_DOWN_FILE.unlink(missing_ok=True)
    server = _server()
    settings = _autoscaler(server).update(min_containers=0)
    if not now:
        typer.echo(f"min_containers={settings.min_containers}; idle containers stop after {cfg.SCALEDOWN_S // 60} min")
        return
    # `modal container stop` gets the container replaced while the autoscaler still wants one; a
    # 2-second idle window makes the autoscaler itself retire it, with no replacement.
    _autoscaler(server).update(scaledown_window=2)
    try:
        t0 = time.monotonic()
        while _containers() and time.monotonic() - t0 < 300:
            typer.echo(f"\r  waiting for the container to stop ... {int(time.monotonic() - t0)}s", nl=False, err=True)
            time.sleep(10)
        typer.echo("", err=True)
    finally:
        _autoscaler(server).update(scaledown_window=cfg.SCALEDOWN_S)
    typer.echo("stopped" if not _containers() else "still running (busy?); it stops after the idle window")


@app.command()
def status() -> None:
    """Autoscaler settings, running containers and the configured GPU."""
    server = _server()
    try:
        settings = _autoscaler(server).settings()
        typer.echo(
            f"autoscaler: min={settings.min_containers} max={settings.max_containers} "
            f"buffer={settings.buffer_containers} scaledown_window={settings.scaledown_window}s"
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"autoscaler settings unavailable: {exc}", err=True)
    containers = _containers()
    typer.echo(f"running containers: {len(containers)}" + (f" ({', '.join(c.get('Container ID', c.get('container_id', '?')) for c in containers)})" if containers else ""))
    typer.echo(f"deploy-time gpu spec: {cfg.GPU_SPEC} ({cfg.hourly_cost_text()} per container)")
    if AUTO_DOWN_FILE.exists():
        typer.echo(f"pending auto_down: {AUTO_DOWN_FILE.read_text().strip()}")
    if cfg.WEB_SERVER:
        try:
            typer.echo(f"openai-compatible url: {server.api.get_web_url()}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"web url unavailable: {exc}", err=True)


@app.command()
def logs(
    tail: Annotated[int, typer.Option("--tail", "-n", help="Lines from the end")] = 200,
    list_logs: Annotated[bool, typer.Option("--list", help="List the stored boot logs instead")] = False,
    name: Annotated[str | None, typer.Option("--name", help="A specific log file name from --list")] = None,
) -> None:
    """Show the SGLang log of the latest (or a given) boot, stored on the cache Volume."""
    import modal

    volume = modal.Volume.from_name(cfg.CACHE_VOLUME)
    rel = str(cfg.LOG_DIR.relative_to(cfg.CACHE_DIR))
    try:
        entries = sorted(e.path for e in volume.listdir(rel))
    except Exception as exc:  # noqa: BLE001
        _fail(f"no logs yet ({exc})")
    if list_logs:
        for entry in entries:
            typer.echo(entry)
        return
    if not entries:
        _fail("no logs yet")
    path = f"{rel}/{name}" if name else entries[-1]
    text = b"".join(volume.read_file(path)).decode("utf-8", errors="replace")
    typer.echo("\n".join(text.splitlines()[-tail:]))


@jobs_app.command("status")
def jobs_status(job_id: str) -> None:
    """Show whether a spawned generation has finished."""
    import modal

    call = modal.FunctionCall.from_id(job_id)
    try:
        raw = call.get(timeout=0)
    except TimeoutError:
        typer.echo(f"{job_id}: running")
        return
    except modal.exception.RemoteError as exc:
        _fail(f"{job_id} failed: {exc}")
    result = GenerationResult.model_validate(raw)
    typer.echo(f"{job_id}: done  finish={result.finish_reason}  stats={result.stats}")


@jobs_app.command("fetch")
def jobs_fetch(
    job_id: str,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    poll: Annotated[float, typer.Option("--poll")] = 15.0,
) -> None:
    """Wait for a spawned generation (if needed) and print or save its answer."""
    import modal

    result = GenerationResult.model_validate(_wait(modal.FunctionCall.from_id(job_id), poll, "waiting"))
    _Printer(show_reasoning=False, quiet=True).finish(result, "?")
    _emit_result(result, out=out, as_json=as_json, printed=False)


@jobs_app.command("list")
def jobs_list(limit: Annotated[int, typer.Option("--limit", "-n")] = 20) -> None:
    """List generations spawned from this machine (newest last)."""
    if not JOBS_FILE.exists():
        typer.echo("no jobs recorded yet")
        return
    for line in JOBS_FILE.read_text(encoding="utf-8").splitlines()[-limit:]:
        entry = json.loads(line)
        typer.echo(f"{entry['created_at']}  {entry['job_id']}  {entry['prompt']}")


def main() -> None:  # console_scripts entry point
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
