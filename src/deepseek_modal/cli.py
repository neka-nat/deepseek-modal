"""`dsv4`: command line client for the deployed Modal app.

Only needs `modal`, `typer` and `pydantic` locally; SGLang and the weights live on Modal. Streaming
goes over Modal's own RPC (`remote_gen`), so the only credential involved is your Modal token.
"""

from __future__ import annotations

import json
import sys
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


def _running_containers(handle: Any) -> int | None:
    try:
        stats = handle.get_current_stats()
    except Exception:  # noqa: BLE001 - best effort
        return None
    return getattr(stats, "num_total_runners", None)


def _cold_start_hint(server: Any) -> None:
    if _running_containers(server.generate) == 0:
        _note(f"no warm container: cold start ahead (loading 510 GB of weights, 10-25 min, {cfg.hourly_cost_text()})")


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
) -> None:
    """Keep one server container running (min_containers=1), so calls skip the cold start."""
    server = _server()
    settings = server.generate.update_autoscaler(min_containers=1)
    typer.echo(f"min_containers={settings.min_containers} ({cfg.hourly_cost_text()})")
    if auto_down:
        call = _function("auto_down").spawn(int(hours * 3600))
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        AUTO_DOWN_FILE.write_text(json.dumps({"call_id": call.object_id, "hours": hours}))
        typer.echo(f"auto_down job {call.object_id} will reset min_containers to 0 in {hours:g} h; `dsv4 down` does it now")


@app.command()
def down() -> None:
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
    settings = _server().generate.update_autoscaler(min_containers=0)
    typer.echo(f"min_containers={settings.min_containers}; idle containers stop after {cfg.SCALEDOWN_S // 60} min")


@app.command()
def status() -> None:
    """Autoscaler settings, running containers and the configured GPU."""
    handle = _server().generate
    try:
        settings = handle.autoscaler.settings()
        typer.echo(
            f"autoscaler: min={settings.min_containers} max={settings.max_containers} "
            f"buffer={settings.buffer_containers} scaledown_window={settings.scaledown_window}s"
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"autoscaler settings unavailable: {exc}", err=True)
    running = _running_containers(handle)
    typer.echo(f"running containers: {running if running is not None else 'unknown'}")
    typer.echo(f"deploy-time gpu spec: {cfg.GPU_SPEC} ({cfg.hourly_cost_text()} per container)")
    if AUTO_DOWN_FILE.exists():
        typer.echo(f"pending auto_down: {AUTO_DOWN_FILE.read_text().strip()}")
    if cfg.WEB_SERVER:
        try:
            typer.echo(f"openai-compatible url: {_server().api.get_web_url()}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"web url unavailable: {exc}", err=True)


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
