"""Interactive chat REPL for MORIE -- Claude Code-like terminal experience.

Provides a multi-turn conversational interface with streaming LLM responses,
slash commands for module execution, and agent persona switching.  Uses only
``rich`` (core dependency) -- does NOT require Textual.

Usage::

    morie chat                        # default assistant
    morie chat --agent morie-architect # agent-specific persona

Slash commands::

    /run <module>       Run an analysis module with progress display
    /list               List available modules
    /doctor             Run environment diagnostics
    /profile <csv>      Profile a dataset
    /inspect <path>     Inspect output CSV(s)
    /verify <path>      Verify statistical outputs
    /agent <name>       Switch to an agent persona
    /agents             List available agents
    /provider           Show current LLM provider
    /history            Show conversation history
    /clear              Clear conversation history
    /help               Show available commands
    /quit               Exit the REPL
"""

from __future__ import annotations

import os
import signal
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .llm import (
    _MORIE_SYSTEM_PROMPT_TEMPLATE,
    _format_context_block,
    ask_multi,
    build_morie_context,
    detect_available_provider,
)
from .modules import MODULE_SPECS, list_modules

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single message in the conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str
    timestamp: float = 0.0
    provider: str | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class SlashCommand:
    """Registry entry for a slash command."""

    name: str
    description: str
    handler: Callable[[list[str]], str | None]


# ---------------------------------------------------------------------------
# Agent loader
# ---------------------------------------------------------------------------

_AGENTS_DIR_NAMES = [
    ".claude/agents",
]


def _find_agents_dir() -> Path | None:
    """Locate the .claude/agents/ directory."""
    # Walk up from this file to find the repo root.
    current = Path(__file__).resolve()
    for _ in range(8):
        current = current.parent
        for subdir in _AGENTS_DIR_NAMES:
            candidate = current / subdir
            if candidate.is_dir():
                return candidate
    return None


def list_agents() -> list[dict[str, str]]:
    """List available agent definitions.

    Returns
    -------
    list[dict[str, str]]
        Each dict has ``name`` and ``description`` keys.
    """
    agents_dir = _find_agents_dir()
    if agents_dir is None:
        return []

    agents = []
    for md_file in sorted(agents_dir.glob("*.md")):
        name = md_file.stem
        # Read first few lines for description.
        desc = ""
        try:
            text = md_file.read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.startswith("#") and not line.startswith("##"):
                    desc = line.lstrip("#").strip()
                    break
            if not desc:
                desc = name
        except OSError:
            desc = name
        agents.append({"name": name, "description": desc})
    return agents


def load_agent_prompt(agent_name: str) -> str | None:
    """Load an agent's markdown file and return it as a system prompt.

    Parameters
    ----------
    agent_name : str
        Agent filename stem (e.g., ``"morie-architect"``).

    Returns
    -------
    str | None
        The agent file content, or ``None`` if not found.
    """
    agents_dir = _find_agents_dir()
    if agents_dir is None:
        return None

    path = agents_dir / f"{agent_name}.md"
    if not path.is_file():
        return None

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------


class ChatSession:
    """Manages a multi-turn chat conversation.

    Parameters
    ----------
    system_prompt : str | None
        Custom system prompt.  When ``None``, the default MORIE agent
        prompt is used (with module/data context injected).
    agent : str | None
        Agent name to load.  Overrides ``system_prompt`` if provided.
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        agent: str | None = None,
    ) -> None:
        self.history: list[ChatMessage] = []
        self.agent_name = agent
        self.slash_commands: dict[str, SlashCommand] = {}
        self._cancelled = False
        self.dataset_context: str | None = None

        # Build system prompt.
        if agent:
            agent_text = load_agent_prompt(agent)
            if agent_text:
                self.system_prompt = agent_text
            else:
                self.system_prompt = self._default_system_prompt()
        elif system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self._default_system_prompt()

        self._register_default_commands()

    @staticmethod
    def _default_system_prompt() -> str:
        context = build_morie_context()
        context_block = _format_context_block(context)
        return _MORIE_SYSTEM_PROMPT_TEMPLATE.format(context_block=context_block)

    # ------------------------------------------------------------------
    # Slash command registration
    # ------------------------------------------------------------------

    def _register_default_commands(self) -> None:
        """Register built-in slash commands."""

        def _cmd_list(_args: list[str]) -> str:
            lines = []
            for spec in list_modules():
                lines.append(f"  {spec['name']}")
                lines.append(f"    {spec['description']}")
            return "\n".join(lines) if lines else "No modules found."

        def _cmd_doctor(_args: list[str]) -> str:
            from .doctor import run_checks

            results = run_checks()
            lines = ["MORIE Doctor:"]
            for check in results["checks"]:
                status = "OK" if check["passed"] else ("FAIL" if check["required"] else "WARN")
                lines.append(f"  [{status}] {check['label']}: {check['detail']}")
            return "\n".join(lines)

        def _cmd_run(args: list[str]) -> str:
            if not args:
                return "Usage: /run <module-name>"
            module_name = args[0]
            if module_name not in MODULE_SPECS:
                return f"Unknown module: {module_name}. Use /list to see available modules."
            try:
                from .progress import PipelineTracker

                tracker = PipelineTracker(
                    [module_name],
                    cpads_csv="data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv",
                    track_carbon=False,
                )
                results = tracker.run()
                r = results[0]
                if r.status == "success":
                    return (
                        f"Module '{module_name}' completed in {r.elapsed_seconds:.1f}s"
                        f" ({r.output_files_actual} files written)"
                    )
                else:
                    return f"Module '{module_name}' failed: {r.error_message}"
            except Exception as exc:
                return f"Error running module '{module_name}': {exc}"

        def _cmd_profile(args: list[str]) -> str:
            if not args:
                return "Usage: /profile <csv-path>"
            try:
                from .dataset import load_dataset, profile_dataset

                df = load_dataset(args[0])
                profile = profile_dataset(df)
                return profile.summary_table()
            except Exception as exc:
                return f"Error profiling dataset: {exc}"

        def _cmd_inspect(args: list[str]) -> str:
            if not args:
                return "Usage: /inspect <path>"
            try:
                from pathlib import Path as _P

                from .inspector import inspect_directory, inspect_output

                target = _P(args[0])
                if target.is_file():
                    r = inspect_output(target)
                    return f"{r.file_path}: {r.rows} rows x {r.columns} cols\nColumns: {', '.join(r.column_names)}"
                elif target.is_dir():
                    results = inspect_directory(target)
                    lines = [f"Found {len(results)} CSV files:"]
                    for r in results:
                        lines.append(f"  {r.file_path}: {r.rows} rows x {r.columns} cols")
                    return "\n".join(lines)
                else:
                    return f"Path not found: {args[0]}"
            except Exception as exc:
                return f"Error inspecting: {exc}"

        def _cmd_verify(args: list[str]) -> str:
            if not args:
                return "Usage: /verify <path>"
            try:
                from pathlib import Path as _P

                from .inspector import verify_directory, verify_statistical_output

                target = _P(args[0])
                if target.is_file():
                    report = verify_statistical_output(target)
                    lines = [f"Verification: {report.file_path}"]
                    for c in report.checks:
                        status = "OK" if c.passed else ("FAIL" if c.severity == "error" else "WARN")
                        lines.append(f"  [{status}] {c.name}: {c.message}")
                    lines.append(
                        "PASSED" if report.passed else f"{report.error_count} errors, {report.warning_count} warnings"
                    )
                    return "\n".join(lines)
                elif target.is_dir():
                    reports = verify_directory(target)
                    lines = [f"Verified {len(reports)} files:"]
                    for report in reports:
                        status = "PASS" if report.passed else "FAIL"
                        lines.append(f"  [{status}] {report.file_path}")
                    return "\n".join(lines)
                else:
                    return f"Path not found: {args[0]}"
            except Exception as exc:
                return f"Error verifying: {exc}"

        def _cmd_agent(args: list[str]) -> str:
            if not args:
                return "Usage: /agent <agent-name>"
            name = args[0]
            prompt = load_agent_prompt(name)
            if prompt is None:
                available = [a["name"] for a in list_agents()]
                return f"Agent '{name}' not found. Available: {', '.join(available)}"
            self.system_prompt = prompt
            self.agent_name = name
            self.history.clear()
            return f"Switched to agent: {name} (history cleared)"

        def _cmd_agents(_args: list[str]) -> str:
            agents = list_agents()
            if not agents:
                return "No agents found in .claude/agents/"
            lines = ["Available agents:"]
            for a in agents:
                lines.append(f"  {a['name']}: {a['description']}")
            return "\n".join(lines)

        def _cmd_provider(_args: list[str]) -> str:
            provider = detect_available_provider()
            return f"Current LLM provider: {provider}"

        def _cmd_history(_args: list[str]) -> str:
            if not self.history:
                return "No conversation history."
            lines = []
            for msg in self.history:
                prefix = "You" if msg.role == "user" else "MORIE"
                lines.append(f"[{prefix}] {msg.content[:120]}")
            return "\n".join(lines)

        def _cmd_clear(_args: list[str]) -> str:
            self.history.clear()
            return "Conversation history cleared."

        def _cmd_help(_args: list[str]) -> str:
            lines = ["Available commands:"]
            for cmd in self.slash_commands.values():
                lines.append(f"  /{cmd.name:<20} {cmd.description}")
            return "\n".join(lines)

        commands = [
            SlashCommand("run", "Run an analysis module", _cmd_run),
            SlashCommand("list", "List available modules", _cmd_list),
            SlashCommand("doctor", "Run environment diagnostics", _cmd_doctor),
            SlashCommand("profile", "Profile a dataset", _cmd_profile),
            SlashCommand("inspect", "Inspect output CSV(s)", _cmd_inspect),
            SlashCommand("verify", "Verify statistical outputs", _cmd_verify),
            SlashCommand("agent", "Switch agent persona", _cmd_agent),
            SlashCommand("agents", "List available agents", _cmd_agents),
            SlashCommand("provider", "Show current LLM provider", _cmd_provider),
            SlashCommand("history", "Show conversation history", _cmd_history),
            SlashCommand("clear", "Clear conversation history", _cmd_clear),
            SlashCommand("help", "Show this help message", _cmd_help),
            SlashCommand("quit", "Exit the REPL", lambda _: None),
            SlashCommand("exit", "Exit the REPL", lambda _: None),
        ]
        for cmd in commands:
            self.slash_commands[cmd.name] = cmd

    # ------------------------------------------------------------------
    # Dataset context
    # ------------------------------------------------------------------

    def set_dataset_context(self, df: pd.DataFrame, name: str = "loaded") -> None:
        """Inject current dataset metadata so the LLM knows what data is loaded."""
        lines = [
            f"The user has loaded a dataset '{name}':",
            f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"  Columns: {', '.join(str(c) for c in df.columns.tolist()[:50])}",
            f"  Dtypes: {dict(df.dtypes.value_counts())}",
        ]
        try:
            lines.append(f"  Sample (first 3 rows):\n{df.head(3).to_string()}")
        except Exception:
            pass
        self.dataset_context = "\n".join(lines)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the full messages array for the LLM."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.dataset_context:
            messages.append({"role": "system", "content": self.dataset_context})
        for msg in self.history:
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    def send(
        self,
        user_input: str,
        *,
        stream: bool = True,
    ) -> str | Iterator[str]:
        """Send user input and get a response.

        Handles slash commands internally.  For regular messages, sends to
        the LLM provider chain.

        Parameters
        ----------
        user_input : str
            The user's message or slash command.
        stream : bool
            If True, return a streaming iterator for LLM responses.

        Returns
        -------
        str | Iterator[str]
            The response text or streaming iterator.
        """
        stripped = user_input.strip()

        # Handle slash commands.
        if stripped.startswith("/"):
            parts = stripped[1:].split(maxsplit=1)
            cmd_name = parts[0].lower()
            cmd_args = parts[1].split() if len(parts) > 1 else []

            if cmd_name in ("quit", "exit"):
                raise EOFError("User requested exit")

            if cmd_name in self.slash_commands:
                result = self.slash_commands[cmd_name].handler(cmd_args)
                return result or ""
            else:
                return f"Unknown command: /{cmd_name}. Type /help for available commands."

        # Regular message: add to history, send to LLM.
        self.history.append(ChatMessage(role="user", content=stripped))
        messages = self._build_messages()

        response = ask_multi(messages, stream=stream)

        if stream:
            # Wrap the iterator to capture the full response for history.
            return self._capture_stream(response)
        else:
            assert isinstance(response, str)
            self.history.append(
                ChatMessage(
                    role="assistant",
                    content=response,
                    provider=detect_available_provider(),
                )
            )
            return response

    def _capture_stream(self, stream_iter: Iterator[str]) -> Iterator[str]:
        """Wrap a stream iterator to capture the full text for history."""
        chunks: list[str] = []
        for chunk in stream_iter:
            chunks.append(chunk)
            yield chunk
        full_text = "".join(chunks)
        self.history.append(
            ChatMessage(
                role="assistant",
                content=full_text,
                provider=detect_available_provider(),
            )
        )

    def cancel_stream(self) -> None:
        """Signal that the current stream should be cancelled."""
        self._cancelled = True


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

# Tracks whether we're currently streaming (for SIGINT handling).
_current_streaming = False


def run_chat_repl(*, agent: str | None = None) -> int:
    """Launch the interactive chat REPL.

    Parameters
    ----------
    agent : str | None
        Agent name to load at startup.

    Returns
    -------
    int
        Exit code (always 0 for normal exit).
    """
    if not sys.stdin.isatty():
        print("Error: morie chat requires an interactive terminal.", file=sys.stderr)
        return 1

    import readline  # Only import in CLI REPL -- conflicts with Textual's terminal input.

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    session = ChatSession(agent=agent)

    # Welcome message.
    provider = detect_available_provider()
    agent_label = f" ({session.agent_name})" if session.agent_name else ""
    console.print(
        Panel(
            f"[bold cyan]MORIE Chat{agent_label}[/bold cyan]\n"
            f"LLM provider: [bold]{provider}[/bold]\n"
            f"Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.\n"
            f"Press [bold]Ctrl+C[/bold] to cancel streaming, [bold]Ctrl+D[/bold] to exit.",
            border_style="cyan",
        )
    )

    # Set up readline history.
    histfile = os.path.expanduser("~/.morie_chat_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)

    # SIGINT handling.
    global _current_streaming

    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        global _current_streaming
        if _current_streaming:
            _current_streaming = False
            console.print("\n[dim](cancelled)[/dim]")
        else:
            raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            try:
                prompt = f"[bold green]morie{agent_label}>[/bold green] "
                user_input = console.input(prompt)
            except EOFError:
                console.print("\n[dim]Goodbye.[/dim]")
                break
            except KeyboardInterrupt:
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user_input.strip():
                continue

            try:
                response = session.send(user_input, stream=True)
            except EOFError:
                console.print("[dim]Goodbye.[/dim]")
                break

            if isinstance(response, str):
                # Slash command output.
                if response:
                    console.print(response)
            else:
                # Streaming LLM response.
                _current_streaming = True
                collected = []
                try:
                    for chunk in response:
                        if not _current_streaming:
                            break
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        collected.append(chunk)
                    sys.stdout.write("\n")
                except Exception:
                    pass
                finally:
                    _current_streaming = False
                console.print()

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        try:
            readline.write_history_file(histfile)
        except OSError:
            pass

    return 0
