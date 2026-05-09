"""Full-screen Textual terminal IDE for MOIRAIS.

Requires ``pip install moirais[interactive]`` (textual >= 0.85).

Provides a Claude Code-inspired TUI with multiple screens:

- **ChatScreen** — streaming LLM chat with slash commands
- **PipelineScreen** — run analysis modules with live progress
- **DoctorScreen** — environment diagnostics
- **DatasetScreen** — browse and profile datasets
- **HelpScreen** — built-in documentation and command reference
- **DebugScreen** — debug log viewer and diagnostics
- **ReplScreen** — Python/R interactive REPL
- **StatScreen** — run statistical analyses (70+ commands)

Launch with::

    moirais tui        # explicit
    moirais            # auto-launches TUI when interactive terminal detected

Keybindings::

    c  Chat         h  Help/Docs     s  Stats
    p  Pipeline     g  Debug         r  REPL
    d  Doctor       i  Inspect       q  Quit
"""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Guarded import — textual is optional
# ---------------------------------------------------------------------------

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.screen import Screen
    from textual.suggester import SuggestFromList
    from textual.widgets import (
        Footer,
        Header,
        Input,
        RichLog,
        Static,
        TextArea,
        Tree,
    )
    from textual.worker import Worker, WorkerState

    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False


def _check_textual() -> bool:
    """Return True if Textual is available."""
    return _TEXTUAL_AVAILABLE


# ---------------------------------------------------------------------------
# Screens (only defined when textual is available)
# ---------------------------------------------------------------------------

if _TEXTUAL_AVAILABLE:
    # ==================================================================
    # CopyableRichLog — RichLog with plain-text shadow buffer
    # ==================================================================

    import re as _re

    # Matches Rich console markup tags like [bold], [red], [/bold cyan],
    # [dim], [bold yellow], [#ff0000], [link=...] etc.  Does NOT match
    # plain-text brackets such as "[40931 rows x 394 columns]" or
    # "[PYTHON]" because those contain digits, spaces, or uppercase words
    # that do not look like Rich style directives.
    _RICH_TAG_RE = _re.compile(
        r"\[/?(?:bold|dim|italic|underline|strike|reverse|blink|overline"
        r"|red|green|blue|cyan|magenta|yellow|white|black"
        r"|bright_\w+|on \w+|link[^\]]*|#[0-9a-fA-F]+)"
        r"(?:\s[^\]]*)?\]"
    )

    class CopyableRichLog(RichLog):
        """RichLog that keeps a plain-text shadow buffer for clipboard copy.

        Press **ctrl+o** to copy all text, **ctrl+b** for errors, **ctrl+l** for last 5.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._text_buffer: list[str] = []

        def write(
            self,
            content="",
            width=None,
            expand=False,
            shrink=True,
            scroll_end=None,
            animate=False,
        ):
            plain = _RICH_TAG_RE.sub("", str(content))
            self._text_buffer.append(plain)
            return super().write(
                content,
                width=width,
                expand=expand,
                shrink=shrink,
                scroll_end=scroll_end,
                animate=animate,
            )

        def clear(self):
            self._text_buffer.clear()
            return super().clear()

        def get_all_text(self) -> str:
            """Return all buffered text joined by newlines."""
            return "\n".join(self._text_buffer)

        @property
        def buffer_size(self) -> int:
            """Number of entries in the shadow buffer."""
            return len(self._text_buffer)

        def get_errors(self) -> str:
            """Return only error lines (tracebacks, Error:, etc.)."""
            error_re = _re.compile(
                r"(Error|Exception|Traceback|raise |File \"|"
                r"KeyError|ValueError|TypeError|NameError|"
                r"AttributeError|ImportError|RuntimeError|"
                r"IndexError|SyntaxError|ZeroDivision|"
                r"ModuleNotFoundError|FileNotFoundError|"
                r"error:|failed|FAILED)",
                _re.IGNORECASE,
            )
            lines = [l for l in self._text_buffer if error_re.search(l)]
            return "\n".join(lines) if lines else ""

        def get_last_n_exchanges(self, n: int = 5) -> str:
            """Return the last *n* command/output exchanges.

            Splits the buffer at prompt markers (``>>>``, ``R>``, etc.)
            and returns the last *n* groups.
            """
            if not self._text_buffer:
                return ""
            prompt_re = _re.compile(r"(>>>|R>|\[PYTHON\]|\[R\]|moirais>)")
            prompts = [i for i, line in enumerate(self._text_buffer) if prompt_re.search(line)]
            if not prompts:
                return "\n".join(self._text_buffer[-50:])
            start = prompts[-n] if len(prompts) >= n else prompts[0]
            return "\n".join(self._text_buffer[start:])

    # ==================================================================
    # ChatScreen — LLM agent with slash commands
    # ==================================================================

    # Suggestion lists for auto-completion across screens.
    _CHAT_SUGGESTIONS = SuggestFromList(
        [
            "/help",
            "/list",
            "/run",
            "/doctor",
            "/profile",
            "/inspect",
            "/verify",
            "/agent",
            "/agents",
            "/provider",
            "/history",
            "/clear",
            "/quit",
            "/exit",
        ],
        case_sensitive=False,
    )

    _ANALYSIS_SUGGESTIONS = SuggestFromList(
        [
            "help",
            "ttest",
            "ttest2",
            "paired",
            "anova",
            "chi2",
            "corr",
            "ks",
            "mannwhitney",
            "describe",
            "profile",
            "missing",
            "head",
            "columns",
            "evalue",
            "bh",
            "bootstrap",
            "power",
            "cohens_d",
            "kaplan_meier",
            "km",
            "cox",
            "logrank",
            "did",
            "rdd",
            "tsls",
            "match",
            "mcar",
            "impute",
            "vif",
            "rosenbaum",
            "odds_ratio",
            "nnt",
            "table1",
            "propensity",
            "ate",
            "ipw",
            "modules",
            "run",
            "save",
            "export",
            "hedges_g",
            "risk_ratio",
            "bonferroni",
            "jackknife",
            "permtest",
            "event_study",
            "fuzzy_rdd",
            "strobe",
            "residuals",
            "cooks",
            "fisher",
            "shapiro",
            "wilcoxon",
            "levene",
            "normality",
            "aipw",
            "att",
            "atc",
            "cate",
            "gate",
            "late",
            "irm",
            "dml",
            "plr",
            "pliv",
            "ps_nn",
            "ps_subclass",
            "balance",
            "overlap",
            "ess",
            "design_effect",
        ],
        case_sensitive=False,
    )

    _REPL_SUGGESTIONS = SuggestFromList(
        [
            "load(",
            "head(",
            "tail(",
            "shape(",
            "cols(",
            "describe(",
            "sample(",
            "unique(",
            "value_counts(",
            "freq(",
            "missing(",
            "summary(",
            "filter_rows(",
            "select_cols(",
            "rename_col(",
            "dropna(",
            "save(",
            "crosstab(",
            "pivot(",
            "groupby(",
            "modules()",
            "run_module(",
            "ttest(",
            "ttest2(",
            "paired_t(",
            "corr(",
            "spearman(",
            "anova(",
            "chi2(",
            "mannwhitney(",
            "ks_test(",
            "bootstrap_ci(",
            "bh(",
            "power(",
            "effect_size(",
            "evalue(",
            "sensitivity(",
            "ate(",
            "propensity(",
            "ipw(",
            "ebac(",
            "profile()",
            "plan()",
            "selftest()",
            "doctor()",
            "version()",
            "ls()",
            "who()",
            "view(",
            "clear()",
            "help_repl()",
            "kaplan_meier(",
            "cox(",
            "logrank(",
            "did(",
            "rdd(",
            "iv_tsls(",
            "match(",
            "mcar()",
            "impute(",
            "vif(",
            "odds_ratio(",
            "nnt(",
            "rosenbaum(",
            "table1(",
            "hedges_g(",
            "risk_ratio(",
            "bonferroni(",
            "jackknife(",
            "permtest(",
            "event_study(",
            "fuzzy_rdd(",
            "/python",
            "/r",
            "/shell",
            "/auto",
            "/reset",
            "/history",
            "import ",
            "from moirais import ",
        ],
        case_sensitive=False,
    )

    _HELP_SUGGESTIONS = SuggestFromList(
        [
            "overview",
            "modules",
            "causal",
            "did",
            "rdd",
            "iv",
            "matching",
            "survival",
            "statistics",
            "effects",
            "missing",
            "sensitivity",
            "bootstrap",
            "weights",
            "reporting",
            "chat",
            "keybindings",
            "cli",
            "version",
        ],
        case_sensitive=False,
    )

    _DATASET_SUGGESTIONS = SuggestFromList(
        [
            "ls",
        ],
        case_sensitive=False,
    )

    class ChatScreen(Screen):
        """Chat interface — same pattern as DatasetScreen/HelpScreen."""

        BINDINGS = [
            Binding("escape", "app.pop_screen", "Back"),
            Binding("tab", "accept_suggestion", "Complete", show=False),
        ]

        DEFAULT_CSS = """
        /* ELIA-inspired chat layout (#76).
         * darrenburns/elia uses a slim header, generous message
         * padding, a subtle 1-cell border on the log, and a focused
         * input docked at the bottom. We approximate that with the
         * Textual primitives we already have.
         */
        ChatScreen { layout: vertical; background: $surface; }
        #chat-log {
            height: 1fr;
            border: heavy $accent 60%;
            padding: 1 2;
            background: $surface-darken-1;
            scrollbar-size: 1 1;
        }
        #chat-log:focus {
            border: heavy $accent;
        }
        #chat-input {
            border: heavy $primary 50%;
            background: $surface;
            margin: 1 0 0 0;
            padding: 0 2;
        }
        #chat-input:focus {
            border: heavy $accent;
            background: $surface-darken-1;
        }
        Header {
            background: $surface-darken-2;
        }
        Footer {
            background: $surface-darken-2;
            color: $text-muted;
        }
        """

        def __init__(self, agent: str | None = None) -> None:
            super().__init__()
            self._agent = agent
            self._session = None
            self._busy = False  # True while an LLM call is in flight.

        def compose(self) -> ComposeResult:
            yield Header()
            yield CopyableRichLog(id="chat-log", highlight=True, markup=True, wrap=True)
            yield Input(
                placeholder="Ask MOIRAIS anything... (/help for commands, Tab to complete)",
                id="chat-input",
                suggester=_CHAT_SUGGESTIONS,
            )
            yield Footer()

        def on_mount(self) -> None:
            log = self.query_one("#chat-log", RichLog)
            log.can_focus = False

            # Create session — wrapped in try/except so focus() always runs.
            try:
                from .chat import ChatSession

                self._session = ChatSession(agent=self._agent)
            except Exception as exc:
                log.write(f"[red]Chat session init failed: {exc}[/red]")
                self._session = None

            agent_label = f" ({self._agent})" if self._agent else ""
            log.write(f"[bold cyan]MOIRAIS Chat{agent_label}[/bold cyan]")
            log.write("Type a message and press Enter. Type [bold]/help[/bold] for slash commands.\n")

            # Detect provider in the background, just to show status.
            self.run_worker(self._show_provider(), name="detect-provider")

            self.query_one("#chat-input", Input).focus()

        def on_screen_resume(self) -> None:
            """Re-focus input and sync dataset context when returning."""
            self.query_one("#chat-input", Input).focus()
            # If a dataset was loaded in DatasetScreen, inject its metadata.
            if self._session and getattr(self.app, "loaded_df", None) is not None:
                self._session.set_dataset_context(self.app.loaded_df, self.app.loaded_df_name)

        def action_accept_suggestion(self) -> None:
            """Accept the current auto-completion suggestion (Tab key)."""
            inp = self.query_one("#chat-input", Input)
            if inp._suggestion:
                inp.value = inp._suggestion
                inp.cursor_position = len(inp.value)

        async def _show_provider(self) -> None:
            """Detect LLM provider in background and display it."""
            try:
                from .llm import detect_provider_and_model

                provider, model_label = await asyncio.to_thread(detect_provider_and_model)
                log = self.query_one("#chat-log", RichLog)
                log.write(f"[dim]LLM: {model_label}[/dim]")
                if provider == "local":
                    log.write("[dim]No LLM available. Set moiraisfam env var or install ollama.[/dim]")
            except Exception:
                pass  # Non-critical — don't break the screen.

        def on_input_submitted(self, event: Input.Submitted) -> None:
            user_input = event.value.strip()
            if not user_input:
                return
            event.input.value = ""
            event.input.focus()
            log = self.query_one("#chat-log", RichLog)
            log.write(f"[bold green]you>[/bold green] {user_input}")

            if self._session is None:
                log.write("[red]Chat session not available. Press Esc and try again.[/red]")
                log.write("")
                return

            # Slash commands — synchronous, no LLM needed.
            if user_input.startswith("/"):
                try:
                    response = self._session.send(user_input, stream=False)
                except EOFError:
                    self.app.pop_screen()
                    return
                except Exception as exc:
                    log.write(f"[red]Error: {exc}[/red]")
                    log.write("")
                    return
                if response:
                    for line in response.splitlines():
                        log.write(line)
                log.write("")
                return

            # Regular message — run LLM call in background thread.
            from .llm import detect_provider_and_model, pick_thinking_word

            _, _mod = detect_provider_and_model()
            _tw = pick_thinking_word(user_input)
            log.write(f"[dim]\\[{_mod}] {_tw}.....[/dim]")
            self._busy = True
            self.run_worker(
                self._ask_llm(user_input),
                name="chat-ask",
                exclusive=True,
            )

        async def _ask_llm(self, user_input: str) -> None:
            """Send to LLM in a thread, streaming chunks progressively."""
            log = self.query_one("#chat-log", RichLog)
            _sentinel = object()
            try:
                stream_iter = await asyncio.to_thread(lambda: self._session.send(user_input, stream=True))
                if isinstance(stream_iter, str):
                    # Slash command or non-streaming fallback returned a string.
                    if stream_iter:
                        log.write("[bold cyan]moirais>[/bold cyan]")
                        for line in stream_iter.splitlines():
                            log.write(f"  {line}")
                    else:
                        log.write("[dim](no response)[/dim]")
                else:
                    # Stream: accumulate chunks into paragraphs, flush each
                    # paragraph as a block so output stays readable.
                    buf = ""
                    wrote_header = False
                    while True:
                        chunk = await asyncio.to_thread(next, stream_iter, _sentinel)
                        if chunk is _sentinel:
                            break
                        buf += chunk
                        # Flush on double-newline (paragraph break) to keep
                        # numbered lists and long paragraphs intact.
                        while "\n\n" in buf:
                            para, buf = buf.split("\n\n", 1)
                            if not wrote_header:
                                log.write("[bold cyan]moirais>[/bold cyan]")
                                wrote_header = True
                            for line in para.strip().splitlines():
                                log.write(f"  {line}")
                            log.write("")  # blank line between paragraphs
                    # Flush remaining text.
                    if buf.strip():
                        if not wrote_header:
                            log.write("[bold cyan]moirais>[/bold cyan]")
                            wrote_header = True
                        for line in buf.strip().splitlines():
                            log.write(f"  {line}")
                    if not wrote_header:
                        log.write("[dim](no response)[/dim]")
            except asyncio.CancelledError:
                # Worker cancelled by exclusive=True when user sent a new msg.
                return
            except BaseException as exc:
                log.write(f"[red]Error: {exc}[/red]")
            finally:
                self._busy = False
            log.write("")
            self.query_one("#chat-input", Input).focus()

    # ==================================================================
    # PipelineScreen — run analysis modules
    # ==================================================================

    class PipelineScreen(Screen):
        """Pipeline execution with live progress display."""

        BINDINGS = [
            Binding("escape", "app.pop_screen", "Back"),
            Binding("r", "run_all", "Run All"),
        ]

        DEFAULT_CSS = """
        PipelineScreen { layout: vertical; }
        #pipeline-log { height: 1fr; border: heavy $success; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            yield CopyableRichLog(id="pipeline-log", highlight=True, markup=True, wrap=True)
            yield Input(
                placeholder="Module name to run (or press 'r' for all)...",
                id="pipeline-input",
                suggester=SuggestFromList(
                    [s.name for s in __import__("moirais.modules", fromlist=["MODULE_SPECS"]).MODULE_SPECS.values()],
                    case_sensitive=False,
                ),
            )
            yield Footer()

        def on_mount(self) -> None:
            from .modules import list_modules

            log = self.query_one("#pipeline-log", RichLog)
            log.can_focus = False
            log.write("[bold cyan]MOIRAIS Pipeline[/bold cyan]")
            log.write("Press [bold]r[/bold] to run all modules, or type a module name below.\n")

            modules = list_modules()
            log.write(f"[bold]{len(modules)} modules available:[/bold]")
            for spec in modules:
                log.write(f"  [bold]{spec['name']}[/bold]: {spec['description']}")
            self.query_one("#pipeline-input", Input).focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            module_name = event.value.strip()
            if not module_name:
                return
            event.input.value = ""
            event.input.focus()
            self.run_worker(
                self._run_single_module(module_name),
                name="pipeline-single",
                exclusive=True,
            )

        async def _run_single_module(self, module_name: str) -> None:
            from .modules import MODULE_SPECS
            from .progress import PipelineTracker

            log = self.query_one("#pipeline-log", RichLog)

            if module_name not in MODULE_SPECS:
                log.write(f"[red]Unknown module: {module_name}[/red]")
                return

            log.write(f"\n[yellow]Running: {module_name}...[/yellow]")
            tracker = PipelineTracker(
                [module_name],
                cpads_csv="data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv",
                use_live=False,
                track_carbon=False,
            )
            result = tracker.run_single(module_name)
            if result.status == "success":
                log.write(f"  [green]OK[/green] {result.elapsed_seconds:.1f}s ({result.output_files_actual} files)")
            else:
                log.write(f"  [red]FAIL[/red] {result.error_message}")

        def action_run_all(self) -> None:
            self.run_worker(self._execute_pipeline(), name="pipeline-run", exclusive=True)

        async def _execute_pipeline(self) -> None:
            from .modules import list_modules
            from .progress import PipelineTracker

            log = self.query_one("#pipeline-log", RichLog)
            log.write("\n[bold yellow]Starting full pipeline...[/bold yellow]\n")

            module_names = [s["name"] for s in list_modules()]
            tracker = PipelineTracker(
                module_names,
                cpads_csv="data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv",
                use_live=False,
                track_carbon=False,
            )

            for i, name in enumerate(module_names, 1):
                log.write(f"[{i}/{len(module_names)}] Running: {name}...")
                result = tracker.run_single(name)
                if result.status == "success":
                    log.write(f"  [green]OK[/green] {result.elapsed_seconds:.1f}s ({result.output_files_actual} files)")
                else:
                    log.write(f"  [red]FAIL[/red] {result.error_message}")

            log.write("\n[bold green]Pipeline complete.[/bold green]")

    # ==================================================================
    # DoctorScreen — diagnostics
    # ==================================================================

    class DoctorScreen(Screen):
        BINDINGS = [
            Binding("escape", "app.pop_screen", "Back"),
            Binding("r", "refresh_doctor", "Refresh"),
        ]

        DEFAULT_CSS = """
        DoctorScreen { layout: vertical; }
        #doctor-log { height: 1fr; border: heavy $warning; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            yield CopyableRichLog(id="doctor-log", highlight=True, markup=True, wrap=True)
            yield Footer()

        def on_mount(self) -> None:
            self._run_diagnostics()

        def action_refresh_doctor(self) -> None:
            log = self.query_one("#doctor-log", RichLog)
            log.clear()
            self._run_diagnostics()

        def _run_diagnostics(self) -> None:
            from .doctor import run_checks

            log = self.query_one("#doctor-log", RichLog)
            log.can_focus = False
            log.write("[bold cyan]MOIRAIS Doctor — Environment Diagnostics[/bold cyan]\n")

            results = run_checks()
            for check in results["checks"]:
                if check["passed"]:
                    status = "[green]  OK [/green]"
                elif check["required"]:
                    status = "[red] FAIL[/red]"
                else:
                    status = "[yellow] WARN[/yellow]"
                log.write(f"  {status} {check['label']:<30} {check['detail']}")

            log.write("")
            if results["all_required_passed"]:
                log.write("[green]All required checks passed.[/green]")
            else:
                log.write("[red]Some required checks failed.[/red]")

    # ==================================================================
    # DatasetScreen — browse and profile
    # ==================================================================

    class DatasetScreen(Screen):
        BINDINGS = [
            Binding("escape", "app.pop_screen", "Back"),
        ]

        DEFAULT_CSS = """
        DatasetScreen { layout: vertical; }
        #dataset-log { height: 1fr; border: heavy $primary; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            yield CopyableRichLog(id="dataset-log", highlight=True, markup=True, wrap=True)
            yield Input(
                placeholder="CSV path to inspect (or 'ls' to list files)...",
                id="dataset-input",
                suggester=_DATASET_SUGGESTIONS,
            )
            yield Footer()

        def on_mount(self) -> None:
            log = self.query_one("#dataset-log", RichLog)
            log.can_focus = False
            log.write("[bold cyan]MOIRAIS Dataset Inspector[/bold cyan]")
            log.write("Type a CSV file path below, or 'ls' to list data files.\n")

            # Auto-list CSV files in data directory and build dynamic suggestions.
            data_dir = Path(__file__).resolve().parents[2] / "data" / "files" / "csv" / "survey"
            suggestions = ["ls"]
            if data_dir.is_dir():
                csvs = sorted(data_dir.glob("*.csv"))[:30]
                if csvs:
                    log.write(f"[bold]Available CSV files ({len(csvs)} shown):[/bold]")
                    for f in csvs:
                        log.write(f"  {f.name}")
                        suggestions.append(f.name)
                        suggestions.append(str(f))  # full path too
            inp = self.query_one("#dataset-input", Input)
            inp.suggester = SuggestFromList(suggestions, case_sensitive=False)
            inp.focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            user_input = event.value.strip()
            if not user_input:
                return
            event.input.value = ""
            event.input.focus()
            log = self.query_one("#dataset-log", RichLog)

            if user_input == "ls":
                data_dir = Path(__file__).resolve().parents[2] / "data" / "files" / "csv" / "survey"
                if data_dir.is_dir():
                    csvs = sorted(data_dir.glob("*.csv"))
                    for f in csvs:
                        log.write(f"  {f.name}")
                    log.write(f"\n[dim]{len(csvs)} files[/dim]")
                return

            try:
                from .inspector import inspect_output

                result = inspect_output(user_input)
                log.write(f"\n[bold]{result.file_path}[/bold]")
                log.write(f"  {result.rows} rows x {result.columns} columns\n")
                log.write("[bold]Columns:[/bold]")
                for col in result.column_names:
                    missing = result.missing_counts.get(col, 0)
                    miss_str = f" [red]({missing} missing)[/red]" if missing > 0 else ""
                    log.write(f"  {col}: {result.dtypes[col]}{miss_str}")
                if result.summary_stats is not None:
                    log.write("\n[bold]Summary statistics:[/bold]")
                    log.write(result.summary_stats.to_string())
                # Store loaded DataFrame for data-aware chat.
                try:
                    import pandas as pd

                    df = pd.read_csv(result.file_path)
                    self.app.loaded_df = df
                    self.app.loaded_df_name = Path(result.file_path).stem
                    log.write("\n[dim]Dataset stored — Chat will know about this data.[/dim]")
                except Exception:
                    pass  # Non-critical: inspector worked, CSV re-read failed.
            except FileNotFoundError:
                log.write(f"[red]File not found: {user_input}[/red]")
            except Exception as exc:
                log.write(f"[red]Error: {exc}[/red]")
            log.write("")

    # ==================================================================
    # HelpScreen — documentation and command reference
    # ==================================================================

    class HelpScreen(Screen):
        BINDINGS = [
            Binding("escape", "app.pop_screen", "Back"),
        ]

        DEFAULT_CSS = """
        HelpScreen { layout: vertical; }
        #help-log { height: 1fr; border: heavy $primary; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            yield CopyableRichLog(id="help-log", highlight=True, markup=True, wrap=True)
            yield Input(
                placeholder="Topic (modules, causal, survival, chat, ...) or 'q' to go back",
                id="help-input",
                suggester=_HELP_SUGGESTIONS,
            )
            yield Footer()

        def on_mount(self) -> None:
            log = self.query_one("#help-log", RichLog)
            log.can_focus = False
            log.write("[bold cyan]MOIRAIS Help & Documentation[/bold cyan]\n")
            log.write("[bold]Available topics:[/bold]")
            log.write("  [bold]overview[/bold]     — What is MOIRAIS?")
            log.write("  [bold]modules[/bold]      — List all 21 analysis modules")
            log.write("  [bold]causal[/bold]       — Causal inference methods (IPW, AIPW)")
            log.write("  [bold]did[/bold]          — Difference-in-differences")
            log.write("  [bold]rdd[/bold]          — Regression discontinuity")
            log.write("  [bold]iv[/bold]           — Instrumental variables")
            log.write("  [bold]matching[/bold]     — Matching methods (PSM, CEM, entropy)")
            log.write("  [bold]survival[/bold]     — Survival analysis (KM, Cox, AFT)")
            log.write("  [bold]statistics[/bold]   — Hypothesis testing (t-test, ANOVA, chi-sq)")
            log.write("  [bold]effects[/bold]      — Effect sizes (Cohen's d, OR, RR, NNT)")
            log.write("  [bold]missing[/bold]      — Missing data (MICE, imputation)")
            log.write("  [bold]sensitivity[/bold]  — Sensitivity analysis (E-value, Rosenbaum)")
            log.write("  [bold]bootstrap[/bold]    — Bootstrap and resampling methods")
            log.write("  [bold]weights[/bold]      — Survey weights (raking, GREG, post-strat)")
            log.write("  [bold]reporting[/bold]    — Tables, APA formatting, export")
            log.write("  [bold]chat[/bold]         — Chat REPL slash commands")
            log.write("  [bold]keybindings[/bold]  — TUI keyboard shortcuts")
            log.write("  [bold]cli[/bold]          — Command-line reference")
            log.write("  [bold]version[/bold]      — Version and system info")
            log.write("")
            log.write("[dim]Type a topic name below, or press Esc to go back.[/dim]")
            self.query_one("#help-input", Input).focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            topic = event.value.strip().lower()
            if not topic:
                return
            event.input.value = ""
            event.input.focus()
            log = self.query_one("#help-log", RichLog)
            log.write("")
            self._show_topic(log, topic)

        def _show_topic(self, log: RichLog, topic: str) -> None:
            if topic in ("q", "quit", "back"):
                self.app.pop_screen()
                return

            if topic == "overview":
                log.write("[bold cyan]MOIRAIS Overview[/bold cyan]")
                log.write("")
                log.write("MOIRAIS (Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation) is a")
                log.write("terminal-first scientific computing IDE for epidemiological")
                log.write("and statistical modeling.")
                log.write("")
                log.write("[bold]Core capabilities:[/bold]")
                log.write("  - 45 Python modules, 987+ functions")
                log.write("  - 21 analysis pipeline modules")
                log.write("  - Causal inference: IPW, AIPW, DML, DiD, RDD, IV, matching")
                log.write("  - Survival analysis: Kaplan-Meier, Cox, AFT, competing risks")
                log.write("  - Multiple testing corrections (BH, BY, Holm, Storey)")
                log.write("  - Missing data: MICE, pattern analysis, sensitivity")
                log.write("  - Bootstrap and resampling inference")
                log.write("  - Publication-ready tables and figures")
                log.write("  - LLM-powered agent (Ollama, Gemini, OpenAI)")
                log.write("  - Docker container support")

            elif topic == "modules":
                from .modules import list_modules

                specs = list_modules()
                log.write(f"[bold cyan]Analysis Modules ({len(specs)})[/bold cyan]")
                log.write("")
                for spec in specs:
                    log.write(f"  [bold]{spec['name']}[/bold]")
                    log.write(f"    {spec['description']}")
                    log.write(f"    Outputs: {', '.join(spec['output_files'])}")

            elif topic == "causal":
                log.write("[bold cyan]Causal Inference Methods[/bold cyan]")
                log.write("")
                log.write("[bold]Estimands:[/bold]")
                log.write("  ATE  — Average Treatment Effect")
                log.write("  ATT  — Average Treatment Effect on the Treated")
                log.write("  ATC  — Average Treatment Effect on the Controls")
                log.write("  GATE — Group Average Treatment Effect")
                log.write("  CATE — Conditional Average Treatment Effect")
                log.write("  LATE — Local Average Treatment Effect")
                log.write("")
                log.write("[bold]Methods (moirais.causal):[/bold]")
                log.write("  estimate_ate()    — Hajek IPW estimator")
                log.write("  estimate_aipw()   — Doubly-robust AIPW")
                log.write("  estimate_irm()    — Interactive regression model (DML)")
                log.write("")
                log.write("[bold]DiD (moirais.did):[/bold]  did_2x2, event_study, staggered_did, bacon_decomposition")
                log.write("[bold]RDD (moirais.rdd):[/bold]  sharp_rdd, fuzzy_rdd, bandwidth_cct, mccrary_test")
                log.write("[bold]IV (moirais.iv):[/bold]    tsls, liml, gmm_iv, weak instrument tests")
                log.write("[bold]Matching:[/bold]        match_nearest_neighbor, match_cem, entropy_balance")

            elif topic == "survival":
                log.write("[bold cyan]Survival Analysis (moirais.survival)[/bold cyan]")
                log.write("")
                log.write("[bold]Non-parametric:[/bold]")
                log.write("  kaplan_meier()       — KM estimator with CI")
                log.write("  nelson_aalen()       — Cumulative hazard")
                log.write("  logrank_test()       — Compare survival curves")
                log.write("")
                log.write("[bold]Semi-parametric:[/bold]")
                log.write("  cox_ph()             — Cox proportional hazards")
                log.write("  schoenfeld_residuals() — PH assumption test")
                log.write("")
                log.write("[bold]Parametric:[/bold]")
                log.write("  weibull_model, lognormal_model, gompertz_model")
                log.write("  aft_weibull, aft_lognormal, aft_loglogistic")

            elif topic in ("statistics", "stats"):
                log.write("[bold cyan]Statistical Tests (moirais.statistics)[/bold cyan]")
                log.write("")
                log.write("[bold]T-tests:[/bold]    one_sample_ttest, two_sample_ttest, paired_ttest, welch_ttest")
                log.write("[bold]ANOVA:[/bold]     one_way_anova, two_way_anova, kruskal_wallis, friedman_test")
                log.write("[bold]Chi-sq:[/bold]    chi2_independence, mcnemar_test, cochrans_q")
                log.write("[bold]Corr:[/bold]      pearson, spearman, kendall, partial_correlation")
                log.write("[bold]Normal:[/bold]    shapiro_wilk, dagostino_pearson, jarque_bera")
                log.write("[bold]Nonpar:[/bold]    mann_whitney_u, wilcoxon_signed_rank, ks_test")

            elif topic in ("effects", "effect_sizes"):
                log.write("[bold cyan]Effect Sizes (moirais.effect_sizes)[/bold cyan]")
                log.write("")
                log.write("  cohens_d, hedges_g, glass_delta — standardized mean diff")
                log.write("  odds_ratio, risk_ratio, risk_difference — binary outcomes")
                log.write("  number_needed_to_treat, number_needed_to_harm")
                log.write("  cramers_v, phi_coefficient — contingency")
                log.write("  cliffs_delta, vargha_delaney_a — nonparametric")
                log.write("  fixed_effects_meta, random_effects_meta — meta-analysis")

            elif topic == "chat":
                log.write("[bold cyan]Chat Commands[/bold cyan]")
                log.write("")
                log.write("  /run <module>    — Run an analysis module")
                log.write("  /list            — List available modules")
                log.write("  /doctor          — Run diagnostics")
                log.write("  /profile <csv>   — Profile a dataset")
                log.write("  /inspect <path>  — Inspect output CSV(s)")
                log.write("  /verify <path>   — Verify statistical outputs")
                log.write("  /agent <name>    — Switch agent persona")
                log.write("  /agents          — List available agents")
                log.write("  /models          — List available LLM models")
                log.write("  /model <alias>   — Switch model (e.g. /model dq)")
                log.write("  /provider        — Show LLM provider")
                log.write("  /history         — Show conversation history")
                log.write("  /clear           — Clear history")
                log.write("  /help            — Show all commands")

            elif topic == "keybindings":
                log.write("[bold cyan]TUI Keybindings[/bold cyan]")
                log.write("")
                log.write("  [bold]c[/bold]   — Chat (LLM agent)")
                log.write("  [bold]p[/bold]   — Pipeline (run modules)")
                log.write("  [bold]d[/bold]   — Doctor (diagnostics)")
                log.write("  [bold]i[/bold]   — Inspect (dataset browser)")
                log.write("  [bold]h[/bold]   — Help (documentation)")
                log.write("  [bold]g[/bold]   — Debug (log viewer)")
                log.write("  [bold]a[/bold]   — Analysis (run stats)")
                log.write("  [bold]e[/bold]   — REPL (Python/R console)")
                log.write("  [bold]q[/bold]   — Quit")
                log.write("  [bold]Esc[/bold] — Go back to previous screen")

            elif topic == "cli":
                log.write("[bold cyan]CLI Reference[/bold cyan]")
                log.write("")
                log.write("  moirais                    — Launch TUI")
                log.write("  moirais chat               — Interactive chat REPL")
                log.write("  moirais pipeline --all -y  — Run all modules")
                log.write("  moirais doctor             — Diagnostics")
                log.write("  moirais inspect <path>     — Inspect CSV files")
                log.write("  moirais verify <path>      — Verify statistical outputs")
                log.write("  moirais selftest           — Smoke test all subsystems")
                log.write("  moirais data profile <csv> — Profile dataset (native)")
                log.write("  moirais install            — Self-bootstrap")

            elif topic == "version":
                import moirais

                log.write("[bold cyan]Version Info[/bold cyan]")
                log.write(f"  moirais: {moirais.__version__}")
                log.write(f"  Python: {sys.version.split()[0]}")
                import numpy
                import pandas
                import scipy

                log.write(f"  pandas: {pandas.__version__}")
                log.write(f"  numpy: {numpy.__version__}")
                log.write(f"  scipy: {scipy.__version__}")
                from .llm import detect_provider_and_model

                _, model_label = detect_provider_and_model()
                log.write(f"  LLM: {model_label}")

            elif topic in ("missing", "imputation"):
                log.write("[bold cyan]Missing Data (moirais.missing)[/bold cyan]")
                log.write("")
                log.write("  missing_profile()   — Summarize missingness")
                log.write("  littles_mcar_test() — Test MCAR assumption")
                log.write("  mice()              — Multiple imputation (MICE)")
                log.write("  rubins_rules()      — Pool multiply imputed estimates")
                log.write("  tipping_point_analysis() — Sensitivity to MNAR")

            elif topic in ("sensitivity", "sens"):
                log.write("[bold cyan]Sensitivity Analysis (moirais.sensitivity)[/bold cyan]")
                log.write("")
                log.write("  e_value_rr()           — E-value for unmeasured confounding")
                log.write("  rosenbaum_bounds()     — Sensitivity to hidden bias")
                log.write("  tipping_point_analysis() — Missing data sensitivity")
                log.write("  omitted_variable_bias() — Cinelli-Hazlett OVB")
                log.write("  specification_curve()  — Multi-specification robustness")

            elif topic in ("bootstrap", "resampling"):
                log.write("[bold cyan]Bootstrap (moirais.bootstrap_methods)[/bold cyan]")
                log.write("")
                log.write("  bootstrap()           — Nonparametric bootstrap (BCa, percentile)")
                log.write("  wild_bootstrap()      — For heteroskedasticity")
                log.write("  block_bootstrap()     — For time series/panel")
                log.write("  jackknife()           — Delete-one jackknife")
                log.write("  permutation_test()    — Two-sample permutation")
                log.write("  cross_validate()      — K-fold CV")

            elif topic == "did":
                log.write("[bold cyan]Difference-in-Differences (moirais.did)[/bold cyan]")
                log.write("")
                log.write("  did_2x2()                — Classic 2x2 DiD estimator")
                log.write("  did_doubly_robust()      — Doubly-robust DiD")
                log.write("  did_continuous_treatment()— Continuous treatment DiD")
                log.write("  bacon_decomposition()    — Goodman-Bacon decomposition")
                log.write("  did_chaisemartin_dhaultfoeuille() — Staggered DiD")
                log.write("  did_diagnostics()        — Pre-trends and parallel trends tests")

            elif topic == "rdd":
                log.write("[bold cyan]Regression Discontinuity (moirais.rdd)[/bold cyan]")
                log.write("")
                log.write("  sharp_rdd()              — Sharp RDD with local polynomial")
                log.write("  fuzzy_rdd()              — Fuzzy RDD (IV at cutoff)")
                log.write("  donut_rdd()              — Donut-hole RDD")
                log.write("  bandwidth_ik()           — Imbens-Kalyanaraman bandwidth")
                log.write("  bandwidth_cct()          — Calonico-Cattaneo-Titiunik bandwidth")
                log.write("  cattaneo_density_test()  — McCrary/Cattaneo manipulation test")
                log.write("  covariate_balance_rdd()  — Covariate balance at cutoff")

            elif topic == "iv":
                log.write("[bold cyan]Instrumental Variables (moirais.iv)[/bold cyan]")
                log.write("")
                log.write("  tsls()                   — Two-stage least squares")
                log.write("  liml()                   — Limited information maximum likelihood")
                log.write("  gmm_iv()                 — GMM estimator")
                log.write("  cragg_donald_test()      — Weak instrument test")
                log.write("  anderson_rubin_test()    — Anderson-Rubin weak-IV-robust test")
                log.write("  durbin_wu_hausman()      — Endogeneity test")
                log.write("  control_function()       — Control function approach")

            elif topic == "matching":
                log.write("[bold cyan]Matching Methods (moirais.matching)[/bold cyan]")
                log.write("")
                log.write("  propensity_score_matching() — PSM (nearest-neighbor, caliper)")
                log.write("  match_nearest_neighbor()    — NN matching on covariates")
                log.write("  match_cem()                 — Coarsened exact matching")
                log.write("  entropy_balance()           — Entropy balancing weights")
                log.write("  doubly_robust_matching()    — DR matching estimator")
                log.write("  balance_diagnostics()       — Post-matching balance checks")
                log.write("  common_support()            — Common support diagnostics")

            elif topic in ("weights", "survey_weights"):
                log.write("[bold cyan]Survey Weights (moirais.weights)[/bold cyan]")
                log.write("")
                log.write("  design_weights()         — Base design weights")
                log.write("  post_stratify()          — Post-stratification")
                log.write("  rake()                   — Iterative proportional fitting")
                log.write("  greg_calibrate()         — GREG calibration")
                log.write("  replicate_weights()      — BRR/jackknife/bootstrap replicates")
                log.write("  trimmed_weights()        — Weight trimming")

            elif topic == "reporting":
                log.write("[bold cyan]Reporting & Export (moirais.reporting, moirais.export)[/bold cyan]")
                log.write("")
                log.write("[bold]Tables:[/bold]")
                log.write("  table1()                 — Baseline characteristics table")
                log.write("  regression_table()       — Regression results table")
                log.write("  hazard_ratio_table()     — HR table for Cox models")
                log.write("[bold]Formatting:[/bold]")
                log.write("  apa_format()             — APA-style formatting")
                log.write("  strobe_checklist()       — STROBE compliance check")
                log.write("  consort_checklist()      — CONSORT compliance check")
                log.write("[bold]Export:[/bold]")
                log.write("  export_latex()           — LaTeX output")
                log.write("  export_html()            — HTML output")
                log.write("  export_docx()            — Word document output")

            else:
                log.write(f"[yellow]Unknown topic: {topic}[/yellow]")
                log.write("Type 'overview' for available topics.")

    # ==================================================================
    # SettingsScreen — edit moirais.conf in-TUI (like nano/vim)
    # ==================================================================

    class SettingsScreen(Screen):
        BINDINGS = [
            Binding("escape", "app.pop_screen", "Back"),
            Binding("ctrl+s", "save_config", "Save", show=True),
        ]

        DEFAULT_CSS = """
        SettingsScreen { layout: vertical; }
        #settings-header { height: 1; background: $accent; color: $text; }
        #settings-editor { height: 1fr; }
        #settings-footer { height: 1; background: $surface; color: $text-muted; }
        """

        def _config_path(self) -> Path:
            """Find the moirais.conf file."""
            # Check package directory first
            pkg_conf = Path(__file__).parent / "moirais.conf"
            if pkg_conf.exists():
                return pkg_conf
            # Check project root
            try:
                from .data import _project_root

                root_conf = _project_root() / "moirais.conf"
                if root_conf.exists():
                    return root_conf
            except Exception:
                pass
            # Create default at package dir
            return pkg_conf

        def compose(self) -> ComposeResult:
            yield Static("[bold]moirais.conf[/bold] — ctrl+s Save | esc Back", id="settings-header")
            conf = self._config_path()
            text = conf.read_text() if conf.exists() else "# moirais.conf — no config found\n"
            yield TextArea(text, id="settings-editor", language="toml", show_line_numbers=True)
            yield Static(
                "[dim]Edit settings, then ctrl+s to save. Changes apply on next TUI restart.[/dim]",
                id="settings-footer",
            )

        def action_save_config(self) -> None:
            editor = self.query_one("#settings-editor", TextArea)
            conf = self._config_path()
            conf.write_text(editor.text)
            footer = self.query_one("#settings-footer", Static)
            footer.update(f"[bold green]Saved to {conf}[/bold green]")

    # ==================================================================
    # DebugScreen — log viewer and diagnostics
    # ==================================================================

    class DebugScreen(Screen):
        BINDINGS = [
            Binding("escape", "app.pop_screen", "Back"),
            Binding("r", "refresh", "Refresh"),
            Binding("s", "run_selftest", "Selftest"),
        ]

        DEFAULT_CSS = """
        DebugScreen { layout: vertical; }
        #debug-log { height: 1fr; border: heavy $primary 50%; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            yield CopyableRichLog(id="debug-log", highlight=True, markup=True, wrap=True)
            yield Footer()

        def on_mount(self) -> None:
            self._show_debug_info()

        def action_refresh(self) -> None:
            log = self.query_one("#debug-log", RichLog)
            log.clear()
            self._show_debug_info()

        def action_run_selftest(self) -> None:
            self.run_worker(self._selftest(), name="selftest", exclusive=True)

        async def _selftest(self) -> None:
            log = self.query_one("#debug-log", RichLog)
            log.write("\n[bold yellow]Running selftest...[/bold yellow]\n")

            from .selftest import _results

            _results.clear()

            tests = [
                ("Core imports", lambda: __import__("moirais") and "OK"),
                ("Module registry", lambda: f"{len(__import__('moirais').list_modules())} modules"),
                (
                    "LLM detection",
                    lambda: __import__("moirais.llm", fromlist=["detect_provider_and_model"]).detect_provider_and_model()[
                        1
                    ],
                ),
            ]

            for name, fn in tests:
                try:
                    result = fn()
                    log.write(f"  [green]OK[/green]   {name}: {result}")
                except Exception as e:
                    log.write(f"  [red]FAIL[/red] {name}: {e}")

            log.write("\n[dim]Press 's' to run full selftest, 'r' to refresh.[/dim]")

        def _show_debug_info(self) -> None:
            log = self.query_one("#debug-log", RichLog)
            log.can_focus = False
            log.write("[bold cyan]MOIRAIS Debug Console[/bold cyan]\n")

            log.write("[bold]System:[/bold]")
            import platform

            log.write(f"  Platform: {platform.platform()}")
            log.write(f"  Python:   {sys.version.split()[0]}")
            log.write(f"  CWD:      {os.getcwd()}")
            log.write(f"  PID:      {os.getpid()}")
            log.write("")

            log.write("[bold]MOIRAIS:[/bold]")
            try:
                import moirais

                log.write(f"  Version:  {moirais.__version__}")
                log.write(f"  Package:  {Path(moirais.__file__).parent}")
                log.write(f"  Modules:  {len(moirais.list_modules())}")
            except Exception as e:
                log.write(f"  [red]Error: {e}[/red]")

            log.write("")
            log.write("[bold]Loaded Python modules (moirais.*):[/bold]")
            moirais_mods = sorted(k for k in sys.modules if k.startswith("moirais"))
            for m in moirais_mods:
                log.write(f"  {m}")

            log.write("")
            log.write("[bold]Environment:[/bold]")
            for var in [
                "MOIRAIS_HOME",
                "MOIRAIS_DEBUG",
                "OLLAMA_BASE_URL",
                "GEMINI_API_KEY",
                "LLM_API_BASE_URL",
                "OPENAI_API_KEY",
            ]:
                val = os.environ.get(var, "")
                if "KEY" in var and val:
                    val = val[:4] + "..." + val[-4:]
                log.write(f"  {var}={val or '(not set)'}")

            log.write("")
            log.write("[dim]Press 's' for selftest, 'r' to refresh.[/dim]")

    # ==================================================================
    # ReplScreen — Python/R/Shell interactive console
    # ==================================================================

    class ReplScreen(Screen):
        """Real interactive console with Python (code.InteractiveConsole),
        persistent R subprocess, and shell execution."""

        BINDINGS = [
            Binding("escape", "app.pop_screen", "Back"),
            Binding("ctrl+j", "submit_editor", "Run", show=True, priority=True),
            Binding("f3", "history_back", "Prev", show=True),
            Binding("f4", "history_forward", "Next", show=True),
            Binding("ctrl+o", "copy_selection", "Copy", show=False),
        ]

        DEFAULT_CSS = """
        ReplScreen { layout: vertical; }
        #repl-body { height: 1fr; }
        #repl-tree { width: 28; min-width: 22; max-width: 40; height: 100%;
          border-right: solid $primary 50%; overflow-y: auto; }
        #repl-right { width: 1fr; height: 100%; }
        #repl-log { height: 2fr; border: heavy $primary; overflow-y: auto; }
        #repl-editor { height: 1fr; border: heavy $accent; min-height: 5; }
        #repl-mode { height: 1; padding: 0 1; dock: bottom; }
        """

        # R-like syntax patterns for auto-detection
        # Only patterns that are UNAMBIGUOUS R (not valid Python syntax)
        _R_PATTERNS = {
            "<-",  # R assignment — not valid Python
            "%>%",
            "%in%",  # R pipe/membership — not valid Python
            "library(",
            "require(",  # R-specific
            "data.frame(",  # R-specific (Python: pd.DataFrame)
            "read.csv(",  # R-specific (Python: pd.read_csv)
            "install.packages(",  # R-specific
            "dplyr::",
            "tidyr::",  # R namespace operator
            "ggplot(",
            "aes(",  # R ggplot
            "tibble(",  # R-specific
            "lm(",
            "glm(",  # R linear models
            "t.test(",
            "chisq.test(",  # R tests (note the dot)
        }

        # Shell-like patterns
        _SHELL_PATTERNS = {
            "ls ",
            "cd ",
            "pwd",
            "cat ",
            "grep ",
            "find ",
            "echo ",
            "mkdir ",
            "rm ",
            "cp ",
            "mv ",
            "chmod ",
            "chown ",
            "git ",
            "docker ",
            "brew ",
            "apt ",
            "pip ",
            "npm ",
            "curl ",
            "wget ",
            "ssh ",
            "scp ",
            "tar ",
            "zip ",
        }

        def __init__(self, lang: str = "auto") -> None:
            super().__init__()
            self._lang = lang if lang != "auto" else "python"
            self._auto_detect = lang == "auto"
            self._history: list[str] = []
            self._history_idx: int = -1
            self._py_console = None  # code.InteractiveConsole
            self._r_proc = None  # persistent R subprocess
            self._py_console_ns: dict = {}
            self._polyglot: bool = False  # P↔R↔Shell variable bridge
            self._current_model: str | None = None  # override FreeAPI model
            # Detect user's shell
            self._user_shell = os.environ.get("SHELL", "/bin/bash")

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="repl-body"):
                tree: Tree[str] = Tree("MOIRAIS", id="repl-tree")
                tree.root.expand()
                yield tree
                with Vertical(id="repl-right"):
                    yield CopyableRichLog(id="repl-log", highlight=True, markup=True, wrap=True)
                    yield TextArea(id="repl-editor", language="python")
            yield Static("", id="repl-mode")
            yield Footer()

        def _detect_language(self, code: str) -> str:
            """Auto-detect language from syntax patterns."""
            stripped = code.strip()

            # Explicit shell commands (starts with ! or $)
            if stripped.startswith("!") or stripped.startswith("$"):
                return "shell"

            # Check R patterns
            for pat in self._R_PATTERNS:
                if pat in code:
                    return "r"

            # Check shell patterns
            for pat in self._SHELL_PATTERNS:
                if stripped.startswith(pat) or stripped.startswith(pat.strip()):
                    return "shell"

            # Starts with a known shell command name
            first_word = stripped.split()[0] if stripped.split() else ""
            if first_word in {
                "ls",
                "cd",
                "pwd",
                "cat",
                "grep",
                "find",
                "echo",
                "mkdir",
                "rm",
                "cp",
                "mv",
                "git",
                "docker",
                "brew",
                "which",
                "whoami",
                "date",
                "env",
                "export",
                "source",
                "man",
                "head",
                "tail",
                "wc",
                "sort",
                "uniq",
                "sed",
                "awk",
                "cut",
                "tr",
                "xargs",
                "tee",
                "touch",
            }:
                return "shell"

            # Default to current mode
            return self._lang

        def on_mount(self) -> None:
            import code as code_module

            log = self.query_one("#repl-log", RichLog)
            log.can_focus = False

            shell_name = Path(self._user_shell).name
            log.write("[bold cyan]MOIRAIS Interactive Console[/bold cyan]")
            log.write(
                f"[dim]Full Python + R + {shell_name} | "
                f"Auto-detects language | "
                f"/python /r /shell /auto /polyglot /reset | Up/Down history[/dim]"
            )
            log.write("[dim]Prefix with ! for shell, R> for R — or just type and let auto-detect handle it[/dim]")
            log.write("[dim]Commands: /list /run /help /history | load('cpads'), head(), cols(), summary()[/dim]")
            log.write("[dim]Stats: ttest('col'), corr('c1','c2'), evalue(rr) | help_repl() for 1249 commands[/dim]")
            log.write("")

            # Initialize Python InteractiveConsole
            self._py_console_ns = {"__name__": "__console__", "__builtins__": __builtins__}
            self._py_console = code_module.InteractiveConsole(self._py_console_ns)
            self._inject_repl_helpers()
            try:
                self._py_console.runsource("import moirais; import pandas as pd; import numpy as np")
                self._py_console.runsource("from moirais import *")
                log.write("[green]Python: moirais, pandas (pd), numpy (np) loaded[/green]")
            except Exception as e:
                log.write(f"[yellow]Pre-import warning: {e}[/yellow]")

            # Try to start persistent R process
            self._start_r_process(log)

            log.write("")
            self._update_mode_indicator()

            # Populate the helper tree
            tree = self.query_one("#repl-tree", Tree)
            data_node = tree.root.add("Data Loading", expand=True)
            for h in [
                "load(path)",
                "head(df)",
                "tail(df)",
                "shape(df)",
                "cols(df)",
                "describe(df)",
                "sample(df)",
                "missing()",
            ]:
                data_node.add_leaf(h)
            wrangle = tree.root.add("Wrangling", expand=False)
            for h in [
                "filter_rows(cond)",
                "select_cols(c1,c2)",
                "rename_col(old,new)",
                "dropna(col)",
                "crosstab(c1,c2)",
                "pivot(idx,col,val)",
                "groupby(grp,col,fn)",
            ]:
                wrangle.add_leaf(h)
            stats_node = tree.root.add("Statistics", expand=True)
            for h in [
                "ttest(col)",
                "ttest2(col,grp)",
                "anova(col,grp)",
                "chi2(c1,c2)",
                "corr(c1,c2)",
                "mannwhitney(col,grp)",
                "ks_test(col)",
                "shapiro(col)",
            ]:
                stats_node.add_leaf(h)
            effect = tree.root.add("Effect Sizes", expand=False)
            for h in [
                "effect_size(col,grp)",
                "hedges_g(c1,c2)",
                "odds_ratio(c1,c2)",
                "power(d,n,alpha)",
                "bootstrap_ci(col)",
                "bh(p1,p2,...)",
            ]:
                effect.add_leaf(h)
            causal = tree.root.add("Causal Inference", expand=False)
            for h in ["ate(outcome,trt)", "propensity(trt,covs)", "ipw(treatment)", "evalue(rr)", "ebac(drinks,wt,hr)"]:
                causal.add_leaf(h)
            survival = tree.root.add("Survival", expand=False)
            for h in ["kaplan_meier(t,e)", "cox(t,e,covs)", "logrank(t,e,grp)"]:
                survival.add_leaf(h)
            did_node = tree.root.add("DiD / RDD / IV", expand=False)
            for h in ["did(y,treat,post)", "rdd(y,running,c)", "iv_tsls(y,endog,iv)"]:
                did_node.add_leaf(h)
            diag = tree.root.add("Diagnostics", expand=False)
            for h in ["vif(cols)", "nnt(c1,c2)", "rosenbaum(y,trt)", "table1(group)"]:
                diag.add_leaf(h)
            sys_node = tree.root.add("System", expand=False)
            for h in [
                "modules()",
                "run_module(name)",
                "selftest()",
                "doctor()",
                "version()",
                "ls()",
                "who()",
                "clear()",
            ]:
                sys_node.add_leaf(h)
            slash = tree.root.add("Slash Commands", expand=False)
            for h in ["/python", "/r", "/shell", "/auto", "/reset", "/history"]:
                slash.add_leaf(h)

            # Dynamic categories from stat_commands registry
            try:
                from .stat_commands import commands_by_category

                existing = {
                    "Data Loading",
                    "Wrangling",
                    "Statistics",
                    "Effect Sizes",
                    "Causal Inference",
                    "Survival",
                    "DiD / RDD / IV",
                    "Diagnostics",
                    "System",
                    "Slash Commands",
                }
                for cat, cmds in sorted(commands_by_category().items()):
                    if cat not in existing:
                        node = tree.root.add(f"{cat} ({len(cmds)})", expand=False)
                        for c in cmds[:25]:
                            sig = c.usage.split(" ", 1)[-1] if " " in c.usage else "()"
                            node.add_leaf(f"{c.name}({sig})")
                        if len(cmds) > 25:
                            node.add_leaf(f"... +{len(cmds) - 25} more")
            except ImportError:
                pass

            self.query_one("#repl-editor", TextArea).focus()

        def _inject_repl_helpers(self) -> None:
            """Inject convenience functions into the Python REPL namespace."""
            import pprint as _pprint

            # Resolve project root: py-package/moirais/tui.py -> project root
            import moirais as _moirais_mod

            _proj_root = Path(_moirais_mod.__file__).resolve().parent.parent.parent

            ns = self._py_console_ns

            def _view(obj=None):
                """Pretty-print any object. DataFrames use .to_string()."""
                if obj is None:
                    obj = ns.get("df")
                    if obj is None:
                        print("Usage: view(variable)")
                        return
                if hasattr(obj, "to_string"):
                    print(obj.to_string())
                else:
                    _pprint.pprint(obj)

            def _ls():
                """List user-defined variables with their types."""
                skip = {"__name__", "__builtins__", "__doc__", "view", "ls", "clear", "who", "moirais", "pd", "np"}
                user_vars = {k: type(v).__name__ for k, v in ns.items() if not k.startswith("_") and k not in skip}
                if user_vars:
                    for name, typ in sorted(user_vars.items()):
                        print(f"  {name}: {typ}")
                else:
                    print("  (no user variables)")

            def _who():
                """List variables with types and values (like MATLAB whos)."""
                skip = {"__name__", "__builtins__", "__doc__", "view", "ls", "clear", "who", "moirais", "pd", "np"}
                user_vars = {k: v for k, v in ns.items() if not k.startswith("_") and k not in skip}
                if user_vars:
                    for name, val in sorted(user_vars.items()):
                        typ = type(val).__name__
                        rep = repr(val)
                        if len(rep) > 60:
                            rep = rep[:57] + "..."
                        print(f"  {name} ({typ}): {rep}")
                else:
                    print("  (no user variables)")

            def _clear():
                """Clear user variables from namespace (keeps imports)."""
                skip = {"__name__", "__builtins__", "__doc__", "view", "ls", "clear", "who", "moirais", "pd", "np"}
                to_remove = [k for k in ns if not k.startswith("_") and k not in skip]
                for k in to_remove:
                    del ns[k]
                print(f"Cleared {len(to_remove)} variable(s)")

            def _load(path_or_name=None):
                """Load a dataset. Use load('cpads') or load('path/to/file.csv')."""
                import pandas as _pd

                from moirais.data import list_datasets, load_dataset

                if path_or_name is None:
                    print("Usage: load('name') or load('path.csv')")
                    print("\n  Available datasets (from built-in DB + cache):")
                    try:
                        datasets = list_datasets()
                        for d in datasets:
                            tag = f" ({d['rows']} rows)" if d.get("rows") else ""
                            cached = " [cached]" if d.get("cached") else ""
                            print(f"    load('{d['key']}')  — {d['name']}{tag}{cached}")
                    except Exception as e:
                        print(f"    (error listing: {e})")
                    return None
                # Try catalog name first (fuzzy-matched)
                try:
                    df = load_dataset(path_or_name)
                    var_name = path_or_name.replace("-", "_").replace(" ", "_").lower()
                    ns[var_name] = df
                    ns["df"] = df
                    print(f"Loaded {path_or_name}: {df.shape[0]} rows x {df.shape[1]} cols → '{var_name}' and 'df'")
                    return df
                except (KeyError, FileNotFoundError):
                    pass
                # Fall back to file path
                _proj = _proj_root
                candidates = [
                    Path(path_or_name),
                    _proj / path_or_name,
                    _proj / "data" / path_or_name,
                    _proj / "data" / "datasets" / path_or_name,
                    Path.cwd() / path_or_name,
                ]
                for p in candidates:
                    if p.exists():
                        df = _pd.read_csv(p, low_memory=False)
                        name = p.stem.replace("-", "_").replace(" ", "_")
                        ns[name] = df
                        ns["df"] = df
                        print(f"Loaded {p.name}: {df.shape[0]} rows x {df.shape[1]} cols → '{name}' and 'df'")
                        return df
                print(f"Not found: {path_or_name}")
                print("  Use load() with no args to see available datasets")
                return None

            def _head(obj=None, n=10):
                """Show first n rows of a DataFrame (default: last loaded 'df')."""
                if obj is None:
                    obj = ns.get("df")
                if obj is None:
                    print("No data loaded. Use load('file.csv') first.")
                    return
                if hasattr(obj, "head"):
                    print(obj.head(n).to_string())
                else:
                    print(repr(obj)[:500])

            def _shape(obj=None):
                """Show shape of a DataFrame."""
                if obj is None:
                    obj = ns.get("df")
                if obj is None:
                    print("No data loaded.")
                    return
                if hasattr(obj, "shape"):
                    print(f"  {obj.shape[0]} rows x {obj.shape[1]} columns")
                else:
                    print(f"  type: {type(obj).__name__}, len: {len(obj) if hasattr(obj, '__len__') else '?'}")

            def _cols(obj=None):
                """List columns and dtypes of a DataFrame."""
                if obj is None:
                    obj = ns.get("df")
                if obj is None:
                    print("No data loaded.")
                    return
                if hasattr(obj, "dtypes"):
                    for col in obj.columns:
                        n_miss = obj[col].isna().sum()
                        miss_str = f" ({n_miss} missing)" if n_miss > 0 else ""
                        print(f"  {col}: {obj[col].dtype}{miss_str}")
                else:
                    print(repr(obj)[:200])

            def _describe(obj=None):
                """Descriptive statistics for a DataFrame."""
                if obj is None:
                    obj = ns.get("df")
                if obj is None:
                    print("No data loaded.")
                    return
                if hasattr(obj, "describe"):
                    print(obj.describe().to_string())
                else:
                    print(repr(obj))

            def _modules():
                """List available MOIRAIS analysis modules."""
                from moirais.modules import list_modules

                mods = list_modules()
                print(f"  {len(mods)} modules available:")
                for m in mods:
                    print(f"    {m['name']}: {m['description']}")

            def _run_module(name=None, **kwargs):
                """Run an MOIRAIS analysis module by name."""
                if name is None:
                    print("Usage: run_module('module-name')")
                    _modules()
                    return
                from moirais.modules import run_module

                print(f"Running module: {name}...")
                result = run_module(name, **kwargs)
                print(f"  Status: {result.get('status', 'unknown')}")
                return result

            def _ttest(col=None, mu=0, data=None):
                """One-sample t-test. Usage: ttest('column_name') or ttest('col', mu=5, data=df)."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None:
                    print("Usage: ttest('column_name', mu=0)")
                    _nc = [c for c in data.columns if data[c].dtype.kind in "iuf"][:10]
                    if _nc:
                        print(f"  Numeric columns: {', '.join(_nc)}")
                    return
                from moirais.statistics import one_sample_ttest

                vals = data[col].dropna().values
                r = one_sample_ttest(vals, mu0=mu)
                print(f"  One-sample t-test: {col} vs mu={mu}")
                print(f"  n={len(vals)}, t={r.test_statistic:.4f}, p={r.p_value:.6f}, df={r.df:.0f}")
                if hasattr(r, "ci_lower"):
                    print(f"  95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
                return r

            def _corr(col1=None, col2=None, data=None):
                """Pearson correlation. Usage: corr('col1', 'col2')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col1 is None or col2 is None:
                    print("Usage: corr('col1', 'col2')")
                    _nc = [c for c in data.columns if data[c].dtype.kind in "iuf"][:10]
                    if _nc:
                        print(f"  Numeric columns: {', '.join(_nc)}")
                    return
                from moirais.statistics import pearson_correlation

                x = data[col1].dropna().values
                y = data[col2].dropna().values
                n = min(len(x), len(y))
                r = pearson_correlation(x[:n], y[:n])
                print(f"  Pearson r({col1}, {col2}): r={r.test_statistic:.4f}, p={r.p_value:.6f}")
                return r

            def _evalue(rr=None, ci_lower=None):
                """E-value for unmeasured confounding. Usage: evalue(2.5) or evalue(2.5, 1.1)."""
                if rr is None:
                    print("Usage: evalue(rr) or evalue(rr, ci_lower)")
                    print("  rr: risk ratio (e.g. 2.5)")
                    return
                from moirais.sensitivity import e_value_rr

                r = e_value_rr(rr, ci_lower=ci_lower)
                print(f"  E-value (RR={rr}): point={r.e_value_point:.2f}, CI={r.e_value_ci:.2f}")
                print(f"  {r.interpretation}")
                return r

            def _ate(outcome=None, treatment=None, weights_col="weight", data=None):
                """Estimate ATE via IPW-weighted OLS. Usage: ate('outcome', 'treatment')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if outcome is None or treatment is None:
                    print("Usage: ate('outcome_col', 'treatment_col')")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                from moirais.effects import estimate_ate

                coef, se = estimate_ate(data, outcome, treatment, weights_col)
                print(f"  ATE({outcome} ~ {treatment}): coef={coef:.4f}, SE={se:.4f}")
                return {"ate": coef, "se": se}

            def _help_repl():
                """Show all available REPL helper functions."""
                sections = {
                    "Data Loading & I/O": [
                        ("load(path)", "Load CSV/CPADS → 'df'"),
                        ("save(path)", "Save to .csv/.xlsx/.json/.py/.r/.qmd/.md/.html/.tex"),
                        ("export(path, fmt)", "Export analysis bundle"),
                    ],
                    "Exploration": [
                        ("head(df, n)", "First n rows"),
                        ("tail(df, n)", "Last n rows"),
                        ("sample(df, n)", "Random n rows"),
                        ("shape(df)", "Dimensions"),
                        ("cols(df)", "Columns + types + missing"),
                        ("describe(df)", "Descriptive statistics"),
                        ("summary(col)", "R-like summary"),
                        ("profile()", "Full dataset profile"),
                        ("plan()", "Suggest analysis plan"),
                        ("missing()", "Missing data report"),
                    ],
                    "Column Inspection": [
                        ("unique(col)", "Unique values"),
                        ("value_counts(col)", "Frequency table"),
                        ("freq(col)", "Freq table + cumulative %"),
                    ],
                    "Data Wrangling": [
                        ("filter_rows(cond)", "Filter rows by condition"),
                        ("select_cols(c1,c2)", "Select columns"),
                        ("rename_col(old,new)", "Rename column"),
                        ("dropna(col)", "Drop NA rows"),
                        ("crosstab(c1, c2)", "Cross-tabulation"),
                        ("pivot(idx,col,val)", "Pivot table"),
                        ("groupby(grp,col,fn)", "Group-by aggregation"),
                    ],
                    "Statistical Tests": [
                        ("ttest(col, mu)", "One-sample t-test"),
                        ("ttest2(col, grp)", "Two-sample t-test"),
                        ("paired_t(c1, c2)", "Paired t-test"),
                        ("wilcoxon(c1, c2)", "Wilcoxon signed-rank"),
                        ("anova(col, grp)", "One-way ANOVA"),
                        ("chi2(c1, c2)", "Chi-square independence"),
                        ("fisher(a,b,c,d)", "Fisher's exact test"),
                        ("mannwhitney(col,grp)", "Mann-Whitney U"),
                        ("ks_test(col)", "KS normality test"),
                        ("shapiro(col)", "Shapiro-Wilk normality"),
                        ("normality(col)", "Normality test battery"),
                        ("levene(col, grp)", "Levene's variance test"),
                        ("corr(c1, c2)", "Pearson correlation"),
                        ("spearman(c1, c2)", "Spearman correlation"),
                        ("cramers_v(c1, c2)", "Cramér's V association"),
                    ],
                    "Effect Sizes & Power": [
                        ("effect_size(col,grp)", "Cohen's d"),
                        ("hedges_g(c1, c2)", "Hedges' g"),
                        ("cohens_f(col, grp)", "Cohen's f (ANOVA)"),
                        ("odds_ratio(c1,c2)", "Odds ratio"),
                        ("risk_ratio(a,b,c,d)", "Risk ratio"),
                        ("power(d, n, alpha)", "Power analysis"),
                        ("bootstrap_ci(col)", "Bootstrap CI for mean"),
                        ("bh(p1, p2, ...)", "Benjamini-Hochberg"),
                    ],
                    "Causal Inference": [
                        ("ate(outcome, trt)", "Average treatment effect"),
                        ("propensity(trt,covs)", "Propensity scores"),
                        ("ipw(treatment)", "IPW weights"),
                        ("evalue(rr)", "E-value sensitivity"),
                        ("ebac(drinks,wt,hr)", "eBAC calculation"),
                    ],
                    "Survival Analysis": [
                        ("kaplan_meier(t,e)", "Kaplan-Meier curve"),
                        ("cox(t,e,covs)", "Cox proportional hazards"),
                        ("logrank(t,e,grp)", "Log-rank test"),
                    ],
                    "DiD / RDD / IV": [
                        ("did(y,treat,post)", "Difference-in-differences"),
                        ("rdd(y,running,c)", "Sharp RDD"),
                        ("iv_tsls(y,endog,iv)", "2SLS instrumental variables"),
                    ],
                    "Matching & Missing": [
                        ("match(trt,covs)", "PS matching"),
                        ("mcar()", "Little's MCAR test"),
                        ("impute(m)", "MICE imputation"),
                    ],
                    "Diagnostics & Reporting": [
                        ("vif(cols)", "VIF collinearity"),
                        ("odds_ratio(c1,c2)", "Odds ratio"),
                        ("nnt(c1,c2)", "Number needed to treat"),
                        ("rosenbaum(y,trt)", "Rosenbaum bounds"),
                        ("table1(group)", "Publication Table 1"),
                    ],
                    "System": [
                        ("modules()", "List MOIRAIS modules"),
                        ("run_module(name)", "Run a module"),
                        ("selftest()", "Run self-test"),
                        ("doctor()", "Environment check"),
                        ("version()", "Version info"),
                        ("ls()", "List variables"),
                        ("who()", "Variables + values"),
                        ("view(x)", "Pretty-print object"),
                        ("clear()", "Clear variables"),
                        ("help_repl()", "This help"),
                    ],
                }
                print("  MOIRAIS REPL — 85+ Helper Functions:")
                print()
                for section, helpers in sections.items():
                    print(f"  [{section}]")
                    for name, desc in helpers:
                        print(f"    {name:25s} {desc}")
                    print()

            # --- Data wrangling helpers (17-26) ---

            def _tail(obj=None, n=10):
                """Last n rows of a DataFrame."""
                if obj is None:
                    obj = ns.get("df")
                if obj is None:
                    print("No data. Use load() first.")
                    return
                if hasattr(obj, "tail"):
                    print(obj.tail(n).to_string())
                else:
                    print(repr(obj)[-500:])

            def _sample(obj=None, n=5):
                """Random sample of n rows."""
                if obj is None:
                    obj = ns.get("df")
                if obj is None:
                    print("No data.")
                    return
                s = obj.sample(n=min(n, len(obj)))
                print(s.to_string())
                return s

            def _unique(col=None, data=None):
                """Unique values of a column."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None:
                    print("Usage: unique('column_name')")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                vals = data[col].unique()
                print(f"  {col}: {len(vals)} unique values")
                if len(vals) <= 30:
                    for v in sorted(vals, key=str):
                        print(f"    {v}")
                else:
                    for v in sorted(vals, key=str)[:15]:
                        print(f"    {v}")
                    print(f"    ... ({len(vals) - 15} more)")
                return vals

            def _value_counts(col=None, data=None, n=20):
                """Frequency table for a column."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None:
                    print("Usage: value_counts('column_name')")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                vc = data[col].value_counts().head(n)
                total = len(data)
                print(f"  {col} — top {min(n, len(vc))} values:")
                for val, count in vc.items():
                    pct = count / total * 100
                    bar = "#" * int(pct / 2)
                    print(f"    {str(val):20s} {count:6d} ({pct:5.1f}%) {bar}")
                return vc

            def _filter(condition=None, data=None):
                """Filter rows. Usage: filter('col > 5') or filter(df.col > 5)."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if condition is None:
                    print("Usage: filter_rows('condition')")
                    print("  Example: filter_rows('age_groups > 2')")
                    return
                if isinstance(condition, str):
                    result = data.query(condition)
                else:
                    result = data[condition]
                ns["filtered"] = result
                print(f"  Filtered: {len(result)}/{len(data)} rows → 'filtered'")
                return result

            def _select(*columns, data=None):
                """Select columns. Usage: select('col1', 'col2')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data.")
                    return
                result = data[list(columns)]
                ns["selected"] = result
                print(f"  Selected {len(columns)} cols, {len(result)} rows → 'selected'")
                return result

            def _rename(old=None, new=None, data=None):
                """Rename a column. Usage: rename('old_name', 'new_name')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if old is None or new is None:
                    print("Usage: rename_col('old_name', 'new_name')")
                    return
                data = data.rename(columns={old: new})
                ns["df"] = data
                print(f"  Renamed '{old}' → '{new}'")
                return data

            def _dropna(col=None, data=None):
                """Drop rows with missing values. Usage: dropna() or dropna('col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data.")
                    return
                before = len(data)
                if col:
                    data = data.dropna(subset=[col])
                else:
                    data = data.dropna()
                ns["df"] = data
                print(f"  Dropped {before - len(data)} rows with NaN ({len(data)} remaining)")
                return data

            def _save(path="output.csv", data=None):
                """Save DataFrame to CSV. Usage: save('output.csv')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data.")
                    return
                data.to_csv(path, index=False)
                print(f"  Saved {len(data)} rows to {path}")

            # --- Statistical test helpers (27-42) ---

            def _ttest2(col=None, group_col=None, data=None):
                """Two-sample t-test (Welch). Usage: ttest2('col', 'group_col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None or group_col is None:
                    print("Usage: ttest2('value_col', 'group_col')")
                    _nc = [c for c in data.columns if data[c].dtype.kind in "iuf"][:10]
                    if _nc:
                        print(f"  Numeric columns: {', '.join(_nc)}")
                    return
                from moirais.statistics import two_sample_ttest

                groups = data[group_col].dropna().unique()
                if len(groups) < 2:
                    print(f"Need 2+ groups in '{group_col}'")
                    return
                x = data.loc[data[group_col] == groups[0], col].dropna().values
                y = data.loc[data[group_col] == groups[1], col].dropna().values
                r = two_sample_ttest(x, y, equal_var=False)
                print(f"  Two-sample t-test: {col} by {group_col}")
                print(f"  {groups[0]} (n={len(x)}) vs {groups[1]} (n={len(y)})")
                print(f"  t={r.test_statistic:.4f}, p={r.p_value:.6f}")
                return r

            def _paired_t(col1=None, col2=None, data=None):
                """Paired t-test. Usage: paired_t('pre', 'post')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col1 is None or col2 is None:
                    print("Usage: paired_t('col1', 'col2')")
                    _nc = [c for c in data.columns if data[c].dtype.kind in "iuf"][:10]
                    if _nc:
                        print(f"  Numeric columns: {', '.join(_nc)}")
                    return
                from moirais.statistics import paired_ttest

                x = data[col1].dropna().values
                y = data[col2].dropna().values
                n = min(len(x), len(y))
                r = paired_ttest(x[:n], y[:n])
                print(f"  Paired t-test: {col1} vs {col2} (n={n})")
                print(f"  t={r.test_statistic:.4f}, p={r.p_value:.6f}")
                return r

            def _anova(col=None, group_col=None, data=None):
                """One-way ANOVA. Usage: anova('value_col', 'group_col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None or group_col is None:
                    print("Usage: anova('value_col', 'group_col')")
                    _nc = [c for c in data.columns if data[c].dtype.kind in "iuf"][:10]
                    if _nc:
                        print(f"  Numeric columns: {', '.join(_nc)}")
                    return
                from moirais.statistics import one_way_anova

                groups = [g[col].dropna().values for _, g in data.groupby(group_col)]
                r = one_way_anova(*groups)
                print(f"  One-way ANOVA: {col} by {group_col} ({len(groups)} groups)")
                print(f"  F={r.test_statistic:.4f}, p={r.p_value:.6f}")
                return r

            def _chi2(col1=None, col2=None, data=None):
                """Chi-square test of independence. Usage: chi2('col1', 'col2')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col1 is None or col2 is None:
                    print("Usage: chi2('col1', 'col2')")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                import pandas as _pd

                from moirais.statistics import chi2_independence

                ct = _pd.crosstab(data[col1], data[col2])
                r = chi2_independence(ct.values)
                print(f"  Chi-square: {col1} x {col2}")
                print(f"  χ²={r.test_statistic:.4f}, p={r.p_value:.6f}, df={r.df:.0f}")
                return r

            def _mannwhitney(col=None, group_col=None, data=None):
                """Mann-Whitney U test. Usage: mannwhitney('col', 'group_col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None or group_col is None:
                    print("Usage: mannwhitney('value_col', 'group_col')")
                    return
                from moirais.statistics import mann_whitney_u

                groups = data[group_col].dropna().unique()
                x = data.loc[data[group_col] == groups[0], col].dropna().values
                y = data.loc[data[group_col] == groups[1], col].dropna().values
                r = mann_whitney_u(x, y)
                print(f"  Mann-Whitney U: {col} by {group_col}")
                print(f"  U={r.test_statistic:.4f}, p={r.p_value:.6f}")
                return r

            def _ks_test(col=None, data=None):
                """Kolmogorov-Smirnov normality test. Usage: ks_test('col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None:
                    print("Usage: ks_test('column_name')")
                    _nc = [c for c in data.columns if data[c].dtype.kind in "iuf"][:10]
                    if _nc:
                        print(f"  Numeric columns: {', '.join(_nc)}")
                    return
                from moirais.statistics import ks_test_one_sample

                vals = data[col].dropna().values
                r = ks_test_one_sample(vals)
                normal = "Yes" if r.p_value > 0.05 else "No"
                print(f"  KS normality: {col} (n={len(vals)})")
                print(f"  D={r.test_statistic:.4f}, p={r.p_value:.6f}, Normal={normal}")
                return r

            def _spearman(col1=None, col2=None, data=None):
                """Spearman rank correlation. Usage: spearman('col1', 'col2')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col1 is None or col2 is None:
                    print("Usage: spearman('col1', 'col2')")
                    _nc = [c for c in data.columns if data[c].dtype.kind in "iuf"][:10]
                    if _nc:
                        print(f"  Numeric columns: {', '.join(_nc)}")
                    return
                from moirais.statistics import spearman_correlation

                x = data[col1].dropna().values
                y = data[col2].dropna().values
                n = min(len(x), len(y))
                r = spearman_correlation(x[:n], y[:n])
                print(f"  Spearman rho({col1}, {col2}): rho={r.test_statistic:.4f}, p={r.p_value:.6f}")
                return r

            def _bootstrap_ci(col=None, data=None, n_boot=2000):
                """Bootstrap CI for mean. Usage: bootstrap_ci('col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None:
                    print("Usage: bootstrap_ci('column_name')")
                    _nc = [c for c in data.columns if data[c].dtype.kind in "iuf"][:10]
                    if _nc:
                        print(f"  Numeric columns: {', '.join(_nc)}")
                    return
                import numpy as _np

                from moirais.bootstrap_methods import bootstrap as _bs

                vals = data[col].dropna().values.astype(float)
                r = _bs(vals, _np.mean, n_boot=n_boot, ci_method="bca")
                print(f"  Bootstrap CI for mean of '{col}' ({n_boot} replicates)")
                print(f"  Mean={r.estimate:.4f}, SE={r.se:.4f}")
                print(f"  95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
                return r

            def _bh(*p_values):
                """Benjamini-Hochberg correction. Usage: bh(0.01, 0.04, 0.03, 0.20)."""
                import numpy as _np

                from moirais.multiple_testing import benjamini_hochberg

                pv = _np.array(p_values)
                r = benjamini_hochberg(pv)
                print(f"  BH correction ({r.n_tests} tests, {r.n_rejected} rejected):")
                for i, (o, a, rej) in enumerate(zip(r.original, r.adjusted, r.rejected)):
                    status = "REJECT" if rej else "retain"
                    print(f"    [{i + 1}] p={o:.4f} → adjusted={a:.4f} ({status})")
                return r

            def _power(d=None, n=None, alpha=0.05):
                """Power analysis for t-test. Usage: power(0.5, 100)."""
                if d is None or n is None:
                    print("Usage: power(effect_size, sample_size, alpha=0.05)")
                    print("  Example: power(0.5, 100)")
                    return
                from moirais.inference import power_t_test

                pwr = power_t_test(d=d, n=n, alpha=alpha)
                adequate = "Adequate" if pwr >= 0.8 else "UNDERPOWERED"
                print(f"  Power analysis: d={d}, n={n}, α={alpha}")
                print(f"  Power = {pwr:.4f} ({adequate})")
                return pwr

            def _effect_size(col=None, group_col=None, data=None):
                """Cohen's d effect size. Usage: effect_size('col', 'group_col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None or group_col is None:
                    print("Usage: effect_size('value_col', 'group_col')")
                    return
                import numpy as _np

                groups = data[group_col].dropna().unique()
                x = data.loc[data[group_col] == groups[0], col].dropna().values.astype(float)
                y = data.loc[data[group_col] == groups[1], col].dropna().values.astype(float)
                pooled = _np.sqrt(((len(x) - 1) * x.std() ** 2 + (len(y) - 1) * y.std() ** 2) / (len(x) + len(y) - 2))
                d = (x.mean() - y.mean()) / pooled if pooled > 0 else 0
                mag = (
                    "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
                )
                print(f"  Cohen's d: {col} by {group_col}")
                print(f"  d={d:.4f} ({mag})")
                return d

            # --- Causal inference helpers (43-50) ---

            def _propensity(treatment=None, covariates=None, data=None):
                """Propensity scores. Usage: propensity('treatment', ['cov1','cov2'])."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if treatment is None or covariates is None:
                    print("Usage: propensity('treatment_col', ['cov1', 'cov2'])")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                from moirais.causal import compute_propensity_scores

                if isinstance(covariates, str):
                    covariates = [c.strip() for c in covariates.split(",")]
                scores = compute_propensity_scores(data, treatment, covariates)
                ns["ps"] = scores
                print(f"  Propensity scores: treatment={treatment}")
                print(f"  Covariates: {', '.join(covariates)}")
                print(f"  Mean={scores.mean():.4f}, Range=[{scores.min():.4f}, {scores.max():.4f}]")
                print("  → stored in 'ps'")
                return scores

            def _ipw(treatment=None, data=None):
                """IPW weights. Usage: ipw('treatment_col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if treatment is None:
                    print("Usage: ipw('treatment_col')")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                from moirais.causal import calculate_ipw_weights

                w = calculate_ipw_weights(data, treatment)
                ns["ipw_weights"] = w
                ess = (w.sum() ** 2) / (w**2).sum()
                print(f"  IPW weights: treatment={treatment}")
                print(f"  n={len(w)}, mean={w.mean():.4f}, ESS={ess:.1f}")
                print("  → stored in 'ipw_weights'")
                return w

            def _ebac(drinks=None, weight_lbs=None, hours=None, gender="male"):
                """Calculate eBAC. Usage: ebac(5, 180, 3, 'male')."""
                if drinks is None or weight_lbs is None or hours is None:
                    print("Usage: ebac(drinks, weight_lbs, hours, gender='male')")
                    print("  Example: ebac(5, 180, 3, 'male')")
                    return
                from moirais.ebac import calculate_ebac, is_over_legal_limit

                gc = 0.68 if gender.lower() == "male" else 0.55
                bac = calculate_ebac(drinks, weight_lbs, hours, gc)
                over = is_over_legal_limit(bac)
                print(f"  eBAC: {drinks} drinks, {weight_lbs}lbs, {hours}h, {gender}")
                print(f"  BAC = {bac:.4f} {'(OVER LIMIT)' if over else '(under limit)'}")
                return bac

            def _crosstab(col1=None, col2=None, data=None):
                """Cross-tabulation. Usage: crosstab('col1', 'col2')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col1 is None or col2 is None:
                    print("Usage: crosstab('col1', 'col2')")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                import pandas as _pd

                ct = _pd.crosstab(data[col1], data[col2], margins=True)
                print(f"  Crosstab: {col1} x {col2}")
                print(ct.to_string())
                ns["ct"] = ct
                return ct

            def _missing(data=None):
                """Missing data report."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data.")
                    return
                total = len(data)
                print(f"  Missing data report ({total} rows):")
                any_miss = False
                for col in data.columns:
                    n = data[col].isna().sum()
                    if n > 0:
                        any_miss = True
                        pct = n / total * 100
                        bar = "█" * int(pct / 2)
                        print(f"    {col:30s} {n:6d} ({pct:5.1f}%) {bar}")
                if not any_miss:
                    print("    No missing values!")

            def _summary(col=None, data=None):
                """R-like summary. Usage: summary(), summary('col'), or summary(df)."""
                import pandas as _pd

                # If col is a DataFrame, user called summary(df) — treat as data arg
                if isinstance(col, _pd.DataFrame):
                    data = col
                    col = None
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data loaded. Use load('cpads') first.")
                    return
                if col is not None and isinstance(col, str):
                    s = data[col]
                    if s.dtype in ("float64", "int64", "float32", "int32"):
                        print(f"  {col} (numeric):")
                        print(f"    Min:    {s.min()}")
                        print(f"    1st Qu: {s.quantile(0.25):.4f}")
                        print(f"    Median: {s.median():.4f}")
                        print(f"    Mean:   {s.mean():.4f}")
                        print(f"    3rd Qu: {s.quantile(0.75):.4f}")
                        print(f"    Max:    {s.max()}")
                        print(f"    NA's:   {s.isna().sum()}")
                    else:
                        vc = s.value_counts().head(10)
                        print(f"  {col} (categorical, {s.nunique()} levels):")
                        for v, c in vc.items():
                            print(f"    {v}: {c}")
                else:
                    for c in data.columns[:20]:
                        _summary(c, data)
                        print()

            def _freq(col=None, data=None):
                """Frequency table with percentages. Usage: freq('col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None:
                    print("Usage: freq('column_name')")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                vc = data[col].value_counts()
                total = vc.sum()
                cum = 0
                print(f"  Frequency table: {col}")
                print(f"  {'Value':20s} {'Count':>8s} {'%':>8s} {'Cum%':>8s}")
                print(f"  {'-' * 48}")
                for val, count in vc.items():
                    pct = count / total * 100
                    cum += pct
                    print(f"  {str(val):20s} {count:8d} {pct:7.1f}% {cum:7.1f}%")

            def _pivot(index=None, columns=None, values=None, aggfunc="mean", data=None):
                """Pivot table. Usage: pivot('row_col', 'col_col', 'val_col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if index is None or columns is None or values is None:
                    print("Usage: pivot('row_col', 'col_col', 'value_col')")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                import pandas as _pd

                pt = _pd.pivot_table(data, values=values, index=index, columns=columns, aggfunc=aggfunc)
                ns["pivot_result"] = pt
                print(pt.to_string())
                return pt

            def _groupby(group_col=None, agg_col=None, func="mean", data=None):
                """Group-by aggregation. Usage: groupby('group', 'value', 'mean')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if group_col is None or agg_col is None:
                    print("Usage: groupby('group_col', 'value_col', 'mean')")
                    print(f"  Columns: {', '.join(data.columns[:15])}")
                    return
                result = data.groupby(group_col)[agg_col].agg(func)
                print(f"  {func}({agg_col}) by {group_col}:")
                print(result.to_string())
                return result

            def _sensitivity(rr=None, ci_lower=None):
                """Sensitivity analysis (E-value). Usage: sensitivity(2.5, 1.1)."""
                return _evalue(rr, ci_lower)

            def _profile(data=None):
                """Full dataset profile. Usage: profile()."""
                from moirais.dataset import profile_dataset

                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data.")
                    return
                p = profile_dataset(data)
                print(p.summary_table())
                return p

            def _plan(data=None):
                """Suggest analysis plan. Usage: plan()."""
                from moirais.dataset import profile_dataset, suggest_analysis_plan

                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data.")
                    return
                p = profile_dataset(data)
                suggestions = suggest_analysis_plan(p)
                print("  Suggested analysis plan:")
                for i, s in enumerate(suggestions, 1):
                    print(f"    {i}. {s}")
                return suggestions

            def _selftest():
                """Run MOIRAIS self-test."""
                from moirais.selftest import run_selftest

                results = run_selftest()
                passed = sum(1 for r in results["checks"] if r["passed"])
                total = len(results["checks"])
                print(f"  Self-test: {passed}/{total} passed")
                for r in results["checks"]:
                    status = "OK" if r["passed"] else "FAIL"
                    print(f"    [{status}] {r['name']}")
                return results

            def _doctor():
                """Run environment diagnostics."""
                from moirais.doctor import run_checks

                results = run_checks()
                for check in results["checks"]:
                    status = "OK" if check["passed"] else "FAIL"
                    print(f"  [{status}] {check['name']}: {check.get('message', '')}")
                return results

            def _version():
                """Show MOIRAIS version info."""
                import moirais as _e

                print(f"  MOIRAIS {_e.__version__}")
                print(f"  Package: {Path(_e.__file__).parent}")
                print(f"  Python: {sys.version.split()[0]}")
                import numpy
                import pandas
                import scipy

                print(f"  pandas: {pandas.__version__}")
                print(f"  numpy: {numpy.__version__}")
                print(f"  scipy: {scipy.__version__}")

            # --- Survival analysis helpers ---

            def _kaplan_meier(time_col=None, event_col=None, data=None):
                """Kaplan-Meier curve. Usage: kaplan_meier('time', 'event')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if time_col is None or event_col is None:
                    print("Usage: kaplan_meier('time_col', 'event_col')")
                    return
                from moirais.survival import kaplan_meier_curve

                r = kaplan_meier_curve(data[time_col].values, data[event_col].values)
                print(f"  Kaplan-Meier: events={int(data[event_col].sum())}/{len(data)}")
                print(f"  Median survival: {r.median_survival}")
                ns["km_result"] = r
                return r

            def _cox(time_col=None, event_col=None, covariates=None, data=None):
                """Cox PH model. Usage: cox('time', 'event', ['age','sex'])."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if time_col is None or event_col is None or covariates is None:
                    print("Usage: cox('time_col', 'event_col', ['cov1', 'cov2'])")
                    return
                from moirais.survival import cox_ph

                if isinstance(covariates, str):
                    covariates = [c.strip() for c in covariates.split(",")]
                r = cox_ph(data, time_col, event_col, covariates)
                print(f"  Cox PH: concordance={r.concordance:.4f}")
                for name, hr, p in zip(r.covariate_names, r.hazard_ratios, r.p_values):
                    print(f"    {name}: HR={hr:.4f}, p={p:.4f}")
                ns["cox_result"] = r
                return r

            def _logrank(time_col=None, event_col=None, group_col=None, data=None):
                """Log-rank test. Usage: logrank('time', 'event', 'group')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if time_col is None or event_col is None or group_col is None:
                    print("Usage: logrank('time_col', 'event_col', 'group_col')")
                    return
                from moirais.survival import log_rank_test

                r = log_rank_test(data[time_col].values, data[event_col].values, data[group_col].values)
                print(f"  Log-rank test: chi2={r.test_statistic:.4f}, p={r.p_value:.6f}")
                return r

            # --- DiD / RDD / IV helpers ---

            def _did(outcome=None, treatment=None, post=None, data=None):
                """DiD 2x2. Usage: did('outcome', 'treatment', 'post')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if outcome is None or treatment is None or post is None:
                    print("Usage: did('outcome_col', 'treatment_col', 'post_col')")
                    return
                from moirais.did import did_2x2

                r = did_2x2(data, outcome, treatment, post)
                print(f"  DiD: ATT={r.att:.4f}, SE={r.se:.4f}, p={r.p_value:.6f}")
                print(f"  95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
                return r

            def _rdd(outcome=None, running=None, cutoff=0.0, data=None):
                """Sharp RDD. Usage: rdd('outcome', 'running_var', cutoff=0)."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if outcome is None or running is None:
                    print("Usage: rdd('outcome_col', 'running_var_col', cutoff=0)")
                    return
                from moirais.rdd import sharp_rdd

                r = sharp_rdd(data, outcome, running, cutoff=cutoff)
                print(f"  RDD: LATE={r.estimate:.4f}, SE={r.se:.4f}, p={r.p_value:.6f}")
                print(f"  Bandwidth: {r.bandwidth:.4f}")
                return r

            def _iv_tsls(outcome=None, endogenous=None, instruments=None, data=None):
                """2SLS IV. Usage: iv_tsls('outcome', ['endog'], ['instrument'])."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if outcome is None or endogenous is None or instruments is None:
                    print("Usage: iv_tsls('outcome', ['endogenous'], ['instruments'])")
                    return
                from moirais.iv import tsls

                if isinstance(endogenous, str):
                    endogenous = [endogenous]
                if isinstance(instruments, str):
                    instruments = [instruments]
                r = tsls(data, outcome, endogenous, instruments)
                for name, coef, p in zip(r.covariate_names, r.coefficients, r.p_values):
                    print(f"  {name}: coef={coef:.4f}, p={p:.4f}")
                return r

            # --- Matching / missing helpers ---

            def _match(treatment=None, covariates=None, data=None):
                """PS matching. Usage: match('treatment', ['cov1','cov2'])."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if treatment is None or covariates is None:
                    print("Usage: match('treatment_col', ['cov1', 'cov2'])")
                    return
                from moirais.matching import propensity_score_matching

                if isinstance(covariates, str):
                    covariates = [c.strip() for c in covariates.split(",")]
                r = propensity_score_matching(data, treatment, covariates)
                print(f"  Matched pairs: {r.n_matched}, ATT={r.att:.4f}")
                ns["match_result"] = r
                return r

            def _mcar(data=None):
                """Little's MCAR test. Usage: mcar()."""
                from moirais.missing import littles_mcar_test

                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data.")
                    return
                r = littles_mcar_test(data)
                verdict = "MCAR plausible" if r.p_value > 0.05 else "MCAR rejected"
                print(f"  Little's MCAR: chi2={r.test_statistic:.4f}, p={r.p_value:.6f}")
                print(f"  Conclusion: {verdict}")
                return r

            def _impute(m=5, data=None):
                """MICE imputation. Usage: impute(m=5)."""
                from moirais.missing import mice

                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data.")
                    return
                r = mice(data, m=m)
                print(f"  MICE: {r.m} imputations, {len(r.imputed_columns)} columns imputed")
                ns["mice_result"] = r
                return r

            # --- Diagnostics helpers ---

            def _vif(columns=None, data=None):
                """VIF diagnostics. Usage: vif(['col1','col2','col3'])."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if columns is None:
                    print("Usage: vif(['col1', 'col2', 'col3'])")
                    _nc = [c for c in data.columns if data[c].dtype.kind in "iuf"][:10]
                    if _nc:
                        print(f"  Numeric columns: {', '.join(_nc)}")
                    return

                from moirais.diagnostics import collinearity_diagnostics

                if isinstance(columns, str):
                    columns = [c.strip() for c in columns.split(",")]
                X = data[columns].dropna().values.astype(float)
                r = collinearity_diagnostics(X, column_names=columns)
                for name, v in zip(r.variable_names, r.vif):
                    flag = " (HIGH)" if v > 10 else ""
                    print(f"  {name}: VIF={v:.2f}{flag}")
                return r

            def _odds_ratio(col1=None, col2=None, data=None):
                """Odds ratio from 2x2 table. Usage: odds_ratio('outcome', 'exposure')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col1 is None or col2 is None:
                    print("Usage: odds_ratio('outcome_col', 'exposure_col')")
                    return
                import pandas as _pd

                from moirais.effect_sizes import odds_ratio

                ct = _pd.crosstab(data[col1], data[col2])
                if ct.shape != (2, 2):
                    print(f"  Error: need 2x2 table, got {ct.shape}")
                    return
                a, b, c, d = ct.iloc[0, 0], ct.iloc[0, 1], ct.iloc[1, 0], ct.iloc[1, 1]
                r = odds_ratio(int(a), int(b), int(c), int(d))
                print(f"  OR={r.estimate:.4f}, 95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
                return r

            def _nnt(col1=None, col2=None, data=None):
                """Number needed to treat. Usage: nnt('outcome', 'treatment')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col1 is None or col2 is None:
                    print("Usage: nnt('outcome_col', 'treatment_col')")
                    return
                import pandas as _pd

                from moirais.effect_sizes import number_needed_to_treat

                ct = _pd.crosstab(data[col1], data[col2])
                if ct.shape != (2, 2):
                    print(f"  Error: need 2x2 table, got {ct.shape}")
                    return
                a, b, c, d = ct.iloc[0, 0], ct.iloc[0, 1], ct.iloc[1, 0], ct.iloc[1, 1]
                r = number_needed_to_treat(int(a), int(b), int(c), int(d))
                print(f"  NNT={r.estimate:.2f}, 95% CI: [{r.ci_lower:.2f}, {r.ci_upper:.2f}]")
                return r

            def _rosenbaum(outcome_col=None, treatment_col=None, data=None):
                """Rosenbaum bounds. Usage: rosenbaum('outcome', 'treatment')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if outcome_col is None or treatment_col is None:
                    print("Usage: rosenbaum('outcome_col', 'treatment_col')")
                    return
                from moirais.sensitivity import rosenbaum_bounds

                treated = data.loc[data[treatment_col] == 1, outcome_col].dropna().values
                control = data.loc[data[treatment_col] == 0, outcome_col].dropna().values
                r = rosenbaum_bounds(treated, control)
                for gamma, p_upper in zip(r.gamma_values, r.p_upper):
                    sig = "sig" if p_upper < 0.05 else "n.s."
                    print(f"  Gamma={gamma:.1f}: p_upper={p_upper:.4f} ({sig})")
                return r

            def _table1(group_col=None, data=None):
                """Publication Table 1. Usage: table1('group_col')."""
                from moirais.tables_pub import table1

                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data.")
                    return
                r = table1(data, group_col=group_col)
                print(r.to_string())
                return r

            def _hedges_g(col1=None, col2=None, data=None):
                """Hedges' g. Usage: hedges_g('col1', 'col2')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col1 is None or col2 is None:
                    print("Usage: hedges_g('col1', 'col2')")
                    return
                from moirais.effect_sizes import hedges_g

                r = hedges_g(data[col1].dropna().values, data[col2].dropna().values)
                print(f"  g={r.estimate:.4f}, 95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
                return r

            def _risk_ratio(a=None, b=None, c=None, d=None):
                """Risk ratio from 2x2 table. Usage: risk_ratio(a, b, c, d)."""
                if a is None or b is None or c is None or d is None:
                    print("Usage: risk_ratio(a, b, c, d)")
                    print("  a,b,c,d are cells of a 2x2 table")
                    return
                from moirais.effect_sizes import risk_ratio

                r = risk_ratio(int(a), int(b), int(c), int(d))
                print(f"  RR={r.estimate:.4f}, 95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
                return r

            def _bonferroni(*pvals):
                """Bonferroni correction. Usage: bonferroni(0.01, 0.04, 0.06)."""
                import numpy as _np

                from moirais.multiple_testing import bonferroni

                r = bonferroni(_np.array(pvals))
                for i, (o, a) in enumerate(zip(r.original, r.adjusted)):
                    sig = "reject" if r.reject[i] else "retain"
                    print(f"  p{i + 1}: {o:.4f} -> {a:.4f} ({sig})")
                return r

            def _jackknife(col=None, data=None):
                """Jackknife (mean). Usage: jackknife('col')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None:
                    print("Usage: jackknife('column_name')")
                    return
                import numpy as _np

                from moirais.bootstrap_methods import jackknife

                r = jackknife(data[col].dropna().values, statistic=_np.mean)
                print(f"  Est={r.estimate:.4f}, SE={r.se:.4f}, CI=[{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
                return r

            def _permtest(col=None, group_col=None, data=None):
                """Permutation test. Usage: permtest('col', 'group')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if col is None or group_col is None:
                    print("Usage: permtest('value_col', 'group_col')")
                    return
                from moirais.bootstrap_methods import permutation_test

                groups = data[group_col].dropna().unique()
                g1 = data.loc[data[group_col] == groups[0], col].dropna().values
                g2 = data.loc[data[group_col] == groups[1], col].dropna().values
                r = permutation_test(g1, g2)
                print(f"  Observed={r.observed:.4f}, p={r.p_value:.6f}")
                return r

            def _event_study(outcome=None, unit=None, time=None, treatment_time=None, data=None):
                """Event study. Usage: event_study('y', 'id', 'time', 'treat_time')."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if outcome is None or unit is None or time is None or treatment_time is None:
                    print("Usage: event_study('outcome', 'unit_id', 'time', 'treatment_time')")
                    return
                from moirais.did import event_study

                r = event_study(data, outcome=outcome, unit=unit, time=time, treatment_time=treatment_time)
                for c in r.coefficients:
                    sig = "*" if c.p_value < 0.05 else ""
                    print(f"  t={c.period:+d}: {c.estimate:+.4f} (SE={c.se:.4f}) p={c.p_value:.4f}{sig}")
                return r

            def _fuzzy_rdd(outcome=None, running=None, treatment=None, cutoff=0.0, data=None):
                """Fuzzy RDD. Usage: fuzzy_rdd('y', 'running', 'treat', cutoff=0)."""
                if data is None:
                    data = ns.get("df")
                if data is None:
                    print("No data. Use load('cpads') first.")
                    return
                if outcome is None or running is None or treatment is None:
                    print("Usage: fuzzy_rdd('outcome', 'running_var', 'treatment', cutoff=0)")
                    return
                from moirais.rdd import fuzzy_rdd

                r = fuzzy_rdd(data, outcome=outcome, running=running, treatment=treatment, cutoff=cutoff)
                print(f"  LATE={r.estimate:.4f}, SE={r.se:.4f}, p={r.p_value:.6f}")
                return r

            # Register all helpers
            ns["view"] = _view
            ns["ls"] = _ls
            ns["who"] = _who
            ns["clear"] = _clear
            ns["load"] = _load
            ns["head"] = _head
            ns["tail"] = _tail
            ns["shape"] = _shape
            ns["cols"] = _cols
            ns["describe"] = _describe
            ns["sample"] = _sample
            ns["unique"] = _unique
            ns["value_counts"] = _value_counts
            ns["freq"] = _freq
            ns["missing"] = _missing
            ns["summary"] = _summary
            ns["filter_rows"] = _filter
            ns["select_cols"] = _select
            ns["rename_col"] = _rename
            ns["dropna"] = _dropna
            ns["save"] = _save
            ns["crosstab"] = _crosstab
            ns["pivot"] = _pivot
            ns["groupby"] = _groupby
            ns["modules"] = _modules
            ns["run_module"] = _run_module
            ns["ttest"] = _ttest
            ns["ttest2"] = _ttest2
            ns["paired_t"] = _paired_t
            ns["corr"] = _corr
            ns["spearman"] = _spearman
            ns["anova"] = _anova
            ns["chi2"] = _chi2
            ns["mannwhitney"] = _mannwhitney
            ns["ks_test"] = _ks_test
            ns["bootstrap_ci"] = _bootstrap_ci
            ns["bh"] = _bh
            ns["power"] = _power
            ns["effect_size"] = _effect_size
            ns["evalue"] = _evalue
            ns["sensitivity"] = _sensitivity
            ns["ate"] = _ate
            ns["propensity"] = _propensity
            ns["ipw"] = _ipw
            ns["ebac"] = _ebac
            ns["profile"] = _profile
            ns["plan"] = _plan
            ns["selftest"] = _selftest
            ns["doctor"] = _doctor
            ns["version"] = _version
            ns["help_repl"] = _help_repl
            # Survival
            ns["kaplan_meier"] = _kaplan_meier
            ns["cox"] = _cox
            ns["logrank"] = _logrank
            # DiD / RDD / IV
            ns["did"] = _did
            ns["rdd"] = _rdd
            ns["iv_tsls"] = _iv_tsls
            # Matching / Missing
            ns["match"] = _match
            ns["mcar"] = _mcar
            ns["impute"] = _impute
            # Diagnostics / Effect Sizes / Sensitivity
            ns["vif"] = _vif
            ns["odds_ratio"] = _odds_ratio
            ns["nnt"] = _nnt
            ns["rosenbaum"] = _rosenbaum
            ns["table1"] = _table1
            # New helpers
            ns["hedges_g"] = _hedges_g
            ns["risk_ratio"] = _risk_ratio
            ns["bonferroni"] = _bonferroni
            ns["jackknife"] = _jackknife
            ns["permtest"] = _permtest
            ns["event_study"] = _event_study
            ns["fuzzy_rdd"] = _fuzzy_rdd

            # Inject stat_commands registry (existing helpers take priority)
            try:
                from .stat_commands import ALIAS_MAP, COMMAND_REGISTRY

                for _cmd_name, _cmd in COMMAND_REGISTRY.items():
                    if _cmd_name not in ns:
                        ns[_cmd_name] = _cmd.handler_repl
                for _alias, _canonical in ALIAS_MAP.items():
                    if _alias not in ns and _canonical in COMMAND_REGISTRY:
                        ns[_alias] = COMMAND_REGISTRY[_canonical].handler_repl
            except ImportError:
                pass

        def _start_r_process(self, log: RichLog) -> None:
            """Start a persistent R subprocess for interactive use."""
            try:
                self._r_proc = subprocess.Popen(
                    ["R", "--no-echo", "--no-save", "--no-restore"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0,
                )
                log.write("[green]R: persistent session started[/green]")
            except FileNotFoundError:
                self._r_proc = None
                log.write("[yellow]R: not found — /r mode will use one-shot Rscript[/yellow]")

        def _update_mode_indicator(self, detected: str = "") -> None:
            """Update the mode label below the body."""
            mode_widget = self.query_one("#repl-mode", Static)
            shell_name = Path(self._user_shell).name
            colors = {"python": "green", "r": "blue", "shell": "yellow"}
            auto_tag = " [dim](auto)[/dim]" if self._auto_detect else ""
            det_tag = f" [dim]detected: {detected}[/dim]" if detected and detected != self._lang else ""
            poly_tag = " [bold magenta][polyglot][/bold magenta]" if self._polyglot else ""
            mode_widget.update(
                f"[bold {colors[self._lang]}]{self._lang.upper()}[/bold {colors[self._lang]}]"
                f"{auto_tag}{det_tag}{poly_tag}"
                f" | [dim]{len(self._history)} cmds | {shell_name} | Enter to run[/dim]"
            )

        def action_history_back(self) -> None:
            if not self._history:
                return
            editor = self.query_one("#repl-editor", TextArea)
            if self._history_idx == -1:
                self._history_idx = len(self._history) - 1
            elif self._history_idx > 0:
                self._history_idx -= 1
            editor.load_text(self._history[self._history_idx])

        def action_history_forward(self) -> None:
            editor = self.query_one("#repl-editor", TextArea)
            if self._history_idx == -1:
                return
            if self._history_idx < len(self._history) - 1:
                self._history_idx += 1
                editor.load_text(self._history[self._history_idx])
            else:
                self._history_idx = -1
                editor.load_text("")

        def action_copy_selection(self) -> None:
            """Copy TextArea selection to clipboard, or fall back to log."""
            editor = self.query_one("#repl-editor", TextArea)
            selected = editor.selected_text
            if selected:
                self.app._copy_text(selected)
                self.app.notify("Copied selection", timeout=2)
            else:
                self.app.action_copy_all()

        def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
            """Insert clicked helper text into the editor."""
            label = str(event.node.label)
            # Only insert leaf nodes (actual commands), not category headers
            if event.node.is_root or event.node.children:
                return
            editor = self.query_one("#repl-editor", TextArea)
            editor.insert(label)
            editor.focus()

        def action_submit_editor(self) -> None:
            """Execute the contents of the TextArea editor."""
            editor = self.query_one("#repl-editor", TextArea)
            user_code = editor.text.strip()
            if not user_code:
                return
            editor.load_text("")
            editor.focus()
            self._history_idx = -1
            log = self.query_one("#repl-log", RichLog)

            # LLM chat via ? prefix (with optional ?XX model alias)
            if user_code.startswith("?"):
                rest = user_code[1:].strip()
                if not rest:
                    log.write("[dim]Usage: ?prompt  or  ?<alias> prompt  (e.g. ?dq what is DML)[/dim]")
                    return
                # Check if first 2 chars are a model alias
                inline_model = None
                from .llm import _build_alias_table

                alias_table = _build_alias_table()
                if len(rest) >= 3 and rest[:2] in alias_table and rest[2] == " ":
                    inline_model = alias_table[rest[:2]]
                    query = rest[3:].strip()
                else:
                    query = rest
                if not query:
                    log.write("[dim]Usage: ?<alias> your question[/dim]")
                    return
                model_to_use = inline_model or self._current_model
                if inline_model:
                    # Show which model we're using for this query
                    from .llm import list_freeapi_models

                    mlabel = inline_model
                    for m in list_freeapi_models():
                        if m["model"] == inline_model:
                            mlabel = m["label"]
                            break
                    log.write(f"[bold magenta]you>[/bold magenta] {query} [dim]({mlabel})[/dim]")
                else:
                    log.write(f"[bold magenta]you>[/bold magenta] {query}")
                from .llm import detect_provider_and_model, pick_thinking_word

                _mod = model_to_use or self._current_model or detect_provider_and_model()[1]
                _tw = pick_thinking_word(query)
                log.write(f"[dim]\\[{_mod}] {_tw}.....[/dim]")
                self._history.append(user_code)
                self.run_worker(
                    self._ask_llm(query, model=model_to_use),
                    name="repl-llm",
                    exclusive=True,
                )
                return

            # Slash commands for mode switching
            if user_code == "/python":
                self._lang = "python"
                self._auto_detect = False
                log.write("[cyan]Locked to Python mode[/cyan]")
                self._update_mode_indicator()
                return
            if user_code == "/r":
                self._lang = "r"
                self._auto_detect = False
                log.write("[cyan]Locked to R mode[/cyan]")
                self._update_mode_indicator()
                return
            if user_code == "/shell":
                shell_name = Path(self._user_shell).name
                self._lang = "shell"
                self._auto_detect = False
                log.write(f"[cyan]Locked to Shell mode ({shell_name})[/cyan]")
                self._update_mode_indicator()
                return
            if user_code == "/auto":
                self._auto_detect = True
                log.write("[cyan]Auto-detect mode enabled — language detected per command[/cyan]")
                self._update_mode_indicator()
                return
            if user_code == "/reset":
                self._reset_console(log)
                return
            if user_code == "/history":
                if self._history:
                    for i, h in enumerate(self._history[-20:], 1):
                        log.write(f"  {i:3d}  {h}")
                else:
                    log.write("[dim]No history yet[/dim]")
                return
            if user_code in ("/list", "/datasets"):
                try:
                    from moirais.data import list_datasets

                    datasets = list_datasets()
                    log.write("[bold cyan]Available datasets:[/bold cyan]")
                    for d in datasets:
                        tag = f" ({d['rows']} rows)" if d.get("rows") else ""
                        log.write(f"  load('{d['key']}'){tag} — {d['name']}")
                except Exception as e:
                    log.write(f"[red]Error listing datasets: {e}[/red]")
                return
            if user_code in ("/run", "/modules"):
                try:
                    from moirais.modules import list_modules

                    mods = list_modules()
                    log.write(f"[bold cyan]{len(mods)} analysis modules:[/bold cyan]")
                    for m in mods:
                        log.write(f"  {m['name']}: {m['description']}")
                    log.write("[dim]Use Pipeline screen (p from home) to run modules.[/dim]")
                except Exception as e:
                    log.write(f"[red]Error: {e}[/red]")
                return
            if user_code == "/help":
                log.write("[bold cyan]MOIRAIS REPL — Commands:[/bold cyan]")
                log.write("[bold]Mode:[/bold] /python /r /shell /auto /polyglot")
                log.write("[bold]Info:[/bold] /list /run /help /history /reset")
                log.write("[bold]Data:[/bold] load('cpads'), head(), cols(), summary(), describe()")
                log.write("[bold]Stats:[/bold] ttest('col'), chi2('c1','c2'), corr('c1','c2')")
                log.write("[bold]Wrangling:[/bold] filter_rows(), select_cols(), groupby()")
                log.write("[bold]Causal:[/bold] ate(), propensity(), ipw(), evalue(rr)")
                log.write("[bold]System:[/bold] who(), ls(), clear(), version(), modules()")
                log.write("[bold]Prefix:[/bold] ? for LLM, ! for shell, R> for R")
                log.write("[dim]help_repl() for full list of 1249 commands[/dim]")
                return
            if user_code == "/polyglot":
                self._polyglot = not self._polyglot
                state = "ON" if self._polyglot else "OFF"
                log.write(f"[cyan]Polyglot mode {state}[/cyan]")
                if self._polyglot:
                    log.write("[dim]Variables auto-bridge across Python, R, and Shell after each command[/dim]")
                    log.write("[dim]Supports: scalars, vectors, strings, booleans, DataFrames.[/dim]")
                else:
                    log.write("[dim]P↔R↔Shell variable bridge disabled[/dim]")
                self._update_mode_indicator()
                return
            if user_code == "/models":
                from .llm import detect_provider_and_model, list_freeapi_models

                _, current_label = detect_provider_and_model()
                current = self._current_model or ""
                log.write("[bold cyan]Available models[/bold cyan]")
                # Show local Ollama models first
                try:
                    from .loc import LocalOllama

                    client = LocalOllama()
                    if client.is_running():
                        local_models = client.list_models()
                        if local_models:
                            log.write("[dim]  Local (Ollama)[/dim]")
                            for m in local_models:
                                marker = " ◀" if m.name == current or (not current and m.name == client.model) else ""
                                log.write(f"    {m.label:22s} ({m.name}) {m.quantization}{marker}")
                except Exception:
                    pass
                # Show FreeAPI models
                freeapi_models = list_freeapi_models()
                if freeapi_models:
                    log.write("[dim]  Remote (FreeAPI, no key)[/dim]")
                    log.write("[dim]  Alias  Model                  Size[/dim]")
                    for m in freeapi_models:
                        marker = " ◀" if m["model"] == current else ""
                        log.write(f"    [bold]{m['alias']:4s}[/bold]   {m['label']:22s} ({m['model']}){marker}")
                log.write("")
                log.write("[dim]Switch: /model <alias>  |  Inline: ?<alias> <prompt>[/dim]")
                return
            if user_code.startswith("/model "):
                alias = user_code[7:].strip().lower()
                from .llm import _build_alias_table, list_freeapi_models

                table = _build_alias_table()
                if alias in table:
                    self._current_model = table[alias]
                    label = alias
                    for m in list_freeapi_models():
                        if m["model"] == self._current_model:
                            label = m["label"]
                            break
                    log.write(f"[bold green]Model: {label}[/bold green]")
                else:
                    # Try exact model name match
                    for m in list_freeapi_models():
                        if m["model"] == alias:
                            self._current_model = alias
                            log.write(f"[bold green]Model: {m['label']}[/bold green]")
                            return
                    log.write(f"[yellow]Unknown model alias: {alias}[/yellow]")
                    log.write("[dim]Type /models to see available models[/dim]")
                return
            # Catch-all for unknown slash commands
            if user_code.startswith("/"):
                log.write(f"[yellow]Unknown command: {user_code}[/yellow]")
                log.write("[dim]Type /help for available commands[/dim]")
                return

            self._history.append(user_code)

            # Auto-detect or use locked mode
            if self._auto_detect:
                detected = self._detect_language(user_code)
            else:
                detected = self._lang

            # Handle ! prefix for explicit shell
            actual_code = user_code
            if user_code.startswith("!"):
                detected = "shell"
                actual_code = user_code[1:].strip()

            # Handle R> prefix for explicit R
            if user_code.startswith("R>") or user_code.startswith("r>"):
                detected = "r"
                actual_code = user_code[2:].strip()

            # For multiline code, execute all lines together
            if detected == "python":
                self._eval_python(log, actual_code)
                if self._polyglot and self._r_proc and self._r_proc.poll() is None:
                    self._bridge_python_to_r(log, actual_code)
            elif detected == "r":
                self._eval_r(log, actual_code)
            else:
                self._eval_shell(log, actual_code)
                if self._polyglot:
                    self._bridge_shell_to_python(log, actual_code)

            self._update_mode_indicator(detected)

        def _reset_console(self, log: RichLog) -> None:
            import code as code_module

            self._py_console_ns = {"__name__": "__console__", "__builtins__": __builtins__}
            self._py_console = code_module.InteractiveConsole(self._py_console_ns)
            self._inject_repl_helpers()
            self._py_console.runsource("import moirais; import pandas as pd; import numpy as np")
            self._py_console.runsource("from moirais import *")
            if self._r_proc:
                self._r_proc.terminate()
                self._start_r_process(log)
            log.write("[cyan]Console reset — all modes reinitialized[/cyan]")
            self._update_mode_indicator()

        async def _ask_llm(self, query: str, model: str | None = None) -> None:
            """Send a question to the LLM and stream the response."""
            log = self.query_one("#repl-log", RichLog)
            _sentinel = object()
            try:
                from .llm import ask, build_moirais_context, detect_provider_and_model

                provider, _unused = await asyncio.to_thread(detect_provider_and_model)
                ctx = await asyncio.to_thread(build_moirais_context)
                stream_iter = await asyncio.to_thread(
                    ask,
                    query,
                    ctx,
                    stream=True,
                    provider=provider,
                    model=model or self._current_model,
                )
                if isinstance(stream_iter, str):
                    log.write("[bold cyan]moirais>[/bold cyan]")
                    for line in stream_iter.splitlines():
                        log.write(f"  {line}")
                else:
                    log.write("[bold cyan]moirais>[/bold cyan]")
                    buf = ""
                    while True:
                        chunk = await asyncio.to_thread(next, stream_iter, _sentinel)
                        if chunk is _sentinel:
                            break
                        buf += chunk
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            if line.strip():
                                log.write(f"  {line}")
                    if buf.strip():
                        log.write(f"  {buf.strip()}")
            except Exception as exc:
                log.write(f"[red]LLM error: {exc}[/red]")

        def _eval_python(self, log: RichLog, code: str) -> None:
            """Execute Python — uses exec() for multiline, push() for single."""
            first_line = code.split("\n", 1)[0]
            log.write(f"[bold green][P][/bold green] {first_line}")
            if "\n" in code:
                for extra in code.split("\n")[1:]:
                    log.write(f"[bold green][P][/bold green] {extra}")
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                if "\n" in code:
                    # Multiline: compile as 'exec' block
                    compiled = compile(code, "<editor>", "exec")
                    exec(compiled, self._py_console_ns)
                else:
                    # Single line: use InteractiveConsole for expression display
                    self._py_console.push(code)
                sys.stdout = old_stdout
                sys.stderr = old_stderr

                out = stdout_capture.getvalue()
                err = stderr_capture.getvalue()

                if out.strip():
                    for line in out.strip().splitlines()[:50]:
                        log.write(line)
                if err.strip():
                    for line in err.strip().splitlines():
                        stripped = line.strip()
                        if stripped.startswith("Traceback (most recent"):
                            continue
                        if stripped.startswith('File "<'):
                            continue
                        if stripped == "":
                            continue
                        log.write(f"[red]{line}[/red]")
            except Exception:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                tb = traceback.format_exc()
                tb_lines = tb.strip().splitlines()
                if tb_lines:
                    log.write(f"[red]{tb_lines[-1]}[/red]")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        def _eval_r(self, log: RichLog, code: str) -> None:
            """Execute R code — persistent subprocess if available, fallback to one-shot."""
            log.write(f"[bold blue][R][/bold blue] {code}")

            if self._r_proc and self._r_proc.poll() is None:
                # Persistent R session: write code, read output
                try:
                    # Use a sentinel to know when output ends
                    sentinel = f"__MOIRAIS_DONE_{id(code)}__"
                    full_cmd = f"{code}\ncat('{sentinel}\\n')\n"
                    self._r_proc.stdin.write(full_cmd)
                    self._r_proc.stdin.flush()

                    lines = []
                    while True:
                        raw_line = self._r_proc.stdout.readline()
                        if not raw_line:
                            break  # Process died
                        if sentinel in raw_line:
                            break
                        lines.append(raw_line.rstrip("\n"))

                    for line in lines:
                        if line:  # skip blank lines from R
                            log.write(line)
                except Exception as e:
                    log.write(f"[red]R session error: {e}[/red]")
                    log.write("[yellow]Restarting R...[/yellow]")
                    self._r_proc.terminate()
                    self._start_r_process(log)
            else:
                # Fallback: one-shot Rscript
                try:
                    result = subprocess.run(
                        ["Rscript", "-e", code],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.stdout.strip():
                        for line in result.stdout.strip().splitlines():
                            log.write(line)
                    if result.stderr.strip():
                        for line in result.stderr.strip().splitlines():
                            log.write(f"[yellow]{line}[/yellow]")
                except FileNotFoundError:
                    log.write("[red]R not found. Install R: https://cran.r-project.org[/red]")
                except subprocess.TimeoutExpired:
                    log.write("[red]R command timed out (30s limit)[/red]")
                except Exception as e:
                    log.write(f"[red]Error: {e}[/red]")

            # Polyglot: bridge R variables to Python
            if self._polyglot and self._r_proc and self._r_proc.poll() is None:
                self._bridge_r_to_python(log, code)

        def _bridge_r_to_python(self, log: RichLog, code: str) -> None:
            """Extract R assignments and inject into Python namespace."""
            import re as _re

            assignments = _re.findall(r"(\w+)\s*<-", code)
            if not assignments:
                return
            for var_name in assignments:
                try:
                    sentinel2 = f"__MOIRAIS_POLY_{var_name}__"
                    extract_cmd = (
                        f"tryCatch({{"
                        f"  .v <- {var_name};"
                        f"  cat('{sentinel2}\\n');"
                        f"  if(is.data.frame(.v) || is.matrix(.v)) {{"
                        "    .csv <- paste(capture.output("
                        "write.csv(as.data.frame(.v), '', row.names=FALSE)), collapse='\\x01');"
                        f"    cat(paste0('DF:', .csv))"
                        f"  }} else if(is.numeric(.v) && length(.v)==1) cat(as.character(.v))"
                        f"  else if(is.character(.v) && length(.v)==1) cat(paste0('S:', .v))"
                        f"  else if(is.logical(.v) && length(.v)==1) cat(paste0('L:', .v))"
                        f"  else if(is.numeric(.v)) cat(paste0('V:', paste(.v, collapse=',')))"
                        f"  else if(is.character(.v)) cat(paste0('SV:', paste(.v, collapse='\\x01')))"
                        f"  else cat('__COMPLEX__');"
                        f"  cat('{sentinel2}_END\\n')"
                        f"}}, error=function(e) cat('{sentinel2}\\n__ERR__\\n{sentinel2}_END\\n'))\n"
                    )
                    self._r_proc.stdin.write(extract_cmd)
                    self._r_proc.stdin.flush()

                    capturing = False
                    val_lines: list[str] = []
                    while True:
                        raw_line = self._r_proc.stdout.readline()
                        if not raw_line:  # actual EOF — process died
                            break
                        line = raw_line.rstrip("\n")
                        if f"{sentinel2}_END" in line:
                            break
                        if sentinel2 in line:
                            capturing = True
                            continue
                        if capturing:
                            val_lines.append(line)

                    val_str = "\n".join(val_lines).strip()
                    if not val_str or val_str in ("__COMPLEX__", "__ERR__"):
                        continue

                    if val_str.startswith("DF:"):
                        import pandas as pd

                        csv_text = val_str[3:].replace("\x01", "\n")
                        self._py_console_ns[var_name] = pd.read_csv(io.StringIO(csv_text))
                    elif val_str.startswith("SV:"):
                        self._py_console_ns[var_name] = val_str[3:].split("\x01")
                    elif val_str.startswith("V:"):
                        import numpy as np

                        self._py_console_ns[var_name] = np.array([float(x) for x in val_str[2:].split(",")])
                    elif val_str.startswith("S:"):
                        self._py_console_ns[var_name] = val_str[2:]
                    elif val_str.startswith("L:"):
                        self._py_console_ns[var_name] = val_str[2:].upper() == "TRUE"
                    else:
                        try:
                            self._py_console_ns[var_name] = float(val_str)
                        except ValueError:
                            self._py_console_ns[var_name] = val_str
                    log.write(f"[dim][polyglot] {var_name} → Python[/dim]")
                except Exception:
                    pass  # Best-effort — don't break the REPL

        def _bridge_python_to_r(self, log: RichLog, code: str) -> None:
            """Extract Python assignments and inject into R session."""
            import re as _re

            # Match simple assignments: x = ..., but not ==, !=, <=, >=, +=, -=, etc.
            assignments = _re.findall(r"^(\w+)\s*(?<![!=<>+\-*/])=(?!=)", code, _re.MULTILINE)
            if not assignments:
                return
            for var_name in assignments:
                if var_name.startswith("_"):
                    continue
                val = self._py_console_ns.get(var_name)
                if val is None and var_name not in self._py_console_ns:
                    continue
                try:
                    r_cmd = None
                    if isinstance(val, bool):
                        r_cmd = f"{var_name} <- {'TRUE' if val else 'FALSE'}"
                    elif isinstance(val, (int, float)):
                        r_cmd = f"{var_name} <- {val}"
                    elif isinstance(val, str):
                        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
                        r_cmd = f'{var_name} <- "{escaped}"'
                    elif isinstance(val, (list, tuple)):
                        if all(isinstance(v, (int, float)) for v in val):
                            r_cmd = f"{var_name} <- c({','.join(str(v) for v in val)})"
                        elif all(isinstance(v, str) for v in val):
                            items = ",".join(f'"{v}"' for v in val)
                            r_cmd = f"{var_name} <- c({items})"
                    else:
                        # numpy array
                        try:
                            import numpy as np

                            if isinstance(val, np.ndarray) and val.ndim == 1:
                                r_cmd = f"{var_name} <- c({','.join(str(v) for v in val)})"
                        except ImportError:
                            pass
                        # pandas DataFrame
                        if r_cmd is None:
                            try:
                                import pandas as pd

                                if isinstance(val, pd.DataFrame):
                                    csv_str = val.to_csv(index=False)
                                    csv_escaped = csv_str.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                                    r_cmd = (
                                        f'{var_name} <- read.csv('
                                        f'textConnection("{csv_escaped}"), stringsAsFactors=FALSE)'
                                    )
                            except ImportError:
                                pass

                    if r_cmd is None:
                        continue

                    sentinel3 = f"__MOIRAIS_PY2R_{var_name}__"
                    full_cmd = f"{r_cmd}\ncat('{sentinel3}_END\\n')\n"
                    self._r_proc.stdin.write(full_cmd)
                    self._r_proc.stdin.flush()

                    while True:
                        raw_line = self._r_proc.stdout.readline()
                        if not raw_line or f"{sentinel3}_END" in raw_line:
                            break

                    log.write(f"[dim][polyglot] {var_name} → R[/dim]")
                except Exception:
                    pass

        def _bridge_shell_to_python(self, log: RichLog, cmd: str) -> None:
            """Extract shell variable assignments and inject into Python (and R)."""
            import re as _re

            patterns = _re.findall(r'(?:export\s+)?(\w+)=(["\']?)(.+?)\2(?:\s|$)', cmd)
            for var_name, _, val in patterns:
                if var_name.startswith("_"):
                    continue
                try:
                    try:
                        self._py_console_ns[var_name] = int(val)
                    except ValueError:
                        try:
                            self._py_console_ns[var_name] = float(val)
                        except ValueError:
                            self._py_console_ns[var_name] = val
                    log.write(f"[dim][polyglot] {var_name} → Python[/dim]")
                    # Also bridge to R if available
                    if self._r_proc and self._r_proc.poll() is None:
                        py_val = self._py_console_ns[var_name]
                        if isinstance(py_val, (int, float)):
                            r_cmd = f"{var_name} <- {py_val}"
                        else:
                            escaped = str(py_val).replace('"', '\\"')
                            r_cmd = f'{var_name} <- "{escaped}"'
                        sentinel = f"__MOIRAIS_SH2R_{var_name}__"
                        self._r_proc.stdin.write(f"{r_cmd}\ncat('{sentinel}_END\\n')\n")
                        self._r_proc.stdin.flush()
                        while True:
                            raw = self._r_proc.stdout.readline()
                            if not raw or f"{sentinel}_END" in raw:
                                break
                        log.write(f"[dim][polyglot] {var_name} → R[/dim]")
                except Exception:
                    pass

        def _eval_shell(self, log: RichLog, cmd: str) -> None:
            """Execute shell command via user's actual shell (zsh/bash)."""
            shell_name = Path(self._user_shell).name
            _shell_tags = {"zsh": "Z", "bash": "B", "sh": "S"}
            tag = _shell_tags.get(shell_name, "SH")
            log.write(f"[bold yellow][{tag}][/bold yellow] {cmd}")
            try:
                # Polyglot: inject Python vars as env vars for shell access
                env = dict(os.environ)
                if self._polyglot:
                    for k, v in self._py_console_ns.items():
                        if isinstance(v, (int, float, str, bool)) and not k.startswith("_"):
                            env[k] = str(v)
                result = subprocess.run(
                    [self._user_shell, "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(Path.cwd()),
                    env=env,
                )
                if result.stdout.strip():
                    for line in result.stdout.strip().splitlines()[:100]:
                        log.write(line)
                if result.stderr.strip():
                    for line in result.stderr.strip().splitlines()[:20]:
                        log.write(f"[yellow]{line}[/yellow]")
                if result.returncode != 0:
                    log.write(f"[red]Exit code: {result.returncode}[/red]")
            except subprocess.TimeoutExpired:
                log.write("[red]Command timed out (30s limit)[/red]")
            except Exception as e:
                log.write(f"[red]Error: {e}[/red]")

        def on_unmount(self) -> None:
            """Clean up persistent R process."""
            if self._r_proc and self._r_proc.poll() is None:
                self._r_proc.terminate()

    # ==================================================================
    # StatScreen — run statistical analyses (48 commands)
    # ==================================================================

    class StatScreen(Screen):
        BINDINGS = [
            Binding("escape", "app.pop_screen", "Back"),
        ]

        _last_result = None  # Stores last analysis output (str or DataFrame)
        _last_result_text: str = ""  # Plain-text version for save

        DEFAULT_CSS = """
        StatScreen { layout: vertical; }
        #analysis-log { height: 1fr; border: solid $success 50%; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            yield CopyableRichLog(id="analysis-log", highlight=True, markup=True, wrap=True)
            yield Input(
                placeholder="Analysis command (try 'help', Tab to complete)...",
                id="analysis-input",
                suggester=_ANALYSIS_SUGGESTIONS,
            )
            yield Footer()

        def _show_help(self, log: RichLog) -> None:
            """Display help text for analysis commands."""
            log.write("\n[bold cyan]MOIRAIS Analysis Console — Commands:[/bold cyan]")
            log.write("")
            log.write("[bold]Data:[/bold]")
            log.write("  [bold]describe[/bold] <csv>                — Descriptive statistics")
            log.write("  [bold]profile[/bold] <csv>                 — Full dataset profile")
            log.write("  [bold]missing[/bold] <csv>                 — Missing data analysis")
            log.write("  [bold]head[/bold] <csv> [n]                — First n rows (default 10)")
            log.write("  [bold]columns[/bold] <csv>                 — List columns and types")
            log.write("")
            log.write("[bold]Statistical Tests:[/bold]")
            log.write("  [bold]ttest[/bold] <csv> <col> [mu]        — One-sample t-test")
            log.write("  [bold]ttest2[/bold] <csv> <col> <group>    — Two-sample t-test (Welch)")
            log.write("  [bold]paired[/bold] <csv> <col1> <col2>    — Paired t-test")
            log.write("  [bold]anova[/bold] <csv> <col> <group>     — One-way ANOVA")
            log.write("  [bold]chi2[/bold] <csv> <col1> <col2>      — Chi-square independence")
            log.write("  [bold]fisher[/bold] <csv> <col1> <col2>    — Fisher exact test (2x2)")
            log.write("  [bold]corr[/bold] <csv> <col1> <col2>      — Pearson correlation")
            log.write("  [bold]ks[/bold] <csv> <col>                — Kolmogorov-Smirnov normality")
            log.write("  [bold]shapiro[/bold] <csv> <col>           — Shapiro-Wilk normality")
            log.write("  [bold]wilcoxon[/bold] <csv> <col1> <col2>  — Wilcoxon signed-rank test")
            log.write("  [bold]levene[/bold] <csv> <col> <group>    — Levene variance equality test")
            log.write("  [bold]normality[/bold] <csv> <col>         — Normality battery")
            log.write("  [bold]mannwhitney[/bold] <csv> <col> <grp> — Mann-Whitney U test")
            log.write("")
            log.write("[bold]Causal Inference:[/bold]")
            log.write("  [bold]propensity[/bold] <csv> <trt> <covs> — Propensity scores")
            log.write("  [bold]ate[/bold] <csv> <outcome> <trt>     — Average treatment effect")
            log.write("  [bold]ipw[/bold] <csv> <outcome> <trt>     — IPW-weighted estimate")
            log.write("  [bold]aipw[/bold] <csv> <outcome> <trt> <covs> — Doubly-robust AIPW")
            log.write("  [bold]att[/bold] <csv> <outcome> <trt> <covs>  — ATT via Hajek IPW")
            log.write("  [bold]atc[/bold] <csv> <outcome> <trt> <covs>  — ATC via Hajek IPW")
            log.write("  [bold]cate[/bold] <csv> <outcome> <trt> <covs> — Conditional treatment effects")
            log.write("  [bold]gate[/bold] <csv> <outcome> <trt> <group> <covs> — Group treatment effects")
            log.write("  [bold]late[/bold] <csv> <outcome> <trt> <iv> [covs] — Local average treatment effect")
            log.write("  [bold]irm[/bold] <csv> <outcome> <trt> <covs> — DoubleML interactive regression")
            log.write("  [bold]dml[/bold] <csv> <outcome> <trt> <covs> — DoubleML PLR (RF)")
            log.write("  [bold]plr[/bold] <csv> <outcome> <trt> <covs> — PLR via DoubleML")
            log.write("  [bold]pliv[/bold] <csv> <outcome> <trt> <iv> <covs> — PLIV via DoubleML")
            log.write("")
            log.write("[bold]Effect Sizes & Sensitivity:[/bold]")
            log.write("  [bold]evalue[/bold] <rr> [ci_lower]        — E-value for confounding")
            log.write("  [bold]cohens_d[/bold] <csv> <col> <group>  — Cohen's d effect size")
            log.write("  [bold]power[/bold] <d> <n> [alpha]          — Power analysis")
            log.write("")
            log.write("[bold]Resampling:[/bold]")
            log.write("  [bold]bootstrap[/bold] <csv> <col>          — Bootstrap CI for mean")
            log.write("  [bold]bh[/bold] <p1,p2,...>                 — Benjamini-Hochberg correction")
            log.write("")
            log.write("[bold]Survival Analysis:[/bold]")
            log.write("  [bold]kaplan_meier[/bold] <csv> <time> <event> — Kaplan-Meier survival curve")
            log.write("  [bold]cox[/bold] <csv> <time> <event> <covs>  — Cox proportional hazards")
            log.write("  [bold]logrank[/bold] <csv> <time> <event> <grp>— Log-rank test")
            log.write("")
            log.write("[bold]Causal Inference (DiD, RDD, IV):[/bold]")
            log.write("  [bold]did[/bold] <csv> <outcome> <treat> <post>— Difference-in-differences")
            log.write("  [bold]rdd[/bold] <csv> <outcome> <running> [c] — Sharp RDD")
            log.write("  [bold]tsls[/bold] <csv> <outcome> <endog> <iv> — Two-stage least squares")
            log.write("")
            log.write("[bold]Matching & Missing Data:[/bold]")
            log.write("  [bold]match[/bold] <csv> <treat> <covs>       — Propensity score matching")
            log.write("  [bold]ps_nn[/bold] <csv> <treat> <outcome> <covs> — PS nearest-neighbour matching")
            log.write("  [bold]ps_subclass[/bold] <csv> <treat> <outcome> <covs> — PS subclassification")
            log.write("  [bold]balance[/bold] <csv> <treat> <covs> [weights] — Balance diagnostics")
            log.write("  [bold]overlap[/bold] <csv> <treat> <covs>     — Positivity / overlap diagnostics")
            log.write("  [bold]mcar[/bold] <csv>                       — Little's MCAR test")
            log.write("  [bold]impute[/bold] <csv> [m]                 — MICE imputation")
            log.write("")
            log.write("[bold]Diagnostics & Sensitivity:[/bold]")
            log.write("  [bold]vif[/bold] <csv> <cols>                 — VIF collinearity diagnostics")
            log.write("  [bold]rosenbaum[/bold] <csv> <outcome> <treat>— Rosenbaum sensitivity bounds")
            log.write("  [bold]odds_ratio[/bold] <csv> <col1> <col2>   — Odds ratio (2x2)")
            log.write("  [bold]nnt[/bold] <csv> <outcome> <treat>      — Number needed to treat")
            log.write("  [bold]table1[/bold] <csv> <group>             — Publication Table 1")
            log.write("  [bold]residuals[/bold] <csv> <y> <x1> [x2..] — Residual diagnostics")
            log.write("  [bold]cooks[/bold] <csv> <y> <x1> [x2..]     — Cook's D / influence diagnostics")
            log.write("  [bold]ess[/bold] <csv> <weight_col>           — Effective sample size")
            log.write("  [bold]design_effect[/bold] <csv> <weight_col> — Weight design effect")
            log.write("")
            log.write("[bold]Advanced Methods:[/bold]")
            log.write("  [bold]hedges_g[/bold] <csv> <col1> <col2>    — Hedges' g effect size")
            log.write("  [bold]risk_ratio[/bold] <a> <b> <c> <d>      — Risk ratio (2x2 table)")
            log.write("  [bold]bonferroni[/bold] <p1> <p2> ...        — Bonferroni correction")
            log.write("  [bold]jackknife[/bold] <csv> <col>           — Jackknife estimation (mean)")
            log.write("  [bold]permtest[/bold] <csv> <col> <group>    — Permutation test")
            log.write("  [bold]event_study[/bold] <csv> <y> <unit> <t> [treat_t] — Event study (DiD)")
            log.write("  [bold]fuzzy_rdd[/bold] <csv> <y> <run> <treat> [cut] — Fuzzy RDD")
            log.write("  [bold]strobe[/bold] <report.txt>             — STROBE compliance check")
            log.write("")
            log.write("[bold]Psychometrics (moirais.psymet):[/bold]")
            log.write("  [bold]crba[/bold] <csv>                       — Cronbach's alpha (all numeric cols)")
            log.write("  [bold]mcdo[/bold] <csv> [nf]                  — McDonald's omega (default nf=1)")
            log.write("  [bold]kmo[/bold] <csv>                        — Kaiser-Meyer-Olkin MSA")
            log.write("  [bold]bart[/bold] <csv>                       — Bartlett's sphericity test")
            log.write("  [bold]paran[/bold] <csv>                      — Parallel analysis (# factors)")
            log.write("  [bold]splhf[/bold] <csv>                      — Split-half reliability")
            log.write("  [bold]itcor[/bold] <csv>                      — Item-total correlations")
            log.write("  [bold]adel[/bold] <csv>                       — Alpha if item deleted")
            log.write("  [bold]idisc[/bold] <csv>                      — Item discrimination index")
            log.write("  [bold]crel[/bold] <l1,l2,...>                  — Composite reliability from loadings")
            log.write("  [bold]ave[/bold] <l1,l2,...>                   — AVE from loadings")
            log.write("")
            log.write("[bold]Correctional (moirais.otis):[/bold]")
            log.write("  [bold]rplace[/bold] <csv> <year> [sex]        — Regional placement analysis")
            log.write("  [bold]astcmb[/bold] <csv>                     — Alert-state combo encoding")
            log.write("  [bold]volat[/bold] <csv>                      — Regional volatility metric")
            log.write("  [bold]rctrnd[/bold] <csv>                     — Restrictive confinement trends")
            log.write("  [bold]otdesc[/bold] <csv>                     — OTIS descriptive statistics")
            log.write("  [bold]otdml[/bold] <csv> <outcome> <treat>    — DML IRM (ATE/ATT)")
            log.write("")
            log.write("[bold]Pipeline:[/bold]")
            log.write("  [bold]modules[/bold]                        — List analysis modules")
            log.write("  [bold]run[/bold] <module-name>              — Run an analysis module")
            log.write("")
            log.write("[bold]Export:[/bold]")
            log.write("  [bold]save[/bold] <path>                    — Save last result as text")
            log.write("  [bold]export[/bold] <path> [csv|md|latex|html] — Export last result")
            log.write("")
            log.write("  [bold]help[/bold]                           — Show this help")
            log.write("")

        def on_mount(self) -> None:
            log = self.query_one("#analysis-log", RichLog)
            log.can_focus = False
            self._show_help(log)
            self.query_one("#analysis-input", Input).focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            cmd = event.value.strip()
            if not cmd:
                return
            event.input.value = ""
            event.input.focus()
            log = self.query_one("#analysis-log", RichLog)

            self.run_worker(
                self._run_analysis(cmd),
                name="analysis",
                exclusive=True,
            )

        def _store(self, text: str, df=None) -> None:
            """Store last result for save/export commands."""
            self._last_result_text = text
            self._last_result = df if df is not None else text

        async def _run_analysis(self, cmd: str) -> None:
            log = self.query_one("#analysis-log", RichLog)
            parts = cmd.split()
            action = parts[0].lower()

            # Capture output lines for save/export.
            _capture: list[str] = []
            _orig_write = log.write

            def _capturing_write(text="", **kw):
                _capture.append(str(text))
                return _orig_write(text, **kw)

            if action not in ("help", "save", "export"):
                log.write(f"[dim]Running {action}...[/dim]")
                log.write = _capturing_write  # type: ignore[assignment]

            try:
                if action == "help":
                    self._show_help(log)
                    return

                elif action == "ttest" and len(parts) >= 3:
                    csv_path = parts[1]
                    col = parts[2]
                    mu0 = float(parts[3]) if len(parts) > 3 else 0.0
                    import pandas as pd

                    from .statistics import one_sample_ttest

                    df = pd.read_csv(csv_path)
                    data = df[col].dropna().values
                    result = one_sample_ttest(data, mu0=mu0)
                    log.write(f"\n[bold]One-sample t-test: {col} vs mu={mu0}[/bold]")
                    log.write(f"  n = {len(data)}")
                    log.write(f"  t = {result.test_statistic:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")
                    log.write(f"  df = {result.df:.0f}")
                    if hasattr(result, "ci_lower"):
                        log.write(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

                elif action == "ttest2" and len(parts) >= 4:
                    csv_path = parts[1]
                    col = parts[2]
                    group_col = parts[3]
                    import pandas as pd

                    from .statistics import two_sample_ttest

                    df = pd.read_csv(csv_path)
                    groups = df[group_col].dropna().unique()
                    if len(groups) < 2:
                        log.write(f"[red]Need at least 2 groups in '{group_col}', found {len(groups)}[/red]")
                    else:
                        g1, g2 = groups[0], groups[1]
                        x = df.loc[df[group_col] == g1, col].dropna().values
                        y = df.loc[df[group_col] == g2, col].dropna().values
                        result = two_sample_ttest(x, y, equal_var=False)
                        log.write(f"\n[bold]Two-sample t-test (Welch): {col} by {group_col}[/bold]")
                        log.write(f"  Groups: {g1} (n={len(x)}) vs {g2} (n={len(y)})")
                        log.write(f"  t = {result.test_statistic:.4f}")
                        log.write(f"  p = {result.p_value:.6f}")
                        log.write(f"  df = {result.df:.1f}")
                        if hasattr(result, "ci_lower"):
                            log.write(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
                        if len(groups) > 2:
                            log.write(f"  [dim](Used first 2 of {len(groups)} groups: {g1}, {g2})[/dim]")

                elif action == "describe" and len(parts) >= 2:
                    import pandas as pd

                    df = pd.read_csv(parts[1])
                    log.write(f"\n[bold]Descriptive Statistics: {parts[1]}[/bold]")
                    log.write(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
                    desc = df.describe()
                    log.write(desc.to_string())

                elif action == "profile" and len(parts) >= 2:
                    from .dataset import load_dataset, profile_dataset

                    df = load_dataset(parts[1])
                    profile = profile_dataset(df)
                    log.write(f"\n[bold]Dataset Profile: {parts[1]}[/bold]")
                    log.write(profile.summary_table())

                elif action == "corr" and len(parts) >= 4:
                    import pandas as pd

                    from .statistics import pearson_correlation

                    df = pd.read_csv(parts[1])
                    x = df[parts[2]].dropna().values
                    y = df[parts[3]].dropna().values
                    min_len = min(len(x), len(y))
                    result = pearson_correlation(x[:min_len], y[:min_len])
                    log.write(f"\n[bold]Pearson Correlation: {parts[2]} vs {parts[3]}[/bold]")
                    log.write(f"  r = {result.test_statistic:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")

                elif action == "evalue":
                    rr = float(parts[1])
                    ci_lo = float(parts[2]) if len(parts) > 2 else None
                    from .sensitivity import e_value_rr

                    result = e_value_rr(rr, ci_lower=ci_lo)
                    log.write(f"\n[bold]E-value (RR = {rr})[/bold]")
                    log.write(f"  E-value (point): {result.e_value_point:.2f}")
                    log.write(f"  E-value (CI):    {result.e_value_ci:.2f}")
                    log.write(f"  {result.interpretation}")

                elif action == "bh":
                    import numpy as np

                    from .multiple_testing import benjamini_hochberg

                    p_str = parts[1] if len(parts) > 1 else ""
                    p_values = np.array([float(x) for x in p_str.split(",")])
                    result = benjamini_hochberg(p_values)
                    log.write("\n[bold]Benjamini-Hochberg Correction[/bold]")
                    log.write(f"  Tests: {result.n_tests}")
                    log.write(f"  Rejected: {result.n_rejected}")
                    for i, (orig, adj, rej) in enumerate(zip(result.original, result.adjusted, result.rejected)):
                        status = "[green]reject[/green]" if rej else "[dim]retain[/dim]"
                        log.write(f"  [{i + 1}] p={orig:.4f} -> adjusted={adj:.4f} {status}")

                elif action == "bootstrap" and len(parts) >= 3:
                    import numpy as np
                    import pandas as pd

                    from .bootstrap_methods import bootstrap

                    df = pd.read_csv(parts[1])
                    col = parts[2]
                    data = df[col].dropna().values.astype(float)
                    result = bootstrap(data, np.mean, n_boot=2000, ci_method="bca")
                    log.write(f"\n[bold]Bootstrap CI for mean of '{col}'[/bold]")
                    log.write(f"  Mean:   {result.estimate:.4f}")
                    log.write(f"  SE:     {result.se:.4f}")
                    log.write(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
                    log.write(f"  Bias:   {result.bias:.4f}")
                    log.write(f"  Method: BCa ({result.n_boot} replicates)")

                elif action == "missing" and len(parts) >= 2:
                    import pandas as pd

                    df = pd.read_csv(parts[1])
                    total = len(df)
                    log.write(f"\n[bold]Missing Data Report: {parts[1]}[/bold]")
                    log.write(f"  Total rows: {total}\n")
                    for col in df.columns:
                        n_miss = df[col].isna().sum()
                        pct = n_miss / total * 100
                        if n_miss > 0:
                            color = "red" if pct > 20 else "yellow"
                            log.write(f"  [{color}]{col}[/{color}]: {n_miss}/{total} ({pct:.1f}%)")
                        else:
                            log.write(f"  [green]{col}[/green]: 0 missing")

                elif action == "head" and len(parts) >= 2:
                    import pandas as pd

                    n = int(parts[2]) if len(parts) > 2 else 10
                    df = pd.read_csv(parts[1])
                    log.write(f"\n[bold]First {n} rows: {parts[1]}[/bold]")
                    log.write(df.head(n).to_string())

                elif action == "columns" and len(parts) >= 2:
                    import pandas as pd

                    df = pd.read_csv(parts[1])
                    log.write(f"\n[bold]Columns: {parts[1]}[/bold]")
                    for col in df.columns:
                        n_miss = df[col].isna().sum()
                        pct = n_miss / len(df) * 100
                        miss = f" ({n_miss} missing, {pct:.1f}%)" if n_miss > 0 else ""
                        log.write(f"  {col}: {df[col].dtype}{miss}")

                elif action == "paired" and len(parts) >= 4:
                    import pandas as pd

                    from .statistics import paired_ttest

                    df = pd.read_csv(parts[1])
                    x = df[parts[2]].dropna().values
                    y = df[parts[3]].dropna().values
                    n = min(len(x), len(y))
                    result = paired_ttest(x[:n], y[:n])
                    log.write(f"\n[bold]Paired t-test: {parts[2]} vs {parts[3]}[/bold]")
                    log.write(f"  n = {n}")
                    log.write(f"  t = {result.test_statistic:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")

                elif action == "anova" and len(parts) >= 4:
                    import pandas as pd

                    from .statistics import one_way_anova

                    df = pd.read_csv(parts[1])
                    col, grp = parts[2], parts[3]
                    groups = [g[col].dropna().values for _, g in df.groupby(grp)]
                    result = one_way_anova(*groups)
                    log.write(f"\n[bold]One-way ANOVA: {col} by {grp}[/bold]")
                    log.write(f"  F = {result.test_statistic:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")
                    if hasattr(result, "effect_size") and result.effect_size is not None:
                        log.write(f"  eta² = {result.effect_size:.4f}")

                elif action == "chi2" and len(parts) >= 4:
                    import pandas as pd

                    from .statistics import chi2_independence

                    df = pd.read_csv(parts[1])
                    ct = pd.crosstab(df[parts[2]], df[parts[3]])
                    result = chi2_independence(ct.values)
                    log.write(f"\n[bold]Chi-square independence: {parts[2]} x {parts[3]}[/bold]")
                    log.write(f"  χ² = {result.test_statistic:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")
                    log.write(f"  df = {result.df:.0f}")

                elif action == "ks" and len(parts) >= 3:
                    import pandas as pd

                    from .statistics import ks_test_one_sample

                    df = pd.read_csv(parts[1])
                    data = df[parts[2]].dropna().values
                    result = ks_test_one_sample(data)
                    log.write(f"\n[bold]Kolmogorov-Smirnov normality: {parts[2]}[/bold]")
                    log.write(f"  D = {result.test_statistic:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")
                    log.write(f"  Normal: {'Yes' if result.p_value > 0.05 else 'No'} (α=0.05)")

                elif action == "mannwhitney" and len(parts) >= 4:
                    import pandas as pd

                    from .statistics import mann_whitney_u

                    df = pd.read_csv(parts[1])
                    col, grp = parts[2], parts[3]
                    groups = df[grp].dropna().unique()
                    if len(groups) < 2:
                        log.write(f"[red]Need 2+ groups in '{grp}'[/red]")
                    else:
                        x = df.loc[df[grp] == groups[0], col].dropna().values
                        y = df.loc[df[grp] == groups[1], col].dropna().values
                        result = mann_whitney_u(x, y)
                        log.write(f"\n[bold]Mann-Whitney U: {col} by {grp}[/bold]")
                        log.write(f"  U = {result.test_statistic:.4f}")
                        log.write(f"  p = {result.p_value:.6f}")

                elif action == "propensity" and len(parts) >= 4:
                    import pandas as pd

                    from .causal import compute_propensity_scores

                    df = pd.read_csv(parts[1])
                    trt = parts[2]
                    covs = parts[3].split(",")
                    scores = compute_propensity_scores(df, trt, covs)
                    log.write(f"\n[bold]Propensity Scores: treatment={trt}[/bold]")
                    log.write(f"  Covariates: {', '.join(covs)}")
                    log.write(f"  Mean score: {scores.mean():.4f}")
                    log.write(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")

                elif action == "ate" and len(parts) >= 4:
                    import pandas as pd

                    from .effects import estimate_ate

                    df = pd.read_csv(parts[1])
                    outcome, trt = parts[2], parts[3]
                    result = estimate_ate(df, outcome, trt)
                    log.write(f"\n[bold]ATE: {outcome} ~ {trt}[/bold]")
                    log.write(f"  Estimate: {result}")

                elif action == "ipw" and len(parts) >= 4:
                    import pandas as pd

                    from .causal import calculate_ipw_weights

                    df = pd.read_csv(parts[1])
                    outcome, trt = parts[2], parts[3]
                    weights = calculate_ipw_weights(df, trt)
                    log.write(f"\n[bold]IPW Weights: treatment={trt}[/bold]")
                    log.write(f"  n = {len(weights)}")
                    log.write(f"  Mean weight: {weights.mean():.4f}")
                    log.write(f"  ESS: {(weights.sum() ** 2) / (weights**2).sum():.1f}")

                elif action == "cohens_d" and len(parts) >= 4:
                    import numpy as np
                    import pandas as pd

                    df = pd.read_csv(parts[1])
                    col, grp = parts[2], parts[3]
                    groups = df[grp].dropna().unique()
                    if len(groups) < 2:
                        log.write(f"[red]Need 2+ groups in '{grp}'[/red]")
                    else:
                        x = df.loc[df[grp] == groups[0], col].dropna().values.astype(float)
                        y = df.loc[df[grp] == groups[1], col].dropna().values.astype(float)
                        pooled_std = np.sqrt(
                            ((len(x) - 1) * x.std() ** 2 + (len(y) - 1) * y.std() ** 2) / (len(x) + len(y) - 2)
                        )
                        d = (x.mean() - y.mean()) / pooled_std if pooled_std > 0 else 0
                        log.write(f"\n[bold]Cohen's d: {col} by {grp}[/bold]")
                        log.write(f"  d = {d:.4f}")
                        magnitude = (
                            "negligible"
                            if abs(d) < 0.2
                            else "small"
                            if abs(d) < 0.5
                            else "medium"
                            if abs(d) < 0.8
                            else "large"
                        )
                        log.write(f"  Magnitude: {magnitude}")

                elif action == "power" and len(parts) >= 3:
                    from .inference import power_t_test

                    d = float(parts[1])
                    n = int(parts[2])
                    alpha = float(parts[3]) if len(parts) > 3 else 0.05
                    result = power_t_test(d=d, n=n, alpha=alpha)
                    log.write("\n[bold]Power Analysis[/bold]")
                    log.write(f"  Effect size d = {d}")
                    log.write(f"  Sample size n = {n}")
                    log.write(f"  Alpha = {alpha}")
                    log.write(f"  Power = {result:.4f}")
                    log.write(f"  {'Adequate' if result >= 0.8 else 'Underpowered'} (threshold: 0.80)")

                # ── Survival Analysis ────────────────────────────
                elif action in ("kaplan_meier", "km") and len(parts) >= 4:
                    import pandas as pd

                    from .survival import kaplan_meier_curve

                    df = pd.read_csv(parts[1])
                    time_col, event_col = parts[2], parts[3]
                    result = kaplan_meier_curve(df[time_col].values, df[event_col].values)
                    log.write(f"\n[bold]Kaplan-Meier: time={time_col}, event={event_col}[/bold]")
                    log.write(f"  Events: {int(df[event_col].sum())} / {len(df)}")
                    log.write(f"  Median survival: {result.median_survival}")
                    if hasattr(result, "times") and len(result.times) > 0:
                        log.write(f"  Time range: [{result.times[0]:.2f}, {result.times[-1]:.2f}]")

                elif action == "cox" and len(parts) >= 5:
                    import pandas as pd

                    from .survival import cox_ph

                    df = pd.read_csv(parts[1])
                    time_col, event_col = parts[2], parts[3]
                    covs = parts[4].split(",")
                    result = cox_ph(df, time_col, event_col, covs)
                    log.write(f"\n[bold]Cox PH: {time_col} ~ {' + '.join(covs)}[/bold]")
                    log.write(f"  Concordance: {result.concordance:.4f}")
                    for name, hr, p in zip(result.covariate_names, result.hazard_ratios, result.p_values):
                        sig = "*" if p < 0.05 else ""
                        log.write(f"  {name}: HR={hr:.4f}, p={p:.4f} {sig}")

                elif action == "logrank" and len(parts) >= 5:
                    import pandas as pd

                    from .survival import log_rank_test

                    df = pd.read_csv(parts[1])
                    result = log_rank_test(df[parts[2]].values, df[parts[3]].values, df[parts[4]].values)
                    log.write(f"\n[bold]Log-rank test: {parts[2]} by {parts[4]}[/bold]")
                    log.write(f"  chi2 = {result.test_statistic:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")

                # ── Causal Inference (DiD, RDD, IV) ─────────────
                elif action == "did" and len(parts) >= 5:
                    import pandas as pd

                    from .did import did_2x2

                    df = pd.read_csv(parts[1])
                    result = did_2x2(df, outcome=parts[2], treatment=parts[3], post=parts[4])
                    log.write(f"\n[bold]DiD 2x2: {parts[2]} ~ {parts[3]} x {parts[4]}[/bold]")
                    log.write(f"  ATT = {result.att:.4f}")
                    log.write(f"  SE = {result.se:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")
                    log.write(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

                elif action == "rdd" and len(parts) >= 4:
                    import pandas as pd

                    from .rdd import sharp_rdd

                    df = pd.read_csv(parts[1])
                    cutoff = float(parts[4]) if len(parts) > 4 else 0.0
                    result = sharp_rdd(df, outcome=parts[2], running=parts[3], cutoff=cutoff)
                    log.write(f"\n[bold]Sharp RDD: {parts[2]} ~ {parts[3]} (cutoff={cutoff})[/bold]")
                    log.write(f"  LATE = {result.estimate:.4f}")
                    log.write(f"  SE = {result.se:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")
                    log.write(f"  Bandwidth: {result.bandwidth:.4f}")

                elif action == "tsls" and len(parts) >= 5:
                    import pandas as pd

                    from .iv import tsls

                    df = pd.read_csv(parts[1])
                    endog = parts[3].split(",")
                    instruments = parts[4].split(",")
                    result = tsls(df, outcome=parts[2], endogenous=endog, instruments=instruments)
                    log.write(f"\n[bold]2SLS: {parts[2]} ~ {parts[3]} | {parts[4]}[/bold]")
                    for name, coef, se, p in zip(
                        result.covariate_names, result.coefficients, result.se, result.p_values
                    ):
                        sig = "*" if p < 0.05 else ""
                        log.write(f"  {name}: coef={coef:.4f}, SE={se:.4f}, p={p:.4f} {sig}")
                    if hasattr(result, "first_stage_f"):
                        log.write(f"  First-stage F: {result.first_stage_f:.2f}")

                # ── Matching ────────────────────────────────────
                elif action == "match" and len(parts) >= 4:
                    import pandas as pd

                    from .matching import propensity_score_matching

                    df = pd.read_csv(parts[1])
                    covs = parts[3].split(",")
                    result = propensity_score_matching(df, treatment=parts[2], covariates=covs)
                    log.write(f"\n[bold]PS Matching: treatment={parts[2]}[/bold]")
                    log.write(f"  Matched pairs: {result.n_matched}")
                    log.write(f"  ATT = {result.att:.4f}")
                    if hasattr(result, "att_se"):
                        log.write(f"  SE = {result.att_se:.4f}")

                # ── Missing Data ────────────────────────────────
                elif action == "mcar" and len(parts) >= 2:
                    import pandas as pd

                    from .missing import littles_mcar_test

                    df = pd.read_csv(parts[1])
                    result = littles_mcar_test(df)
                    log.write(f"\n[bold]Little's MCAR Test: {parts[1]}[/bold]")
                    log.write(f"  chi2 = {result.test_statistic:.4f}")
                    log.write(f"  df = {result.df}")
                    log.write(f"  p = {result.p_value:.6f}")
                    mcar = "MCAR plausible" if result.p_value > 0.05 else "MCAR rejected (data is MAR or MNAR)"
                    log.write(f"  Conclusion: {mcar}")

                elif action == "impute" and len(parts) >= 2:
                    import pandas as pd

                    from .missing import mice

                    df = pd.read_csv(parts[1])
                    m = int(parts[2]) if len(parts) > 2 else 5
                    result = mice(df, m=m)
                    log.write(f"\n[bold]MICE Imputation: {parts[1]} (m={m})[/bold]")
                    log.write(f"  Imputed datasets: {result.m}")
                    log.write(f"  Columns imputed: {len(result.imputed_columns)}")
                    for col in result.imputed_columns[:10]:
                        log.write(f"    {col}")

                # ── Diagnostics ─────────────────────────────────
                elif action == "vif" and len(parts) >= 3:
                    import numpy as np
                    import pandas as pd

                    from .diagnostics import collinearity_diagnostics

                    df = pd.read_csv(parts[1])
                    cols = parts[2].split(",")
                    X = df[cols].dropna().values.astype(float)
                    result = collinearity_diagnostics(X, column_names=cols)
                    log.write("\n[bold]VIF / Collinearity Diagnostics[/bold]")
                    for name, vif in zip(result.variable_names, result.vif):
                        flag = " [red](HIGH)[/red]" if vif > 10 else ""
                        log.write(f"  {name}: VIF={vif:.2f}{flag}")
                    log.write(f"  Condition number: {result.condition_number:.2f}")

                # ── Effect Sizes ────────────────────────────────
                elif action == "odds_ratio" and len(parts) >= 4:
                    import pandas as pd

                    from .effect_sizes import odds_ratio

                    df = pd.read_csv(parts[1])
                    ct = pd.crosstab(df[parts[2]], df[parts[3]])
                    if ct.shape == (2, 2):
                        a, b, c, d = ct.iloc[0, 0], ct.iloc[0, 1], ct.iloc[1, 0], ct.iloc[1, 1]
                        result = odds_ratio(int(a), int(b), int(c), int(d))
                        log.write(f"\n[bold]Odds Ratio: {parts[2]} x {parts[3]}[/bold]")
                        log.write(f"  OR = {result.estimate:.4f}")
                        log.write(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
                    else:
                        log.write(f"[red]Need a 2x2 table, got {ct.shape}[/red]")

                elif action == "nnt" and len(parts) >= 4:
                    import pandas as pd

                    from .effect_sizes import number_needed_to_treat

                    df = pd.read_csv(parts[1])
                    ct = pd.crosstab(df[parts[2]], df[parts[3]])
                    if ct.shape == (2, 2):
                        a, b, c, d = ct.iloc[0, 0], ct.iloc[0, 1], ct.iloc[1, 0], ct.iloc[1, 1]
                        result = number_needed_to_treat(int(a), int(b), int(c), int(d))
                        log.write(f"\n[bold]NNT: {parts[2]} x {parts[3]}[/bold]")
                        log.write(f"  NNT = {result.estimate:.2f}")
                        log.write(f"  95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
                    else:
                        log.write(f"[red]Need a 2x2 table, got {ct.shape}[/red]")

                # ── Sensitivity Analysis ────────────────────────
                elif action == "rosenbaum" and len(parts) >= 4:
                    import numpy as np
                    import pandas as pd

                    from .sensitivity import rosenbaum_bounds

                    df = pd.read_csv(parts[1])
                    outcome_col, treat_col = parts[2], parts[3]
                    treated = df.loc[df[treat_col] == 1, outcome_col].dropna().values
                    control = df.loc[df[treat_col] == 0, outcome_col].dropna().values
                    result = rosenbaum_bounds(treated, control)
                    log.write(f"\n[bold]Rosenbaum Bounds: {outcome_col} by {treat_col}[/bold]")
                    for gamma, p_upper in zip(result.gamma_values, result.p_upper):
                        sig = "[green]sig[/green]" if p_upper < 0.05 else "[red]n.s.[/red]"
                        log.write(f"  Gamma={gamma:.1f}: p_upper={p_upper:.4f} {sig}")

                # ── Publication Tables ──────────────────────────
                elif action == "table1" and len(parts) >= 3:
                    import pandas as pd

                    from .tables_pub import table1

                    df = pd.read_csv(parts[1])
                    group_col = parts[2]
                    result = table1(df, group_col=group_col)
                    log.write(f"\n[bold]Table 1: by {group_col}[/bold]")
                    log.write(result.to_string())

                # ── Additional Effect Sizes ────────────────────
                elif action == "hedges_g" and len(parts) >= 4:
                    import pandas as pd

                    from .effect_sizes import hedges_g

                    df = pd.read_csv(parts[1])
                    x = df[parts[2]].dropna().values
                    y = df[parts[3]].dropna().values
                    result = hedges_g(x, y)
                    log.write(f"\n[bold]Hedges' g: {parts[2]} vs {parts[3]}[/bold]")
                    log.write(f"  g = {result.estimate:.4f}")
                    log.write(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

                elif action == "risk_ratio" and len(parts) >= 5:
                    from .effect_sizes import risk_ratio

                    a, b, c, d = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    result = risk_ratio(a, b, c, d)
                    log.write(f"\n[bold]Risk Ratio (2x2 table: {a},{b},{c},{d})[/bold]")
                    log.write(f"  RR = {result.estimate:.4f}")
                    log.write(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

                # ── Multiple Testing ───────────────────────────
                elif action == "bonferroni" and len(parts) >= 2:
                    import numpy as np

                    from .multiple_testing import bonferroni

                    pvals = np.array([float(p) for p in parts[1:]])
                    result = bonferroni(pvals)
                    log.write(f"\n[bold]Bonferroni correction ({len(pvals)} tests)[/bold]")
                    for i, (orig, adj) in enumerate(zip(result.original, result.adjusted)):
                        sig = "[green]reject[/green]" if result.reject[i] else "[dim]retain[/dim]"
                        log.write(f"  p{i + 1}: {orig:.4f} → {adj:.4f} {sig}")

                # ── Advanced Resampling ────────────────────────
                elif action == "jackknife" and len(parts) >= 3:
                    import numpy as np
                    import pandas as pd

                    from .bootstrap_methods import jackknife

                    df = pd.read_csv(parts[1])
                    data = df[parts[2]].dropna().values
                    result = jackknife(data, statistic=np.mean)
                    log.write(f"\n[bold]Jackknife: mean of {parts[2]}[/bold]")
                    log.write(f"  Estimate = {result.estimate:.4f}")
                    log.write(f"  SE = {result.se:.4f}")
                    log.write(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
                    log.write(f"  Bias = {result.bias:.6f}")

                elif action == "permtest" and len(parts) >= 4:
                    import pandas as pd

                    from .bootstrap_methods import permutation_test

                    df = pd.read_csv(parts[1])
                    col = parts[2]
                    group_col = parts[3]
                    groups = df[group_col].dropna().unique()
                    g1 = df.loc[df[group_col] == groups[0], col].dropna().values
                    g2 = df.loc[df[group_col] == groups[1], col].dropna().values
                    result = permutation_test(g1, g2)
                    log.write(f"\n[bold]Permutation test: {col} by {group_col}[/bold]")
                    log.write(f"  Observed stat = {result.observed:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")
                    log.write(f"  Permutations = {result.n_permutations}")

                # ── Advanced DiD / RDD ─────────────────────────
                elif action == "event_study" and len(parts) >= 5:
                    import pandas as pd

                    from .did import event_study

                    df = pd.read_csv(parts[1])
                    result = event_study(
                        df,
                        outcome=parts[2],
                        unit=parts[3],
                        time=parts[4],
                        treatment_time=parts[5] if len(parts) > 5 else parts[4],
                    )
                    log.write(f"\n[bold]Event Study: {parts[2]}[/bold]")
                    for coef in result.coefficients:
                        sig = "*" if coef.p_value < 0.05 else ""
                        log.write(
                            f"  t={coef.period:+d}: {coef.estimate:+.4f} (SE={coef.se:.4f}) p={coef.p_value:.4f}{sig}"
                        )

                elif action == "fuzzy_rdd" and len(parts) >= 5:
                    import pandas as pd

                    from .rdd import fuzzy_rdd

                    df = pd.read_csv(parts[1])
                    cutoff = float(parts[5]) if len(parts) > 5 else 0.0
                    result = fuzzy_rdd(df, outcome=parts[2], running=parts[3], treatment=parts[4], cutoff=cutoff)
                    log.write(f"\n[bold]Fuzzy RDD: {parts[2]} ~ {parts[3]} (treatment={parts[4]})[/bold]")
                    log.write(f"  LATE = {result.estimate:.4f}")
                    log.write(f"  SE = {result.se:.4f}")
                    log.write(f"  p = {result.p_value:.6f}")
                    log.write(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
                    log.write(f"  Bandwidth = {result.bandwidth:.4f}")

                # ── STROBE Reporting ───────────────────────────
                elif action == "strobe" and len(parts) >= 2:
                    from .reporting import check_strobe_compliance

                    text = Path(parts[1]).read_text()
                    checks = check_strobe_compliance(text)
                    met = sum(1 for c in checks if c.met)
                    log.write(f"\n[bold]STROBE Compliance: {met}/{len(checks)} items met[/bold]")
                    for c in checks:
                        status = "[green]MET[/green]" if c.met else "[red]NOT MET[/red]"
                        log.write(f"  {status} {c.item}: {c.description}")

                # ── Model Diagnostics ──────────────────────────
                elif action == "residuals" and len(parts) >= 4:
                    import numpy as np
                    import pandas as pd

                    from .diagnostics import compute_residuals

                    df = pd.read_csv(parts[1])
                    y = df[parts[2]].values
                    X_cols = parts[3:]
                    X = df[X_cols].values
                    from numpy.linalg import lstsq

                    beta, _, _, _ = lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
                    y_hat = np.column_stack([np.ones(len(X)), X]) @ beta
                    result = compute_residuals(y, y_hat, np.column_stack([np.ones(len(X)), X]))
                    log.write(f"\n[bold]Residual Diagnostics: {parts[2]} ~ {' + '.join(X_cols)}[/bold]")
                    log.write(f"  Durbin-Watson = {result.durbin_watson:.4f}")
                    log.write(f"  Shapiro-Wilk (residuals) p = {result.normality_p:.4f}")
                    log.write(f"  Breusch-Pagan p = {result.heteroscedasticity_p:.4f}")

                elif action == "cooks" and len(parts) >= 4:
                    import pandas as pd

                    from .diagnostics import compute_influence

                    df = pd.read_csv(parts[1])
                    y = df[parts[2]].values
                    X_cols = parts[3:]
                    import numpy as np

                    X = np.column_stack([np.ones(len(df)), df[X_cols].values])
                    result = compute_influence(y, X)
                    high = sum(1 for d in result.cooks_distance if d > 4 / len(y))
                    log.write(f"\n[bold]Influence Diagnostics: {parts[2]} ~ {' + '.join(X_cols)}[/bold]")
                    log.write(f"  High Cook's D (>4/n): {high} observations")
                    log.write(f"  Max Cook's D = {max(result.cooks_distance):.4f}")
                    log.write(f"  Max leverage = {max(result.leverage):.4f}")
                    log.write(
                        f"  High leverage (>2p/n): {sum(1 for h in result.leverage if h > 2 * X.shape[1] / len(y))}"
                    )

                elif action == "modules":
                    from .modules import list_modules

                    mods = list_modules()
                    log.write(f"\n[bold]{len(mods)} analysis modules available:[/bold]")
                    for m in mods:
                        log.write(f"  [bold]{m['name']}[/bold]: {m['description']}")

                elif action == "run" and len(parts) >= 2:
                    from .modules import MODULE_SPECS, run_module

                    mod_name = parts[1]
                    if mod_name not in MODULE_SPECS:
                        log.write(f"[red]Unknown module: {mod_name}[/red]")
                        log.write("Type 'modules' to see available modules.")
                    else:
                        log.write(f"\n[yellow]Running module: {mod_name}...[/yellow]")
                        try:
                            result = run_module(mod_name)
                            log.write(f"  [green]Done[/green]: {result}")
                        except Exception as e:
                            log.write(f"  [red]Failed[/red]: {e}")

                elif action == "save" and len(parts) >= 2:
                    path = parts[1]
                    if not self._last_result_text:
                        log.write("[yellow]No result to save. Run an analysis first.[/yellow]")
                    else:
                        Path(path).write_text(self._last_result_text)
                        log.write(f"[green]Saved to {path}[/green]")

                elif action == "export" and len(parts) >= 2:
                    path = parts[1]
                    fmt = parts[2] if len(parts) > 2 else "csv"
                    if self._last_result is None:
                        log.write("[yellow]No result to export. Run an analysis first.[/yellow]")
                    else:
                        import pandas as pd

                        if isinstance(self._last_result, pd.DataFrame):
                            if fmt == "csv":
                                self._last_result.to_csv(path, index=False)
                            elif fmt in ("md", "markdown"):
                                from .export import df_to_markdown

                                Path(path).write_text(df_to_markdown(self._last_result))
                            elif fmt == "latex":
                                from .export import df_to_latex

                                Path(path).write_text(df_to_latex(self._last_result))
                            elif fmt == "html":
                                from .export import df_to_html

                                Path(path).write_text(df_to_html(self._last_result))
                            else:
                                log.write(f"[yellow]Unknown format: {fmt}. Use csv, md, latex, html.[/yellow]")
                                return
                            log.write(f"[green]Exported {fmt} to {path}[/green]")
                        else:
                            Path(path).write_text(str(self._last_result))
                            log.write(f"[green]Saved to {path}[/green]")

                elif action == "commands":
                    from .stat_commands import commands_by_category

                    cats = commands_by_category()
                    log.write("\n[bold cyan]All MOIRAIS Commands by Category[/bold cyan]")
                    for cat, cmds in sorted(cats.items()):
                        log.write(f"\n[bold]{cat}[/bold] ({len(cmds)})")
                        for c in cmds:
                            a = f" ({', '.join(c.aliases)})" if c.aliases else ""
                            log.write(f"  [bold]{c.name}[/bold]{a} — {c.description}")

                else:
                    # Fallback: dispatch through stat_commands registry
                    from .stat_commands import resolve as _resolve_cmd

                    _cmd = _resolve_cmd(action)
                    if _cmd is not None:
                        _cmd.handler_stat(parts, log, self._store)
                    else:
                        log.write(f"[yellow]Unknown command: {action}[/yellow]")
                        log.write("Type 'help' for available commands.")
                        log.write("[dim]Type 'commands' to see all 1200+ commands.[/dim]")

            except FileNotFoundError as e:
                log.write(f"[red]File not found: {e}[/red]")
            except Exception as e:
                log.write(f"[red]Error: {e}[/red]")
            finally:
                log.write = _orig_write  # type: ignore[assignment]
                if _capture:
                    # Strip Rich markup for plain-text save (precise pattern to preserve data brackets).
                    import re

                    _RICH_TAG_RE = re.compile(
                        r"\[/?"
                        r"(?:bold|dim|italic|underline|strike|reverse|"
                        r"red|green|yellow|blue|cyan|magenta|white|"
                        r"on \w+|link[^\]]*|#[0-9a-fA-F]{6})"
                        r"[^\]]*\]"
                    )
                    plain = "\n".join(_RICH_TAG_RE.sub("", line) for line in _capture)
                    self._store(plain)

            log.write("")

    # ==================================================================
    # Main Application
    # ==================================================================

    class MOIRAISApp(App):
        """MOIRAIS Terminal IDE — Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation."""

        TITLE = "MOIRAIS"
        SUB_TITLE = "Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation"

        ENABLE_COMMAND_PALETTE = False

        DEFAULT_CSS = """
        /* ── App-level design tokens — ATOM-style heavy borders ─── */
        /* Inspired by rustysys-dev/rsd-terminal-ui — clear, contrast-
         * balanced panel boundaries so panes don't visually merge into
         * each other. Textual offers 'heavy', 'double', 'thick',
         * 'tall' as the heaviest stroke options; we use heavy for
         * focusable widgets and double for the frame-style containers
         * (chat-log etc.) that should read as "this is a panel". */

        /* Shared Input styling — all screens */
        Input {
            dock: bottom;
            margin: 1 0 0 0;
            background: $surface;
            border: heavy $primary 60%;
            padding: 0 1;
        }
        Input:focus {
            border: heavy $accent;
            background: $surface-darken-1;
        }
        Input.-invalid {
            border: heavy $error;
        }

        /* Shared TextArea styling */
        TextArea {
            border: heavy $primary 50%;
        }
        TextArea:focus {
            border: heavy $accent;
        }

        /* Shared RichLog defaults — consistent padding */
        RichLog {
            padding: 1;
            scrollbar-size: 1 1;
        }

        /* REPL sidebar — subtle background to differentiate panels */
        #repl-tree {
            background: $surface-darken-1;
            padding: 0 1;
        }

        /* REPL mode indicator bar — visible background */
        #repl-mode {
            background: $surface;
            color: $text;
            border-top: solid $primary 30%;
        }

        /* Settings header/footer — better contrast */
        #settings-header {
            border-bottom: solid $primary 30%;
            padding: 0 2;
        }
        #settings-footer {
            border-top: solid $primary 30%;
            padding: 0 2;
        }

        /* HomeScreen layout */
        #home-main {
            padding: 2 4;
        }
        """

        # Shared state: loaded dataset available across screens.
        loaded_df = None
        loaded_df_name: str = ""

        BINDINGS = [
            Binding("q", "quit", "Quit", show=True),
            Binding("f2", "toggle_copy_mode", "Copy Mode", show=True),
            Binding("ctrl+o", "copy_all", "Copy All", show=True),
            Binding("ctrl+b", "copy_errors", "Copy Errors", show=False),
            Binding("ctrl+l", "copy_last5", "Copy Last 5", show=False),
        ]

        _copy_mode: bool = False

        SCREENS = {
            "chat": ChatScreen,
            "pipeline": PipelineScreen,
            "doctor": DoctorScreen,
            "dataset": DatasetScreen,
            "help": HelpScreen,
            "debug": DebugScreen,
            "stats": StatScreen,
            "repl": lambda: ReplScreen(lang="auto"),
            "settings": SettingsScreen,
        }

        def get_default_screen(self) -> Screen:
            """Use HomeScreen as the base screen — no push, stack depth 1."""
            return HomeScreen()

        def on_mount(self) -> None:
            """Pre-warm the Python import graph in a background worker.

            Symptom this fixes: every screen mount took "minutes" because
            ChatScreen/PipelineScreen/DatasetScreen each triggered
            first-time imports of llm.py / chat.py / data.py / modules.py
            — and those transitively pull numpy, pandas, scipy, sklearn,
            httpx, etc. The first navigation after launch felt frozen.

            By kicking off `_warmup_imports` immediately on app mount,
            the heavy imports run in parallel with the user reading the
            HomeScreen banner. By the time they press 'c' / 'p' / 'd',
            sys.modules is already populated and the destination screen
            mounts instantly.

            Wrapped in run_worker so it doesn't block the event loop.
            """
            self.run_worker(self._warmup_imports(), name="moirais-warmup")

        async def _warmup_imports(self) -> None:
            import importlib

            modules = [
                "moirais.chat",
                "moirais.llm",
                "moirais.data",
                "moirais.modules",
                "moirais.causal",
                "moirais.effects",
                "moirais.investigation",
                "moirais.fn",
                "moirais.runner",
                "moirais.psymet",
            ]
            for mod in modules:
                try:
                    await asyncio.to_thread(importlib.import_module, mod)
                except Exception:
                    # Don't fail the whole warmup if one module is busted —
                    # the user can still navigate; the broken screen will
                    # surface its own error when opened.
                    pass

        # -- Clipboard helpers -----------------------------------------

        def _copy_text(self, text: str) -> None:
            """Copy *text* to the system clipboard (platform-aware).

            macOS Terminal.app does not support OSC 52, so on Darwin we
            use the built-in ``/usr/bin/pbcopy`` binary (absolute path
            to prevent PATH injection).  On all other platforms we fall
            back to Textual's native OSC 52 implementation.
            """
            if sys.platform == "darwin":
                try:
                    subprocess.run(
                        ["/usr/bin/pbcopy"],
                        input=text.encode("utf-8"),
                        check=True,
                        timeout=3,
                    )
                    return
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
            # Fallback: OSC 52 (iTerm2, kitty, WezTerm, Windows Terminal…)
            self.copy_to_clipboard(text)

        def action_toggle_copy_mode(self) -> None:
            """Toggle Copy Mode — disables Textual mouse capture so the
            terminal's native text selection works (click-drag, Cmd+C)."""
            self._copy_mode = not self._copy_mode
            if self._driver is not None:
                if self._copy_mode:
                    self._driver._disable_mouse_support()
                    self.sub_title = "COPY MODE — select text with mouse, Cmd+C to copy"
                    self.notify(
                        "Copy Mode ON — select with mouse, Cmd+C to copy",
                        timeout=3,
                    )
                else:
                    self._driver._enable_mouse_support()
                    self.sub_title = "Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation"
                    self.notify("Copy Mode OFF", timeout=2)

        def action_copy_all(self) -> None:
            """Copy entire log from start to finish — all input, output, errors."""
            results = self.screen.query(CopyableRichLog)
            log = results.first() if results else None
            if log is None:
                self.notify("No log to copy", severity="warning", timeout=2)
                return
            text = log.get_all_text()
            if not text.strip():
                self.notify("Log is empty", severity="warning", timeout=2)
                return
            self._copy_text(text)
            n = len(text.splitlines())
            self.notify(f"Copied entire log ({n} lines)", timeout=3)

        def action_copy_errors(self) -> None:
            """Copy only error lines from the active screen's log."""
            results = self.screen.query(CopyableRichLog)
            log = results.first() if results else None
            if log is None:
                self.notify("No log to copy", severity="warning", timeout=2)
                return
            text = log.get_errors()
            if not text.strip():
                self.notify("No errors found in log", severity="information", timeout=2)
                return
            self._copy_text(text)
            self.notify(f"Copied {len(text.splitlines())} error lines", timeout=2)

        def action_copy_last5(self) -> None:
            """Copy the last 5 command/output exchanges from the log."""
            results = self.screen.query(CopyableRichLog)
            log = results.first() if results else None
            if log is None:
                self.notify("No log to copy", severity="warning", timeout=2)
                return
            text = log.get_last_n_exchanges(5)
            if not text.strip():
                self.notify("Nothing to copy", severity="warning", timeout=2)
                return
            self._copy_text(text)
            self.notify(f"Copied last 5 exchanges ({len(text.splitlines())} lines)", timeout=2)

    class HomeScreen(Screen):
        """Home screen — single vertical layout."""

        # priority=True is critical — without it, Textual lets focused
        # widgets (Header, Footer, VerticalScroll) absorb single-letter
        # keys before the screen-level binding fires. Reported symptom:
        # pressing 'c' did nothing because VerticalScroll's default key
        # handler grabbed it. Priority bindings run before widget routing.
        BINDINGS = [
            Binding("c", "go('chat')", "Chat", show=True, priority=True),
            Binding("p", "go('pipeline')", "Pipeline", show=True, priority=True),
            Binding("d", "go('doctor')", "Doctor", show=True, priority=True),
            Binding("i", "go('dataset')", "Inspect", show=True, priority=True),
            Binding("h", "go('help')", "Help", show=True, priority=True),
            Binding("g", "go('debug')", "Debug", show=True, priority=True),
            Binding("s", "go('stats')", "Stats", show=True, priority=True),
            Binding("e", "go('repl')", "REPL", show=True, priority=True),
            Binding("comma", "go('settings')", "Settings", show=True, priority=True),
            Binding("q", "quit_app", "Quit", show=True, priority=True),
        ]

        def action_go(self, screen_name: str) -> None:
            self.app.push_screen(screen_name)

        def action_quit_app(self) -> None:
            self.app.exit()

        def on_mount(self) -> None:
            # Take focus off the VerticalScroll so it can't intercept
            # single-letter keys before our priority bindings fire.
            self.set_focus(None)
            self.run_worker(self._load_status(), name="home-status")

        DEFAULT_CSS = """
        HomeScreen { layout: vertical; }
        #home-main { height: 1fr; content-align: center middle; }
        #home-main Static { text-align: center; }
        """

        def compose(self) -> ComposeResult:
            self.styles.overflow_y = "hidden"

            yield Header()
            with VerticalScroll(id="home-main"):
                yield Static(
                    "[bold cyan]⚡ MOIRAIS  Terminal IDE[/bold cyan]\n"
                    "[dim]Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation[/dim]\n\n"
                    "[bold cyan]Navigation[/bold cyan]\n"
                    "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                    "┃  [bold yellow]c[/bold yellow]  Chat          LLM agent      ┃\n"
                    "┃  [bold yellow]p[/bold yellow]  Pipeline      Run modules    ┃\n"
                    "┃  [bold yellow]d[/bold yellow]  Doctor        Diagnostics    ┃\n"
                    "┃  [bold yellow]i[/bold yellow]  Inspect       Browse datasets┃\n"
                    "┃  [bold yellow]h[/bold yellow]  Help          Documentation  ┃\n"
                    "┃  [bold yellow]g[/bold yellow]  Debug         System info    ┃\n"
                    "┃  [bold yellow]s[/bold yellow]  Stats         Statistics     ┃\n"
                    "┃  [bold yellow]e[/bold yellow]  REPL          Python/R       ┃\n"
                    "┃  [bold red]q[/bold red]  Quit          Exit MOIRAIS   ┃\n"
                    "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n",
                )
                yield Static("[dim]Loading…[/dim]", id="home-status-content")
            yield Footer()

        async def _load_status(self) -> None:
            """Detect LLM provider in background so the event loop stays free.

            CRITICAL: this runs in a background worker so the user can
            navigate (c/p/d/i) the moment the screen mounts — even before
            this finishes. We deliberately DO NOT call agent.warmup() here:
            warmup loads the Ollama model into RAM which takes 30s–3min on
            cold start and would freeze the system-status line. The user
            can trigger warmup manually from the chat screen if they want.
            """
            from .data import list_datasets
            from .llm import detect_model_display
            from .modules import list_modules

            # Cap LLM detection at 2 seconds — if Ollama isn't running,
            # don't make the user wait on a TCP timeout (default 30s+).
            try:
                info = await asyncio.wait_for(
                    asyncio.to_thread(detect_model_display), timeout=2.0
                )
            except (asyncio.TimeoutError, Exception):
                info = {"inner": "offline", "outer": "no-llm"}

            try:
                modules = list_modules()
            except Exception:
                modules = []
            try:
                datasets = list_datasets()
                n_cached = sum(1 for d in datasets if d["cached"])
            except Exception:
                n_cached = 0

            content = self.query_one("#home-status-content", Static)
            content.update(
                f"[bold cyan]System Status[/bold cyan]\n"
                f"  LLM: [bold green]{info['inner']}[/bold green] "
                f"[dim]\\[{info['outer']}][/dim]  |  "
                f"Modules: [bold]{len(modules)}[/bold]  |  "
                f"Datasets: [bold]{n_cached}[/bold] built-in  |  "
                f"Python: [bold]{sys.version.split()[0]}[/bold]\n\n"
                f"MOIRAIS ships 32 Canadian public health datasets and "
                f"48 statistical analysis commands.\n"
                f"Press [bold yellow]i[/bold yellow] to browse datasets, "
                f"[bold yellow]s[/bold yellow] for stats, "
                f"[bold yellow]c[/bold yellow] to chat with an LLM, or "
                f"[bold yellow]h[/bold yellow] for help."
            )


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------


def launch_tui(*, agent: str | None = None) -> int:
    """Launch the full-screen MOIRAIS terminal IDE."""
    if not _TEXTUAL_AVAILABLE:
        try:
            from rich.console import Console

            Console(stderr=True).print(
                "[red]Textual is required for the TUI.[/red]\nInstall with: [bold]pip install moirais[interactive][/bold]"
            )
        except ImportError:
            print(
                "Textual is required for the TUI.\nInstall with: pip install moirais[interactive]",
                file=sys.stderr,
            )
        return 1

    app = MOIRAISApp()
    app.run(mouse=True)
    return 0
