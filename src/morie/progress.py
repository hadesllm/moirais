"""Rich live-progress wrapper for MORIE pipeline execution.

Provides :class:`PipelineTracker` which wraps :func:`morie.modules.run_module`
with ``rich.progress`` bars, live status tables, and output-file validation.
Falls back to plain ``print()`` statements when stdout is not a TTY (pipes,
CI runners, redirected output).

Usage
-----
Programmatic::

    from morie.progress import PipelineTracker
    tracker = PipelineTracker(["power-design", "logistic-models"], cpads_csv="data.csv")
    results = tracker.run()

CLI (automatic when TTY detected)::

    morie pipeline --all -y   # rich progress bars in interactive terminals
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

from .modules import DEFAULT_CPADS_CSV, MODULE_SPECS, list_modules, run_module

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ModuleResult:
    """Outcome of a single module execution."""

    name: str
    status: str = "pending"  # pending | running | success | error | skipped
    elapsed_seconds: float = 0.0
    output_files_expected: int = 0
    output_files_actual: int = 0
    error_message: str | None = None
    outputs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline tracker
# ---------------------------------------------------------------------------


class PipelineTracker:
    """Execute analysis modules with live progress display.

    Parameters
    ----------
    module_names : list[str]
        Module names to execute (must be keys in ``MODULE_SPECS``).
    cpads_csv : str
        Path to the CPADS CSV input file.
    output_dir : str | None
        Directory for CSV outputs.  ``None`` uses the module default.
    track_carbon : bool
        If True and codecarbon is installed, track CO2 emissions.
    use_live : bool | None
        Force live display on/off.  ``None`` auto-detects from TTY.
    """

    def __init__(
        self,
        module_names: list[str],
        *,
        cpads_csv: str = DEFAULT_CPADS_CSV,
        output_dir: str | None = None,
        track_carbon: bool = True,
        use_live: bool | None = None,
    ) -> None:
        self.module_names = module_names
        self.cpads_csv = cpads_csv
        self.output_dir = output_dir
        self.track_carbon = track_carbon
        self.use_live = use_live if use_live is not None else sys.stdout.isatty()
        self.results: list[ModuleResult] = []

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def run(self) -> list[ModuleResult]:
        """Execute all modules, returning a list of :class:`ModuleResult`."""
        self.results = [
            ModuleResult(
                name=name,
                output_files_expected=len(MODULE_SPECS[name].output_files) if name in MODULE_SPECS else 0,
            )
            for name in self.module_names
        ]

        tracker = self._start_carbon_tracker()

        if self.use_live:
            self._run_with_rich()
        else:
            self._run_plain()

        self._stop_carbon_tracker(tracker)
        return self.results

    def run_single(self, module_name: str) -> ModuleResult:
        """Run one module with optional spinner."""
        result = ModuleResult(
            name=module_name,
            output_files_expected=len(MODULE_SPECS[module_name].output_files) if module_name in MODULE_SPECS else 0,
        )

        if self.use_live:
            self._run_single_rich(result)
        else:
            self._run_single_plain(result)

        return result

    # ------------------------------------------------------------------
    # Rich live display
    # ------------------------------------------------------------------

    def _run_with_rich(self) -> None:
        from rich import box
        from rich.console import Console
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        from rich.table import Table

        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        overall_task = progress.add_task("Pipeline", total=len(self.results))

        def _build_status_table() -> Table:
            table = Table(
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold cyan",
                expand=True,
            )
            table.add_column("Module", style="bold", min_width=30)
            table.add_column("Status", width=10, justify="center")
            table.add_column("Time", width=8, justify="right")
            table.add_column("Files", width=10, justify="center")

            for r in self.results:
                if r.status == "success":
                    status = "[green]OK[/green]"
                elif r.status == "error":
                    status = "[red]FAIL[/red]"
                elif r.status == "running":
                    status = "[yellow]...[/yellow]"
                elif r.status == "skipped":
                    status = "[dim]SKIP[/dim]"
                else:
                    status = "[dim]--[/dim]"

                elapsed = f"{r.elapsed_seconds:.1f}s" if r.elapsed_seconds > 0 else ""
                files = f"{r.output_files_actual}/{r.output_files_expected}" if r.status in ("success", "error") else ""
                table.add_row(r.name, status, elapsed, files)
            return table

        from rich.console import Group

        with Live(
            Group(progress, _build_status_table()),
            console=console,
            refresh_per_second=4,
        ) as live:
            for idx, result in enumerate(self.results):
                result.status = "running"
                progress.update(overall_task, description=f"[{idx + 1}/{len(self.results)}] {result.name}")
                live.update(Group(progress, _build_status_table()))

                self._execute_module(result)

                progress.update(overall_task, advance=1)
                live.update(Group(progress, _build_status_table()))

        console.print()
        self._print_summary_rich(console)

    def _run_single_rich(self, result: ModuleResult) -> None:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress.add_task(result.name, total=None)
            self._execute_module(result)

        if result.status == "success":
            console.print(f"[green]OK[/green] {result.name} ({result.elapsed_seconds:.1f}s)")
        else:
            console.print(f"[red]FAIL[/red] {result.name}: {result.error_message}")

    # ------------------------------------------------------------------
    # Plain text fallback
    # ------------------------------------------------------------------

    def _run_plain(self) -> None:
        total = len(self.results)
        for idx, result in enumerate(self.results, start=1):
            result.status = "running"
            print(f"[{idx}/{total}] Running module: {result.name}", flush=True)
            self._execute_module(result)
            if result.status == "success":
                print(
                    f"[{idx}/{total}] Finished module: {result.name} "
                    f"({result.elapsed_seconds:.1f}s, "
                    f"{result.output_files_actual}/{result.output_files_expected} files)",
                    flush=True,
                )
            else:
                print(
                    f"[{idx}/{total}] FAILED module: {result.name}: {result.error_message}",
                    flush=True,
                )

        print()
        self._print_summary_plain()

    def _run_single_plain(self, result: ModuleResult) -> None:
        print(f"Running module: {result.name}", flush=True)
        self._execute_module(result)
        if result.status == "success":
            print(f"Finished: {result.name} ({result.elapsed_seconds:.1f}s)")
        else:
            print(f"FAILED: {result.name}: {result.error_message}")

    # ------------------------------------------------------------------
    # Module execution (shared logic)
    # ------------------------------------------------------------------

    def _execute_module(self, result: ModuleResult) -> None:
        """Run a single module and populate the result."""
        t0 = time.monotonic()
        try:
            outputs = run_module(
                result.name,
                cpads_csv=self.cpads_csv,
                output_dir=self.output_dir,
            )
            result.outputs = outputs
            result.status = "success"
            result.output_files_actual = len(outputs)
        except Exception as exc:
            result.status = "error"
            result.error_message = str(exc)
        finally:
            result.elapsed_seconds = time.monotonic() - t0

    # ------------------------------------------------------------------
    # Carbon tracking
    # ------------------------------------------------------------------

    def _start_carbon_tracker(self):
        if not self.track_carbon:
            return None
        try:
            from morie.emissions import EmissionsTracker

            emissions_dir = (
                os.path.join(self.output_dir, "emissions") if self.output_dir else "data/manifest/outputs/emissions"
            )
            os.makedirs(emissions_dir, exist_ok=True)
            tracker = EmissionsTracker(
                project_name="morie-pipeline",
                output_dir=emissions_dir,
                log_level="error",
            )
            tracker.start()
            return tracker
        except (ImportError, Exception):
            return None

    def _stop_carbon_tracker(self, tracker) -> None:
        if tracker is None:
            return
        try:
            emissions = tracker.stop()
            if emissions is not None:
                msg = f"Pipeline CO2 emissions: {emissions:.6f} kg CO2eq"
                if self.use_live:
                    from rich.console import Console

                    Console().print(f"[dim]{msg}[/dim]")
                else:
                    print(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------

    def summary_table(self):
        """Build a ``rich.table.Table`` summarising all results."""
        from rich import box
        from rich.table import Table

        table = Table(
            title="Pipeline Summary",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Module", style="bold", min_width=30)
        table.add_column("Status", width=10, justify="center")
        table.add_column("Time", width=8, justify="right")
        table.add_column("Files", width=10, justify="center")

        for r in self.results:
            if r.status == "success":
                status = "[green]OK[/green]"
            elif r.status == "error":
                status = "[red]FAIL[/red]"
            elif r.status == "skipped":
                status = "[dim]SKIP[/dim]"
            else:
                status = "[dim]--[/dim]"

            elapsed = f"{r.elapsed_seconds:.1f}s" if r.elapsed_seconds > 0 else ""
            files = f"{r.output_files_actual}/{r.output_files_expected}"
            table.add_row(r.name, status, elapsed, files)

        return table

    def _print_summary_rich(self, console=None) -> None:
        if console is None:
            from rich.console import Console

            console = Console()

        console.print(self.summary_table())

        succeeded = sum(1 for r in self.results if r.status == "success")
        failed = sum(1 for r in self.results if r.status == "error")
        total = len(self.results)
        total_time = sum(r.elapsed_seconds for r in self.results)

        if failed == 0:
            console.print(f"[green]Pipeline completed: {succeeded}/{total} modules in {total_time:.1f}s[/green]")
        else:
            console.print(
                f"[red]Pipeline finished with errors: {succeeded} OK, {failed} FAILED ({total_time:.1f}s)[/red]"
            )

    def _print_summary_plain(self) -> None:
        succeeded = sum(1 for r in self.results if r.status == "success")
        failed = sum(1 for r in self.results if r.status == "error")
        total = len(self.results)
        total_time = sum(r.elapsed_seconds for r in self.results)

        print(f"Pipeline completed: {succeeded}/{total} modules in {total_time:.1f}s")
        if failed > 0:
            print(f"  {failed} module(s) failed:")
            for r in self.results:
                if r.status == "error":
                    print(f"    - {r.name}: {r.error_message}")


# ---------------------------------------------------------------------------
# Drop-in replacement for runner.execute_pipeline
# ---------------------------------------------------------------------------


def execute_pipeline_with_progress(
    modules: list[str] | None = None,
    *,
    cpads_csv: str = DEFAULT_CPADS_CSV,
    dataset_key: str | None = None,
    output_dir: str | None = None,
    silent: bool = False,
    track_carbon: bool = True,
) -> int:
    """Run the MORIE pipeline with rich progress display.

    Drop-in replacement for :func:`morie.runner.execute_pipeline`.  Auto-detects
    TTY for rich vs plain output.

    Parameters
    ----------
    modules : list[str] | None
        Module names to run.  ``None`` runs all implemented modules.
    cpads_csv : str
        Path to the CPADS CSV.
    output_dir : str | None
        Output directory for CSVs.
    silent : bool
        Skip confirmation prompt.
    track_carbon : bool
        Track CO2 emissions via CodeCarbon.

    Returns
    -------
    int
        ``0`` on success, ``1`` on abort or failure.
    """
    selected = modules or [item["name"] for item in list_modules()]

    if not silent:
        print("Selected modules:", ", ".join(selected))
        try:
            confirm = input(f"Run {len(selected)} modules? [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            print("\nPipeline aborted.")
            return 1
        if confirm.lower() != "y":
            print("Pipeline aborted.")
            return 1

    tracker = PipelineTracker(
        selected,
        cpads_csv=cpads_csv,
        output_dir=output_dir,
        track_carbon=track_carbon,
    )
    results = tracker.run()

    failed = sum(1 for r in results if r.status == "error")
    return 1 if failed > 0 else 0
