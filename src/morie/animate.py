# SPDX-License-Identifier: GPL-2.0-only
"""Rich-based animated UI helpers for long MORIE callables.

Pairs with the DoubleML-style training-output experience: live spinners,
progress bars, and streaming per-iteration tables. Wired into the slow
callables (Kulldorff MC permutations, longitudinal simulator subject
loop, DML cross-fit folds) via an opt-in `animate=True` kwarg.

Falls back to plain `print()` if `rich` is not installed -- the helpers
are non-fatal decoration, never a hard dependency.

Public API:
    animated_bar(total, description) – context-manager Progress with
        spinner + bar + percentage + elapsed/remaining time columns.
    streaming_table(columns)         – rich.live.Live wrapper around
        a per-iteration table that updates in place.
    morie_banner()                   – one-shot ASCII banner for CLI
        entrypoints (used by morie.demo).

Demo:
    python -m morie.demo

"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterable


try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
    )
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


__all__ = [
    "animated_bar",
    "streaming_table",
    "morie_banner",
    "rich_available",
]


def rich_available() -> bool:
    """True if rich is importable and animated output is possible."""
    return _HAS_RICH


@contextmanager
def animated_bar(
    total: int,
    description: str = "morie",
    *,
    transient: bool = False,
    spinner: str = "dots",
):
    """Yield a progress handle with `.advance()` method.

    Rich version emits a live-updating spinner + bar + time columns.
    Fallback version prints periodic dots.

    Example:
        >>> with animated_bar(199, "Kulldorff MC perms") as bar:
        ...     for _ in range(199):
        ...         do_one_permutation()
        ...         bar.advance()
    """
    if _HAS_RICH:
        cols = [
            SpinnerColumn(style="cyan", spinner_name=spinner),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ]
        with Progress(*cols, transient=transient) as p:
            task_id = p.add_task(description, total=total)

            class _Handle:
                def advance(self, n: int = 1):
                    p.advance(task_id, n)

            yield _Handle()
    else:
        threshold = max(1, total // 20)
        counter = [0]

        class _Handle:
            def advance(self, n: int = 1):
                counter[0] += n
                if counter[0] % threshold == 0:
                    sys.stderr.write(".")
                    sys.stderr.flush()

        sys.stderr.write(f"{description} ")
        yield _Handle()
        sys.stderr.write(" done\n")


@contextmanager
def streaming_table(columns: Iterable[str], *, title: str = ""):
    """Yield a callable that appends a row to a live-updating table.

    Example:
        >>> with streaming_table(["iter", "log_LRT", "best"]) as add:
        ...     for i in range(199):
        ...         lrt = scan_permutation(i)
        ...         add(i, f"{lrt:.2f}", f"{best:.2f}")
    """
    cols = list(columns)
    if _HAS_RICH:
        table = Table(title=title or None)
        for c in cols:
            table.add_column(c)
        with Live(table, refresh_per_second=10):
            def _add(*row):
                table.add_row(*[str(x) for x in row])
            yield _add
    else:
        print(" | ".join(cols))
        def _add(*row):
            print(" | ".join(str(x) for x in row))
        yield _add


def morie_banner() -> None:
    """Print a one-shot MORIE banner at the start of a CLI session."""
    if _HAS_RICH:
        Console().print(Panel.fit(
            "[bold cyan]MORIE[/bold cyan] -- Multi-domain Open Research\n"
            "and Inferential Estimation\n"
            "[dim]v0.2.0 · GPL-2.0-only · https://github.com/hadesllm/morie[/dim]",
            border_style="cyan",
        ))
    else:
        print("MORIE -- Multi-domain Open Research and Inferential Estimation")
        print("v0.2.0 · GPL-2.0-only · https://github.com/hadesllm/morie")
