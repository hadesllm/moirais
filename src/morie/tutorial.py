"""morie.tutorial — interactive first-time-user walkthrough.

Reached via ``morie tutorial`` on the CLI.  Walks the user through
the same sequence documented in TUTORIAL.md but executes each step
live and prompts before continuing, so a non-coder doesn't have to
copy-paste from a markdown file in another window.

Design notes:

  * Each step prints a short explanation, runs its command(s)
    in-process, then pauses for ``[Enter] continue / [s]kip / [q]uit``.
  * Commands are invoked via subprocess against the same `morie`
    binary that's running this tutorial (sys.executable + -m morie.runner),
    so the user sees the same output they would type by hand.
  * No external state — output lands in ``~/morie-tutorial-<date>/``
    so re-running the tutorial doesn't overwrite a previous attempt.
"""

from __future__ import annotations

import datetime as _dt
import shutil as _shutil
import subprocess
import sys
from pathlib import Path


def _print_step(n: int, title: str, body: str) -> None:
    width = min(_shutil.get_terminal_size((80, 20)).columns, 80)
    print()
    print("=" * width)
    print(f"STEP {n}  —  {title}")
    print("=" * width)
    print(body.strip())
    print()


def _prompt() -> str:
    try:
        ans = input("[Enter] continue · [s] skip · [q] quit: ").strip().lower()
    except EOFError:
        ans = "q"
    return ans


def _run(cmd: list[str]) -> int:
    """Run a morie subcommand in-process, streaming its output."""
    full = [sys.executable, "-m", "morie.runner", *cmd]
    print(f"  $ morie {' '.join(cmd)}")
    print()
    return subprocess.run(full, check=False).returncode


def run() -> int:
    """Entry point.  Returns CLI exit code."""
    out_root = Path.home() / f"morie-tutorial-{_dt.date.today().isoformat()}"
    out_root.mkdir(exist_ok=True)

    print(f"""
Welcome to the morie tutorial.

This walks you through one full analysis from a clean install to
interpretable output.  Every step is a real command, and we'll run
each one for you.  Output lands in:

    {out_root}

You can quit at any prompt (press q) and resume by re-running
`morie tutorial`.  If you've used morie before, you may want
`morie cheatsheet` instead — a one-page command reference.
""")
    if _prompt() == "q":
        return 0

    # Step 1 — list-modules
    _print_step(
        1,
        "What does morie know how to do?",
        "morie ships ~23 analysis modules.  Each has a short description and a "
        "list of output files.  Let's see them:",
    )
    ans = _prompt()
    if ans == "q":
        return 0
    if ans != "s":
        _run(["list-modules"])

    # Step 2 — doctor
    _print_step(
        2,
        "Is everything healthy?",
        "`morie doctor` checks that every dependency is present and reports "
        "what's working.  Useful when something breaks later — you re-run "
        "this to find what's missing.",
    )
    ans = _prompt()
    if ans == "q":
        return 0
    if ans != "s":
        _run(["doctor"])

    # Step 3 — power-design on synthetic data
    out_dir = out_root / "power-design"
    _print_step(
        3,
        "Run a real analysis on the bundled synthetic dataset",
        f"Now we'll run the `power-design` module.  This computes how many "
        f"participants you'd need to survey to detect a given effect.  Output "
        f"lands in:\n\n    {out_dir}\n\n"
        f"You'll see a 'synthetic data' warning — expected, because the "
        f"bundled CPADS frame is a 1,200-row toy file.  When you have the "
        f"real Statistics Canada PUMF, you'd add `--cpads-csv /path/to/real.csv`.",
    )
    ans = _prompt()
    if ans == "q":
        return 0
    if ans != "s":
        _run(["run-module", "power-design", "--output-dir", str(out_dir)])

    # Step 4 — read the output
    _print_step(
        4,
        "What did we just produce?",
        f"The module wrote ~13 CSVs to:\n\n    {out_dir}\n\n"
        f"The one that answers \"how many participants do I need?\" is "
        f"`power_two_proportion_gender.csv`.  Let's peek at it:",
    )
    ans = _prompt()
    if ans == "q":
        return 0
    if ans != "s":
        csv_path = out_dir / "power_two_proportion_gender.csv"
        if csv_path.exists():
            with csv_path.open() as f:
                for _i, line in enumerate(f):
                    if _i >= 8:
                        print("...")
                        break
                    print("  " + line.rstrip())
        else:
            print(f"  (file not produced — did the previous step fail? check {out_dir}/)")

    # Step 5 — pull a real TPS dataset
    _print_step(
        5,
        "Pull real Toronto Police data (one line)",
        "morie ships `morie pull` for the most common Canadian sociolegal "
        "feeds.  Let's grab the layer index — no API URLs to remember.",
    )
    ans = _prompt()
    if ans == "q":
        return 0
    if ans != "s":
        _run(["pull", "tps-layers"])

    # Step 6 — what next
    _print_step(
        6,
        "What to do next",
        "You now have everything you need to:\n\n"
        "  • Run any of the 23 modules:   morie run-module <name>\n"
        "  • Pull any TPS or CPADS feed:  morie pull <name>\n"
        "  • Ask the agent for help:      morie ask \"...\"\n"
        "  • Browse the cheat sheet:      morie cheatsheet\n"
        "  • Read the full tutorial:      cat TUTORIAL.md\n"
        "  • File an issue if stuck:      https://github.com/hadesllm/morie/issues\n\n"
        "Welcome aboard.",
    )
    _prompt()
    return 0
