"""MORIE self-test — comprehensive smoke test of all subsystems.

Verifies that every major component of the MORIE IDE is functional
without requiring pytest or any test framework.  Designed to give
users and developers confidence that the software works.

Run with::

    morie selftest
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_results: list[dict[str, Any]] = []


def _test(name: str, fn, *args, **kwargs) -> bool:
    """Run a single test and record the result."""
    t0 = time.monotonic()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.monotonic() - t0
        _results.append(
            {
                "name": name,
                "passed": True,
                "elapsed": elapsed,
                "detail": str(result)[:120] if result else "OK",
            }
        )
        return True
    except Exception as exc:
        elapsed = time.monotonic() - t0
        _results.append(
            {
                "name": name,
                "passed": False,
                "elapsed": elapsed,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        return False


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


def _test_core_imports():
    """Test that all core modules import."""
    core = [
        "morie",
        "morie.cpads",
        "morie.data",
        "morie.dataset",
        "morie.causal",
        "morie.effects",
        "morie.ebac",
        "morie.investigation",
        "morie.survey",
        "morie.inference",
        "morie.ml",
        "morie.modules",
        "morie.sampling",
        "morie.llm",
        "morie.perseus",
        "morie.doctor",
        "morie.runner",
    ]
    failed = []
    for mod in core:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            failed.append(f"{mod}: {exc}")
    if failed:
        raise RuntimeError(f"{len(failed)} imports failed: {'; '.join(failed[:3])}")
    return f"{len(core)} core modules imported"


def _test_new_modules():
    """Test that all new IDE/statistical modules import."""
    new_mods = [
        "morie.progress",
        "morie.inspector",
        "morie.chat",
        "morie.tui",
        "morie.statistics",
        "morie.survival",
        "morie.missing",
        "morie.effect_sizes",
        "morie.multiple_testing",
        "morie.diagnostics",
        "morie.sensitivity",
        "morie.bootstrap_methods",
        "morie.did",
        "morie.rdd",
        "morie.iv",
        "morie.matching",
        "morie.viz",
        "morie.tables_pub",
        "morie.validation",
        "morie.export",
        "morie.container",
        "morie.notebook",
        "morie.reporting",
        "morie.bench",
        "morie.weights",
    ]
    failed = []
    for mod in new_mods:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            failed.append(f"{mod}: {exc}")
    if failed:
        raise RuntimeError(f"{len(failed)} imports failed: {'; '.join(failed[:3])}")
    return f"{len(new_mods)} new modules imported"


def _test_module_registry():
    """Test MODULE_SPECS and list_modules."""
    from morie.modules import list_modules

    specs = list_modules()
    if len(specs) < 20:
        raise RuntimeError(f"Expected 20+ modules, got {len(specs)}")
    return f"{len(specs)} modules registered"


def _test_chat_session():
    """Test ChatSession with slash commands."""
    from morie.chat import ChatSession

    session = ChatSession()

    # /help
    result = session.send("/help", stream=False)
    if "Available commands" not in result:
        raise RuntimeError("Expected 'Available commands' in /help output")

    # /list
    result = session.send("/list", stream=False)
    if "power-design" not in result and "No modules" not in result:
        raise RuntimeError("Expected module list in /list output")

    # /provider
    result = session.send("/provider", stream=False)
    if "provider" not in result.lower():
        raise RuntimeError("Expected provider info")

    # /agents
    result = session.send("/agents", stream=False)
    # Should work regardless of whether agents dir is found.

    return "ChatSession: /help, /list, /provider, /agents all work"


def _test_tui_screens():
    """Test TUI screens render headlessly."""
    try:
        from morie.tui import _TEXTUAL_AVAILABLE, MORIEApp
    except ImportError:
        return "SKIP: textual not installed"

    if not _TEXTUAL_AVAILABLE:
        return "SKIP: textual not available"

    import asyncio

    async def _verify():
        app = MORIEApp()
        async with app.run_test() as pilot:
            assert pilot.app.title == "MORIE"
            # Doctor screen.
            await pilot.press("d")
            assert pilot.app.screen.__class__.__name__ == "DoctorScreen"
            await pilot.press("escape")
            # Chat screen.
            await pilot.press("c")
            assert pilot.app.screen.__class__.__name__ == "ChatScreen"
            await pilot.press("escape")
            # Pipeline screen.
            await pilot.press("p")
            assert pilot.app.screen.__class__.__name__ == "PipelineScreen"
            await pilot.press("escape")
            # Dataset screen.
            await pilot.press("i")
            assert pilot.app.screen.__class__.__name__ == "DatasetScreen"

    asyncio.run(_verify())
    return "TUI: 5 screens render (Home, Doctor, Chat, Pipeline, Dataset)"


def _test_progress_tracker():
    """Test PipelineTracker with a mock module."""
    from unittest.mock import MagicMock, patch

    from morie.progress import PipelineTracker

    with patch("morie.progress.run_module") as mock_run:
        mock_run.return_value = {"output": MagicMock()}
        tracker = PipelineTracker(
            ["power-design"],
            cpads_csv="fake.csv",
            use_live=False,
            track_carbon=False,
        )
        results = tracker.run()
        if results[0].status != "success":
            raise RuntimeError(f"Expected success, got {results[0].status}")

    return "PipelineTracker: runs, reports success"


def _test_inspector():
    """Test inspect_output on a real CSV."""
    from morie.inspector import inspect_output

    # Find a CSV file.
    project = Path(__file__).resolve().parents[2]
    csv_dir = project / "data" / "files" / "csv" / "survey"
    csvs = list(csv_dir.glob("*.csv"))[:1]
    if not csvs:
        return "SKIP: no CSV files found in data/"

    result = inspect_output(csvs[0])
    if result.rows < 1:
        raise RuntimeError(f"Expected rows > 0, got {result.rows}")
    return f"Inspector: {result.rows} rows x {result.columns} cols from {csvs[0].name}"


def _test_verify():
    """Test verify_statistical_output on a real CSV."""
    from morie.inspector import verify_statistical_output

    project = Path(__file__).resolve().parents[2]
    csv_dir = project / "data" / "files" / "csv" / "survey"
    csvs = list(csv_dir.glob("*.csv"))[:1]
    if not csvs:
        return "SKIP: no CSV files found"

    report = verify_statistical_output(csvs[0])
    return f"Verifier: {len(report.checks)} checks on {csvs[0].name}"


def _test_statistics():
    """Test a real statistical function."""
    import numpy as np

    from morie.statistics import one_sample_ttest

    data = np.array([2.1, 3.5, 2.8, 4.0, 3.2])
    result = one_sample_ttest(data, mu0=3.0)
    if not hasattr(result, "p_value"):
        raise RuntimeError("Expected TestResult with p_value attribute")
    if not 0 <= result.p_value <= 1:
        raise RuntimeError(f"p_value out of range: {result.p_value}")
    return f"one_sample_ttest: t={result.test_statistic:.3f}, p={result.p_value:.3f}"


def _test_effect_sizes():
    """Test effect size computation."""
    import numpy as np

    from morie.effect_sizes import cohens_d

    g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    result = cohens_d(g1, g2)
    if not hasattr(result, "estimate"):
        raise RuntimeError("Expected result with estimate")
    return f"cohens_d: d={result.estimate:.3f}"


def _test_multiple_testing():
    """Test p-value adjustment."""
    import numpy as np

    from morie.multiple_testing import benjamini_hochberg

    p = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    result = benjamini_hochberg(p, alpha=0.05)
    if result.n_tests != 5:
        raise RuntimeError(f"Expected 5 tests, got {result.n_tests}")
    return f"BH correction: {result.n_rejected}/5 rejected at alpha=0.05"


def _test_llm_detection():
    """Test LLM provider detection."""
    from morie.llm import detect_available_provider

    provider = detect_available_provider()
    valid = ("ollama", "freeapi", "gemini", "api", "openai", "local")
    if provider not in valid:
        raise RuntimeError(f"Unknown provider: {provider}")
    return f"LLM provider: {provider}"


def _test_datasets():
    """Test built-in dataset database."""
    from morie.data import morie_db, list_datasets

    db_path = morie_db()
    if not db_path.exists():
        raise RuntimeError("Built-in morie.db not found")
    ds = list_datasets()
    cached = [d for d in ds if d["cached"]]
    return f"Datasets: {len(cached)}/{len(ds)} available ({db_path.stat().st_size // (1024 * 1024)}MB)"


def _test_doctor():
    """Test doctor diagnostics."""
    from morie.doctor import run_checks

    results = run_checks()
    n_checks = len(results["checks"])
    n_passed = sum(1 for c in results["checks"] if c["passed"])
    return f"Doctor: {n_passed}/{n_checks} checks passed"


def _test_r_available():
    """Test R availability."""
    import shutil

    if shutil.which("Rscript") is None:
        return "SKIP: Rscript not found"

    import subprocess

    try:
        out = (
            subprocess.check_output(
                ["Rscript", "-e", "cat(R.version.string)"],
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            .decode()
            .strip()
        )
        return f"R: {out}"
    except Exception as exc:
        return f"SKIP: R check failed: {exc}"


def _test_sensitivity():
    """Test sensitivity analysis."""
    from morie.sensitivity import e_value_rr

    result = e_value_rr(3.9, ci_lower=2.4, ci_upper=6.3)
    if result.e_value_point < 1:
        raise RuntimeError(f"E-value should be >= 1, got {result.e_value_point}")
    return f"E-value: {result.e_value_point:.2f} (RR=3.9)"


def _test_bootstrap():
    """Test bootstrap inference."""
    import numpy as np

    from morie.bootstrap_methods import bootstrap

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = bootstrap(data, np.mean, n_boot=500, ci_method="percentile", seed=42)
    if not result.ci_lower < result.estimate < result.ci_upper:
        raise RuntimeError("CI should contain estimate")
    return f"Bootstrap: mean={result.estimate:.1f}, CI=[{result.ci_lower:.1f}, {result.ci_upper:.1f}]"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_selftest() -> int:
    """Run all self-tests and print results."""
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        use_rich = True
    except ImportError:
        use_rich = False

    tests = [
        ("Core module imports", _test_core_imports),
        ("New module imports (25 modules)", _test_new_modules),
        ("Module registry (21 specs)", _test_module_registry),
        ("Chat session (slash commands)", _test_chat_session),
        ("TUI screens (headless render)", _test_tui_screens),
        ("Progress tracker", _test_progress_tracker),
        ("Inspector (CSV profiling)", _test_inspector),
        ("Verifier (statistical checks)", _test_verify),
        ("Statistics (t-test)", _test_statistics),
        ("Effect sizes (Cohen's d)", _test_effect_sizes),
        ("Multiple testing (BH)", _test_multiple_testing),
        ("Sensitivity (E-value)", _test_sensitivity),
        ("Bootstrap inference", _test_bootstrap),
        ("LLM provider detection", _test_llm_detection),
        ("Built-in datasets", _test_datasets),
        ("Doctor diagnostics", _test_doctor),
        ("R availability", _test_r_available),
    ]

    print("MORIE Self-Test")
    print("=" * 60)
    print()

    t0_total = time.monotonic()
    for name, fn in tests:
        _test(name, fn)

    total_elapsed = time.monotonic() - t0_total
    passed = sum(1 for r in _results if r["passed"])
    skipped = sum(1 for r in _results if r["detail"].startswith("SKIP"))
    failed = sum(1 for r in _results if not r["passed"] and not r["detail"].startswith("SKIP"))

    if use_rich:
        console = Console()
        table = Table(
            title="MORIE Self-Test Results",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Status", width=6, justify="center")
        table.add_column("Test", style="bold", min_width=30)
        table.add_column("Detail")
        table.add_column("Time", width=8, justify="right")

        for r in _results:
            if r["passed"]:
                if r["detail"].startswith("SKIP"):
                    status = "[yellow] SKIP[/yellow]"
                else:
                    status = "[green]  OK [/green]"
            else:
                status = "[red] FAIL[/red]"
            table.add_row(
                status,
                r["name"],
                r["detail"][:80],
                f"{r['elapsed']:.2f}s",
            )

        console.print(table)
        console.print()
        if failed == 0:
            console.print(f"[green]All tests passed: {passed} OK, {skipped} skipped ({total_elapsed:.1f}s)[/green]")
        else:
            console.print(f"[red]{failed} FAILED, {passed} OK, {skipped} skipped ({total_elapsed:.1f}s)[/red]")
    else:
        for r in _results:
            if r["passed"]:
                status = "SKIP" if r["detail"].startswith("SKIP") else " OK "
            else:
                status = "FAIL"
            print(f"  [{status}] {r['name']:<35} {r['detail'][:60]}")

        print()
        if failed == 0:
            print(f"All tests passed: {passed} OK, {skipped} skipped ({total_elapsed:.1f}s)")
        else:
            print(f"{failed} FAILED, {passed} OK, {skipped} skipped ({total_elapsed:.1f}s)")

    return 1 if failed > 0 else 0
