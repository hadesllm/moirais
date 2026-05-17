"""MORIE environment self-diagnostics.

``morie doctor`` checks that all required components are installed and
reachable before the user runs analysis modules.  Each check prints a
pass/fail row; the overall exit code is 0 only if all required checks pass.

Uses :mod:`rich` for formatted terminal output when available, with a plain
text fallback for minimal environments.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Individual check functions -- each returns (bool, str) = (passed, message)
# ---------------------------------------------------------------------------


def _check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    label = f"{v.major}.{v.minor}.{v.micro}"
    return ok, label if ok else f"{label} (need >= 3.10)"


def _check_import(package: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, "__version__", "?")
        return True, version
    except ImportError:
        return False, "not installed"


def _check_r() -> tuple[bool, str]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False, "Rscript not found (optional)"
    try:
        out = subprocess.check_output(["Rscript", "--version"], stderr=subprocess.STDOUT, timeout=5).decode().strip()
        # "R scripting front-end version 4.4.1 (2024-06-14)"
        version = out.split("version")[-1].strip().split()[0] if "version" in out else out
        return True, version
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False, "Rscript found but failed to run"


def _check_ollama() -> tuple[bool, str]:
    try:
        from .loc import LocalOllama

        client = LocalOllama()
        if not client.is_running():
            return False, f"not reachable at {client.base_url} (optional)"
        models = client.list_models()
        if not models:
            return True, "running (no models pulled)"
        labels = [f"{m.name} ({m.quantization})" if m.quantization else m.name for m in models[:3]]
        return True, ", ".join(labels)
    except Exception:
        url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return False, f"not reachable at {url} (optional)"


def _check_gemini() -> tuple[bool, str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return True, f"key set, model={model}"
    return False, "GEMINI_API_KEY not set (optional)"


def _check_openai_compat() -> tuple[bool, str]:
    base = os.environ.get("LLM_API_BASE_URL", "").strip()
    key = os.environ.get("LLM_API_KEY", "").strip()
    if base and key:
        return True, base
    return False, "LLM_API_BASE_URL / LLM_API_KEY not set (optional)"


def _check_openai() -> tuple[bool, str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return (True, "key set") if key else (False, "OPENAI_API_KEY not set (optional)")


def _check_freeapi() -> tuple[bool, str]:
    """Check OllamaFreeAPI SDK -- free LLM access, no API key needed."""
    try:
        from .fam import OllamaFreeAPI

        client = OllamaFreeAPI()
        models = client.list_models()
        names = [m if isinstance(m, str) else str(m) for m in models[:5]]
        return True, f"{len(models)} models ({', '.join(names)}...)"
    except ImportError:
        return False, "morie.fam not available"
    except Exception as e:
        return False, f"error: {e}"


def _check_datasets() -> tuple[bool, str]:
    """Check built-in MORIE datasets database."""
    try:
        from .data import morie_db, list_datasets

        db_path = morie_db()
        if not db_path.exists():
            return False, "morie.db not found -- reinstall morie"
        size_mb = db_path.stat().st_size // (1024 * 1024)
        ds = list_datasets()
        cached = [d for d in ds if d["cached"]]
        return True, f"{len(cached)}/{len(ds)} datasets ({size_mb}MB built-in DB)"
    except Exception as e:
        return False, f"error: {e}"


def _check_docker() -> tuple[bool, str]:
    docker = shutil.which("docker")
    if docker is None:
        return False, "docker not found (optional)"
    try:
        out = subprocess.check_output(["docker", "--version"], stderr=subprocess.STDOUT, timeout=5).decode().strip()
        return True, out
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False, "docker found but not running (optional)"


def _check_morie_version() -> tuple[bool, str]:
    """Check whether a newer morie release is available on PyPI.

    Uses the daily-cached result from the import-time update check, so
    this never makes a network call itself.
    """
    try:
        import morie

        from ._update_check import _parse_version, _read_cache

        installed = getattr(morie, "__version__", "0.0.0+unknown")
        latest = _read_cache().get("latest")
        if not latest:
            return True, f"{installed} (latest not yet checked)"
        if _parse_version(latest) > _parse_version(installed):
            return False, f"{installed} -- {latest} available; run `morie update`"
        return True, f"{installed} (up to date)"
    except Exception as exc:  # noqa: BLE001
        return True, f"version check unavailable ({exc})"


# ---------------------------------------------------------------------------
# Master check list
# ---------------------------------------------------------------------------

_REQUIRED_IMPORTS = [
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "statsmodels",
    "httpx",
    "rich",
]

_OPTIONAL_IMPORTS = ["doubleml", "openai", "textual"]


def run_checks() -> dict[str, Any]:
    """Run all diagnostics and return a structured results dict."""
    results: dict[str, Any] = {"checks": [], "all_required_passed": True}

    def _add(label: str, passed: bool, detail: str, required: bool = True) -> None:
        results["checks"].append({"label": label, "passed": passed, "detail": detail, "required": required})
        if required and not passed:
            results["all_required_passed"] = False

    # Python
    ok, detail = _check_python_version()
    _add("Python version", ok, detail, required=True)

    # morie itself -- newer release available?
    ok, detail = _check_morie_version()
    _add("morie version", ok, detail, required=False)

    # Required Python packages
    for pkg in _REQUIRED_IMPORTS:
        ok, detail = _check_import(pkg)
        _add(f"import {pkg}", ok, detail, required=True)

    # Optional Python packages
    for pkg in _OPTIONAL_IMPORTS:
        ok, detail = _check_import(pkg)
        _add(f"import {pkg}", ok, detail, required=False)

    # R
    ok, detail = _check_r()
    _add("R (Rscript)", ok, detail, required=False)

    # LLM providers
    ok, detail = _check_freeapi()
    _add("OllamaFreeAPI", ok, detail, required=False)

    ok, detail = _check_ollama()
    _add("Ollama (local)", ok, detail, required=False)

    ok, detail = _check_gemini()
    _add("Gemini API key", ok, detail, required=False)

    ok, detail = _check_openai_compat()
    _add("OpenAI-compat API", ok, detail, required=False)

    ok, detail = _check_openai()
    _add("OpenAI API key", ok, detail, required=False)

    # Data
    ok, detail = _check_datasets()
    _add("Built-in datasets", ok, detail, required=False)

    # Infrastructure
    ok, detail = _check_docker()
    _add("Docker", ok, detail, required=False)

    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_plain(results: dict[str, Any]) -> None:
    print("MORIE Doctor -- environment diagnostics")
    print("=" * 50)
    for check in results["checks"]:
        status = "OK " if check["passed"] else ("FAIL" if check["required"] else "WARN")
        print(f"  [{status}] {check['label']:<30} {check['detail']}")
    print()
    if results["all_required_passed"]:
        print("All required checks passed.")
    else:
        print("Some required checks failed. See FAIL rows above.")


def _render_rich(results: dict[str, Any]) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title="MORIE Doctor -- environment diagnostics",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Status", width=6, justify="center")
    table.add_column("Check", style="bold", min_width=28)
    table.add_column("Detail")

    for check in results["checks"]:
        if check["passed"]:
            status = "[green]  OK [/green]"
        elif check["required"]:
            status = "[red] FAIL[/red]"
        else:
            status = "[yellow] WARN[/yellow]"
        table.add_row(status, check["label"], check["detail"])

    console.print(table)

    if results["all_required_passed"]:
        console.print("[green]All required checks passed.[/green]")
    else:
        console.print("[red]Some required checks failed. See FAIL rows above.[/red]")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


# import-name -> pip-install-name, where the two differ
_PIP_NAME = {"sklearn": "scikit-learn"}


def _render(results: dict[str, Any]) -> None:
    """Render the results table, picking rich or plain output.

    Respects ``NO_COLOR`` (https://no-color.org) and falls back to the
    plain renderer when stdout isn't a TTY, so screen readers and
    pipelines get a clean linear stream instead of box-drawing.
    """
    no_color = bool(os.environ.get("NO_COLOR"))
    not_tty = not sys.stdout.isatty()
    if no_color or not_tty:
        _render_plain(results)
    else:
        try:
            _render_rich(results)
        except ImportError:
            _render_plain(results)


def _heal(results: dict[str, Any]) -> bool:
    """Attempt to remediate failed checks (``morie doctor --fix``).

    Auto-fixable: missing Python packages (``pip install``) and the
    morie cache directory.  Everything else gets an actionable hint
    rather than a silent failure.  Returns True if anything was fixed.
    """
    print()
    print("Healing -- attempting to fix failed checks ...")

    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME")
        or os.path.join(os.path.expanduser("~"), ".cache"),
        "morie")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"  [ok]   cache directory ready: {cache_dir}")
    except OSError as exc:
        print(f"  [fail] could not create {cache_dir}: {exc}")

    fixed_any = False
    for check in results["checks"]:
        if check["passed"]:
            continue
        label = check["label"]
        if label.startswith("import "):
            pkg = label[len("import "):]
            pip_name = _PIP_NAME.get(pkg, pkg)
            print(f"  ...    installing {pip_name} via pip ...")
            rc = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name]
            ).returncode
            print(f"  [{'ok' if rc == 0 else 'fail'}]   pip install {pip_name}")
            fixed_any = fixed_any or rc == 0
        elif label == "morie version":
            print("  [hint] run `morie update` to upgrade morie itself.")
        elif label == "Built-in datasets":
            print("  [hint] reinstall to restore the built-in DB: "
                  "pip install --force-reinstall morie")
        elif label == "R (Rscript)":
            print("  [hint] install R from https://www.r-project.org/ "
                  "(optional -- only the R bridge needs it).")
        else:
            print(f"  [hint] {label}: optional component -- configure it "
                  "only if you need that feature.")

    if fixed_any:
        importlib.invalidate_caches()
    return fixed_any


def run_doctor(fix: bool = False) -> int:
    """Run diagnostics and print a summary table.

    Parameters
    ----------
    fix : bool
        When True (``morie doctor --fix``), attempt to remediate failed
        checks -- install missing Python packages, create the cache
        directory -- then re-run and re-render the diagnostics.

    Returns
    -------
    int
        ``0`` if all required checks pass, ``1`` otherwise.
    """
    results = run_checks()
    _render(results)

    if fix:
        healed = _heal(results)
        if healed:
            print()
            print("Re-running diagnostics after fixes ...")
        results = run_checks()
        _render(results)

    return 0 if results["all_required_passed"] else 1
