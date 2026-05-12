"""Numerical correctness eval harness for morie.fn.

Goal: gate every fn behind a golden test before PyPI release. Without a
gate, the fn/ tree balloons faster than correctness can be verified, and
the package risks shipping silent-NaN stubs.

Source-of-truth references:
    - scipy.stats / scipy.special / scipy.linalg
    - statsmodels
    - R packages (frozen JSON values, since R is not always installed in CI)

Design (v0):
    - GoldenCase: one input/expected pair with tolerances + provenance.
    - JSON files at fn/_golden/<fn_name>.json hold the cases (data, not code).
    - run_golden(fn, name) -> list[CaseResult] runs everything and reports.
    - Stub detection (returns NaN-only) is reported as "stub" not "fail" so
      we can xfail unfilled fn cleanly while gating filled ones.

Adding a golden:
    1. Implement (or stub-out) fn/<name>.py.
    2. Generate reference values from scipy/statsmodels/R, save to
       fn/_golden/<name>.json (schema below).
    3. tests/pytests/test_fn_golden.py auto-discovers + asserts.

JSON schema:
    {
      "fn":        "<short name>",
      "callable":  "<python attribute path to call, eg 'morie.fn.dnorm.dnorm'>",
      "source":    "<reference impl, eg 'scipy.stats.norm.pdf'>",
      "cases": [
        {"inputs": {...kw}, "expected": <scalar|list|dict>,
         "rtol": 1e-7, "atol": 0, "notes": ""}
      ]
    }
"""
from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

GOLDEN_DIR = Path(__file__).parent / "fn" / "_golden"


@dataclass(frozen=True)
class GoldenCase:
    inputs: dict[str, Any]
    expected: Any
    source: str
    rtol: float = 1e-5
    atol: float = 1e-8
    notes: str = ""


@dataclass
class CaseResult:
    case: GoldenCase
    actual: Any
    status: str  # "pass" | "fail" | "stub" | "error"
    error: float | None = None
    detail: str = ""


@dataclass
class FnReport:
    fn_name: str
    callable_path: str
    source: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r.status == "fail")

    @property
    def n_stub(self) -> int:
        return sum(1 for r in self.results if r.status == "stub")

    @property
    def n_error(self) -> int:
        return sum(1 for r in self.results if r.status == "error")


def _load_callable(path: str) -> Callable:
    mod_path, attr = path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def load_golden_file(path: Path) -> tuple[str, str, str, list[GoldenCase]]:
    raw = json.loads(path.read_text())
    cases = [
        GoldenCase(
            inputs=c["inputs"],
            expected=c["expected"],
            source=c.get("source", raw.get("source", "")),
            rtol=c.get("rtol", 1e-5),
            atol=c.get("atol", 1e-8),
            notes=c.get("notes", ""),
        )
        for c in raw["cases"]
    ]
    return raw["fn"], raw["callable"], raw.get("source", ""), cases


def _is_stub(actual: Any) -> bool:
    """A return is a stub if every numeric leaf is NaN."""
    if actual is None:
        return True
    if isinstance(actual, dict):
        if not actual:
            return False
        return all(_is_stub(v) for v in actual.values())
    if isinstance(actual, (list, tuple)):
        if not actual:
            return False
        return all(_is_stub(v) for v in actual)
    try:
        f = float(actual)
        return math.isnan(f)
    except (TypeError, ValueError):
        return False


def _compare(actual: Any, expected: Any, rtol: float, atol: float) -> tuple[bool, float | None, str]:
    """Recursive close-comparison for scalar/list/tuple/dict."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, None, f"type mismatch: expected dict, got {type(actual).__name__}"
        max_err = 0.0
        for k, ev in expected.items():
            av = actual.get(k)
            ok, err, detail = _compare(av, ev, rtol, atol)
            if not ok:
                return False, err, f"key '{k}': {detail}"
            if err is not None and err > max_err:
                max_err = err
        return True, max_err, ""
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            try:
                actual = list(actual)
            except TypeError:
                return False, None, f"type mismatch: expected list, got {type(actual).__name__}"
        if len(actual) != len(expected):
            return False, None, f"length mismatch: actual={len(actual)} expected={len(expected)}"
        max_err = 0.0
        for i, (a, e) in enumerate(zip(actual, expected)):
            ok, err, detail = _compare(a, e, rtol, atol)
            if not ok:
                return False, err, f"idx {i}: {detail}"
            if err is not None and err > max_err:
                max_err = err
        return True, max_err, ""
    # scalar
    try:
        a = float(actual)
        e = float(expected)
    except (TypeError, ValueError):
        return actual == expected, None, "non-numeric equality" if actual != expected else ""
    if math.isnan(e):
        return math.isnan(a), None, "expected NaN"
    if math.isinf(e):
        return math.isinf(a) and (math.copysign(1, a) == math.copysign(1, e)), None, "expected inf"
    err = abs(a - e)
    tol = atol + rtol * abs(e)
    return err <= tol, err, ""


def run_golden(callable_path: str, fn_name: str, source: str, cases: Iterable[GoldenCase]) -> FnReport:
    """Run all golden cases for one fn and produce a report."""
    report = FnReport(fn_name=fn_name, callable_path=callable_path, source=source)
    try:
        fn = _load_callable(callable_path)
    except (ImportError, AttributeError) as e:
        for case in cases:
            report.results.append(
                CaseResult(case=case, actual=None, status="error", detail=f"import: {e}")
            )
        return report
    for case in cases:
        try:
            actual = fn(**case.inputs)
        except Exception as e:  # noqa: BLE001 -- eval gate must capture all
            report.results.append(
                CaseResult(case=case, actual=None, status="error", detail=f"{type(e).__name__}: {e}")
            )
            continue
        if _is_stub(actual):
            report.results.append(
                CaseResult(case=case, actual=actual, status="stub", detail="all-NaN return")
            )
            continue
        ok, err, detail = _compare(actual, case.expected, case.rtol, case.atol)
        report.results.append(
            CaseResult(
                case=case,
                actual=actual,
                status="pass" if ok else "fail",
                error=err,
                detail=detail,
            )
        )
    return report


def discover() -> list[Path]:
    """Return all golden JSON files in fn/_golden/."""
    if not GOLDEN_DIR.exists():
        return []
    return sorted(GOLDEN_DIR.glob("*.json"))


def run_all() -> list[FnReport]:
    """Run every golden file in fn/_golden/. Used by CI + the test harness."""
    reports: list[FnReport] = []
    for path in discover():
        fn, callable_path, source, cases = load_golden_file(path)
        reports.append(run_golden(callable_path, fn, source, cases))
    return reports


def summary(reports: Iterable[FnReport]) -> dict[str, int]:
    """Aggregate pass/fail/stub/error counts across reports."""
    s = {"pass": 0, "fail": 0, "stub": 0, "error": 0, "fns": 0}
    for r in reports:
        s["fns"] += 1
        s["pass"] += r.n_pass
        s["fail"] += r.n_fail
        s["stub"] += r.n_stub
        s["error"] += r.n_error
    return s
