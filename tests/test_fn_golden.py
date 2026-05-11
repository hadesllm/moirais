"""Auto-discovering golden-value test for morie.fn.

Reads every JSON in fn/_golden/, calls the corresponding fn, asserts
output matches expected (within tolerance). Stub returns (all-NaN) are
xfail-style: they fail strict mode but pass loose mode.

Strict mode is governed by env var MORIE_GOLDEN_STRICT=1 — set in CI
once enough fn are filled out that drift becomes meaningful.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from morie import eval as meval


def _golden_files() -> list[Path]:
    return meval.discover()


def _ids(files: list[Path]) -> list[str]:
    return [f.stem for f in files]


_files = _golden_files()
STRICT = os.getenv("MORIE_GOLDEN_STRICT", "0") == "1"


@pytest.mark.skipif(not _files, reason="no golden files in fn/_golden/")
@pytest.mark.parametrize("path", _files, ids=_ids(_files))
def test_fn_golden(path: Path) -> None:
    fn, callable_path, source, cases = meval.load_golden_file(path)
    report = meval.run_golden(callable_path, fn, source, cases)
    failures = [r for r in report.results if r.status in {"fail", "error"}]
    if failures:
        msg_lines = [f"{fn}: {len(failures)}/{len(report.results)} cases failed"]
        for r in failures:
            msg_lines.append(
                f"  inputs={r.case.inputs} expected={r.case.expected} "
                f"actual={r.actual} status={r.status} detail={r.detail}"
            )
        pytest.fail("\n".join(msg_lines))
    if report.n_stub > 0:
        if STRICT:
            pytest.fail(
                f"{fn}: {report.n_stub}/{len(report.results)} cases returned NaN-stub "
                f"(MORIE_GOLDEN_STRICT=1, fn must be implemented)"
            )
        pytest.xfail(f"{fn}: {report.n_stub} stub returns (fn not yet implemented)")


def test_eval_module_imports() -> None:
    """Ensure the harness module is importable in CI even with no goldens."""
    assert callable(meval.run_all)
    assert callable(meval.summary)


def test_summary_shape() -> None:
    reports = meval.run_all()
    s = meval.summary(reports)
    assert set(s.keys()) >= {"pass", "fail", "stub", "error", "fns"}
    assert s["fns"] == len(reports)
