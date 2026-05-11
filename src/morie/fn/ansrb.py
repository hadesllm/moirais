# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Ansari-Bradley test for scale."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def ansari_bradley(x: np.ndarray, y: np.ndarray) -> TestResult:
    """Ansari-Bradley two-sample test for equal scale.

    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Each sample needs >= 2 observations.")

    stat, p = sp_stats.ansari(x, y)

    return TestResult(
        test_name="Ansari-Bradley",
        statistic=float(stat),
        p_value=float(p),
        method="Ansari-Bradley",
        n=len(x) + len(y),
        extra={"n_x": len(x), "n_y": len(y)},
    )


ansrb = ansari_bradley


def cheatsheet() -> str:
    return "ansari_bradley({}) -> Ansari-Bradley test for scale."
