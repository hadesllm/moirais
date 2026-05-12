# morie.fn -- function file (hadesllm/morie)
"""Quadrat test for Complete Spatial Randomness (CSR)."""

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def quadrat_test(points: np.ndarray, n_quadrats: int = 4, cdf=None) -> TestResult:
    """
    Quadrat test for Complete Spatial Randomness.

    Divides the study area into a grid and tests whether point counts
    follow a uniform (Poisson) distribution using a chi-squared test.

    :param points: (n, 2) array of coordinates.
    :param n_quadrats: Number of divisions per axis (total = n_quadrats^2).
    :return: TestResult with chi-squared statistic and p-value.
    :raises ValueError: If n_quadrats < 2.

    References
    ----------
    Cressie NAC (1993). Statistics for Spatial Data. Revised ed.
    Wiley-Interscience.
    """
    if n_quadrats < 2:
        raise ValueError("n_quadrats must be >= 2.")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (n, 2).")
    n = pts.shape[0]
    x_edges = np.linspace(pts[:, 0].min(), pts[:, 0].max() + 1e-10, n_quadrats + 1)
    y_edges = np.linspace(pts[:, 1].min(), pts[:, 1].max() + 1e-10, n_quadrats + 1)
    counts = np.zeros((n_quadrats, n_quadrats), dtype=int)
    x_idx = np.digitize(pts[:, 0], x_edges) - 1
    y_idx = np.digitize(pts[:, 1], y_edges) - 1
    x_idx = np.clip(x_idx, 0, n_quadrats - 1)
    y_idx = np.clip(y_idx, 0, n_quadrats - 1)
    for xi, yi in zip(x_idx, y_idx):
        counts[xi, yi] += 1
    observed = counts.ravel().astype(float)
    expected = np.full_like(observed, n / len(observed))
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    df = len(observed) - 1
    pval = float(1 - sp_stats.chi2.cdf(chi2, df))
    return TestResult(
        test_name="quadrat_test",
        statistic=chi2,
        p_value=pval,
        df=float(df),
        method="chi-squared quadrat",
        n=n,
        extra={"counts": counts, "n_quadrats": n_quadrats},
    )


quadr = quadrat_test


def cheatsheet() -> str:
    return "quadrat_test({}) -> Quadrat test for Complete Spatial Randomness (CSR)."
