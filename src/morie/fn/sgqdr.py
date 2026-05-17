"""Quadrat count test for point patterns."""

from __future__ import annotations

from ._containers import DescriptiveResult


def quadrat_count_test(points, window, nx=5, ny=5, cdf=None):
    """Chi-squared quadrat count test for CSR.

    .. epigraph:: Measure what is measurable, and make measurable what is not. -- Galileo Galilei

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    nx, ny : int
        Number of quadrats in x and y.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy import stats

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    xmin, xmax, ymin, ymax = window

    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)

    counts = np.zeros((ny, nx), dtype=int)
    for k in range(n):
        ix = min(int((pts[k, 0] - xmin) / (xmax - xmin) * nx), nx - 1)
        iy = min(int((pts[k, 1] - ymin) / (ymax - ymin) * ny), ny - 1)
        counts[iy, ix] += 1

    expected = n / (nx * ny)
    chi2 = float(np.sum((counts - expected) ** 2 / expected)) if expected > 0 else 0.0
    df = nx * ny - 1
    p_value = 1.0 - stats.chi2.cdf(chi2, df)

    return DescriptiveResult(
        name="quadrat_count_test",
        value=chi2,
        extra={
            "chi2": chi2,
            "p_value": float(p_value),
            "df": df,
            "counts": counts,
            "expected": expected,
            "VMR": float(np.var(counts.ravel()) / expected) if expected > 0 else 0.0,
        },
    )


sgqdr = quadrat_count_test


def cheatsheet() -> str:
    return "quadrat_count_test({}) -> Quadrat count test for point patterns."
