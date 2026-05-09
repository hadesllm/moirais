# moirais.fn — function file (hadesllm/moirais)
"""Gaussian KDE. 'Patience is bitter, but its fruit is sweet. — Aristotle' -- General"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kde(
    x: np.ndarray,
    bw: str | float = "silverman",
    n_grid: int = 200,
) -> DescriptiveResult:
    """Gaussian kernel density estimation.

    Parameters
    ----------
    x : ndarray
        1D sample.
    bw : str or float, default "silverman"
        Bandwidth: "silverman", "scott", or a positive float.
    n_grid : int, default 200
        Number of evaluation grid points.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``grid`` and ``density`` arrays.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 data points")
    std = float(np.std(x, ddof=1))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std

    if isinstance(bw, str):
        if bw == "silverman":
            h = 0.9 * sigma * n ** (-1 / 5)
        elif bw == "scott":
            h = 1.06 * sigma * n ** (-1 / 5)
        else:
            raise ValueError(f"Unknown bandwidth rule: {bw}")
    else:
        h = float(bw)
    if h <= 0:
        h = 1.0

    lo = x.min() - 3 * h
    hi = x.max() + 3 * h
    grid = np.linspace(lo, hi, n_grid)
    density = np.zeros(n_grid)
    for xi in x:
        density += np.exp(-0.5 * ((grid - xi) / h) ** 2)
    density /= n * h * np.sqrt(2 * np.pi)

    return DescriptiveResult(
        name="Gaussian KDE",
        value=float(grid[np.argmax(density)]),
        extra={"grid": grid, "density": density, "bandwidth": h, "n": n},
    )


kern = kde


def cheatsheet() -> str:
    return "kde({}) -> Gaussian KDE. 'Patience is bitter, but its fruit is sweet. — Aristotle' -- General"
