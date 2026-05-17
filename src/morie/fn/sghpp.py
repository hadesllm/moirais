"""Homogeneous Poisson process simulation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def homogeneous_poisson(n, window, seed=None):
    """Simulate a homogeneous Poisson point process.

    .. epigraph:: Give me a place to stand and I will move the earth. -- Archimedes

    Parameters
    ----------
    n : int
        Expected number of points (or exact if int).
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    seed : int, optional
        Random seed.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    xmin, xmax, ymin, ymax = window
    area = (xmax - xmin) * (ymax - ymin)
    rng = np.random.default_rng(seed)

    n_pts = rng.poisson(n) if isinstance(n, (int, float)) else int(n)
    x = rng.uniform(xmin, xmax, n_pts)
    y = rng.uniform(ymin, ymax, n_pts)
    points = np.column_stack([x, y])

    return DescriptiveResult(
        name="homogeneous_poisson",
        value=float(n_pts),
        extra={
            "points": points,
            "n_points": n_pts,
            "intensity": n_pts / area if area > 0 else 0.0,
            "window": window,
        },
    )


sghpp = homogeneous_poisson


def cheatsheet() -> str:
    return "homogeneous_poisson({}) -> Homogeneous Poisson process simulation."
