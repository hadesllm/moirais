"""Cox (doubly stochastic) process simulation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def cox_process(intensity_field, window, seed=None):
    """Simulate a log-Gaussian Cox process.

    .. epigraph:: "Medallion's humming." -- Geralt, The Witcher

    Parameters
    ----------
    intensity_field : array_like
        2D grid of intensity values, shape ``(ny, nx)``.
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    seed : int, optional
        Random seed.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    lam = np.asarray(intensity_field, dtype=np.float64)
    ny, nx = lam.shape
    xmin, xmax, ymin, ymax = window
    rng = np.random.default_rng(seed)

    cell_w = (xmax - xmin) / nx
    cell_h = (ymax - ymin) / ny

    all_x, all_y = [], []
    for iy in range(ny):
        for ix in range(nx):
            n_pts = rng.poisson(max(0, lam[iy, ix]) * cell_w * cell_h)
            if n_pts > 0:
                px = rng.uniform(xmin + ix * cell_w, xmin + (ix + 1) * cell_w, n_pts)
                py = rng.uniform(ymin + iy * cell_h, ymin + (iy + 1) * cell_h, n_pts)
                all_x.extend(px)
                all_y.extend(py)

    points = np.column_stack([all_x, all_y]) if all_x else np.empty((0, 2))

    return DescriptiveResult(
        name="cox_process",
        value=float(len(all_x)),
        extra={
            "points": points,
            "n_points": len(all_x),
            "mean_intensity": float(lam.mean()),
        },
    )


sgcox = cox_process


def cheatsheet() -> str:
    return "cox_process({}) -> Cox (doubly stochastic) process simulation."
