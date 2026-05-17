"""Matern cluster process simulation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def matern_cluster_process(parent_intensity, n_offspring, radius, window, seed=None):
    """Simulate a Matern cluster point process.

    .. epigraph:: It does not matter how slowly you go as long as you do not stop. -- Confucius

    Parameters
    ----------
    parent_intensity : float
        Intensity of parent Poisson process.
    n_offspring : int
        Mean number of offspring per parent.
    radius : float
        Cluster radius.
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

    ext = radius
    n_parents = rng.poisson(parent_intensity * (area + 4 * ext * (xmax - xmin + ymax - ymin)))
    px = rng.uniform(xmin - ext, xmax + ext, n_parents)
    py = rng.uniform(ymin - ext, ymax + ext, n_parents)

    all_x, all_y = [], []
    for i in range(n_parents):
        nc = rng.poisson(n_offspring)
        angles = rng.uniform(0, 2 * np.pi, nc)
        radii = radius * np.sqrt(rng.uniform(0, 1, nc))
        cx = px[i] + radii * np.cos(angles)
        cy = py[i] + radii * np.sin(angles)
        mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
        all_x.extend(cx[mask])
        all_y.extend(cy[mask])

    points = np.column_stack([all_x, all_y]) if all_x else np.empty((0, 2))

    return DescriptiveResult(
        name="matern_cluster_process",
        value=float(len(all_x)),
        extra={
            "points": points,
            "n_points": len(all_x),
            "n_parents": n_parents,
            "intensity": len(all_x) / area if area > 0 else 0.0,
        },
    )


sgmtc = matern_cluster_process


def cheatsheet() -> str:
    return "matern_cluster_process({}) -> Matern cluster process simulation."
