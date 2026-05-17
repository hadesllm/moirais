"""Thomas cluster process simulation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def thomas_process(kappa, mu, sigma, window, seed=None):
    """Simulate a Thomas (modified) cluster process.

    Parents are Poisson(kappa), offspring are N(parent, sigma^2) with mean mu.

    .. epigraph:: Number rules the universe. -- Pythagoras

    Parameters
    ----------
    kappa : float
        Parent intensity.
    mu : float
        Mean offspring per parent.
    sigma : float
        Offspring dispersion (std dev).
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

    ext = 3 * sigma
    n_parents = rng.poisson(kappa * (xmax - xmin + 2 * ext) * (ymax - ymin + 2 * ext))
    px = rng.uniform(xmin - ext, xmax + ext, n_parents)
    py = rng.uniform(ymin - ext, ymax + ext, n_parents)

    all_x, all_y = [], []
    for i in range(n_parents):
        nc = rng.poisson(mu)
        cx = px[i] + rng.normal(0, sigma, nc)
        cy = py[i] + rng.normal(0, sigma, nc)
        mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
        all_x.extend(cx[mask])
        all_y.extend(cy[mask])

    points = np.column_stack([all_x, all_y]) if all_x else np.empty((0, 2))

    return DescriptiveResult(
        name="thomas_process",
        value=float(len(all_x)),
        extra={
            "points": points,
            "n_points": len(all_x),
            "n_parents": n_parents,
            "expected_intensity": kappa * mu,
        },
    )


sgthm = thomas_process


def cheatsheet() -> str:
    return "thomas_process({}) -> Thomas cluster process simulation."
