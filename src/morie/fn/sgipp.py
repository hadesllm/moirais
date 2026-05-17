"""Inhomogeneous Poisson process simulation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def inhomogeneous_poisson(intensity_fn, window, max_intensity=None, seed=None):
    """Simulate an inhomogeneous Poisson process via thinning.

    .. epigraph:: In the midst of chaos, there is also opportunity. -- Sun Tzu

    Parameters
    ----------
    intensity_fn : callable
        ``intensity_fn(x, y) -> float`` giving local intensity.
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    max_intensity : float, optional
        Upper bound on intensity_fn. Estimated if None.
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

    if max_intensity is None:
        gx = np.linspace(xmin, xmax, 50)
        gy = np.linspace(ymin, ymax, 50)
        gxx, gyy = np.meshgrid(gx, gy)
        max_intensity = max(intensity_fn(gxx.ravel(), gyy.ravel()))

    n_propose = rng.poisson(max_intensity * area)
    x_prop = rng.uniform(xmin, xmax, n_propose)
    y_prop = rng.uniform(ymin, ymax, n_propose)

    intensities = np.array([intensity_fn(x_prop[i], y_prop[i]) for i in range(n_propose)])
    probs = intensities / max_intensity
    keep = rng.uniform(0, 1, n_propose) < probs

    points = np.column_stack([x_prop[keep], y_prop[keep]])

    return DescriptiveResult(
        name="inhomogeneous_poisson",
        value=float(points.shape[0]),
        extra={
            "points": points,
            "n_points": points.shape[0],
            "n_proposed": n_propose,
            "acceptance_rate": float(keep.mean()),
        },
    )


sgipp = inhomogeneous_poisson


def cheatsheet() -> str:
    return "inhomogeneous_poisson({}) -> Inhomogeneous Poisson process simulation."
