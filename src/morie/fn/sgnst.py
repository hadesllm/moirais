"""Nested (composite) variogram model."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nested_variogram(h, models):
    """Evaluate a nested variogram as a sum of component models.

    .. epigraph:: Mathematics is the art of giving the same name to different things. -- Henri Poincare

    Parameters
    ----------
    h : array_like
        Lag distances.
    models : list of dict
        Each dict has keys ``'model'``, ``'nugget'``, ``'sill'``, ``'range'``
        (and optionally ``'nu'`` for Matern).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    h = np.asarray(h, dtype=np.float64)
    gamma_total = np.zeros_like(h)

    for m in models:
        kind = m["model"]
        c0 = m.get("nugget", 0.0)
        c = m.get("sill", 1.0)
        a = m.get("range", 1.0)

        if kind == "spherical":
            g = np.where(
                h <= a,
                c0 + (c - c0) * (1.5 * h / a - 0.5 * (h / a) ** 3),
                c,
            )
        elif kind == "exponential":
            g = c0 + (c - c0) * (1.0 - np.exp(-h / a))
        elif kind == "gaussian":
            g = c0 + (c - c0) * (1.0 - np.exp(-((h / a) ** 2)))
        else:
            g = c0 + (c - c0) * (1.0 - np.exp(-h / a))

        g = np.where(h == 0, 0.0, g)
        gamma_total += g

    return DescriptiveResult(
        name="nested_variogram",
        value=float(gamma_total.max()) if len(gamma_total) > 0 else 0.0,
        extra={
            "gamma": gamma_total.tolist(),
            "n_components": len(models),
            "models": [m["model"] for m in models],
        },
    )


sgnst = nested_variogram


def cheatsheet() -> str:
    return "nested_variogram({}) -> Nested (composite) variogram model."
