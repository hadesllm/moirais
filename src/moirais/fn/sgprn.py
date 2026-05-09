"""Practical (effective) range estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def practical_range(model, params, threshold=0.95):
    """Compute the practical range where gamma reaches threshold * sill.

    .. epigraph:: "Git Gud." -- Dark Souls community

    Parameters
    ----------
    model : str
        ``'spherical'``, ``'exponential'``, ``'gaussian'``.
    params : dict
        Must contain ``'nugget'``, ``'sill'``, ``'range'``.
    threshold : float
        Fraction of sill (default 0.95).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    nugget = params["nugget"]
    sill = params["sill"]
    a = params["range"]

    target = nugget + threshold * (sill - nugget)

    if model == "spherical":
        pr = a
    elif model == "exponential":
        pr = -a * np.log(1.0 - threshold)
    elif model == "gaussian":
        pr = a * np.sqrt(-np.log(1.0 - threshold))
    else:
        pr = a

    return DescriptiveResult(
        name="practical_range",
        value=float(pr),
        extra={
            "practical_range": float(pr),
            "model": model,
            "theoretical_range": float(a),
            "threshold": threshold,
            "target_gamma": float(target),
        },
    )


sgprn = practical_range


def cheatsheet() -> str:
    return "practical_range({}) -> Practical (effective) range estimation."
