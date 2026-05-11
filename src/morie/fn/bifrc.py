# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bifurcation diagram data for the logistic map."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bifurcation_data(
    r_range: tuple[float, float] = (2.5, 4.0),
    *,
    n_r: int = 200,
    n_iter: int = 300,
    n_last: int = 100,
    x0: float = 0.5,
) -> DescriptiveResult:
    """Generate bifurcation diagram data for the logistic map.

    Parameters
    ----------
    r_range : (r_min, r_max)
    n_r : int
        Number of r values to sample.
    n_iter : int
        Total iterations per r.
    n_last : int
        Tail iterations to keep (attractors).
    x0 : float
        Initial condition.

    Returns
    -------
    DescriptiveResult
    """
    rs = np.linspace(r_range[0], r_range[1], n_r)
    r_vals = []
    x_vals = []

    for r in rs:
        x = x0
        for _ in range(n_iter - n_last):
            x = r * x * (1 - x)
        for _ in range(n_last):
            x = r * x * (1 - x)
            r_vals.append(r)
            x_vals.append(x)

    feigenbaum_r1 = 3.0
    feigenbaum_r2 = 3.44949
    feigenbaum_rinf = 3.56995

    return DescriptiveResult(
        name="bifurcation",
        value=float(len(r_vals)),
        extra={
            "n_points": len(r_vals),
            "r_range": list(r_range),
            "n_r": n_r,
            "feigenbaum_r1": feigenbaum_r1,
            "feigenbaum_r2": feigenbaum_r2,
            "feigenbaum_r_inf": feigenbaum_rinf,
        },
    )


bifrc = bifurcation_data


def cheatsheet() -> str:
    return "bifurcation_data({}) -> Bifurcation diagram data for the logistic map."
