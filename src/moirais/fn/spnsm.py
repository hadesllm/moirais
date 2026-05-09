"""Smoothing spline. 'The belonging you seek is ahead of you.' -- Maz Kanata"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline

from ._containers import DescriptiveResult


def spline_smooth(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
    s: float | None = None,
) -> DescriptiveResult:
    """
    Fit a smoothing spline to *(x, y)* data.

    Uses :class:`scipy.interpolate.UnivariateSpline` internally.

    :param x: Predictor array (1-D), must be strictly increasing after sort.
    :param y: Response array (1-D), same length as *x*.
    :param k: Degree of the spline (1--5). Default 3 (cubic).
    :param s: Smoothing factor. If None, uses ``len(x)`` (moderate).
    :return: DescriptiveResult with fitted values in extra.
    :raises ValueError: If inputs are invalid.

    References
    ----------
    de Boor, C. (1978). A Practical Guide to Splines. Springer.
    Reinsch, C. H. (1967). Smoothing by spline functions. Numerische
    Mathematik, 10(3), 177--183. doi:10.1007/BF02162161
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < k + 1:
        raise ValueError(f"x and y must be 1-D with length >= {k + 1}.")

    order = np.argsort(x)
    xs, ys = x[order], y[order]

    if s is None:
        s = float(len(x))

    spl = UnivariateSpline(xs, ys, k=k, s=s)
    fitted = spl(xs)
    residual = ys - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))

    return DescriptiveResult(
        name="Smoothing Spline",
        value=rmse,
        extra={
            "x_sorted": xs,
            "fitted": fitted,
            "residuals": residual,
            "rmse": rmse,
            "degree": k,
            "smoothing_factor": s,
            "n_knots": len(spl.get_knots()),
            "n": len(x),
        },
    )


spnsm = spline_smooth


def cheatsheet() -> str:
    return "spline_smooth({}) -> Smoothing spline. 'The belonging you seek is ahead of you.' "
