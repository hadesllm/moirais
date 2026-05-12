# morie.fn -- function file (hadesllm/morie)
"""LOESS/LOWESS smoother. 'Rebellions are built on hope.' -- Jyn Erso"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def loess_smooth(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.3,
) -> DescriptiveResult:
    """
    Locally weighted scatterplot smoothing (LOWESS).

    Implements a simple LOWESS using tricube weights and local linear
    regression at each point.

    :param x: Predictor array (1-D).
    :param y: Response array (1-D), same length as *x*.
    :param frac: Fraction of data used in each local fit. Default 0.3.
    :return: DescriptiveResult with smoothed values in extra.
    :raises ValueError: If inputs are invalid.

    References
    ----------
    Cleveland, W. S. (1979). Robust locally weighted regression and
    smoothing scatterplots. Journal of the American Statistical
    Association, 74(368), 829--836. doi:10.1080/01621459.1979.10481038
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("x and y must be 1-D arrays of equal length >= 3.")
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}.")

    n = len(x)
    k = max(int(np.ceil(frac * n)), 3)
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    def _tricube(d: np.ndarray) -> np.ndarray:
        d = np.clip(d, 0.0, 1.0)
        return (1.0 - d**3) ** 3

    fitted = np.empty(n)
    for i in range(n):
        dists = np.abs(xs - xs[i])
        idx = np.argsort(dists)[:k]
        max_dist = dists[idx[-1]]
        if max_dist == 0.0:
            max_dist = 1.0
        w = _tricube(dists[idx] / max_dist)
        xw = xs[idx]
        yw = ys[idx]
        sw = np.sum(w)
        swx = np.sum(w * xw)
        swx2 = np.sum(w * xw**2)
        swy = np.sum(w * yw)
        swxy = np.sum(w * xw * yw)
        denom = sw * swx2 - swx**2
        if abs(denom) < 1e-15:
            fitted[i] = swy / sw if sw > 0 else ys[i]
        else:
            b1 = (sw * swxy - swx * swy) / denom
            b0 = (swy - b1 * swx) / sw
            fitted[i] = b0 + b1 * xs[i]

    residual = ys - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))

    return DescriptiveResult(
        name="LOWESS Smooth",
        value=rmse,
        extra={
            "x_sorted": xs,
            "fitted": fitted,
            "residuals": residual,
            "rmse": rmse,
            "frac": frac,
            "n": n,
        },
    )


lssmo = loess_smooth


def cheatsheet() -> str:
    return "loess_smooth({}) -> LOESS/LOWESS smoother. 'Rebellions are built on hope.' -- Jy"
