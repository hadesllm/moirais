# morie.fn — function file (hadesllm/morie)
"""Bias-reduced kernel density via geometric extrapolation."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kbias"]


def kbias(
    data: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bw: float | None = None,
    n_grid: int = 512,
) -> dict:
    r"""
    Bias-reduced kernel density estimator (geometric extrapolation).

    Uses two KDE passes at bandwidths *h* and *h/c* to cancel the leading
    bias term via:

    .. math::

        \hat{f}_{\mathrm{BR}}(x) =
        \hat{f}_{h/c}(x)^{c^2/(c^2 - 1)} \;/\;
        \hat{f}_{h}(x)^{1/(c^2 - 1)}

    with c = sqrt(2) following Jones & Foster (1993).

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations.
    x_eval : np.ndarray or None
        Grid of evaluation points.
    bw : float or None
        Bandwidth for the wider kernel. Silverman's rule if None.
    n_grid : int
        Grid size when *x_eval* is None.

    Returns
    -------
    dict
        ``x_eval``, ``density``, ``bw``.

    References
    ----------
    Jones, M. C. & Foster, P. J. (1993). Generalized jackknifing and higher
        order kernel smoothing. *Journal of Nonparametric Statistics*, 3, 81-94.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    sigma = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    if bw is None:
        bw = max(0.9 * s * n ** (-0.2), 1e-10)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    if x_eval is None:
        lo = data.min() - 3 * bw
        hi = data.max() + 3 * bw
        x_eval = np.linspace(lo, hi, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    def _kde(h):
        u = (x_eval[:, None] - data[None, :]) / h
        k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        return k.mean(axis=1) / h

    c = np.sqrt(2.0)
    f_h = _kde(bw)
    f_hc = _kde(bw / c)

    eps = 1e-300
    f_h = np.maximum(f_h, eps)
    f_hc = np.maximum(f_hc, eps)

    alpha = c ** 2 / (c ** 2 - 1)
    beta = 1.0 / (c ** 2 - 1)
    density = np.exp(alpha * np.log(f_hc) - beta * np.log(f_h))

    return RichResult(payload={"x_eval": x_eval, "density": density, "bw": bw})


def cheatsheet() -> str:
    return "kbias({data}) -> Bias-reduced KDE via geometric extrapolation."
