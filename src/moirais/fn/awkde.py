# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Adaptive bandwidth kernel density estimator."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["awkde"]


def awkde(
    data: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bw: float | None = None,
    alpha: float = 0.5,
    n_grid: int = 512,
) -> dict:
    r"""
    Adaptive (variable) bandwidth kernel density estimator.

    Uses Abramson's square-root law to set local bandwidths:

    .. math::

        h_i = h \left[\frac{\tilde{f}(X_i)}{g}\right]^{-\alpha}

    where :math:`\tilde{f}` is a pilot density estimate and
    :math:`g = \exp\!\bigl(\frac{1}{n}\sum\log\tilde{f}(X_i)\bigr)`
    is the geometric mean.

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations.
    x_eval : np.ndarray or None
        Evaluation grid.
    bw : float or None
        Global bandwidth. Silverman's rule if None.
    alpha : float
        Sensitivity parameter in [0, 1]. Default 0.5 (Abramson).
    n_grid : int
        Grid size when *x_eval* is None.

    Returns
    -------
    dict
        ``x_eval``, ``density``, ``bw``, ``alpha``.

    References
    ----------
    Abramson, I. S. (1982). On bandwidth variation in kernel estimates — a
        square root law. *Annals of Statistics*, 10(4), 1217-1223.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

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

    u_pilot = (data[:, None] - data[None, :]) / bw
    pilot_at_data = np.exp(-0.5 * u_pilot ** 2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    pilot_at_data = np.maximum(pilot_at_data, 1e-300)

    log_g = np.mean(np.log(pilot_at_data))
    g = np.exp(log_g)
    lambdas = (pilot_at_data / g) ** (-alpha)
    local_bw = bw * lambdas

    density = np.zeros(len(x_eval))
    for i in range(n):
        hi = local_bw[i]
        u = (x_eval - data[i]) / hi
        density += np.exp(-0.5 * u ** 2) / (hi * np.sqrt(2 * np.pi))
    density /= n

    return RichResult(payload={"x_eval": x_eval, "density": density, "bw": bw, "alpha": alpha})


def cheatsheet() -> str:
    return "awkde({data}) -> Adaptive bandwidth KDE (Abramson square-root law)."
