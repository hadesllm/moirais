# morie.fn — function file (hadesllm/morie)
"""Threshold autoregressive (TAR) model (Tong 1990)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["threshold_autoregression"]


def threshold_autoregression(x, p=1, d=1, n_grid=50):
    r"""Two-regime SETAR(p, d).

    .. math::

        y_t = \begin{cases}
            \phi_0^{(1)} + \sum_{i=1}^p \phi_i^{(1)} y_{t-i} + \epsilon_t
              & y_{t-d}\le c,\\
            \phi_0^{(2)} + \sum_{i=1}^p \phi_i^{(2)} y_{t-i} + \epsilon_t
              & y_{t-d}>c.
        \end{cases}

    Threshold ``c`` is selected by grid search minimising pooled SSE.

    Parameters
    ----------
    x : array-like
        Univariate series.
    p : int, default 1
        AR order in each regime.
    d : int, default 1
        Delay parameter for the threshold variable.
    n_grid : int, default 50
        Number of candidate thresholds (quantile grid).

    Returns
    -------
    RichResult
        keys: ``threshold``, ``phi_lower``, ``phi_upper``, ``p``, ``d``,
        ``regime_sizes``, ``sse``, ``n``, ``method``.

    References
    ----------
    Tong H (1990). *Non-Linear Time Series: A Dynamical System Approach*.
    Oxford UP.
    """
    y = np.asarray(x, dtype=float).ravel()
    n = y.size
    start = max(p, d)
    if n - start < 4 * (p + 1):
        raise ValueError(
            f"Series too short for SETAR(p={p}, d={d}); have {n}.")
    Y = y[start:]
    X = np.column_stack(
        [np.ones(Y.size)] + [y[start - i : n - i] for i in range(1, p + 1)]
    )
    Z = y[start - d : n - d]  # threshold variable
    q_lo, q_hi = np.quantile(Z, [0.15, 0.85])
    grid = np.linspace(q_lo, q_hi, n_grid)
    best = (np.inf, np.nan, None, None, None)
    for c in grid:
        mask_lo = Z <= c
        mask_hi = ~mask_lo
        if mask_lo.sum() < 2 * (p + 1) or mask_hi.sum() < 2 * (p + 1):
            continue
        phi_lo, *_ = np.linalg.lstsq(X[mask_lo], Y[mask_lo], rcond=None)
        phi_hi, *_ = np.linalg.lstsq(X[mask_hi], Y[mask_hi], rcond=None)
        res_lo = Y[mask_lo] - X[mask_lo] @ phi_lo
        res_hi = Y[mask_hi] - X[mask_hi] @ phi_hi
        sse = float(np.sum(res_lo ** 2) + np.sum(res_hi ** 2))
        if sse < best[0]:
            best = (sse, c, phi_lo, phi_hi, (int(mask_lo.sum()), int(mask_hi.sum())))
    sse, c, phi_lo, phi_hi, sizes = best
    if phi_lo is None:
        raise ValueError("Could not find admissible threshold grid point.")
    return RichResult(payload={
        "threshold": float(c),
        "phi_lower": np.asarray(phi_lo),
        "phi_upper": np.asarray(phi_hi),
        "p": int(p), "d": int(d),
        "regime_sizes": sizes,
        "sse": float(sse),
        "n": int(n),
        "method": f"SETAR(p={p}, d={d}) via grid-search OLS",
    })


def cheatsheet():
    return "tarmd: Threshold autoregression / SETAR (Tong 1990)."
