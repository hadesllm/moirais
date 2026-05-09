"""Trimmed weights for causal inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _validate_df


def trimmed_weights(
    data: pd.DataFrame,
    *,
    w: str = "weight",
    method: str = "quantile",
    lower: float = 0.01,
    upper: float = 0.99,
    threshold: float | None = None,
) -> np.ndarray:
    r"""Trim extreme IPW/balancing weights to reduce variance.

    Three methods:

    - ``'quantile'``: clip at the ``lower`` and ``upper`` quantiles.
    - ``'fixed'``: clip at ``threshold`` (symmetric around 1).
    - ``'crump'``: discard units with PS outside [alpha, 1-alpha] where
      alpha is chosen to minimize variance (Crump et al., 2009).

    .. math::

        w_i^{trim} = \max\left(q_\ell,\;\min(w_i, q_u)\right)

    Parameters
    ----------
    data : pd.DataFrame
    w : str
        Weight column name.
    method : str
        One of 'quantile', 'fixed', 'crump'.
    lower, upper : float
        Quantile bounds (for 'quantile' method).
    threshold : float or None
        Fixed threshold (for 'fixed' method).

    Returns
    -------
    np.ndarray
        Trimmed weights.

    References
    ----------
    Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009).
    Dealing with limited overlap in estimation of average treatment effects.
    *Biometrika*, 96(1), 187-199.
    """
    _validate_df(data, w)
    weights = data[w].to_numpy(dtype=float).copy()

    if method == "quantile":
        q_lo = float(np.nanquantile(weights, lower))
        q_hi = float(np.nanquantile(weights, upper))
        weights = np.clip(weights, q_lo, q_hi)
    elif method == "fixed":
        if threshold is None:
            threshold = 10.0
        weights = np.clip(weights, 1.0 / threshold, threshold)
    elif method == "crump":
        if "ps" not in data.columns:
            raise ValueError("Crump trimming requires a 'ps' column")
        ps = data["ps"].to_numpy(dtype=float)
        alpha_opt = _crump_alpha(ps)
        mask = (ps >= alpha_opt) & (ps <= 1 - alpha_opt)
        weights[~mask] = 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    return weights


def _crump_alpha(ps: np.ndarray) -> float:
    """Find optimal Crump threshold that minimizes variance."""
    alphas = np.linspace(0.01, 0.25, 25)
    best_alpha = 0.1
    best_var = np.inf
    for a in alphas:
        mask = (ps >= a) & (ps <= 1 - a)
        if mask.sum() < 10:
            continue
        ps_sub = ps[mask]
        var_est = float(np.mean(1.0 / (ps_sub * (1 - ps_sub))))
        if var_est < best_var:
            best_var = var_est
            best_alpha = float(a)
    return best_alpha


trmwt = trimmed_weights


def cheatsheet() -> str:
    return "trimmed_weights({}) -> Trimmed weights for causal inference."
