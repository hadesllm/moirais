# moirais.fn — function file (hadesllm/moirais)
"""Marginal structural model (MSM) via inverse probability weighting.

Implements the MSM framework for estimating the causal effect of a
time-varying treatment using stabilized IPW weights and a weighted
outcome regression.

References
----------
Robins, J. M., Hernan, M. A., & Brumback, B. (2000). Marginal
structural models and causal inference in epidemiology.
*Epidemiology*, 11(5), 550-560.

Hernan, M. A., & Robins, J. M. (2020). *Causal Inference: What If*.
Chapman & Hall/CRC. Chapter 12.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm as _norm

__all__ = ["msmod"]


def msmod(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    stabilize: bool = True,
    ps_trim: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Fit a marginal structural model via stabilized IPW.

    Stabilized weights are:

    .. math::

        SW_i = \frac{P(T_i)}{P(T_i \mid X_i)}

    where the numerator is the marginal treatment probability and the
    denominator is the propensity score.  The MSM ATE is estimated via
    weighted least squares.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment vector, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    stabilize : bool
        If True, use stabilized weights; otherwise use raw IPW.
    ps_trim : float
        Trim propensity scores to ``[ps_trim, 1 - ps_trim]``.
    alpha : float
        Significance level for confidence interval.

    Returns
    -------
    dict
        ``ate``, ``se``, ``ci_lower``, ``ci_upper``, ``weights``,
        ``effective_n``, ``n``, ``method``.

    References
    ----------
    Robins et al. (2000). Epidemiology, 11(5), 550-560.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)
    if not (len(T) == n == X.shape[0]):
        raise ValueError("Y, T, X must share first dimension.")

    Xa = np.column_stack([np.ones(n), X])
    ps = np.clip(_irls(Xa, T), ps_trim, 1.0 - ps_trim)

    p_marg = float(T.mean())

    if stabilize:
        # Stabilized: numerator is marginal P(T)
        w = np.where(
            T == 1,
            p_marg / ps,
            (1.0 - p_marg) / (1.0 - ps),
        )
    else:
        w = np.where(T == 1, 1.0 / ps, 1.0 / (1.0 - ps))

    # Weighted least squares: regress Y on intercept + T with weights w
    D = np.column_stack([np.ones(n), T])
    W_diag = w
    DtW = D.T * W_diag
    beta_wls = np.linalg.lstsq(DtW @ D, DtW @ Y, rcond=None)[0]
    ate = float(beta_wls[1])

    # Robust (HC1) standard error for WLS
    fitted = D @ beta_wls
    resid = Y - fitted
    meat = float(np.sum(w**2 * T**2 * resid**2))
    bread_sq = float(np.sum(w * T**2))
    se = float(np.sqrt(meat) / bread_sq) if bread_sq > 0 else np.nan

    z = _norm.ppf(1.0 - alpha / 2.0)
    eff_n = float(np.sum(w) ** 2 / np.sum(w**2))

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "weights": w,
        "effective_n": eff_n,
        "n": n,
        "method": "MSM-IPW",
    }


def _irls(X, y, max_iter=25):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = np.clip(expit(X @ beta), 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        A = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        b = (X.T * W) @ (X @ beta + (y - p) / W)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
    return expit(X @ beta)


def cheatsheet() -> str:
    return "msmod(Y, T, X) -> Marginal structural model via stabilized IPW (Robins et al. 2000)."
