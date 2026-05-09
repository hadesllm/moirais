"""Stratified propensity score analysis.

Implements the Cochran (1968) stratification approach: divide the
propensity score distribution into K strata, compute within-stratum
treatment effect estimates, and combine with stratum-size weights.

References
----------
Cochran, W. G. (1968). The effectiveness of adjustment by subclassification
in removing bias in observational studies. *Biometrics*, 24(2), 295-313.

Rosenbaum, P. R., & Rubin, D. B. (1984). Reducing bias in observational
studies using subclassification on the propensity score. *Journal of the
American Statistical Association*, 79(387), 516-524.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm as _norm

__all__ = ["strdp"]


def strdp(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    n_strata: int = 5,
    ps_trim: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate ATE via stratified propensity score analysis.

    Steps
    -----
    1. Estimate :math:`\hat{e}(X) = P(T=1 \mid X)` via logistic regression.
    2. Divide observations into :math:`K` equally-spaced propensity strata.
    3. Within each stratum :math:`k`, compute
       :math:`\hat{\tau}_k = \bar{Y}_{k,1} - \bar{Y}_{k,0}`.
    4. Combine: :math:`\hat{\tau} = \sum_k (n_k / n) \hat{\tau}_k`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment vector, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    n_strata : int
        Number of propensity score strata (Cochran recommends 5).
    ps_trim : float
        Trim propensity to ``[ps_trim, 1 - ps_trim]``.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``stratum_effects``, ``stratum_ns``, ``propensity``,
        ``n``, ``method``.

    References
    ----------
    Rosenbaum & Rubin (1984). JASA, 79(387), 516-524.
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

    # Stratify by propensity score quintiles
    boundaries = np.quantile(ps, np.linspace(0, 1, n_strata + 1))
    boundaries[0] -= 1e-8
    boundaries[-1] += 1e-8

    stratum_effects = np.full(n_strata, np.nan)
    stratum_ns = np.zeros(n_strata, dtype=int)

    for k in range(n_strata):
        in_strat = (ps > boundaries[k]) & (ps <= boundaries[k + 1])
        T_k = T[in_strat]
        Y_k = Y[in_strat]
        stratum_ns[k] = int(in_strat.sum())
        if T_k.sum() >= 1 and (1 - T_k).sum() >= 1:
            stratum_effects[k] = float(Y_k[T_k == 1].mean() - Y_k[T_k == 0].mean())

    # Weighted ATE
    valid = ~np.isnan(stratum_effects)
    weights = stratum_ns[valid] / stratum_ns[valid].sum()
    ate = float(np.sum(weights * stratum_effects[valid]))

    # Variance: weighted sum of within-stratum variances
    var_ate = 0.0
    for k in range(n_strata):
        if np.isnan(stratum_effects[k]):
            continue
        in_strat = (ps > boundaries[k]) & (ps <= boundaries[k + 1])
        T_k, Y_k = T[in_strat], Y[in_strat]
        nk = stratum_ns[k]
        n1, n0 = int(T_k.sum()), int((1 - T_k).sum())
        if n1 < 2 or n0 < 2:
            continue
        v1 = float(np.var(Y_k[T_k == 1], ddof=1))
        v0 = float(np.var(Y_k[T_k == 0], ddof=1))
        var_stratum = v1 / n1 + v0 / n0
        var_ate += (nk / n) ** 2 * var_stratum

    se = float(np.sqrt(var_ate))
    z = _norm.ppf(1.0 - alpha / 2.0)

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "stratum_effects": stratum_effects.tolist(),
        "stratum_ns": stratum_ns.tolist(),
        "propensity": ps,
        "n": n,
        "method": f"stratified-PS-{n_strata}",
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
    return "strdp(Y, T, X) -> Stratified propensity analysis (Rosenbaum & Rubin 1984, JASA)."
