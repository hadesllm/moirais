# moirais.fn — function file (hadesllm/moirais)
"""Entropy balancing weights (Hainmueller, 2012)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._helpers import _validate_df


def entropy_balance(
    data: pd.DataFrame,
    *,
    t: str = "treatment",
    covariates: list[str] | None = None,
    alpha: float = 0.05,
    max_iter: int = 500,
) -> dict:
    r"""Entropy balancing: reweight control group to match treated moments.

    Solves the dual problem:

    .. math::

        \min_{\lambda} \sum_{i: T_i=0} \exp(\lambda^\top X_i)
        \quad \text{s.t.} \quad
        \sum_{i: T_i=0} w_i X_i = \bar{X}_{T=1}

    where :math:`w_i = \exp(\lambda^\top X_i) / \sum_j \exp(\lambda^\top X_j)`.

    Parameters
    ----------
    data : pd.DataFrame
    t : str
        Binary treatment column.
    covariates : list[str]
        Covariate columns to balance on.
    alpha : float
        Not used for weighting but included for API consistency.
    max_iter : int
        Maximum iterations for optimization.

    Returns
    -------
    dict
        Keys: 'weights' (n-length, 1.0 for treated), 'converged',
        'max_smd_after', 'smd_after'.

    References
    ----------
    Hainmueller, J. (2012). Entropy balancing for causal effects.
    *Political Analysis*, 20(1), 25-46.
    """
    if covariates is None or len(covariates) == 0:
        raise ValueError("covariates required for entropy balancing")
    _validate_df(data, t, *covariates)
    df = data[[t] + covariates].dropna()
    T = df[t].to_numpy(dtype=float)
    X = df[covariates].to_numpy(dtype=float)
    n = len(T)

    idx_t = T == 1
    idx_c = T == 0
    n_c = int(idx_c.sum())

    X_c = X[idx_c]
    target = X[idx_t].mean(axis=0)

    def dual_obj(lam):
        logw = X_c @ lam
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        bal = X_c.T @ w - target
        return float(0.5 * bal @ bal)

    def dual_grad(lam):
        logw = X_c @ lam
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        bal = X_c.T @ w - target
        dw = w[:, None] * (X_c - (X_c.T @ w)[None, :])
        return dw.T @ (X_c @ np.zeros_like(lam)) + X_c.T @ (w * (X_c @ np.zeros_like(lam))) if False else (X_c * w[:, None]).T @ X_c @ np.linalg.lstsq(np.eye(len(lam)), bal, rcond=None)[0]

    lam0 = np.zeros(X_c.shape[1])
    result = minimize(dual_obj, lam0, method="L-BFGS-B",
                      options={"maxiter": max_iter})

    logw = X_c @ result.x
    logw -= logw.max()
    w_c = np.exp(logw)
    w_c = w_c / w_c.sum() * n_c

    weights = np.ones(n)
    weights[idx_c] = w_c

    smd_after = {}
    for i, c in enumerate(covariates):
        wm_c = np.average(X_c[:, i], weights=w_c)
        m_t = target[i]
        pooled_sd = X[:, i].std()
        smd_after[c] = float(abs(m_t - wm_c) / pooled_sd) if pooled_sd > 0 else 0.0

    max_smd = max(smd_after.values()) if smd_after else 0.0

    return {
        "weights": weights,
        "converged": result.success,
        "max_smd_after": max_smd,
        "smd_after": smd_after,
        "n": n,
        "n_control": n_c,
    }


ebal = entropy_balance


def cheatsheet() -> str:
    return "entropy_balance({}) -> Entropy balancing weights (Hainmueller, 2012)."
