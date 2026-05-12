# morie.fn -- function file (hadesllm/morie)
"""DML Interactive Regression Model (IRM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def dml_irm(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    covariates: list[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    alpha: float = 0.05,
) -> ESRes:
    r"""DML Interactive Regression Model for ATE.

    The IRM allows for heterogeneous treatment effects by modeling
    :math:`E[Y|T,X]` separately for each treatment arm.

    Neyman-orthogonal score for IRM:

    .. math::

        \psi(W; \theta, \eta) = g_1(X) - g_0(X)
        + \frac{T(Y - g_1(X))}{e(X)}
        - \frac{(1-T)(Y - g_0(X))}{1 - e(X)} - \theta

    where :math:`g_t(x) = E[Y|T=t, X=x]` and :math:`e(x) = P(T=1|X=x)`.

    Uses ridge regression for nuisance estimation.

    Parameters
    ----------
    data : pd.DataFrame
    y, t : str
        Outcome and treatment columns.
    covariates : list[str]
        Covariate columns.
    n_folds : int
        Number of cross-fitting folds.
    seed : int
        Random seed.
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Chernozhukov, V., et al. (2018). Double/debiased machine learning.
    *The Econometrics Journal*, 21(1), C1-C68.
    """
    if covariates is None or len(covariates) == 0:
        raise ValueError("covariates required")
    _validate_df(data, y, t, *covariates)
    df = data[[y, t] + covariates].dropna().reset_index(drop=True)
    n = len(df)

    Y = df[y].to_numpy(dtype=float)
    T_arr = df[t].to_numpy(dtype=float)
    X = df[covariates].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    fold_ids = np.zeros(n, dtype=int)
    perm = rng.permutation(n)
    fold_size = n // n_folds
    for k in range(n_folds):
        s = k * fold_size
        e = (k + 1) * fold_size if k < n_folds - 1 else n
        fold_ids[perm[s:e]] = k

    def _ridge(Xtr, ytr, Xte, lam=1.0):
        p = Xtr.shape[1]
        beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(p), Xtr.T @ ytr)
        return Xte @ beta

    def _logistic_ridge(Xtr, ytr, Xte, lam=1.0):
        p = Xtr.shape[1]
        beta = np.zeros(p)
        for _ in range(30):
            z = Xtr @ beta
            phat = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            phat = np.clip(phat, 1e-8, 1 - 1e-8)
            W = phat * (1 - phat)
            grad = Xtr.T @ (ytr - phat) - lam * beta
            H = Xtr.T @ (Xtr * W[:, None]) + lam * np.eye(p)
            try:
                beta += np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
        z = Xte @ beta
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    g1_hat = np.zeros(n)
    g0_hat = np.zeros(n)
    e_hat = np.zeros(n)

    for k in range(n_folds):
        tr = fold_ids != k
        te = fold_ids == k
        Xtr, Xte = X[tr], X[te]

        mask1 = tr & (T_arr == 1)
        mask0 = tr & (T_arr == 0)
        if mask1.sum() > 0:
            g1_hat[te] = _ridge(X[mask1], Y[mask1], Xte)
        if mask0.sum() > 0:
            g0_hat[te] = _ridge(X[mask0], Y[mask0], Xte)
        e_hat[te] = np.clip(_logistic_ridge(Xtr, T_arr[tr], Xte), 0.01, 0.99)

    psi = (g1_hat - g0_hat
           + T_arr * (Y - g1_hat) / e_hat
           - (1 - T_arr) * (Y - g0_hat) / (1 - e_hat))

    theta = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return ESRes(
        measure="DML-IRM ATE",
        estimate=theta,
        ci_lower=theta - z_crit * se,
        ci_upper=theta + z_crit * se,
        se=se,
        n=n,
        extra={"n_folds": n_folds, "seed": seed},
    )


dmlir = dml_irm


def cheatsheet() -> str:
    return "dml_irm({}) -> DML Interactive Regression Model (IRM)."
