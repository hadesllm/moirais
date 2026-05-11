# morie.fn — function file (hadesllm/morie)
"""DML Partially Linear Model (PLR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def dml_plr(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    covariates: list[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    alpha: float = 0.05,
) -> ESRes:
    r"""DML Partially Linear Regression model.

    The PLR assumes:

    .. math::

        Y = \theta T + g(X) + \epsilon, \quad E[\epsilon|X,T] = 0
        T = m(X) + V, \quad E[V|X] = 0

    The Neyman-orthogonal estimator:

    .. math::

        \hat{\theta} = \frac{\sum_i \hat{V}_i (Y_i - \hat{g}(X_i))}
        {\sum_i \hat{V}_i (T_i - \hat{m}(X_i))}

    where :math:`\hat{V}_i = T_i - \hat{m}(X_i)` and nuisance functions
    are estimated via cross-fitting.

    Parameters
    ----------
    data : pd.DataFrame
    y, t : str
        Outcome and treatment columns.
    covariates : list[str]
        Covariate columns.
    n_folds : int
        Cross-fitting folds.
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
    Robinson, P. M. (1988). Root-n-consistent semiparametric regression.
    *Econometrica*, 56(4), 931-954.
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

    g_hat = np.zeros(n)
    m_hat = np.zeros(n)

    for k in range(n_folds):
        tr = fold_ids != k
        te = fold_ids == k
        g_hat[te] = _ridge(X[tr], Y[tr], X[te])
        m_hat[te] = _ridge(X[tr], T_arr[tr], X[te])

    V_hat = T_arr - m_hat
    Y_tilde = Y - g_hat

    denom = float(np.sum(V_hat ** 2))
    if denom < 1e-15:
        theta = float("nan")
        se = float("nan")
    else:
        theta = float(np.sum(V_hat * Y_tilde) / denom)
        scores = V_hat * (Y_tilde - theta * V_hat) / (denom / n)
        se = float(np.std(scores, ddof=1) / np.sqrt(n))

    z_crit = stats.norm.ppf(1 - alpha / 2)

    return ESRes(
        measure="DML-PLR ATE",
        estimate=theta,
        ci_lower=theta - z_crit * se if np.isfinite(se) else float("nan"),
        ci_upper=theta + z_crit * se if np.isfinite(se) else float("nan"),
        se=se,
        n=n,
        extra={"n_folds": n_folds, "seed": seed},
    )


dmlpl = dml_plr


def cheatsheet() -> str:
    return "dml_plr({}) -> DML Partially Linear Model (PLR)."
