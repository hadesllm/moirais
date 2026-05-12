# morie.fn -- function file (hadesllm/morie)
"""Double debiasing estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def double_debias(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    covariates: list[str] | None = None,
    n_folds: int = 2,
    seed: int = 42,
    alpha: float = 0.05,
) -> ESRes:
    r"""Double debiasing (Belloni, Chernozhukov, Hansen, 2014).

    The double-debiased estimator removes regularization bias by
    partialling out X from both Y and T, then running OLS on residuals:

    1. Regress Y on X to get :math:`\tilde{Y} = Y - \hat{E}[Y|X]`
    2. Regress T on X to get :math:`\tilde{T} = T - \hat{E}[T|X]`
    3. Regress :math:`\tilde{Y}` on :math:`\tilde{T}`:

    .. math::

        \hat{\theta} = \frac{\sum_i \tilde{T}_i \tilde{Y}_i}
        {\sum_i \tilde{T}_i^2}

    Uses sample splitting to avoid overfitting bias.

    Parameters
    ----------
    data : pd.DataFrame
    y, t : str
        Outcome and treatment columns.
    covariates : list[str]
        Covariate columns.
    n_folds : int
        Number of sample splits (2 = single split, >2 = cross-fitting).
    seed : int
        Random seed.
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on
    treatment effects after selection among high-dimensional controls.
    *Review of Economic Studies*, 81(2), 608-650.
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

    Y_tilde = np.zeros(n)
    T_tilde = np.zeros(n)

    for k in range(n_folds):
        tr = fold_ids != k
        te = fold_ids == k
        Y_tilde[te] = Y[te] - _ridge(X[tr], Y[tr], X[te])
        T_tilde[te] = T_arr[te] - _ridge(X[tr], T_arr[tr], X[te])

    denom = float(np.sum(T_tilde ** 2))
    if denom < 1e-15:
        raise ValueError("Treatment residual variance near zero after partialling")

    theta = float(np.sum(T_tilde * Y_tilde) / denom)
    resid = Y_tilde - theta * T_tilde
    se = float(np.sqrt(np.sum(T_tilde ** 2 * resid ** 2) / denom ** 2))

    z = stats.norm.ppf(1 - alpha / 2)
    return ESRes(
        measure="Double-debiased ATE",
        estimate=theta,
        ci_lower=theta - z * se,
        ci_upper=theta + z * se,
        se=se,
        n=n,
        extra={"n_folds": n_folds, "seed": seed},
    )


dbias = double_debias


def cheatsheet() -> str:
    return "double_debias({}) -> Double debiasing estimator."
