# morie.fn -- function file (hadesllm/morie)
"""Firth penalized logistic regression. 'Statistics is the grammar of science. -- Karl Pearson'"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special

from ._containers import RegressionResult
from ._helpers import _validate_df


def firth_logistic(data: pd.DataFrame, cdf=None, *, y: str = "y", x: list[str] | str = "x", max_iter: int = 100, tol: float = 1e-6) -> RegressionResult:
    """Firth penalized logistic regression for rare events.

    Applies Firth's (1993) bias-reduction via Jeffreys prior,
    which penalises the likelihood by |I(beta)|^{1/2}.  This removes
    first-order bias and handles separation.

    :param data: DataFrame with binary outcome and predictors.
    :param y: Name of the binary outcome column.
    :param x: Predictor column name(s).
    :param max_iter: Maximum IRLS iterations.
    :param tol: Convergence tolerance.
    :return: RegressionResult with Firth-corrected coefficients.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    n, p = X.shape

    beta = np.zeros(p)

    for _ in range(max_iter):
        eta = X @ beta
        mu = special.expit(eta)
        W = mu * (1.0 - mu) + 1e-12
        Wsqrt = np.sqrt(W)
        XW = X * Wsqrt[:, None]
        _, s, Vt = np.linalg.svd(XW, full_matrices=False)
        s_inv2 = 1.0 / (s**2 + 1e-12)
        H_diag = np.sum((X @ (Vt.T * s_inv2) @ Vt) * X * W[:, None], axis=1)
        Y_adj = Y + H_diag * (0.5 - mu)
        z = eta + (Y_adj - mu) / W
        WtX = X * W[:, None]
        XtWX = WtX.T @ X
        XtWz = WtX.T @ z
        beta_new = np.linalg.solve(XtWX + np.eye(p) * 1e-10, XtWz)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu_final = special.expit(X @ beta)
    W_final = mu_final * (1.0 - mu_final) + 1e-12
    XtWX = (X * W_final[:, None]).T @ X
    try:
        cov = np.linalg.inv(XtWX)
        se_arr = np.sqrt(np.diag(cov).clip(0))
    except np.linalg.LinAlgError:
        se_arr = np.full(p, float("nan"))

    from scipy import stats as _st

    z_vals = beta / (se_arr + 1e-12)
    pvals = 2 * (1 - _st.norm.cdf(np.abs(z_vals)))

    names = ["(Intercept)", *x]
    return RegressionResult(
        method="Firth logistic",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, pvals)},
        fitted=mu_final,
        residuals=Y - mu_final,
        n=n,
        k=p - 1,
        extra={"method": "Firth (1993) penalised ML"},
    )


firth = firth_logistic


def cheatsheet() -> str:
    return "firth_logistic({}) -> Firth penalized logistic regression. 'Do or do not. There is"
