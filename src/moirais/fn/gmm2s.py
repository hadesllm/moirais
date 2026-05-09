# moirais.fn — function file (hadesllm/moirais)
"""Two-step GMM estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def gmm_twostep(data: pd.DataFrame, cdf=None, *, y: str = "outcome", x: list[str] | str = "x", z: list[str] | str = "z") -> RegressionResult:
    r"""Two-step Generalized Method of Moments (GMM) estimator.

    Step 1 uses identity weight matrix. Step 2 uses the optimal weight
    matrix from step-1 residuals:

    .. math::

        \hat{\beta}_{GMM} = (X^\top Z \hat{W} Z^\top X)^{-1}
        X^\top Z \hat{W} Z^\top Y

    where :math:`\hat{W} = \left(\frac{1}{n} \sum_i \hat{e}_i^2 z_i z_i^\top\right)^{-1}`
    and :math:`\hat{e}_i` are step-1 residuals.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x : list[str] or str
        Endogenous/exogenous regressors.
    z : list[str] or str
        Instruments (must be at least as many as x).

    Returns
    -------
    RegressionResult

    References
    ----------
    Hansen, L. P. (1982). Large sample properties of generalized method of
    moments estimators. *Econometrica*, 50(4), 1029-1054.
    """
    if isinstance(x, str):
        x = [x]
    if isinstance(z, str):
        z = [z]
    _validate_df(data, y, *x, *z)
    df = data[[y] + x + z].dropna()
    n = len(df)

    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), df[x].to_numpy(dtype=float)])
    Z = np.column_stack([np.ones(n), df[z].to_numpy(dtype=float)])

    k = X.shape[1]
    q = Z.shape[1]
    if q < k:
        raise ValueError(f"Need at least {k} instruments, got {q}")

    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    beta1 = np.linalg.lstsq(Z @ ZtZ_inv @ Z.T @ X, Z @ ZtZ_inv @ Z.T @ Y, rcond=None)[0]

    e1 = Y - X @ beta1
    S = np.zeros((q, q))
    for i in range(n):
        zi = Z[i]
        S += e1[i] ** 2 * np.outer(zi, zi)
    S /= n

    try:
        W_opt = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        W_opt = np.linalg.pinv(S)

    XtZ = X.T @ Z
    ZtX = Z.T @ X
    A = XtZ @ W_opt @ ZtX
    b = XtZ @ W_opt @ Z.T @ Y
    try:
        beta2 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta2 = np.linalg.lstsq(A, b, rcond=None)[0]

    e2 = Y - X @ beta2
    S2 = np.zeros((q, q))
    for i in range(n):
        zi = Z[i]
        S2 += e2[i] ** 2 * np.outer(zi, zi)
    S2 /= n
    try:
        W2 = np.linalg.inv(S2)
    except np.linalg.LinAlgError:
        W2 = W_opt

    V = np.linalg.inv(XtZ @ W2 @ ZtX) / n * n
    se_arr = np.sqrt(np.maximum(np.diag(V), 0))

    names = ["(Intercept)"] + list(x)
    coefficients = {nm: float(b) for nm, b in zip(names, beta2)}
    se_dict = {nm: float(s) for nm, s in zip(names, se_arr)}
    p_dict = {}
    for i, nm in enumerate(names):
        if se_arr[i] > 0:
            z_stat = beta2[i] / se_arr[i]
            p_dict[nm] = float(2 * stats.norm.sf(abs(z_stat)))
        else:
            p_dict[nm] = float("nan")

    j_stat = float(n * e2 @ Z @ W2 @ Z.T @ e2 / n ** 2) if q > k else 0.0
    j_df = q - k
    j_pval = float(1 - stats.chi2.cdf(j_stat, j_df)) if j_df > 0 else float("nan")

    return RegressionResult(
        method="2-step GMM",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        n=n,
        k=k - 1,
        extra={
            "j_stat": j_stat,
            "j_df": j_df,
            "j_pval": j_pval,
            "n_instruments": q,
        },
    )


gmm2s = gmm_twostep


def cheatsheet() -> str:
    return "gmm_twostep({}) -> Two-step GMM estimator."
