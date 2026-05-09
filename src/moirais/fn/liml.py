# moirais.fn — function file (hadesllm/moirais)
"""Limited information maximum likelihood (LIML) estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def liml_estimator(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    x_endog: list[str] | str = "x_endog",
    x_exog: list[str] | None = None,
    z: list[str] | str = "z",
) -> RegressionResult:
    r"""Limited Information Maximum Likelihood (LIML) IV estimator.

    LIML is the k-class estimator with :math:`k = \lambda_{\min}`, the
    smallest eigenvalue of:

    .. math::

        (Y^\top M_Z Y)^{-1} (Y^\top M_X Y)

    where :math:`M_Z = I - Z(Z^\top Z)^{-1}Z^\top` and similarly for
    :math:`M_X` (exogenous regressors only).

    LIML is less biased than 2SLS when instruments are weak.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x_endog : list[str] or str
        Endogenous regressor(s).
    x_exog : list[str] or None
        Exogenous controls (included instruments).
    z : list[str] or str
        Excluded instruments.

    Returns
    -------
    RegressionResult

    References
    ----------
    Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters
    of a single equation in a complete system of stochastic equations.
    *Annals of Mathematical Statistics*, 20(1), 46-63.
    """
    if isinstance(x_endog, str):
        x_endog = [x_endog]
    if isinstance(z, str):
        z = [z]
    _validate_df(data, y, *x_endog, *z)

    all_cols = [y] + x_endog + z
    if x_exog is not None:
        _validate_df(data, *x_exog)
        all_cols.extend(x_exog)
    else:
        x_exog = []
    df = data[all_cols].dropna()
    n = len(df)

    Y_vec = df[y].to_numpy(dtype=float)
    X_endog = df[x_endog].to_numpy(dtype=float)
    if X_endog.ndim == 1:
        X_endog = X_endog.reshape(-1, 1)
    Z_excl = df[z].to_numpy(dtype=float)
    if Z_excl.ndim == 1:
        Z_excl = Z_excl.reshape(-1, 1)

    ones = np.ones((n, 1))
    if len(x_exog) > 0:
        X_exog_mat = df[x_exog].to_numpy(dtype=float)
        W = np.column_stack([ones, X_exog_mat])
    else:
        W = ones

    Z_full = np.column_stack([W, Z_excl])

    ZtZ_inv = np.linalg.inv(Z_full.T @ Z_full)
    P_z = Z_full @ ZtZ_inv @ Z_full.T
    M_z = np.eye(n) - P_z

    WtW_inv = np.linalg.inv(W.T @ W)
    P_w = W @ WtW_inv @ W.T
    M_w = np.eye(n) - P_w

    YX = np.column_stack([Y_vec, X_endog])
    A = YX.T @ M_z @ YX
    B = YX.T @ M_w @ YX

    try:
        eigvals = np.linalg.eigvalsh(np.linalg.solve(A, B))
        kappa = float(np.min(eigvals[eigvals > 0]))
    except np.linalg.LinAlgError:
        kappa = 1.0

    X_all = np.column_stack([W, X_endog])
    p = X_all.shape[1]

    PzX = P_z @ X_all
    PzY = P_z @ Y_vec
    X_tilde = (1 - kappa) * X_all + kappa * PzX
    Y_tilde = (1 - kappa) * Y_vec + kappa * PzY

    try:
        beta = np.linalg.lstsq(X_tilde, Y_tilde, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.full(p, np.nan)

    resid = Y_vec - X_all @ beta
    sigma2 = float(resid @ resid) / max(n - p, 1)

    try:
        V = sigma2 * np.linalg.inv(X_tilde.T @ X_tilde)
    except np.linalg.LinAlgError:
        V = np.full((p, p), np.nan)
    se_arr = np.sqrt(np.maximum(np.diag(V), 0))

    names = ["(Intercept)"] + x_exog + x_endog
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}
    se_dict = {nm: float(s) for nm, s in zip(names, se_arr)}
    p_dict = {}
    df_r = n - p
    for i, nm in enumerate(names):
        if se_arr[i] > 0 and df_r > 0:
            t_stat = beta[i] / se_arr[i]
            p_dict[nm] = float(2 * stats.t.sf(abs(t_stat), df_r))
        else:
            p_dict[nm] = float("nan")

    return RegressionResult(
        method="LIML",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        n=n,
        k=p - 1,
        extra={"kappa": kappa},
    )


liml = liml_estimator


def cheatsheet() -> str:
    return "liml_estimator({}) -> Limited information maximum likelihood (LIML)."
