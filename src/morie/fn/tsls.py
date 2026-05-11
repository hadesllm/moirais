"""Two-stage least squares with diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def two_stage_ls(data: pd.DataFrame, cdf=None, *, y: str = "outcome", x_endog: list[str] | str = "x_endog", x_exog: list[str] | None = None, z: list[str] | str = "z") -> RegressionResult:
    r"""Two-stage least squares (2SLS) with first-stage F and Hausman test.

    Stage 1: Regress each endogenous variable on all instruments + exogenous:

    .. math::

        X_{endog} = Z \pi + \nu

    Stage 2: Replace endogenous with fitted values:

    .. math::

        Y = \hat{X}_{endog} \beta_{endog} + X_{exog} \beta_{exog} + \epsilon

    Standard errors use the original (not fitted) regressors for correct
    variance estimation.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x_endog : list[str] or str
        Endogenous regressor column(s).
    x_exog : list[str] or None
        Exogenous control columns.
    z : list[str] or str
        Excluded instrument columns.

    Returns
    -------
    RegressionResult
        Extra dict includes first_stage_F, hausman_stat, hausman_pval.

    References
    ----------
    Angrist, J. D., & Imbens, G. W. (1995). Two-stage least squares
    estimation of average causal effects. *JASA*, 90(430), 431-442.
    """
    if isinstance(x_endog, str):
        x_endog = [x_endog]
    if isinstance(z, str):
        z = [z]
    _validate_df(data, y, *x_endog, *z)
    x_exog = x_exog or []
    if x_exog:
        _validate_df(data, *x_exog)
    df = data[[y] + x_endog + z + x_exog].dropna()
    n = len(df)

    Y = df[y].to_numpy(dtype=float)
    X_end = df[x_endog].to_numpy(dtype=float)
    if X_end.ndim == 1:
        X_end = X_end.reshape(-1, 1)
    Z_mat = np.column_stack([np.ones(n), df[z].to_numpy(dtype=float)])
    if x_exog:
        Z_mat = np.column_stack([Z_mat, df[x_exog].to_numpy(dtype=float)])

    first_stage_f = []
    X_hat = np.zeros_like(X_end)
    for j in range(X_end.shape[1]):
        pi_j = np.linalg.lstsq(Z_mat, X_end[:, j], rcond=None)[0]
        X_hat[:, j] = Z_mat @ pi_j
        resid_fs = X_end[:, j] - X_hat[:, j]
        ss_res_fs = float(resid_fs @ resid_fs)
        ss_tot_fs = float(np.sum((X_end[:, j] - X_end[:, j].mean()) ** 2))
        r2_fs = 1 - ss_res_fs / ss_tot_fs if ss_tot_fs > 0 else 0.0
        k_fs = Z_mat.shape[1]
        f_stat = (r2_fs / max(k_fs - 1, 1)) / (max(1 - r2_fs, 1e-15) / max(n - k_fs, 1))
        first_stage_f.append(float(f_stat))

    if x_exog:
        X_2nd = np.column_stack([np.ones(n), X_hat, df[x_exog].to_numpy(dtype=float)])
        X_orig = np.column_stack([np.ones(n), X_end, df[x_exog].to_numpy(dtype=float)])
    else:
        X_2nd = np.column_stack([np.ones(n), X_hat])
        X_orig = np.column_stack([np.ones(n), X_end])

    beta = np.linalg.lstsq(X_2nd, Y, rcond=None)[0]

    resid = Y - X_orig @ beta
    p = X_orig.shape[1]
    sigma2 = float(resid @ resid) / max(n - p, 1)
    try:
        V = sigma2 * np.linalg.inv(X_2nd.T @ X_2nd)
    except np.linalg.LinAlgError:
        V = np.full((p, p), np.nan)
    se_arr = np.sqrt(np.maximum(np.diag(V), 0))

    names = ["(Intercept)"] + x_endog + x_exog
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

    beta_ols = np.linalg.lstsq(X_orig, Y, rcond=None)[0]
    diff = beta - beta_ols
    resid_ols = Y - X_orig @ beta_ols
    sigma2_ols = float(resid_ols @ resid_ols) / max(n - p, 1)
    try:
        V_ols = sigma2_ols * np.linalg.inv(X_orig.T @ X_orig)
        V_diff = V - V_ols
        hausman_stat = float(diff @ np.linalg.inv(V_diff) @ diff)
        hausman_df = len(x_endog)
        hausman_pval = float(1 - stats.chi2.cdf(hausman_stat, hausman_df))
    except np.linalg.LinAlgError:
        hausman_stat = float("nan")
        hausman_df = len(x_endog)
        hausman_pval = float("nan")

    return RegressionResult(
        method="2SLS",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        n=n,
        k=p - 1,
        extra={
            "first_stage_F": first_stage_f,
            "hausman_stat": hausman_stat,
            "hausman_df": hausman_df,
            "hausman_pval": hausman_pval,
        },
    )


tsls = two_stage_ls


def cheatsheet() -> str:
    return "two_stage_ls({}) -> Two-stage least squares with diagnostics."
