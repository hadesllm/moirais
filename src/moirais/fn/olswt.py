# moirais.fn — function file (hadesllm/moirais)
"""OLS with IPW weights for causal ATE estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def ols_weighted(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    x: list[str] | None = None,
    w: str = "weight",
) -> RegressionResult:
    r"""OLS regression with IPW weights and HC1 robust standard errors.

    Fits:

    .. math::

        Y_i = \beta_0 + \tau T_i + X_i^\top \gamma + \epsilon_i

    with weight matrix :math:`W = \text{diag}(w_1, \ldots, w_n)`.

    HC1 robust variance:

    .. math::

        \hat{V}_{HC1} = \frac{n}{n-k}(X^\top W X)^{-1}
        \left(\sum_{i=1}^n w_i^2 \hat{e}_i^2 x_i x_i^\top\right)
        (X^\top W X)^{-1}

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    t : str
        Treatment column (coefficient = ATE estimate).
    x : list[str] or None
        Additional covariates.
    w : str
        Weight column.

    Returns
    -------
    RegressionResult
        The treatment coefficient is the ATE estimate.

    References
    ----------
    Freedman, D. A. (2006). On the so-called "Huber sandwich estimator"
    and "robust standard errors". *The American Statistician*, 60(4), 299-302.
    """
    _validate_df(data, y, t, w)
    cols = [y, t, w]
    if x is not None:
        _validate_df(data, *x)
        cols.extend(x)
    df = data[cols].dropna()

    Y = df[y].to_numpy(dtype=float)
    T_arr = df[t].to_numpy(dtype=float)
    W = df[w].to_numpy(dtype=float)
    n = len(Y)

    if x is not None and len(x) > 0:
        X_extra = df[x].to_numpy(dtype=float)
        X_design = np.column_stack([np.ones(n), T_arr, X_extra])
        names = ["(Intercept)", t] + list(x)
    else:
        X_design = np.column_stack([np.ones(n), T_arr])
        names = ["(Intercept)", t]
    p = X_design.shape[1]

    W_sqrt = np.sqrt(np.maximum(W, 0))
    Xw = X_design * W_sqrt[:, None]
    Yw = Y * W_sqrt
    beta = np.linalg.lstsq(Xw, Yw, rcond=None)[0]

    resid = Y - X_design @ beta

    XtWX = X_design.T @ (X_design * W[:, None])
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.full((p, p), np.nan)

    meat = np.zeros((p, p))
    for i in range(n):
        xi = X_design[i]
        meat += (W[i] ** 2) * (resid[i] ** 2) * np.outer(xi, xi)
    hc1_factor = n / max(n - p, 1)
    V_hc1 = hc1_factor * XtWX_inv @ meat @ XtWX_inv
    se_arr = np.sqrt(np.maximum(np.diag(V_hc1), 0))

    coefficients = {nm: float(b) for nm, b in zip(names, beta)}
    se_dict = {nm: float(s) for nm, s in zip(names, se_arr)}
    p_dict = {}
    df_resid = n - p
    for i, nm in enumerate(names):
        if se_arr[i] > 0 and df_resid > 0:
            t_stat = beta[i] / se_arr[i]
            p_dict[nm] = float(2 * stats.t.sf(abs(t_stat), df_resid))
        else:
            p_dict[nm] = float("nan")

    ss_res = float(np.sum(W * resid ** 2))
    ss_tot = float(np.sum(W * (Y - np.average(Y, weights=W)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return RegressionResult(
        method="OLS-IPW (HC1)",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        r_squared=r2,
        residuals=resid,
        fitted=X_design @ beta,
        n=n,
        k=p - 1,
    )


olswt = ols_weighted


def cheatsheet() -> str:
    return "ols_weighted({}) -> OLS with IPW weights for causal ATE estimation."
