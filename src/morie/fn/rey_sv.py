# morie.fn — function file (hadesllm/morie)
"""Survey-weighted regression with robust (sandwich) standard errors."""

import numpy as np
from scipy.stats import t as t_dist

from morie.fn._containers import RegressionResult


def rey_sv(
    df,
    y: str = "y",
    x: list | str = "x",
    weights: str = "weight",
    alpha: float = 0.05,
) -> RegressionResult:
    """
    Survey-weighted OLS regression with sandwich (HC1) standard errors.

    Fits the weighted least squares model:

    .. math::

        \\hat{\\beta} = (X' W X)^{-1} X' W y

    with robust (Huber-White sandwich) variance estimator accounting
    for heterogeneous survey weights.

    :param df: DataFrame with response, predictors, and weight column.
    :param y: Response column name.
    :param x: Predictor column name(s) (str or list of str).
    :param weights: Name of the survey weight column.
    :param alpha: Significance level. Default 0.05.
    :return: :class:`RegressionResult` with robust standard errors.
    :raises ValueError: On missing columns or non-positive weights.

    References
    ----------
    Lumley, T. (2010). Complex Surveys: A Guide to Analysis Using R.
    Wiley.

    Binder, D. A. (1983). On the variances of asymptotically normal
    estimators from complex surveys. *International Statistical Review*,
    51(3), 279-292.
    """

    if isinstance(x, str):
        x = [x]
    for col in [y, weights] + x:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame.")

    y_arr = np.asarray(df[y], dtype=float)
    w_arr = np.asarray(df[weights], dtype=float)
    X_arr = np.column_stack([np.ones(len(df))] + [np.asarray(df[c], dtype=float) for c in x])
    n, p = X_arr.shape

    if np.any(w_arr <= 0):
        raise ValueError("All weights must be > 0.")

    # Weighted OLS
    W = np.diag(w_arr)
    XtWX = X_arr.T @ W @ X_arr
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)
    beta = XtWX_inv @ X_arr.T @ W @ y_arr

    fitted = X_arr @ beta
    residuals = y_arr - fitted

    # Sandwich (HC1) variance: (X'WX)^-1 X' W diag(e^2) W X (X'WX)^-1
    e2 = residuals**2
    meat = X_arr.T @ np.diag(w_arr**2 * e2) @ X_arr
    hc1_factor = n / max(n - p, 1)
    V = hc1_factor * XtWX_inv @ meat @ XtWX_inv
    se_beta = np.sqrt(np.maximum(np.diag(V), 0.0))

    df_resid = n - p
    t_vals = beta / np.where(se_beta > 0, se_beta, np.inf)
    p_vals = 2.0 * t_dist.sf(np.abs(t_vals), df=max(df_resid, 1))

    # Weighted R-squared
    y_bar_w = np.average(y_arr, weights=w_arr)
    ss_tot = np.sum(w_arr * (y_arr - y_bar_w) ** 2)
    ss_res = np.sum(w_arr * residuals**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = ["intercept"] + list(x)

    return RegressionResult(
        method="Survey-Weighted OLS",
        coefficients=dict(zip(names, beta.tolist())),
        se=dict(zip(names, se_beta.tolist())),
        p_values=dict(zip(names, p_vals.tolist())),
        r_squared=float(r2),
        residuals=residuals,
        fitted=fitted,
        n=n,
        k=p,
        extra={"se_type": "HC1 sandwich", "df_resid": df_resid},
    )


def cheatsheet() -> str:
    return "rey_sv({}) -> Survey-weighted regression with robust (sandwich) standard e"
