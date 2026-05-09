# moirais.fn — function file (hadesllm/moirais)
"""Hausman specification test (FE vs RE)."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def hausman_test(
    y: np.ndarray,
    X: np.ndarray,
    entity: np.ndarray,
) -> DescriptiveResult:
    """Hausman test for fixed effects vs. random effects.

    Under :math:`H_0` (RE is consistent and efficient), the FE and RE
    estimators converge to the same value.  The test statistic is
    :math:`H = (\\hat{\\beta}_{FE} - \\hat{\\beta}_{RE})^\\top
    [\\widehat{\\text{Var}}(\\hat{\\beta}_{FE}) -
     \\widehat{\\text{Var}}(\\hat{\\beta}_{RE})]^{-1}
    (\\hat{\\beta}_{FE} - \\hat{\\beta}_{RE}) \\sim \\chi^2_p`.

    Parameters
    ----------
    y : (n,) response
    X : (n, p) predictors
    entity : (n,) entity identifiers

    Returns
    -------
    DescriptiveResult
        ``value`` is the chi-squared statistic.

    References
    ----------
    Hausman, J. A. (1978). Specification tests in econometrics.
    *Econometrica*, 46(6), 1251--1271.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    entity = np.asarray(entity).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    unique_ent = np.unique(entity)
    N = len(unique_ent)

    y_dm = y.copy()
    X_dm = X.copy()
    for e in unique_ent:
        idx = entity == e
        y_dm[idx] -= np.mean(y[idx])
        X_dm[idx] -= np.mean(X[idx], axis=0)
    XtX_fe = X_dm.T @ X_dm
    beta_fe = np.linalg.solve(XtX_fe + np.eye(p) * 1e-12, X_dm.T @ y_dm)
    resid_fe = y_dm - X_dm @ beta_fe
    sigma2_fe = float(resid_fe @ resid_fe) / max(n - N - p, 1)
    V_fe = sigma2_fe * np.linalg.inv(XtX_fe + np.eye(p) * 1e-12)

    sigma2_e = sigma2_fe
    y_bar_e = np.zeros(n)
    X_bar_e = np.zeros_like(X)
    for e in unique_ent:
        idx = entity == e
        y_bar_e[idx] = np.mean(y[idx])
        X_bar_e[idx] = np.mean(X[idx], axis=0)
    T_mean = n / N
    sigma2_u = max(0, sigma2_e * 0.5)
    theta = 1 - np.sqrt(sigma2_e / (sigma2_e + T_mean * sigma2_u + 1e-300))

    y_re = y - theta * y_bar_e + theta * np.mean(y)
    X_re = X - theta * X_bar_e + theta * np.mean(X, axis=0)
    X_re_int = np.column_stack([np.ones(n) * (1 - theta), X_re])
    XtX_re = X_re_int.T @ X_re_int
    beta_re_full = np.linalg.solve(XtX_re + np.eye(p + 1) * 1e-12, X_re_int.T @ y_re)
    beta_re = beta_re_full[1:]
    resid_re = y_re - X_re_int @ beta_re_full
    sigma2_re = float(resid_re @ resid_re) / max(n - p - 1, 1)
    V_re = sigma2_re * np.linalg.inv(XtX_re + np.eye(p + 1) * 1e-12)[1:, 1:]

    diff = beta_fe - beta_re
    V_diff = V_fe - V_re
    try:
        H = float(diff @ np.linalg.solve(V_diff + np.eye(p) * 1e-12, diff))
    except np.linalg.LinAlgError:
        H = float("nan")

    H = max(H, 0.0)
    p_value = float(_st.chi2.sf(H, df=p))

    return DescriptiveResult(
        name="Hausman Test",
        value=float(H),
        extra={
            "p_value": p_value,
            "df": p,
            "beta_fe": beta_fe.tolist(),
            "beta_re": beta_re.tolist(),
            "reject_H0": p_value < 0.05,
            "interpretation": "FE preferred" if p_value < 0.05 else "RE preferred",
        },
    )


hausr = hausman_test


def cheatsheet() -> str:
    return "hausman_test({}) -> Hausman FE vs RE specification test."
