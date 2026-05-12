# morie.fn -- function file (hadesllm/morie)
"""Panel data regression (fixed effects / random effects)."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def panel_regression(
    y: np.ndarray,
    X: np.ndarray,
    entity: np.ndarray,
    *,
    method: str = "fe",
    add_intercept: bool = True,
) -> RegressionResult:
    """Panel (longitudinal) regression with fixed or random effects.

    Parameters
    ----------
    y : (n,) response
    X : (n, p) predictors
    entity : (n,) entity (panel unit) identifiers
    method : str
        'fe' for fixed effects (within estimator) or 're' for random effects
        (partial demeaning via theta).
    add_intercept : bool
        For FE, intercept is dropped. For RE, included.

    Returns
    -------
    RegressionResult

    References
    ----------
    Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and
    Panel Data* (2nd ed.). MIT Press.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    entity = np.asarray(entity).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    unique_ent = np.unique(entity)
    N = len(unique_ent)

    if method == "fe":
        y_dm = y.copy()
        X_dm = X.copy()
        for e in unique_ent:
            idx = entity == e
            y_dm[idx] -= np.mean(y[idx])
            X_dm[idx] -= np.mean(X[idx], axis=0)

        XtX = X_dm.T @ X_dm
        try:
            beta = np.linalg.solve(XtX + np.eye(p) * 1e-12, X_dm.T @ y_dm)
        except np.linalg.LinAlgError:
            raise ValueError("X'X singular after demeaning.")

        resid = y_dm - X_dm @ beta
        dof = n - N - p
        if dof <= 0:
            dof = 1
        sigma2 = float(resid @ resid) / dof
        try:
            cov = sigma2 * np.linalg.inv(XtX)
            se_arr = np.sqrt(np.diag(cov).clip(0))
        except np.linalg.LinAlgError:
            se_arr = np.full(p, float("nan"))

        fitted = X @ beta
        for e in unique_ent:
            idx = entity == e
            fitted[idx] += np.mean(y[idx] - X[idx] @ beta)

        names = [f"x{j}" for j in range(p)]

    elif method == "re":
        y_bar_e = np.zeros(n)
        X_bar_e = np.zeros_like(X)
        T_e = np.zeros(n)
        for e in unique_ent:
            idx = entity == e
            ni = np.sum(idx)
            y_bar_e[idx] = np.mean(y[idx])
            X_bar_e[idx] = np.mean(X[idx], axis=0)
            T_e[idx] = ni

        y_dm = y - y_bar_e
        X_dm = X - X_bar_e
        sigma2_e = float((y_dm - X_dm @ np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]) @ (y_dm - X_dm @ np.linalg.lstsq(X_dm, y_dm, rcond=None)[0])) / max(n - N - p, 1)

        y_bar_all = np.mean(y)
        X_bar_all = np.mean(X, axis=0)
        between_y = np.array([np.mean(y[entity == e]) for e in unique_ent])
        between_X = np.array([np.mean(X[entity == e], axis=0) for e in unique_ent])
        bX_dm = between_X - X_bar_all
        by_dm = between_y - y_bar_all
        sigma2_b_resid = float(by_dm @ by_dm - by_dm @ bX_dm @ np.linalg.lstsq(bX_dm, by_dm, rcond=None)[0]) / max(N - p - 1, 1) if p + 1 < N else sigma2_e
        sigma2_u = max(sigma2_b_resid - sigma2_e / np.mean(T_e[:N]), 0)

        theta = np.zeros(n)
        for e in unique_ent:
            idx = entity == e
            ni = np.sum(idx)
            theta[idx] = 1 - np.sqrt(sigma2_e / (sigma2_e + ni * sigma2_u + 1e-300))

        y_re = y - theta * y_bar_e
        X_re = X - theta[:, None] * X_bar_e
        if add_intercept:
            ones_re = 1 - theta
            X_re = np.column_stack([ones_re, X_re])
            kk = p + 1
        else:
            kk = p

        XtX = X_re.T @ X_re
        beta = np.linalg.solve(XtX + np.eye(kk) * 1e-12, X_re.T @ y_re)
        resid = y_re - X_re @ beta
        sigma2 = float(resid @ resid) / max(n - kk, 1)
        try:
            cov = sigma2 * np.linalg.inv(XtX)
            se_arr = np.sqrt(np.diag(cov).clip(0))
        except np.linalg.LinAlgError:
            se_arr = np.full(kk, float("nan"))

        fitted = X_re @ beta
        if add_intercept:
            names = ["(Intercept)"] + [f"x{j}" for j in range(p)]
        else:
            names = [f"x{j}" for j in range(p)]
    else:
        raise ValueError("method must be 'fe' or 're'.")

    t_vals = beta / (se_arr + 1e-300)
    dof_t = max(n - len(beta) - (N if method == "fe" else 0), 1)
    p_vals = 2.0 * _st.t.sf(np.abs(t_vals), df=dof_t)

    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return RegressionResult(
        method=f"Panel ({method.upper()})",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        r_squared=r2,
        residuals=y - fitted,
        fitted=fitted,
        n=n,
        k=p,
        extra={
            "method": method,
            "n_entities": N,
            "sigma2": sigma2,
        },
    )


panel = panel_regression


def cheatsheet() -> str:
    return "panel_regression({}) -> Panel data FE/RE regression."
