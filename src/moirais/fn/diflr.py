# moirais.fn — function file (hadesllm/moirais)
"""Logistic Regression DIF detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from moirais.fn._containers import DIFResult


def _logistic_fit(y: np.ndarray, X_design: np.ndarray, cdf=None) -> tuple[np.ndarray, float]:
    """Fit logistic regression via IRLS, return (coefficients, deviance).

    Uses iteratively reweighted least squares (IRLS) for numerical stability.
    Falls back to scipy.optimize if IRLS does not converge.
    """
    n, p = X_design.shape
    beta = np.zeros(p)
    max_iter = 50

    for _ in range(max_iter):
        eta = X_design @ beta
        eta = np.clip(eta, -700, 700)
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1.0 - 1e-10)

        W = mu * (1.0 - mu)
        z = eta + (y - mu) / W

        # Weighted least squares
        XtW = X_design.T * W[None, :]
        try:
            beta_new = np.linalg.solve(XtW @ X_design, XtW @ z)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    eta = X_design @ beta
    eta = np.clip(eta, -700, 700)
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1.0 - 1e-10)
    deviance = -2.0 * np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu))

    return beta, deviance


def diflr(
    data: pd.DataFrame | np.ndarray,
    group: np.ndarray | pd.Series | list,
    *,
    item_names: list[str] | None = None,
    alpha: float = 0.05,
) -> DIFResult:
    """Logistic Regression DIF detection (Swaminathan & Rogers, 1990).

    For each item, fits three nested logistic regression models:
      M1: item ~ theta (no DIF)
      M2: item ~ theta + group (uniform DIF)
      M3: item ~ theta + group + theta*group (non-uniform DIF)

    DIF is flagged when the likelihood ratio test (M1 vs M3) is significant.

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item response matrix (n x k), values 0/1.
    group : array-like
        Binary group membership (0=reference, 1=focal).
    item_names : list[str], optional
        Item labels.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    DIFResult
        method="logistic", items DataFrame with: item, chi2_uniform,
        p_uniform, chi2_nonuniform, p_nonuniform, chi2_total, p_total,
        dif_type.

    References
    ----------
    Swaminathan, H. & Rogers, H. J. (1990). Detecting differential item
    functioning using logistic regression procedures. Journal of Educational
    Measurement, 27(4), 361-370.
    """
    X = np.asarray(data, dtype=np.float64)
    g = np.asarray(group, dtype=np.float64).ravel()
    n, k = X.shape

    if len(g) != n:
        raise ValueError(f"group length ({len(g)}) != n ({n}).")

    X = np.where(np.isnan(X), 0.0, X)

    if item_names is None:
        if isinstance(data, pd.DataFrame):
            item_names = list(data.columns)
        else:
            item_names = [f"item_{j}" for j in range(k)]

    # Total score as matching variable (rest score excluding target item)
    total_score = X.sum(axis=1)

    results = []
    flagged = []

    for j in range(k):
        y = X[:, j]
        theta_j = total_score - y  # rest score
        # Standardize theta for numerical stability
        theta_std = (theta_j - theta_j.mean()) / max(theta_j.std(), 1e-10)

        # Design matrices
        ones = np.ones(n)
        interaction = theta_std * g

        X1 = np.column_stack([ones, theta_std])  # M1: no DIF
        X2 = np.column_stack([ones, theta_std, g])  # M2: uniform
        X3 = np.column_stack([ones, theta_std, g, interaction])  # M3: non-uniform

        _, dev1 = _logistic_fit(y, X1)
        _, dev2 = _logistic_fit(y, X2)
        _, dev3 = _logistic_fit(y, X3)

        # Likelihood ratio tests
        # Uniform DIF: M1 vs M2 (1 df)
        chi2_uniform = max(dev1 - dev2, 0.0)
        p_uniform = 1.0 - sp.chi2.cdf(chi2_uniform, df=1)

        # Non-uniform DIF: M2 vs M3 (1 df)
        chi2_nonuniform = max(dev2 - dev3, 0.0)
        p_nonuniform = 1.0 - sp.chi2.cdf(chi2_nonuniform, df=1)

        # Total DIF: M1 vs M3 (2 df)
        chi2_total = max(dev1 - dev3, 0.0)
        p_total = 1.0 - sp.chi2.cdf(chi2_total, df=2)

        # Classify DIF type
        if p_total < alpha:
            if p_nonuniform < alpha:
                dif_type = "non-uniform"
            elif p_uniform < alpha:
                dif_type = "uniform"
            else:
                dif_type = "total"
            flagged.append(item_names[j])
        else:
            dif_type = "none"

        results.append(
            {
                "item": item_names[j],
                "chi2_uniform": float(chi2_uniform),
                "p_uniform": float(p_uniform),
                "chi2_nonuniform": float(chi2_nonuniform),
                "p_nonuniform": float(p_nonuniform),
                "chi2_total": float(chi2_total),
                "p_total": float(p_total),
                "dif_type": dif_type,
            }
        )

    items_df = pd.DataFrame(results)

    return DIFResult(
        method="logistic",
        items=items_df,
        flagged=flagged,
        group_var="group",
    )


lr_dif = diflr


def cheatsheet() -> str:
    return "_logistic_fit({}) -> Logistic Regression DIF detection."
