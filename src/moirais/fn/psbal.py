# moirais.fn — function file (hadesllm/moirais)
"""Propensity score balancing weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _validate_df


def ps_balance(
    data: pd.DataFrame,
    *,
    t: str = "treatment",
    covariates: list[str] | None = None,
    ps_col: str | None = None,
    alpha: float = 0.05,
) -> dict:
    r"""Compute propensity score balancing weights and diagnostics.

    If ``ps_col`` is provided, uses those propensity scores directly.
    Otherwise estimates P(T=1|X) via logistic regression on ``covariates``.

    Balancing weights for ATE:

    .. math::

        w_i = \frac{T_i}{e(X_i)} + \frac{1 - T_i}{1 - e(X_i)}

    Returns a dict with weights array and balance diagnostics
    (standardized mean differences before/after weighting).

    Parameters
    ----------
    data : pd.DataFrame
    t : str
        Binary treatment column.
    covariates : list[str] or None
        Covariate columns for PS estimation (required if ps_col is None).
    ps_col : str or None
        Pre-computed propensity score column.
    alpha : float
        Significance level for CI on weighted ATE.

    Returns
    -------
    dict
        Keys: 'weights', 'ps', 'smd_before', 'smd_after'.

    References
    ----------
    Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the
    propensity score in observational studies. *Biometrika*, 70(1), 41-55.
    """
    _validate_df(data, t)
    df = data.dropna(subset=[t])
    T = df[t].to_numpy(dtype=float)

    if ps_col is not None:
        ps = df[ps_col].to_numpy(dtype=float)
    else:
        if covariates is None or len(covariates) == 0:
            raise ValueError("Must provide covariates or ps_col")
        _validate_df(df, *covariates)
        X = df[covariates].to_numpy(dtype=float)
        X_design = np.column_stack([np.ones(len(X)), X])
        beta = np.zeros(X_design.shape[1])
        for _ in range(50):
            p = 1.0 / (1.0 + np.exp(-X_design @ beta))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            W_diag = p * (1 - p)
            grad = X_design.T @ (T - p)
            H = X_design.T @ (X_design * W_diag[:, None])
            try:
                beta += np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
        ps = 1.0 / (1.0 + np.exp(-X_design @ beta))

    ps = np.clip(ps, 0.01, 0.99)
    weights = T / ps + (1 - T) / (1 - ps)

    smd_before = {}
    smd_after = {}
    if covariates is not None:
        for c in covariates:
            x = df[c].to_numpy(dtype=float)
            x1 = x[T == 1]
            x0 = x[T == 0]
            pooled_sd = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
            if pooled_sd > 0:
                smd_before[c] = float((x1.mean() - x0.mean()) / pooled_sd)
            else:
                smd_before[c] = 0.0

            w1 = weights[T == 1]
            w0 = weights[T == 0]
            wm1 = np.average(x[T == 1], weights=w1)
            wm0 = np.average(x[T == 0], weights=w0)
            if pooled_sd > 0:
                smd_after[c] = float((wm1 - wm0) / pooled_sd)
            else:
                smd_after[c] = 0.0

    return {
        "weights": weights,
        "ps": ps,
        "smd_before": smd_before,
        "smd_after": smd_after,
        "n": len(T),
    }


psbal = ps_balance


def cheatsheet() -> str:
    return "ps_balance({}) -> Propensity score balancing weights."
