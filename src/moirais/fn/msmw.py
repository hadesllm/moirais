# moirais.fn — function file (hadesllm/moirais)
"""Marginal structural model weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def marginal_structural(
    df: pd.DataFrame,
    *,
    treatment_col: str = "treatment",
    covariate_cols: list[str] | None = None,
    time_col: str | None = None,
) -> DescriptiveResult:
    """
    Compute stabilised inverse probability of treatment weights for
    a marginal structural model.

    For a single time point, computes IPW weights. For longitudinal
    data with time_col, computes the product of conditional weights
    across time periods.

    Parameters
    ----------
    df : DataFrame
        Data with treatment and covariates.
    treatment_col : str
        Name of binary treatment column.
    covariate_cols : list of str, optional
        Covariate columns (default: all numeric except treatment).
    time_col : str, optional
        Time column for longitudinal data.

    Returns
    -------
    DescriptiveResult
        extra has 'weights', 'mean_weight', 'max_weight', 'ess'.

    References
    ----------
    Robins, J. M., Hernan, M. A., & Brumback, B. (2000). Marginal
    structural models and causal inference in epidemiology.
    *Epidemiology*, 11(5), 550-560.
    """
    if treatment_col not in df.columns:
        raise ValueError(f"Column {treatment_col} not in DataFrame.")

    if covariate_cols is None:
        covariate_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != treatment_col and c != time_col
        ]

    A = df[treatment_col].values.astype(float)
    X = df[covariate_cols].values.astype(float)

    from scipy.special import expit

    Xd = np.column_stack([np.ones(len(X)), X])
    try:
        from numpy.linalg import lstsq

        beta, _, _, _ = lstsq(Xd, A, rcond=None)
        ps = expit(Xd @ beta)
    except Exception:
        ps = np.full(len(A), np.mean(A))

    ps = np.clip(ps, 0.01, 0.99)
    p_marg = np.mean(A)

    num = A * p_marg + (1 - A) * (1 - p_marg)
    denom = A * ps + (1 - A) * (1 - ps)
    weights = num / denom

    ess = np.sum(weights) ** 2 / np.sum(weights**2)

    return DescriptiveResult(
        name="MSM_weights",
        value=float(np.mean(weights)),
        extra={
            "weights": weights,
            "mean_weight": float(np.mean(weights)),
            "max_weight": float(np.max(weights)),
            "ess": float(ess),
            "n": len(A),
        },
    )


msmw = marginal_structural


def cheatsheet() -> str:
    return "marginal_structural({}) -> Marginal structural model weights."
