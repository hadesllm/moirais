# morie.fn — function file (hadesllm/morie)
"""Difference-in-Differences for policy change in OTIS data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats


def otis_did_policy(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", group_col: str = "region", time_col: str = "end_fiscal_year", post_year: int = 2020, treatment_group: str | None = None) -> dict:
    """Estimate DiD effect of a policy change.

    Uses a 2x2 DiD design: pre/post period x treatment/control group.
    If treatment_group is None, treats the first group alphabetically
    as treatment and all others as control.

    Parameters
    ----------
    df : DataFrame
        Panel or repeated cross-section data.
    outcome : str
        Outcome column.
    group_col : str
        Column identifying treatment vs control groups.
    time_col : str
        Column with time period (numeric year).
    post_year : int
        Year at which policy takes effect (inclusive).
    treatment_group : str, optional
        Value in group_col that identifies the treated group.

    Returns
    -------
    dict
        Keys: did_estimate, se, pval, ci_lower, ci_upper, n,
              pre_treat, pre_ctrl, post_treat, post_ctrl.
    """
    data = df[[outcome, group_col, time_col]].dropna().copy()
    data["_post"] = (data[time_col] >= post_year).astype(float)

    groups = sorted(data[group_col].unique())
    if treatment_group is None:
        treatment_group = groups[0]

    data["_treat"] = (data[group_col] == treatment_group).astype(float)
    data["_interact"] = data["_post"] * data["_treat"]

    y = data[outcome].values.astype(np.float64)
    X = np.column_stack(
        [
            np.ones(len(y)),
            data["_post"].values,
            data["_treat"].values,
            data["_interact"].values,
        ]
    )
    n = len(y)

    beta, _, _, _ = lstsq(X, y, rcond=None)
    did = float(beta[3])

    # Robust SE
    resid = y - X @ beta
    bread_inv = np.linalg.inv(X.T @ X)
    meat = X.T @ np.diag(resid**2) @ X
    vcov = bread_inv @ meat @ bread_inv
    se = float(np.sqrt(vcov[3, 3]))
    se = max(se, 1e-10)
    z = did / se
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Cell means
    pre_t = data.loc[(data["_post"] == 0) & (data["_treat"] == 1), outcome].mean()
    pre_c = data.loc[(data["_post"] == 0) & (data["_treat"] == 0), outcome].mean()
    post_t = data.loc[(data["_post"] == 1) & (data["_treat"] == 1), outcome].mean()
    post_c = data.loc[(data["_post"] == 1) & (data["_treat"] == 0), outcome].mean()

    return {
        "did_estimate": did,
        "se": se,
        "pval": pval,
        "ci_lower": did - 1.96 * se,
        "ci_upper": did + 1.96 * se,
        "n": n,
        "pre_treat": float(pre_t) if pd.notna(pre_t) else np.nan,
        "pre_ctrl": float(pre_c) if pd.notna(pre_c) else np.nan,
        "post_treat": float(post_t) if pd.notna(post_t) else np.nan,
        "post_ctrl": float(post_c) if pd.notna(post_c) else np.nan,
    }


def cheatsheet() -> str:
    return "otis_did_policy({}) -> Difference-in-Differences for policy change in OTIS data."
