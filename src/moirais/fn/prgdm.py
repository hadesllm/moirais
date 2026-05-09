# moirais.fn — function file (hadesllm/moirais)
"""DML for program causal effect."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from moirais.fn._containers import ESRes


def program_dml(df, cdf=None, *, outcome_col: str = "outcome", treatment_col: str = "treatment", covariate_cols: list[str] | None = None) -> ESRes:
    """Estimate program causal effect via simplified DML (Partially Linear Regression).

    Uses residual-on-residual regression with OLS as nuisance learner.
    For production use, prefer moirais.fn.plr or moirais.fn.dml with ML nuisance learners.

    Parameters
    ----------
    df : DataFrame
    outcome_col : str
    treatment_col : str
    covariate_cols : list[str], optional

    Returns
    -------
    ESRes
    """

    if outcome_col not in df.columns or treatment_col not in df.columns:
        raise ValueError("outcome and treatment columns required")
    y = df[outcome_col].values.astype(float)
    d = df[treatment_col].values.astype(float)
    if covariate_cols:
        X = df[covariate_cols].values.astype(float)
    else:
        xcols = [c for c in df.columns if c not in [outcome_col, treatment_col]]
        X = df[xcols].values.astype(float) if xcols else np.ones((len(y), 1))

    XtXi = np.linalg.pinv(X.T @ X)
    y_res = y - X @ (XtXi @ (X.T @ y))
    d_res = d - X @ (XtXi @ (X.T @ d))

    denom = d_res @ d_res
    if denom < 1e-12:
        raise ValueError("Treatment has no residual variation after partialling out covariates")
    theta = float((d_res @ y_res) / denom)
    resid = y_res - theta * d_res
    sigma2 = float(np.sum(resid**2) / (len(y) - 1))
    se = float(np.sqrt(sigma2 / denom))
    z = theta / se if se > 0 else float("inf")
    pval = float(2 * (1 - sp_stats.norm.cdf(abs(z))))
    return ESRes(
        measure="program_dml_ate",
        estimate=theta,
        ci_lower=theta - 1.96 * se,
        ci_upper=theta + 1.96 * se,
        se=se,
        n=len(y),
        extra={"z_stat": z, "p_value": pval},
    )


prgdm = program_dml


def cheatsheet() -> str:
    return "program_dml({}) -> DML for program causal effect."
