# moirais.fn — function file (hadesllm/moirais)
"""ATT by region via IPW for OTIS data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def otis_att_region(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", region_col: str = "region", covariates: list[str] | None = None) -> pd.DataFrame:
    """Estimate ATT via IPW within each region.

    Propensity scores estimated via logistic regression (OLS on logit
    scale approximation). ATT weights: treated get weight 1, controls
    get weight e(X)/(1 - e(X)).

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, region, and covariates.
    outcome, treatment, region_col : str
        Column names.
    covariates : list of str, optional
        Covariate columns. Auto-detected from numeric columns if None.

    Returns
    -------
    DataFrame
        Columns: region, att, se, pval, ci_lower, ci_upper, n.
    """
    results = []
    for region, grp in df.groupby(region_col):
        data = grp.select_dtypes(include="number").dropna().copy()
        if outcome not in data.columns or treatment not in data.columns:
            continue
        if len(data) < 10:
            continue

        y = data[outcome].values.astype(np.float64)
        d = data[treatment].values.astype(np.float64)

        if covariates is not None:
            cov_cols = [c for c in covariates if c in data.columns]
        else:
            cov_cols = [c for c in data.columns if c not in (outcome, treatment)]

        if len(cov_cols) == 0:
            X = np.ones((len(y), 1))
        else:
            X = np.column_stack([np.ones(len(y)), data[cov_cols].values.astype(np.float64)])

        # Propensity score via OLS (linear probability model, clipped)
        from numpy.linalg import lstsq

        beta, _, _, _ = lstsq(X, d, rcond=None)
        ps = np.clip(X @ beta, 0.01, 0.99)

        # ATT weights
        n = len(y)
        n1 = d.sum()
        if n1 == 0 or n1 == n:
            continue

        w = np.where(d == 1, 1.0, ps / (1 - ps))
        y1_weighted = np.sum(w * d * y) / np.sum(w * d)
        y0_weighted = np.sum(w * (1 - d) * y) / np.sum(w * (1 - d))
        att = float(y1_weighted - y0_weighted)

        # Influence-function based SE
        psi = w * (d - (1 - d) * ps / (1 - ps)) * (y - d * y1_weighted - (1 - d) * y0_weighted)
        se = float(np.sqrt(np.mean(psi**2) / n))
        se = max(se, 1e-10)
        z = att / se
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))

        results.append(
            {
                "region": region,
                "att": att,
                "se": se,
                "pval": pval,
                "ci_lower": att - 1.96 * se,
                "ci_upper": att + 1.96 * se,
                "n": n,
            }
        )

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_att_region({}) -> ATT by region via IPW for OTIS data."
