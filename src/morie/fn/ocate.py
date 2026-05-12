# morie.fn -- function file (hadesllm/morie)
"""CATE by risk score tercile for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def otis_cate_risk(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", risk_col: str = "Y", n_terciles: int = 3) -> pd.DataFrame:
    """Estimate Conditional ATE by risk-score tercile.

    Splits the risk column into terciles and estimates ATE
    (difference in means) within each.

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, and risk score columns.
    outcome : str
        Outcome column.
    treatment : str
        Treatment column (binary).
    risk_col : str
        Numeric column used to define risk terciles.
    n_terciles : int
        Number of quantile groups (default 3).

    Returns
    -------
    DataFrame
        Columns: tercile, risk_lo, risk_hi, cate, se, pval, ci_lower, ci_upper, n.
    """
    cols = list(dict.fromkeys([outcome, treatment, risk_col]))  # deduplicate
    data = df[cols].dropna().copy()
    data["_tercile"] = pd.qcut(data[risk_col], q=n_terciles, labels=False, duplicates="drop")

    results = []
    for t_label, grp in data.groupby("_tercile"):
        t1 = grp.loc[grp[treatment] == 1, outcome]
        t0 = grp.loc[grp[treatment] == 0, outcome]

        if len(t1) < 2 or len(t0) < 2:
            continue

        cate = float(t1.mean() - t0.mean())
        se = float(np.sqrt(t1.var() / len(t1) + t0.var() / len(t0)))
        z = cate / se if se > 0 else 0.0
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))

        results.append(
            {
                "tercile": int(t_label),
                "risk_lo": float(grp[risk_col].min()),
                "risk_hi": float(grp[risk_col].max()),
                "cate": cate,
                "se": se,
                "pval": pval,
                "ci_lower": cate - 1.96 * se,
                "ci_upper": cate + 1.96 * se,
                "n": len(grp),
            }
        )

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_cate_risk({}) -> CATE by risk score tercile for OTIS correctional data."
