# morie.fn -- function file (hadesllm/morie)
"""An unexamined life is not worth living. -- Socrates"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def diff_in_diff(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    post: str = "post",
) -> ESRes:
    """Difference-in-differences estimator (2x2 design)."""
    _validate_df(data, y, t, post)
    df = data[[y, t, post]].dropna()

    # Four group means
    y_t1_post1 = df.loc[(df[t] == 1) & (df[post] == 1), y].to_numpy(dtype=float)
    y_t1_post0 = df.loc[(df[t] == 1) & (df[post] == 0), y].to_numpy(dtype=float)
    y_t0_post1 = df.loc[(df[t] == 0) & (df[post] == 1), y].to_numpy(dtype=float)
    y_t0_post0 = df.loc[(df[t] == 0) & (df[post] == 0), y].to_numpy(dtype=float)
    for g, name in [
        (y_t1_post1, "T=1,Post=1"),
        (y_t1_post0, "T=1,Post=0"),
        (y_t0_post1, "T=0,Post=1"),
        (y_t0_post0, "T=0,Post=0"),
    ]:
        if len(g) < 1:
            raise ValueError(f"Empty cell: {name}")

    did_est = (y_t1_post1.mean() - y_t1_post0.mean()) - (y_t0_post1.mean() - y_t0_post0.mean())

    # SE via OLS: y = b0 + b1*T + b2*Post + b3*T*Post + e
    T_arr = df[t].to_numpy(dtype=float)
    P_arr = df[post].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), T_arr, P_arr, T_arr * P_arr])
    Y_arr = df[y].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, Y_arr, rcond=None)[0]
    resid = Y_arr - X @ beta
    n = len(df)
    mse = float(np.sum(resid**2)) / (n - 4)
    try:
        se_did = float(np.sqrt(mse * np.linalg.inv(X.T @ X)[3, 3]))
    except np.linalg.LinAlgError:
        se_did = float("nan")

    z = stats.norm.ppf(0.975)
    return ESRes(
        measure="DiD",
        estimate=float(did_est),
        ci_lower=float(did_est - z * se_did),
        ci_upper=float(did_est + z * se_did),
        se=se_did,
        n=n,
    )


did = diff_in_diff


def cheatsheet() -> str:
    return "diff_in_diff({}) -> Difference-in-differences. 'I have the high ground!' -- Obi-"
