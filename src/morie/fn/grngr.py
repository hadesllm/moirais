# morie.fn -- function file (hadesllm/morie)
"""Granger causality test."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import TestResult


def granger_cause(x, y, cdf=None, *, max_lag: int = 4) -> TestResult:
    """Granger causality: does x help predict y beyond y's own history?

    Parameters
    ----------
    x : array-like
        Potential causal series.
    y : array-like
        Outcome series.
    max_lag : int
        Number of lags. Default 4.

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    # Restricted model: y_t = c + sum(a_i * y_{t-i})
    Y_dep = y[max_lag:]
    n_eff = len(Y_dep)
    X_r = np.column_stack([np.ones(n_eff)] + [y[max_lag - i - 1 : n - i - 1] for i in range(max_lag)])
    # Unrestricted: add x lags
    X_u = np.column_stack([X_r] + [x[max_lag - i - 1 : n - i - 1] for i in range(max_lag)])
    beta_r = np.linalg.lstsq(X_r, Y_dep, rcond=None)[0]
    beta_u = np.linalg.lstsq(X_u, Y_dep, rcond=None)[0]
    ss_r = float(np.sum((Y_dep - X_r @ beta_r) ** 2))
    ss_u = float(np.sum((Y_dep - X_u @ beta_u) ** 2))
    df1 = max_lag
    df2 = n_eff - X_u.shape[1]
    f_stat = ((ss_r - ss_u) / df1) / (ss_u / df2) if ss_u > 0 and df2 > 0 else 0
    p_val = float(1 - _st.f.cdf(f_stat, df1, df2))
    return TestResult(
        test_name="Granger causality",
        statistic=float(f_stat),
        p_value=p_val,
        df=float(df1),
        n=n_eff,
        method=f"Granger causality (lag={max_lag})",
        extra={"ss_restricted": ss_r, "ss_unrestricted": ss_u, "max_lag": max_lag},
    )


grngr = granger_cause


def cheatsheet() -> str:
    return "granger_cause({}) -> Granger causality test."
