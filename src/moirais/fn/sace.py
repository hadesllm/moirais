# moirais.fn — function file (hadesllm/moirais)
"""SACE estimator (survivor average causal effect)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def sace(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    s: str = "survival",
    alpha: float = 0.05,
) -> ESRes:
    r"""Survivor Average Causal Effect (SACE).

    Estimates the treatment effect among the *always-survivors* stratum
    (units who would survive under both treatment and control).

    Under monotonicity (treatment does not cause death), the always-survivors
    are identified as survivors in the control group and all survivors in the
    treatment group minus the protected stratum.

    The principal-stratification bound uses:

    .. math::

        SACE = E[Y|T=1, S=1] - E[Y|T=0, S=0] \cdot \frac{P(S=1|T=0)}{P(S=1|T=1)}
              - E[Y|T=0, S=1] \cdot \left(1 - \frac{P(S=1|T=0)}{P(S=1|T=1)}\right)

    Under monotonicity, a simpler trimming estimator is used: trim the
    top ``P(S=1|T=1) - P(S=1|T=0)`` fraction of treated survivors by
    outcome, then compare means.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column (observed only if S=1).
    t : str
        Binary treatment column.
    s : str
        Binary survival/selection indicator (1=observed).
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Rubin, D. B. (2006). Causal inference through potential outcomes and
    principal stratification. *Statistical Science*, 21(3), 299-309.
    Zhang, J. L., & Rubin, D. B. (2003). Estimation of causal effects via
    principal stratification when some outcomes are truncated by death.
    *J. Educ. Behav. Stat.*, 28(4), 353-368.
    """
    _validate_df(data, y, t, s)
    df = data[[y, t, s]].dropna(subset=[t, s])

    t_arr = df[t].to_numpy(dtype=float)
    s_arr = df[s].to_numpy(dtype=float)

    p_s1_t1 = s_arr[t_arr == 1].mean()
    p_s1_t0 = s_arr[t_arr == 0].mean()

    if p_s1_t1 < 1e-10:
        raise ValueError("No survivors in treatment group")
    if p_s1_t0 < 1e-10:
        raise ValueError("No survivors in control group")

    survivors = df[df[s] == 1].copy()
    y_t1 = survivors.loc[survivors[t] == 1, y].to_numpy(dtype=float)
    y_t0 = survivors.loc[survivors[t] == 0, y].to_numpy(dtype=float)

    if len(y_t1) < 2 or len(y_t0) < 2:
        raise ValueError("Need >=2 survivors per group")

    trim_frac = max(0.0, 1.0 - p_s1_t0 / p_s1_t1)
    if trim_frac > 0:
        n_trim = int(np.ceil(trim_frac * len(y_t1)))
        sorted_idx = np.argsort(y_t1)
        keep_idx = sorted_idx[:len(y_t1) - n_trim]
        y_t1_trimmed = y_t1[keep_idx]
    else:
        y_t1_trimmed = y_t1

    if len(y_t1_trimmed) < 2:
        raise ValueError("Too few treated survivors after trimming")

    est = float(y_t1_trimmed.mean() - y_t0.mean())
    se = float(np.sqrt(
        y_t1_trimmed.var(ddof=1) / len(y_t1_trimmed)
        + y_t0.var(ddof=1) / len(y_t0)
    ))
    z = stats.norm.ppf(1 - alpha / 2)
    return ESRes(
        measure="SACE (trimming)",
        estimate=est,
        ci_lower=est - z * se,
        ci_upper=est + z * se,
        se=se,
        n=len(y_t1_trimmed) + len(y_t0),
        extra={
            "p_s1_t1": p_s1_t1,
            "p_s1_t0": p_s1_t0,
            "trim_frac": trim_frac,
            "n_trimmed": len(y_t1_trimmed),
        },
    )


def cheatsheet() -> str:
    return "sace({}) -> SACE estimator (survivor average causal effect)."
