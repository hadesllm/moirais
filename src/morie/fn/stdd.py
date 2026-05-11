"""Staggered difference-in-differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def staggered_did(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    unit: str = "unit",
    time: str = "time",
    treat_time: str = "treat_time",
    alpha: float = 0.05,
) -> ESRes:
    r"""Staggered DiD estimator (simple cohort-weighted average).

    When treatment rolls out at different times across units, naive TWFE
    can be biased. This estimator computes cohort-specific 2x2 DiD
    estimates and averages them:

    .. math::

        \hat{\tau} = \sum_{g} w_g \hat{\tau}_g

    where :math:`\hat{\tau}_g` is the DiD estimate for cohort g
    (units treated at time g) vs. not-yet-treated, and
    :math:`w_g = n_g / \sum n_g`.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with unit, time, outcome, and treatment-onset columns.
    y : str
        Outcome column.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treat_time : str
        Column indicating the first treatment period for each unit.
        Use np.inf or a very large number for never-treated units.
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Goodman-Bacon, A. (2021). Difference-in-differences with variation
    in treatment timing. *J. Econometrics*, 225(2), 254-277.

    Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects
    in event studies. *J. Econometrics*, 225(2), 175-199.
    """
    _validate_df(data, y, unit, time, treat_time)
    df = data[[y, unit, time, treat_time]].dropna()

    times = np.sort(df[time].unique())
    treat_times = df.groupby(unit)[treat_time].first()

    cohorts = treat_times.unique()
    cohorts = cohorts[np.isfinite(cohorts)]
    cohorts = np.sort(cohorts)

    never_treated = treat_times[~np.isfinite(treat_times)].index

    tau_list = []
    w_list = []

    for g in cohorts:
        g_units = treat_times[treat_times == g].index
        if len(g_units) == 0:
            continue

        pre_periods = times[times < g]
        post_periods = times[times >= g]
        if len(pre_periods) == 0 or len(post_periods) == 0:
            continue

        control_units = never_treated
        if len(control_units) == 0:
            not_yet = treat_times[treat_times > post_periods.max()]
            control_units = not_yet.index

        if len(control_units) == 0:
            continue

        last_pre = pre_periods.max()
        first_post = post_periods.min()

        g_pre = df[(df[unit].isin(g_units)) & (df[time] == last_pre)][y].to_numpy(dtype=float)
        g_post = df[(df[unit].isin(g_units)) & (df[time] == first_post)][y].to_numpy(dtype=float)
        c_pre = df[(df[unit].isin(control_units)) & (df[time] == last_pre)][y].to_numpy(dtype=float)
        c_post = df[(df[unit].isin(control_units)) & (df[time] == first_post)][y].to_numpy(dtype=float)

        if len(g_pre) == 0 or len(g_post) == 0 or len(c_pre) == 0 or len(c_post) == 0:
            continue

        tau_g = (g_post.mean() - g_pre.mean()) - (c_post.mean() - c_pre.mean())
        tau_list.append(float(tau_g))
        w_list.append(len(g_units))

    if len(tau_list) == 0:
        raise ValueError("No valid cohort comparisons found")

    w_arr = np.array(w_list, dtype=float)
    w_arr /= w_arr.sum()
    tau_arr = np.array(tau_list)

    tau = float(np.sum(w_arr * tau_arr))

    if len(tau_arr) > 1:
        se = float(np.sqrt(np.sum(w_arr ** 2 * (tau_arr - tau) ** 2)))
    else:
        se = float("nan")

    z = stats.norm.ppf(1 - alpha / 2)
    return ESRes(
        measure="Staggered DiD",
        estimate=tau,
        ci_lower=tau - z * se if np.isfinite(se) else float("nan"),
        ci_upper=tau + z * se if np.isfinite(se) else float("nan"),
        se=se,
        n=len(df),
        extra={
            "n_cohorts": len(tau_list),
            "cohort_estimates": tau_list,
            "cohort_weights": w_arr.tolist(),
        },
    )


stdd = staggered_did


def cheatsheet() -> str:
    return "staggered_did({}) -> Staggered difference-in-differences."
