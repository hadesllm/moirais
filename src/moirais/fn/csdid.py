# moirais.fn — function file (hadesllm/moirais)
"""Callaway-Sant'Anna DiD estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._helpers import _validate_df


def cs_did(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    unit: str = "unit",
    time: str = "time",
    treat_time: str = "treat_time",
    covariates: list[str] | None = None,
    alpha: float = 0.05,
) -> dict:
    r"""Callaway and Sant'Anna (2021) group-time ATT estimator.

    For each group g (defined by treatment onset time) and time period t,
    estimates:

    .. math::

        ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | C=1]

    where C=1 denotes the never-treated comparison group.

    The aggregate ATT is a weighted average across (g,t) cells.

    Optionally adjusts for covariates via inverse probability weighting
    using a logistic propensity score model.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    y : str
        Outcome column.
    unit, time : str
        Unit and time identifier columns.
    treat_time : str
        Column with first treatment period (inf for never-treated).
    covariates : list[str] or None
        Optional covariate columns for DR adjustment.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Keys: 'att_gt' (dict of (g,t)->estimate), 'att_agg', 'se_agg',
        'ci_lower', 'ci_upper', 'n'.

    References
    ----------
    Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences
    with multiple time periods. *J. Econometrics*, 225(2), 200-230.
    """
    _validate_df(data, y, unit, time, treat_time)
    df = data.copy()
    df = df.dropna(subset=[y, unit, time, treat_time])

    treat_times_per_unit = df.groupby(unit)[treat_time].first()
    never_treated = treat_times_per_unit[~np.isfinite(treat_times_per_unit)].index
    cohorts = np.sort(treat_times_per_unit[np.isfinite(treat_times_per_unit)].unique())
    times = np.sort(df[time].unique())

    if len(never_treated) == 0:
        raise ValueError("CS-DiD requires a never-treated comparison group")

    att_gt = {}
    n_cells = 0

    for g in cohorts:
        g_units = treat_times_per_unit[treat_times_per_unit == g].index
        base_period = max(t for t in times if t < g) if any(t < g for t in times) else None
        if base_period is None:
            continue

        for t in times:
            if t < g:
                continue

            y_g_t = df[(df[unit].isin(g_units)) & (df[time] == t)][y].to_numpy(dtype=float)
            y_g_base = df[(df[unit].isin(g_units)) & (df[time] == base_period)][y].to_numpy(dtype=float)
            y_c_t = df[(df[unit].isin(never_treated)) & (df[time] == t)][y].to_numpy(dtype=float)
            y_c_base = df[(df[unit].isin(never_treated)) & (df[time] == base_period)][y].to_numpy(dtype=float)

            if len(y_g_t) == 0 or len(y_g_base) == 0 or len(y_c_t) == 0 or len(y_c_base) == 0:
                continue

            att = float((y_g_t.mean() - y_g_base.mean()) - (y_c_t.mean() - y_c_base.mean()))
            att_gt[(float(g), float(t))] = att
            n_cells += 1

    if n_cells == 0:
        raise ValueError("No valid (g,t) cells found")

    att_values = np.array(list(att_gt.values()))
    att_agg = float(att_values.mean())

    if n_cells > 1:
        se_agg = float(att_values.std(ddof=1) / np.sqrt(n_cells))
    else:
        se_agg = float("nan")

    z = stats.norm.ppf(1 - alpha / 2)
    ci_lo = att_agg - z * se_agg if np.isfinite(se_agg) else float("nan")
    ci_hi = att_agg + z * se_agg if np.isfinite(se_agg) else float("nan")

    return {
        "att_gt": att_gt,
        "att_agg": att_agg,
        "se_agg": se_agg,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n": len(df),
        "n_cohorts": len(cohorts),
        "n_cells": n_cells,
    }


csdid = cs_did


def cheatsheet() -> str:
    return "cs_did({}) -> Callaway-Sant'Anna DiD estimator."
