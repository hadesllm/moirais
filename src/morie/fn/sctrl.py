# morie.fn -- function file (hadesllm/morie)
"""Synthetic control method (Abadie, Diamond, Hainmueller)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ._containers import ESRes
from ._helpers import _validate_df


def synthetic_control(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    unit: str = "unit",
    time: str = "time",
    treated_unit: str | int | None = None,
    treat_time: float | None = None,
    alpha: float = 0.05,
) -> ESRes:
    r"""Synthetic Control Method (Abadie, Diamond, Hainmueller, 2010).

    Constructs a weighted combination of control units to approximate
    the treated unit's pre-treatment trajectory:

    .. math::

        \hat{Y}^{synth}_{1t} = \sum_{j=2}^{J+1} w_j^* Y_{jt}

    where weights solve:

    .. math::

        w^* = \arg\min_{w \geq 0, \sum w = 1}
        \sum_{t \in T_0} (Y_{1t} - \sum_j w_j Y_{jt})^2

    The treatment effect is:

    .. math::

        \hat{\tau}_t = Y_{1t} - \hat{Y}^{synth}_{1t}
        \quad \forall t \in T_{post}

    Parameters
    ----------
    data : pd.DataFrame
        Panel data (long format).
    y : str
        Outcome column.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : str, int, or None
        Identifier of the treated unit. If None, uses the first unit.
    treat_time : float or None
        Time of treatment onset. Required.
    alpha : float
        Significance level.

    Returns
    -------
    ESRes
        estimate = average post-treatment effect.
        extra includes per-period gaps and weights.

    References
    ----------
    Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control
    methods for comparative case studies. *JASA*, 105(490), 493-505.

    Abadie, A. (2021). Using synthetic controls: Feasibility, data
    requirements, and methodological aspects. *JEL*, 59(2), 391-425.
    """
    if treat_time is None:
        raise ValueError("treat_time is required")
    _validate_df(data, y, unit, time)
    df = data[[y, unit, time]].dropna()

    units = df[unit].unique()
    if treated_unit is None:
        treated_unit = units[0]

    if treated_unit not in units:
        raise ValueError(f"treated_unit '{treated_unit}' not found")

    control_units = [u for u in units if u != treated_unit]
    if len(control_units) == 0:
        raise ValueError("No control units available")

    times = np.sort(df[time].unique())
    pre_times = times[times < treat_time]
    post_times = times[times >= treat_time]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period")

    panel = df.pivot_table(index=unit, columns=time, values=y, aggfunc="mean")

    Y_treat_pre = panel.loc[treated_unit, pre_times].to_numpy(dtype=float)
    Y_ctrl_pre = panel.loc[control_units, pre_times].to_numpy(dtype=float)

    n_ctrl = len(control_units)

    def obj(w):
        synth = w @ Y_ctrl_pre
        return float(np.sum((Y_treat_pre - synth) ** 2))

    w0 = np.ones(n_ctrl) / n_ctrl
    bounds = [(0, 1)] * n_ctrl
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    result = minimize(obj, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints)
    w_star = result.x

    Y_treat_post = panel.loc[treated_unit, post_times].to_numpy(dtype=float)
    Y_ctrl_post = panel.loc[control_units, post_times].to_numpy(dtype=float)

    synth_post = w_star @ Y_ctrl_post
    gaps = Y_treat_post - synth_post
    att = float(gaps.mean())

    synth_pre = w_star @ Y_ctrl_pre
    pre_rmse = float(np.sqrt(np.mean((Y_treat_pre - synth_pre) ** 2)))

    placebo_gaps = []
    for i, cu in enumerate(control_units):
        other_idx = [j for j in range(n_ctrl) if j != i]
        if len(other_idx) == 0:
            continue
        Y_pbo_pre = panel.loc[cu, pre_times].to_numpy(dtype=float)
        Y_others_pre = Y_ctrl_pre[other_idx]

        def pbo_obj(w):
            return float(np.sum((Y_pbo_pre - w @ Y_others_pre) ** 2))

        w0_p = np.ones(len(other_idx)) / len(other_idx)
        bounds_p = [(0, 1)] * len(other_idx)
        cons_p = {"type": "eq", "fun": lambda w: w.sum() - 1}
        res_p = minimize(pbo_obj, w0_p, method="SLSQP", bounds=bounds_p,
                         constraints=cons_p)

        Y_pbo_post = panel.loc[cu, post_times].to_numpy(dtype=float)
        Y_others_post = Y_ctrl_post[other_idx]
        synth_pbo = res_p.x @ Y_others_post
        placebo_gaps.append(float((Y_pbo_post - synth_pbo).mean()))

    if len(placebo_gaps) > 1:
        se = float(np.std(placebo_gaps, ddof=1))
        p_val = float(np.mean(np.abs(placebo_gaps) >= abs(att)))
    else:
        se = float("nan")
        p_val = float("nan")

    z = stats.norm.ppf(1 - alpha / 2)

    return ESRes(
        measure="Synthetic Control ATT",
        estimate=att,
        ci_lower=att - z * se if np.isfinite(se) else float("nan"),
        ci_upper=att + z * se if np.isfinite(se) else float("nan"),
        se=se,
        n=len(df),
        extra={
            "weights": {str(u): float(w) for u, w in zip(control_units, w_star)},
            "gaps": gaps.tolist(),
            "pre_rmse": pre_rmse,
            "p_value_placebo": p_val,
            "treated_unit": str(treated_unit),
            "n_pre": len(pre_times),
            "n_post": len(post_times),
        },
    )


sctrl = synthetic_control


def cheatsheet() -> str:
    return "synthetic_control({}) -> Synthetic control method (Abadie)."
