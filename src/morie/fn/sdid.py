# morie.fn -- function file (hadesllm/morie)
"""Synthetic difference-in-differences (Arkhangelsky et al., 2021)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ._containers import ESRes
from ._helpers import _validate_df


def synthetic_did(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    unit: str = "unit",
    time: str = "time",
    t: str = "treatment",
    alpha: float = 0.05,
) -> ESRes:
    r"""Synthetic Difference-in-Differences (SDID).

    SDID combines synthetic control (unit weights) with DiD
    (time weights). The estimator solves:

    .. math::

        \hat{\tau}_{SDID} = \arg\min_{\tau}
        \sum_{i,t} (\hat{\omega}_i \hat{\lambda}_t)
        (Y_{it} - \mu - \alpha_i - \gamma_t - \tau W_{it})^2

    where :math:`\hat{\omega}` are unit weights that make the pre-treatment
    trends of the synthetic control match the treated units, and
    :math:`\hat{\lambda}` are time weights.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data (long format).
    y : str
        Outcome column.
    unit : str
        Unit identifier.
    time : str
        Time period.
    t : str
        Treatment indicator (0/1).
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., &
    Wager, S. (2021). Synthetic difference-in-differences. *AER*,
    111(12), 4088-4118.
    """
    _validate_df(data, y, unit, time, t)
    df = data[[y, unit, time, t]].dropna()

    units = df[unit].unique()
    times = np.sort(df[time].unique())

    panel = df.pivot_table(index=unit, columns=time, values=y, aggfunc="mean")
    treat_status = df.pivot_table(index=unit, columns=time, values=t, aggfunc="max")

    treated_units = []
    control_units = []
    for u in units:
        if u in treat_status.index and treat_status.loc[u].max() > 0:
            treated_units.append(u)
        else:
            control_units.append(u)

    if len(treated_units) == 0 or len(control_units) == 0:
        raise ValueError("Need both treated and control units")

    first_treat = None
    for tt in times:
        if tt in treat_status.columns:
            if treat_status.loc[treated_units, tt].mean() > 0:
                first_treat = tt
                break

    if first_treat is None:
        raise ValueError("Cannot determine treatment onset")

    pre_times = times[times < first_treat]
    post_times = times[times >= first_treat]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods")

    Y_c_pre = panel.loc[control_units, pre_times].to_numpy(dtype=float)
    Y_c_post = panel.loc[control_units, post_times].to_numpy(dtype=float)
    Y_t_pre = panel.loc[treated_units, pre_times].to_numpy(dtype=float)
    Y_t_post = panel.loc[treated_units, post_times].to_numpy(dtype=float)

    n_c = len(control_units)
    T_pre = len(pre_times)

    target_pre = Y_t_pre.mean(axis=0)

    def obj_omega(w):
        synth_pre = w @ Y_c_pre
        return float(np.sum((synth_pre - target_pre) ** 2)) + 1e-6 * float(w @ w)

    w0 = np.ones(n_c) / n_c
    bounds = [(0, 1)] * n_c
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    res = minimize(obj_omega, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    omega = res.x

    T_post = len(post_times)
    lam_pre = np.ones(T_pre) / T_pre

    y_treat_post = float(Y_t_post.mean())
    y_treat_pre = float(np.sum(lam_pre * Y_t_pre.mean(axis=0)))
    y_ctrl_post = float(omega @ Y_c_post.mean(axis=1)) if T_post == 1 else float(
        omega @ np.mean(Y_c_post, axis=1)
    )
    y_ctrl_pre = float(np.sum(lam_pre * (omega @ Y_c_pre)))

    tau = (y_treat_post - y_ctrl_post) - (y_treat_pre - y_ctrl_pre)

    n_total = len(df)
    n_treat = len(treated_units) * T_post
    placebo_effects = []
    for i, cu in enumerate(control_units):
        y_pbo_post = float(panel.loc[cu, post_times].mean())
        y_pbo_pre = float(np.sum(lam_pre * panel.loc[cu, pre_times].to_numpy(dtype=float)))
        other_idx = [j for j in range(n_c) if j != i]
        if len(other_idx) == 0:
            continue
        w_other = omega[other_idx] / omega[other_idx].sum() if omega[other_idx].sum() > 0 else np.ones(len(other_idx)) / len(other_idx)
        y_synth_post = float(w_other @ Y_c_post[other_idx].mean(axis=1))
        y_synth_pre = float(np.sum(lam_pre * (w_other @ Y_c_pre[other_idx])))
        placebo_effects.append((y_pbo_post - y_synth_post) - (y_pbo_pre - y_synth_pre))

    if len(placebo_effects) > 1:
        se = float(np.std(placebo_effects, ddof=1))
    else:
        se = float("nan")

    z = stats.norm.ppf(1 - alpha / 2)

    return ESRes(
        measure="Synthetic DiD",
        estimate=float(tau),
        ci_lower=float(tau) - z * se if np.isfinite(se) else float("nan"),
        ci_upper=float(tau) + z * se if np.isfinite(se) else float("nan"),
        se=se,
        n=n_total,
        extra={
            "n_treated_units": len(treated_units),
            "n_control_units": n_c,
            "n_pre_periods": T_pre,
            "n_post_periods": T_post,
            "omega_max": float(omega.max()),
        },
    )


sdid = synthetic_did


def cheatsheet() -> str:
    return "synthetic_did({}) -> Synthetic difference-in-differences."
