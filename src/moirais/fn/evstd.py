# moirais.fn — function file (hadesllm/moirais)
"""Event study design estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._helpers import _validate_df


def event_study(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    unit: str = "unit",
    time: str = "time",
    treat_time: str = "treat_time",
    pre_window: int = 5,
    post_window: int = 5,
    ref_period: int = -1,
    alpha: float = 0.05,
) -> dict:
    r"""Event study design with dynamic treatment effects.

    Estimates relative-time coefficients:

    .. math::

        Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq ref}
        \tau_k \cdot 1(t - g_i = k) + \epsilon_{it}

    where :math:`g_i` is unit i's treatment time and k indexes
    event-time relative to treatment onset. Period ``ref_period``
    (default -1) is the omitted reference category.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data (long format).
    y : str
        Outcome column.
    unit, time : str
        Unit and time identifier columns.
    treat_time : str
        Column with first treatment period (inf for never-treated).
    pre_window : int
        Number of pre-treatment periods to estimate.
    post_window : int
        Number of post-treatment periods to estimate.
    ref_period : int
        Reference period (relative time, default -1).
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Keys: 'coefficients' (relative_time -> estimate),
        'se', 'ci_lower', 'ci_upper', 'p_values', 'n'.

    References
    ----------
    Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects
    in event studies. *J. Econometrics*, 225(2), 175-199.
    """
    _validate_df(data, y, unit, time, treat_time)
    df = data[[y, unit, time, treat_time]].dropna().copy()

    df["_rel_time"] = df[time].astype(float) - df[treat_time].astype(float)

    never_treated = ~np.isfinite(df["_rel_time"])
    df.loc[never_treated, "_rel_time"] = np.nan

    rel_times = list(range(-pre_window, post_window + 1))
    rel_times = [k for k in rel_times if k != ref_period]

    Y = df[y].to_numpy(dtype=float)
    n = len(df)

    unit_codes = pd.Categorical(df[unit]).codes
    time_codes = pd.Categorical(df[time]).codes
    n_units = len(np.unique(unit_codes))
    n_times = len(np.unique(time_codes))

    unit_means = np.zeros(n)
    time_means = np.zeros(n)
    for i in range(n_units):
        mask = unit_codes == i
        unit_means[mask] = Y[mask].mean()
    for j in range(n_times):
        mask = time_codes == j
        time_means[mask] = Y[mask].mean()
    grand_mean = Y.mean()
    Y_dm = Y - unit_means - time_means + grand_mean

    rel_time_arr = df["_rel_time"].to_numpy(dtype=float)
    D = np.zeros((n, len(rel_times)))
    for j, k in enumerate(rel_times):
        D[:, j] = (rel_time_arr == k).astype(float)

    D_dm = np.zeros_like(D)
    for j in range(D.shape[1]):
        d_j = D[:, j]
        d_unit = np.zeros(n)
        d_time = np.zeros(n)
        for i in range(n_units):
            mask = unit_codes == i
            d_unit[mask] = d_j[mask].mean()
        for t_idx in range(n_times):
            mask = time_codes == t_idx
            d_time[mask] = d_j[mask].mean()
        D_dm[:, j] = d_j - d_unit - d_time + d_j.mean()

    try:
        beta = np.linalg.lstsq(D_dm, Y_dm, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.full(len(rel_times), np.nan)

    resid = Y_dm - D_dm @ beta
    df_resid = max(n - n_units - n_times + 1 - len(rel_times), 1)
    sigma2 = float(resid @ resid) / df_resid

    try:
        cov = sigma2 * np.linalg.inv(D_dm.T @ D_dm)
    except np.linalg.LinAlgError:
        cov = np.full((len(rel_times), len(rel_times)), np.nan)

    se_arr = np.sqrt(np.maximum(np.diag(cov), 0))
    z = stats.norm.ppf(1 - alpha / 2)

    coefficients = {}
    se_dict = {}
    ci_lo = {}
    ci_hi = {}
    p_dict = {}
    for j, k in enumerate(rel_times):
        coefficients[k] = float(beta[j])
        se_dict[k] = float(se_arr[j])
        ci_lo[k] = float(beta[j] - z * se_arr[j])
        ci_hi[k] = float(beta[j] + z * se_arr[j])
        if se_arr[j] > 0 and df_resid > 0:
            t_stat = beta[j] / se_arr[j]
            p_dict[k] = float(2 * stats.t.sf(abs(t_stat), df_resid))
        else:
            p_dict[k] = float("nan")

    coefficients[ref_period] = 0.0
    se_dict[ref_period] = 0.0
    ci_lo[ref_period] = 0.0
    ci_hi[ref_period] = 0.0
    p_dict[ref_period] = float("nan")

    return {
        "coefficients": coefficients,
        "se": se_dict,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_values": p_dict,
        "ref_period": ref_period,
        "n": n,
        "n_units": n_units,
        "n_times": n_times,
    }


evstd = event_study


def cheatsheet() -> str:
    return "event_study({}) -> Event study design estimator."
