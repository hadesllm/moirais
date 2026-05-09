# moirais.fn — function file (hadesllm/moirais)
"""Fuzzy regression discontinuity design."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def fuzzy_rd(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    r: str = "running",
    cutoff: float = 0.0,
    bandwidth: float | None = None,
) -> RegressionResult:
    r"""Fuzzy regression discontinuity design.

    In a fuzzy RD, treatment probability jumps at the cutoff but doesn't
    go from 0 to 1. The estimator is an IV/Wald ratio applied locally:

    .. math::

        \hat{\tau}_{FRD} = \frac{\hat{E}[Y|R=c^+] - \hat{E}[Y|R=c^-]}
        {\hat{E}[T|R=c^+] - \hat{E}[T|R=c^-]}

    Implemented via 2SLS with the above-cutoff indicator as instrument
    for the treatment within the bandwidth window.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    t : str
        Treatment takeup column (binary or continuous).
    r : str
        Running variable column.
    cutoff : float
        RD cutoff value.
    bandwidth : float or None
        If None, uses Silverman's rule-of-thumb.

    Returns
    -------
    RegressionResult
        The 'LATE' coefficient is the fuzzy RD estimate.

    References
    ----------
    Hahn, J., Todd, P., & Van der Klaauw, W. (2001). Identification and
    estimation of treatment effects with a regression-discontinuity design.
    *Econometrica*, 69(1), 201-209.
    """
    _validate_df(data, y, t, r)
    df = data[[y, t, r]].dropna()
    R = df[r].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    T_arr = df[t].to_numpy(dtype=float)

    if bandwidth is None:
        bandwidth = 1.06 * R.std() * len(R) ** (-0.2)

    mask = np.abs(R - cutoff) <= bandwidth
    if mask.sum() < 6:
        raise ValueError("Too few observations within bandwidth")

    R_loc = R[mask] - cutoff
    Y_loc = Y[mask]
    T_loc = T_arr[mask]
    Z_loc = (R_loc >= 0).astype(float)
    n = mask.sum()

    X_fs = np.column_stack([np.ones(n), R_loc, Z_loc, R_loc * Z_loc])
    pi_hat = np.linalg.lstsq(X_fs, T_loc, rcond=None)[0]
    T_hat = X_fs @ pi_hat

    first_stage_jump = float(pi_hat[2])
    if abs(first_stage_jump) < 1e-10:
        raise ValueError("No first-stage jump in treatment at cutoff")

    X_ss = np.column_stack([np.ones(n), R_loc, T_hat, R_loc * Z_loc])
    beta = np.linalg.lstsq(X_ss, Y_loc, rcond=None)[0]

    X_orig = np.column_stack([np.ones(n), R_loc, T_loc, R_loc * Z_loc])
    resid = Y_loc - X_orig @ beta
    p = 4
    mse = float(np.sum(resid ** 2)) / max(n - p, 1)
    try:
        cov = mse * np.linalg.inv(X_ss.T @ X_ss)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se_arr = np.sqrt(np.maximum(np.diag(cov), 0))

    names = ["intercept", "slope", "LATE", "slope_interaction"]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}
    se_dict = {nm: float(s) for nm, s in zip(names, se_arr)}
    p_dict = {}
    df_r = n - p
    for i, nm in enumerate(names):
        if se_arr[i] > 0 and df_r > 0:
            t_stat = beta[i] / se_arr[i]
            p_dict[nm] = float(2 * stats.t.sf(abs(t_stat), df_r))
        else:
            p_dict[nm] = float("nan")

    return RegressionResult(
        method="Fuzzy RDD",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        n=n,
        k=3,
        extra={
            "bandwidth": bandwidth,
            "cutoff": cutoff,
            "first_stage_jump": first_stage_jump,
        },
    )


frd = fuzzy_rd


def cheatsheet() -> str:
    return "fuzzy_rd({}) -> Fuzzy regression discontinuity design."
