# morie.fn — function file (hadesllm/morie)
"""Complier Average Causal Effect (CACE/LATE)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def complier_ate(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    z: str = "instrument",
    alpha: float = 0.05,
) -> ESRes:
    r"""Complier Average Causal Effect via Wald/IV estimator.

    For a binary instrument Z and binary treatment T, the CACE (LATE) is:

    .. math::

        CACE = \frac{E[Y|Z=1] - E[Y|Z=0]}{E[T|Z=1] - E[T|Z=0]}
        = \frac{ITT_Y}{ITT_T}

    This identifies the treatment effect among *compliers* — units
    who take treatment when assigned (Z=1) and don't when not (Z=0).

    Standard error via delta method:

    .. math::

        SE = \frac{1}{ITT_T} \sqrt{Var(Y|Z=1)/n_1 + Var(Y|Z=0)/n_0
        + CACE^2 \cdot [Var(T|Z=1)/n_1 + Var(T|Z=0)/n_0]}

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    t : str
        Treatment received column.
    z : str
        Instrument (randomized assignment) column.
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation
    of local average treatment effects. *Econometrica*, 62(2), 467-475.
    """
    _validate_df(data, y, t, z)
    df = data[[y, t, z]].dropna()

    Z = df[z].to_numpy(dtype=float)
    T_arr = df[t].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)

    z1 = Z == 1
    z0 = Z == 0
    n1 = int(z1.sum())
    n0 = int(z0.sum())

    if n1 < 2 or n0 < 2:
        raise ValueError("Need >=2 observations in each instrument group")

    itt_y = float(Y[z1].mean() - Y[z0].mean())
    itt_t = float(T_arr[z1].mean() - T_arr[z0].mean())

    if abs(itt_t) < 1e-10:
        raise ValueError("First stage too weak: E[T|Z=1] ≈ E[T|Z=0]")

    cace = itt_y / itt_t

    var_y1 = Y[z1].var(ddof=1) / n1
    var_y0 = Y[z0].var(ddof=1) / n0
    var_t1 = T_arr[z1].var(ddof=1) / n1
    var_t0 = T_arr[z0].var(ddof=1) / n0

    se = float(np.sqrt(
        (var_y1 + var_y0 + cace ** 2 * (var_t1 + var_t0)) / itt_t ** 2
    ))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    compliance_rate = float(itt_t)

    return ESRes(
        measure="CACE/LATE",
        estimate=cace,
        ci_lower=cace - z_crit * se,
        ci_upper=cace + z_crit * se,
        se=se,
        n=n1 + n0,
        extra={
            "itt_y": itt_y,
            "itt_t": itt_t,
            "compliance_rate": compliance_rate,
            "n_assigned": n1,
            "n_control": n0,
        },
    )


cmplr = complier_ate


def cheatsheet() -> str:
    return "complier_ate({}) -> Complier Average Causal Effect (CACE/LATE)."
