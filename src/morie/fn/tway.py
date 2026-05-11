"""Two-way fixed effects estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def twfe(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    unit: str = "unit",
    time: str = "time",
) -> RegressionResult:
    r"""Two-way fixed effects (TWFE) regression.

    Estimates:

    .. math::

        Y_{it} = \alpha_i + \gamma_t + \tau D_{it} + \epsilon_{it}

    via the Frisch-Waugh-Lovell theorem (double-demeaning):

    .. math::

        \tilde{Y}_{it} = \tau \tilde{D}_{it} + \tilde{\epsilon}_{it}

    where :math:`\tilde{X}_{it} = X_{it} - \bar{X}_i - \bar{X}_t + \bar{X}`.

    **Warning**: TWFE can be biased with staggered treatment timing and
    heterogeneous effects. Consider using staggered_did or cs_did instead.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data (long format).
    y : str
        Outcome column.
    t : str
        Treatment indicator column.
    unit : str
        Unit identifier column.
    time : str
        Time period column.

    Returns
    -------
    RegressionResult

    References
    ----------
    Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and
    Panel Data*. MIT Press.

    de Chaisemartin, C., & D'Haultfoeuille, X. (2020). Two-way fixed
    effects estimators with heterogeneous treatment effects. *AER*,
    110(9), 2964-2996.
    """
    _validate_df(data, y, t, unit, time)
    df = data[[y, t, unit, time]].dropna()
    n = len(df)

    Y = df[y].to_numpy(dtype=float)
    D = df[t].to_numpy(dtype=float)

    unit_codes = pd.Categorical(df[unit]).codes
    time_codes = pd.Categorical(df[time]).codes
    n_units = len(np.unique(unit_codes))
    n_times = len(np.unique(time_codes))

    unit_means_y = np.zeros(n)
    unit_means_d = np.zeros(n)
    time_means_y = np.zeros(n)
    time_means_d = np.zeros(n)

    for i in range(n_units):
        mask = unit_codes == i
        unit_means_y[mask] = Y[mask].mean()
        unit_means_d[mask] = D[mask].mean()
    for j in range(n_times):
        mask = time_codes == j
        time_means_y[mask] = Y[mask].mean()
        time_means_d[mask] = D[mask].mean()

    grand_mean_y = Y.mean()
    grand_mean_d = D.mean()

    Y_tilde = Y - unit_means_y - time_means_y + grand_mean_y
    D_tilde = D - unit_means_d - time_means_d + grand_mean_d

    denom = float(D_tilde @ D_tilde)
    if denom < 1e-15:
        raise ValueError("No within-unit treatment variation after demeaning")

    tau = float(D_tilde @ Y_tilde / denom)

    resid = Y_tilde - tau * D_tilde
    df_resid = n - n_units - n_times + 1 - 1
    sigma2 = float(resid @ resid) / max(df_resid, 1)
    se = float(np.sqrt(sigma2 / denom))

    ss_res = float(resid @ resid)
    ss_tot = float(Y_tilde @ Y_tilde)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if se > 0 and df_resid > 0:
        t_stat = tau / se
        p_val = float(2 * stats.t.sf(abs(t_stat), df_resid))
    else:
        p_val = float("nan")

    return RegressionResult(
        method="TWFE",
        coefficients={"treatment": tau},
        se={"treatment": se},
        p_values={"treatment": p_val},
        r_squared=r2,
        residuals=resid,
        n=n,
        k=1,
        extra={
            "n_units": n_units,
            "n_times": n_times,
            "df_resid": df_resid,
        },
    )


tway = twfe


def cheatsheet() -> str:
    return "twfe({}) -> Two-way fixed effects estimator."
