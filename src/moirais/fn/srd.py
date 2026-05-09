"""Sharp regression discontinuity with bandwidth selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def sharp_rd(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    r: str = "running",
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    kernel: str = "triangular",
) -> RegressionResult:
    r"""Sharp RDD with kernel weighting and local linear regression.

    Treatment is deterministic at the cutoff: :math:`T_i = 1(R_i \geq c)`.
    The LATE is estimated by local linear regression:

    .. math::

        Y_i = \alpha + \tau T_i + \beta_1 \tilde{R}_i
        + \beta_2 T_i \tilde{R}_i + \epsilon_i

    weighted by kernel :math:`K((R_i - c)/h)`.

    Kernels:
    - triangular: :math:`K(u) = (1 - |u|) \cdot 1(|u| \leq 1)`
    - uniform: :math:`K(u) = 1(|u| \leq 1)`
    - epanechnikov: :math:`K(u) = \frac{3}{4}(1 - u^2) \cdot 1(|u| \leq 1)`

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    r : str
        Running variable column.
    cutoff : float
        RD cutoff.
    bandwidth : float or None
        If None, uses IK rule-of-thumb.
    kernel : str
        Kernel function ('triangular', 'uniform', 'epanechnikov').

    Returns
    -------
    RegressionResult

    References
    ----------
    Imbens, G. W., & Kalyanaraman, K. (2012). Optimal bandwidth choice
    for the regression discontinuity estimator. *Review of Economic
    Studies*, 79(3), 933-959.
    """
    _validate_df(data, y, r)
    df = data[[y, r]].dropna()
    R = df[r].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)

    if bandwidth is None:
        bandwidth = _ik_bandwidth(R, Y, cutoff)

    mask = np.abs(R - cutoff) <= bandwidth
    if mask.sum() < 4:
        raise ValueError("Too few observations within bandwidth")

    R_loc = R[mask] - cutoff
    Y_loc = Y[mask]
    u = R_loc / bandwidth

    if kernel == "triangular":
        K = np.maximum(1 - np.abs(u), 0)
    elif kernel == "uniform":
        K = np.ones_like(u)
    elif kernel == "epanechnikov":
        K = np.maximum(0.75 * (1 - u ** 2), 0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    T_loc = (R_loc >= 0).astype(float)
    X = np.column_stack([np.ones(mask.sum()), R_loc, T_loc, R_loc * T_loc])
    n = mask.sum()
    p = 4

    W_sqrt = np.sqrt(K)
    Xw = X * W_sqrt[:, None]
    Yw = Y_loc * W_sqrt
    beta = np.linalg.lstsq(Xw, Yw, rcond=None)[0]

    resid = Y_loc - X @ beta
    ss_res = float(np.sum(K * resid ** 2))
    mse = ss_res / max(n - p, 1)
    try:
        cov = mse * np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se_arr = np.sqrt(np.maximum(np.diag(cov), 0))

    names = ["intercept", "slope_left", "LATE", "slope_interaction"]
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
        method="Sharp RDD",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        n=n,
        k=3,
        extra={"bandwidth": bandwidth, "cutoff": cutoff, "kernel": kernel},
    )


def _ik_bandwidth(R, Y, c):
    """Imbens-Kalyanaraman rule-of-thumb bandwidth."""
    n = len(R)
    h_pilot = 1.84 * R.std() * n ** (-1 / 5)
    mask = np.abs(R - c) <= h_pilot
    if mask.sum() < 10:
        return h_pilot
    R_loc = R[mask] - c
    Y_loc = Y[mask]
    T_loc = (R_loc >= 0).astype(float)
    X = np.column_stack([np.ones(mask.sum()), R_loc, T_loc, R_loc * T_loc,
                         R_loc ** 2, R_loc ** 2 * T_loc])
    beta = np.linalg.lstsq(X, Y_loc, rcond=None)[0]
    resid = Y_loc - X @ beta
    sigma2 = float(np.sum(resid ** 2)) / max(mask.sum() - 6, 1)
    m2 = abs(beta[4]) + abs(beta[5])
    if m2 < 1e-10:
        return h_pilot
    C_k = 3.4375
    h_opt = C_k * (sigma2 / (m2 ** 2 * n)) ** (1 / 5)
    return float(max(h_opt, h_pilot * 0.1))


srd = sharp_rd


def cheatsheet() -> str:
    return "sharp_rd({}) -> Sharp RDD with bandwidth selection."
