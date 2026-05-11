# morie.fn — function file (hadesllm/morie)
"""RD optimal bandwidth selection (IK/CCT)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _validate_df


def rd_bandwidth(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    r: str = "running",
    cutoff: float = 0.0,
    method: str = "ik",
) -> dict:
    r"""Optimal bandwidth for regression discontinuity designs.

    IK (Imbens-Kalyanaraman, 2012):
    Uses a pilot bandwidth to estimate curvature, then:

    .. math::

        h_{opt} = C_K \left(\frac{\hat{\sigma}^2}{\hat{f}(c)
        (\hat{m}^{(2)+} + \hat{m}^{(2)-})^2 n}\right)^{1/5}

    CCT (Calonico-Cattaneo-Titiunik, 2014):
    Adds bias correction and uses separate bandwidths for bias estimation
    and point estimation.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    r : str
        Running variable column.
    cutoff : float
        RD cutoff.
    method : str
        'ik' (Imbens-Kalyanaraman) or 'cct' (Calonico-Cattaneo-Titiunik).

    Returns
    -------
    dict
        Keys: 'h_opt', 'h_pilot', 'n_left', 'n_right', 'method'.

    References
    ----------
    Imbens, G. W., & Kalyanaraman, K. (2012). Optimal bandwidth choice.
    *Rev. Econ. Stud.*, 79(3), 933-959.

    Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust
    nonparametric confidence intervals for regression-discontinuity
    designs. *Econometrica*, 82(6), 2295-2326.
    """
    _validate_df(data, y, r)
    df = data[[y, r]].dropna()
    R = df[r].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    n = len(R)

    n_left = int((cutoff > R).sum())
    n_right = int((cutoff <= R).sum())

    h_pilot = 1.84 * R.std() * n ** (-1.0 / 5.0)

    if method == "ik":
        h_opt = _ik_bw(R, Y, cutoff, h_pilot, n)
    elif method == "cct":
        h_opt = _cct_bw(R, Y, cutoff, h_pilot, n)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "h_opt": float(h_opt),
        "h_pilot": float(h_pilot),
        "n_left": n_left,
        "n_right": n_right,
        "method": method,
    }


def _ik_bw(R, Y, c, h_pilot, n):
    """IK bandwidth."""
    mask = np.abs(R - c) <= h_pilot
    if mask.sum() < 10:
        return h_pilot

    R_loc = R[mask] - c
    Y_loc = Y[mask]
    T_loc = (R_loc >= 0).astype(float)

    X = np.column_stack([
        np.ones(mask.sum()), R_loc, T_loc, R_loc * T_loc,
        R_loc ** 2, R_loc ** 2 * T_loc
    ])
    beta = np.linalg.lstsq(X, Y_loc, rcond=None)[0]
    resid = Y_loc - X @ beta
    sigma2 = float(np.sum(resid ** 2)) / max(mask.sum() - 6, 1)

    m2_plus = abs(float(beta[4] + beta[5]))
    m2_minus = abs(float(beta[4]))
    m2_sum = m2_plus + m2_minus

    if m2_sum < 1e-10:
        return h_pilot

    f_c = float(mask.sum()) / (2 * h_pilot * n)
    C_k = 3.4375

    h_opt = C_k * (sigma2 / (f_c * m2_sum ** 2 * n)) ** (1.0 / 5.0)
    return max(h_opt, h_pilot * 0.05)


def _cct_bw(R, Y, c, h_pilot, n):
    """CCT bandwidth (simplified)."""
    h_ik = _ik_bw(R, Y, c, h_pilot, n)

    mask = np.abs(R - c) <= 2 * h_ik
    if mask.sum() < 10:
        return h_ik

    R_loc = R[mask] - c
    Y_loc = Y[mask]
    T_loc = (R_loc >= 0).astype(float)

    X = np.column_stack([
        np.ones(mask.sum()), R_loc, T_loc, R_loc * T_loc,
        R_loc ** 2, R_loc ** 2 * T_loc,
        R_loc ** 3, R_loc ** 3 * T_loc,
    ])
    beta = np.linalg.lstsq(X, Y_loc, rcond=None)[0]
    resid = Y_loc - X @ beta
    sigma2 = float(np.sum(resid ** 2)) / max(mask.sum() - 8, 1)

    m3 = abs(float(beta[6])) + abs(float(beta[7]))
    if m3 < 1e-10:
        return h_ik

    h_b = 1.84 * (sigma2 / (m3 ** 2 * n)) ** (1.0 / 7.0)
    rho = h_ik / h_b if h_b > 0 else 1.0
    h_cct = h_ik * (1 + rho ** 2) ** (-1.0 / 5.0)

    return max(float(h_cct), h_pilot * 0.05)


rdopt = rd_bandwidth


def cheatsheet() -> str:
    return "rd_bandwidth({}) -> RD optimal bandwidth (IK/CCT)."
