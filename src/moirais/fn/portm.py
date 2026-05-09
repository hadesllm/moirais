# moirais.fn — function file (hadesllm/moirais)
"""Portmanteau test for weak convergence (Ljung-Box)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def portmanteau_test(
    x: np.ndarray,
    *,
    lags: int = 10,
    method: str = "ljung-box",
) -> dict:
    r"""
    Portmanteau test for serial correlation (weak convergence diagnostic).

    The Ljung-Box statistic tests whether autocorrelations of a time
    series are jointly zero:

    .. math::

        Q_{LB} = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n - k}

    Under :math:`H_0` (no serial correlation), :math:`Q_{LB} \sim \chi^2_h`.

    The Box-Pierce variant uses:

    .. math::

        Q_{BP} = n \sum_{k=1}^{h} \hat{\rho}_k^2

    Used as a diagnostic for weak convergence of empirical processes:
    if the empirical process residuals show serial dependence, the
    Donsker CLT may not apply.

    :param x: 1-D array (time series or residual process).
    :param lags: Number of lags to test. Default 10.
    :param method: ``"ljung-box"`` or ``"box-pierce"``. Default ``"ljung-box"``.
    :return: dict with ``Q`` (test statistic), ``p_value``, ``reject``,
        ``lags``, ``autocorrelations``.
    :raises ValueError: If x has fewer than lags+1 observations.

    References
    ----------
    Ljung, G.M. & Box, G.E.P. (1978). On a measure of lack of fit in
        time series models. *Biometrika*, 65(2), 297--303.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Ch. 7 (weak convergence). Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < lags + 1:
        raise ValueError(f"Need at least {lags + 1} observations, got {n}.")
    if method not in ("ljung-box", "box-pierce"):
        raise ValueError(f"method must be 'ljung-box' or 'box-pierce', got '{method}'.")

    x_centered = x - np.mean(x)
    gamma0 = np.sum(x_centered**2) / n
    if gamma0 == 0:
        return {
            "Q": 0.0,
            "p_value": 1.0,
            "reject": False,
            "lags": lags,
            "autocorrelations": [0.0] * lags,
            "method": method,
        }

    rho = np.empty(lags)
    for k in range(1, lags + 1):
        rho[k - 1] = np.sum(x_centered[k:] * x_centered[:-k]) / (n * gamma0)

    if method == "ljung-box":
        Q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, lags + 1)))
    else:
        Q = n * np.sum(rho**2)

    p_value = float(stats.chi2.sf(Q, df=lags))

    return {
        "Q": float(Q),
        "p_value": p_value,
        "reject": p_value < 0.05,
        "lags": lags,
        "autocorrelations": rho.tolist(),
        "method": method,
    }


portm = portmanteau_test


def cheatsheet() -> str:
    return "portmanteau_test({x}) -> Portmanteau (Ljung-Box) serial correlation test."
