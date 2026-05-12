# morie.fn — function file (hadesllm/morie)
"""Epidemic forecast using the renewal equation."""

from __future__ import annotations

from typing import Any

import numpy as np


def renewal_forecast(
    incidence: np.ndarray,
    serial_interval_pmf: np.ndarray,
    Rt: float,
    *,
    horizon: int = 14,
    seed: int = 42,
    n_sim: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Forecast epidemic incidence using the renewal equation.

    .. math::

        I_t = R_t \\sum_{s=1}^{S} w_s \\, I_{t-s}

    Parameters
    ----------
    incidence : array_like
        Observed daily incidence up to the present.
    serial_interval_pmf : array_like
        Serial interval PMF (lag 1, 2, ...).
    Rt : float
        Assumed constant reproduction number for the forecast period.
    horizon : int, default 14
        Number of days to forecast.
    seed : int, default 42
        Random seed for stochastic simulations.
    n_sim : int, default 0
        If > 0, generate stochastic (Poisson) forecast trajectories
        for prediction intervals.
    alpha : float, default 0.05
        Significance level for prediction intervals.

    Returns
    -------
    dict
        Keys: 'forecast' (deterministic), 'ci_lower', 'ci_upper',
              't_forecast', 'Rt'.

    References
    ----------
    Fraser, C. (2007). Estimating individual and household reproduction
    numbers in an emerging epidemic. PLoS ONE, 2(8), e758.

    Cori, A. et al. (2013). A new framework and software to estimate
    time-varying reproduction numbers during epidemics. American
    Journal of Epidemiology, 178(9), 1505-1512.
    """
    inc = np.asarray(incidence, dtype=float)
    w = np.asarray(serial_interval_pmf, dtype=float)

    if inc.ndim != 1 or inc.size < 1:
        raise ValueError("incidence must be non-empty 1-D array.")
    if w.ndim != 1 or w.size < 1:
        raise ValueError("serial_interval_pmf must be non-empty 1-D.")
    if Rt < 0:
        raise ValueError("Rt must be non-negative.")

    w = w / w.sum()
    max_lag = len(w)
    T = len(inc)

    extended = np.concatenate([inc, np.zeros(horizon)])

    for t in range(T, T + horizon):
        lam = 0.0
        for s in range(min(max_lag, t)):
            lam += w[s] * extended[t - s - 1]
        extended[t] = Rt * lam

    forecast = extended[T:]

    if n_sim > 0:
        rng = np.random.default_rng(seed)
        sims = np.zeros((n_sim, horizon))
        for sim_i in range(n_sim):
            ext_s = np.concatenate([inc, np.zeros(horizon)])
            for t in range(T, T + horizon):
                lam = 0.0
                for s in range(min(max_lag, t)):
                    lam += w[s] * ext_s[t - s - 1]
                mu = max(Rt * lam, 0.0)
                ext_s[t] = rng.poisson(mu)
            sims[sim_i] = ext_s[T:]

        q_lo = alpha / 2
        q_hi = 1 - alpha / 2
        ci_lo = np.quantile(sims, q_lo, axis=0)
        ci_hi = np.quantile(sims, q_hi, axis=0)
    else:
        ci_lo = forecast.copy()
        ci_hi = forecast.copy()

    return {
        "forecast": forecast,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "t_forecast": np.arange(T, T + horizon),
        "Rt": float(Rt),
    }


frcst = renewal_forecast


def cheatsheet() -> str:
    return "renewal_forecast({}) -> Epidemic forecast via renewal equation."
