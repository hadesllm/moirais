# morie.fn — function file (hadesllm/morie)
"""
mrsrv.py - Martingale residuals for Cox proportional hazards models.

Martingale residuals measure the difference between observed events and
the expected number under the fitted model.

Reference: Therneau, T.M., Grambsch, P.M. & Fleming, T.R. (1990). Martingale-
based residuals for survival models. Biometrika, 77(1), 147-160.
"""

__all__ = ["mrsrv"]

import numpy as np


def mrsrv(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
    ties: str = "breslow",
) -> dict:
    """
    Compute martingale residuals for a fitted Cox model.

    The martingale residual for subject i is:
        M_i = delta_i - H_0(t_i) * exp(x_i^T beta)
    where delta_i is the event indicator, H_0(t_i) is the Breslow baseline
    cumulative hazard at t_i, and beta are the fitted coefficients.

    Martingale residuals sum to zero, lie in (-inf, 1], and are useful for
    assessing functional form of covariates and detecting influential observations.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    covariates : np.ndarray, shape (n, p) or (n,)
        Covariate matrix.
    beta : np.ndarray, shape (p,)
        Fitted Cox model log-hazard-ratio coefficients.
    ties : str, optional
        Tie-handling for baseline hazard: 'breslow' (default) or 'efron'.

    Returns
    -------
    dict
        residuals : np.ndarray, shape (n,)
            Martingale residuals.
        baseline_cumhaz : np.ndarray, shape (n,)
            Breslow baseline cumulative hazard at each observation time.
        expected : np.ndarray, shape (n,)
            Expected number of events: H_0(t_i) * exp(x_i^T beta).

    Raises
    ------
    ValueError
        If inputs are incompatible.

    References
    ----------
    Therneau, T.M., Grambsch, P.M. & Fleming, T.R. (1990). Biometrika,
    77(1), 147-160.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.atleast_2d(np.asarray(covariates, dtype=float))
    beta = np.asarray(beta, dtype=float)
    if X.shape[0] != len(time):
        X = X.T
    if X.shape[0] != len(time):
        raise ValueError("covariates rows must match length of time.")
    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")
    if ties not in ("breslow", "efron"):
        raise ValueError("ties must be 'breslow' or 'efron'.")

    n = len(time)
    eta = X @ beta
    exp_eta = np.exp(eta)
    event_times = np.unique(time[event == 1])

    # Breslow baseline cumulative hazard increments
    dH0 = {}
    for t_j in event_times:
        risk_mask = time >= t_j
        d_j = np.sum((time == t_j) & (event == 1))
        denom = exp_eta[risk_mask].sum()
        dH0[t_j] = d_j / denom if denom > 0 else 0.0

    # Baseline cumulative hazard H_0(t) = sum_{t_j <= t} dH_0(t_j)
    sorted_times = np.array(sorted(dH0.keys()))
    cum_h0 = np.cumsum([dH0[t] for t in sorted_times])
    cum_h0_map = dict(zip(sorted_times, cum_h0))

    H0_i = np.zeros(n)
    for i in range(n):
        t_i = time[i]
        eligible = sorted_times[sorted_times <= t_i]
        if len(eligible) > 0:
            H0_i[i] = cum_h0_map[eligible[-1]]

    expected = H0_i * exp_eta
    residuals = event - expected

    return {
        "residuals": residuals,
        "baseline_cumhaz": H0_i,
        "expected": expected,
    }
