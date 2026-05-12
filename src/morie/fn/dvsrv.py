# morie.fn -- function file (hadesllm/morie)
"""
dvsrv.py - Deviance residuals for Cox proportional hazards models.

Deviance residuals are a transformation of martingale residuals that are
approximately normally distributed, making them more suitable for detecting
outliers.

Reference: Therneau, T.M., Grambsch, P.M. & Fleming, T.R. (1990). Martingale-
based residuals for survival models. Biometrika, 77(1), 147-160.
"""

__all__ = ["dvsrv"]

import numpy as np


def dvsrv(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
) -> dict:
    """
    Compute deviance residuals for a fitted Cox proportional hazards model.

    Deviance residuals are defined as:
        D_i = sign(M_i) * sqrt(-2 * [M_i + delta_i * log(delta_i - M_i)])
    where M_i is the martingale residual and delta_i is the event indicator.

    For censored observations (delta_i = 0): D_i = sign(M_i) * sqrt(-2 * M_i)
    For events (delta_i = 1): D_i = sign(M_i) * sqrt(-2 * [M_i + log(1 - M_i)])

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

    Returns
    -------
    dict
        residuals : np.ndarray, shape (n,)
            Deviance residuals.
        martingale : np.ndarray, shape (n,)
            Underlying martingale residuals.

    Raises
    ------
    ValueError
        If inputs are incompatible.

    References
    ----------
    Therneau, T.M., Grambsch, P.M. & Fleming, T.R. (1990). Biometrika,
    77(1), 147-160.
    """
    from morie.fn.mrsrv import mrsrv

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)

    mr_result = mrsrv(time, event, covariates, beta)
    M = mr_result["residuals"]

    # Deviance residuals
    D = np.zeros_like(M)
    evt = event.astype(bool)

    # Censored: D_i = sign(M_i) * sqrt(-2 * M_i)
    # M_i <= 0 for censored, so -2*M_i >= 0
    D[~evt] = np.sign(M[~evt]) * np.sqrt(np.maximum(-2.0 * M[~evt], 0.0))

    # Events: D_i = sign(M_i) * sqrt(-2 * [M_i + log(1 - M_i)])
    # M_i < 1 always (since M_i = 1 - expected, expected > 0 for events)
    log_term = np.where(
        (1.0 - M[evt]) > 0,
        np.log(np.maximum(1.0 - M[evt], 1e-15)),
        -1e15,
    )
    inner = -2.0 * (M[evt] + log_term)
    D[evt] = np.sign(M[evt]) * np.sqrt(np.maximum(inner, 0.0))

    return {
        "residuals": D,
        "martingale": M,
    }
