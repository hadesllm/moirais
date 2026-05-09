# moirais.fn — function file (hadesllm/moirais)
"""Schoenfeld residuals for proportional hazards assumption testing."""

import numpy as np
from scipy.stats import pearsonr


def schon(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> dict:
    """
    Schoenfeld residuals and correlation test for the PH assumption.

    At each event time, the Schoenfeld residual for covariate *j* is:

    .. math::

        r_j(t_i) = x_{ij} - \\bar{x}_j(t_i)

    where :math:`\\bar{x}_j(t_i)` is the risk-set weighted mean of covariate
    *j* at time :math:`t_i`.  A significant correlation between the
    (scaled) residuals and time indicates PH violation.

    :param time: 1-D array of observed times.
    :param event: 1-D binary array (1 = event, 0 = censored).
    :param X: Design matrix (n x p), no intercept.
    :param beta: Coefficient vector (p,) from a Cox-like model.
    :return: dict with ``residuals`` (n_events x p), ``event_times``,
        ``correlation`` (per covariate), ``p_values`` (per covariate),
        ``ph_rejected`` (list of booleans at alpha=0.05).
    :raises ValueError: On shape mismatches.

    References
    ----------
    Schoenfeld, D. (1982). Partial residuals for the proportional hazards
    regression model. *Biometrika*, 69(1), 239-241.

    Grambsch, P. M. & Therneau, T. M. (1994). Proportional hazards tests
    and diagnostics based on weighted residuals. *Biometrika*, 81(3), 515-526.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if len(time) != n or len(event) != n:
        raise ValueError(f"Shape mismatch: time({len(time)}), event({len(event)}), X({n},{p}).")
    if len(beta) != p:
        raise ValueError(f"beta length ({len(beta)}) != number of covariates ({p}).")

    event_mask = event == 1
    event_indices = np.where(event_mask)[0]
    event_times_arr = time[event_indices]

    # Sort events by time
    ev_order = np.argsort(event_times_arr)
    event_indices = event_indices[ev_order]
    event_times_sorted = event_times_arr[ev_order]

    residuals = np.zeros((len(event_indices), p))
    exp_Xbeta = np.exp(X @ beta)

    for idx, ei in enumerate(event_indices):
        t_i = time[ei]
        # Risk set: those with time >= t_i
        risk_set = time >= t_i
        weights = exp_Xbeta[risk_set]
        w_sum = weights.sum()
        if w_sum > 0:
            x_bar = (X[risk_set].T @ weights) / w_sum
        else:
            x_bar = np.zeros(p)
        residuals[idx] = X[ei] - x_bar

    # Correlation test of residuals with event times
    correlations = np.zeros(p)
    p_values = np.zeros(p)
    for j in range(p):
        if len(event_times_sorted) > 2 and np.std(residuals[:, j]) > 0:
            r, pv = pearsonr(event_times_sorted, residuals[:, j])
            correlations[j] = r
            p_values[j] = pv
        else:
            correlations[j] = 0.0
            p_values[j] = 1.0

    covariate_names = [f"x{i}" for i in range(p)]
    return {
        "residuals": residuals,
        "event_times": event_times_sorted,
        "correlation": dict(zip(covariate_names, correlations.tolist())),
        "p_values": dict(zip(covariate_names, p_values.tolist())),
        "ph_rejected": [pv < 0.05 for pv in p_values],
    }


def cheatsheet() -> str:
    return "schon({}) -> Schoenfeld residuals for proportional hazards assumption tes"
