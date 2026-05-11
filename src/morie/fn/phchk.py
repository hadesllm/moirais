# morie.fn — function file (hadesllm/morie)
"""
phchk.py - Proportional hazards assumption check via scaled Schoenfeld residuals.

The test correlates scaled Schoenfeld residuals with a transformation of time.
A significant correlation indicates violation of the PH assumption.

Reference: Grambsch, P.M. & Therneau, T.M. (1994). Proportional hazards tests
and diagnostics based on weighted residuals. Biometrika, 81(3), 515-526.
"""

__all__ = ["phchk"]

import numpy as np


def phchk(time: np.ndarray, event: np.ndarray, covariates: np.ndarray, beta: np.ndarray, transform: str = "km", cdf=None) -> dict:
    """
    Test the proportional hazards assumption using scaled Schoenfeld residuals.

    For each covariate j and each event time t_i, the Schoenfeld residual is:
        r_{ij} = x_{ij} - E[X_j | risk set at t_i]

    The scaled residual (Grambsch-Therneau) is:
        r*_{ij} = d * sigma_j^2 * r_{ij}    (approximately beta_j(t))
    where d is total events and sigma_j^2 is the variance of X_j in the risk set.

    A significant correlation between r*_{ij} and g(t_i) (a function of time)
    indicates time-varying coefficients, i.e., PH violation.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    covariates : np.ndarray, shape (n, p) or (n,)
        Covariate matrix.
    beta : np.ndarray, shape (p,)
        Fitted Cox model coefficients (from cxphr).
    transform : str, optional
        Time transformation for residual correlation: 'km' (Kaplan-Meier,
        default), 'log' (log time), 'identity' (time), or 'rank'.

    Returns
    -------
    dict
        residuals : np.ndarray, shape (n_events, p)
            Schoenfeld residuals at each event time.
        scaled_residuals : np.ndarray, shape (n_events, p)
            Scaled (Grambsch-Therneau) residuals, approximating beta(t).
        event_times : np.ndarray, shape (n_events,)
            Event times at which residuals are computed.
        rho : np.ndarray, shape (p,)
            Spearman correlation of scaled residuals with transformed time.
        chi2 : np.ndarray, shape (p,)
            Test statistic for each covariate.
        p_value : np.ndarray, shape (p,)
            p-values from chi-squared distribution (1 df each).
        global_chi2 : float
            Global test statistic (sum across all covariates).
        global_p_value : float
            p-value for global test (df = p).

    Raises
    ------
    ValueError
        If inputs are incompatible.

    References
    ----------
    Grambsch, P.M. & Therneau, T.M. (1994). Biometrika, 81(3), 515-526.
    """
    from scipy import stats as _stats

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
    if transform not in ("km", "log", "identity", "rank"):
        raise ValueError("transform must be 'km', 'log', 'identity', or 'rank'.")

    n, p = X.shape
    eta = X @ beta
    order = np.argsort(time)
    t_s = time[order]
    e_s = event[order]
    X_s = X[order]
    eta_s = eta[order]

    event_times = np.unique(t_s[e_s == 1])
    n_events = len(event_times)

    residuals = np.zeros((n_events, p))
    # Information matrix to scale residuals: sum over event times of V_j(t)
    info_diag = np.zeros(p)  # diagonal of Fisher information per covariate

    for i, t_j in enumerate(event_times):
        risk_mask = t_s >= t_j
        event_mask = (t_s == t_j) & (e_s == 1)
        exp_eta_r = np.exp(eta_s[risk_mask])
        S0 = exp_eta_r.sum()
        if S0 <= 0:
            continue
        X_r = X_s[risk_mask]
        S1 = (exp_eta_r[:, None] * X_r).sum(axis=0) / S0
        S2_diag = (exp_eta_r[:, None] * X_r ** 2).sum(axis=0) / S0
        var_j = S2_diag - S1 ** 2  # variance of X in risk set

        # Schoenfeld residual for this event time (one observation per event)
        x_event = X_s[event_mask].mean(axis=0)  # average over tied events
        residuals[i] = x_event - S1
        info_diag += var_j

    # Scale residuals: r*_ij = (d * var_j)^{-1} * r_ij  — Grambsch-Therneau
    d = n_events
    scale = np.where(info_diag > 0, d / info_diag, 0.0)
    scaled_residuals = residuals * scale[None, :]

    # KM survival for time transformation
    if transform == "km":
        S = 1.0
        km_at_event = []
        t_arr = t_s
        e_arr = e_s
        for t_j in event_times:
            n_r = np.sum(t_arr >= t_j)
            n_e = np.sum((t_arr == t_j) & (e_arr == 1))
            km_at_event.append(S)
            S *= (1 - n_e / n_r) if n_r > 0 else 1.0
        g_t = np.array(km_at_event)
    elif transform == "log":
        g_t = np.log(event_times)
    elif transform == "identity":
        g_t = event_times
    else:  # rank
        g_t = np.argsort(np.argsort(event_times)).astype(float) + 1

    rho = np.zeros(p)
    chi2 = np.zeros(p)
    p_val = np.zeros(p)
    for j in range(p):
        r_j = scaled_residuals[:, j]
        if np.std(r_j) < 1e-15 or np.std(g_t) < 1e-15:
            continue
        rho[j], _ = _stats.spearmanr(g_t, r_j)
        # Test statistic: rho^2 * (n-2) / (1-rho^2) ~ chi2(1) approximately
        # Use: chi2 = rho^2 * d (approximate, Grambsch-Therneau)
        chi2[j] = rho[j] ** 2 * d
        p_val[j] = float(1 - _stats.chi2.cdf(chi2[j], df=1))

    global_chi2 = float(chi2.sum())
    global_p = float(1 - _stats.chi2.cdf(global_chi2, df=p))

    return {
        "residuals": residuals,
        "scaled_residuals": scaled_residuals,
        "event_times": event_times,
        "rho": rho,
        "chi2": chi2,
        "p_value": p_val,
        "global_chi2": global_chi2,
        "global_p_value": global_p,
    }
