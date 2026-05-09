# moirais.fn — function file (hadesllm/moirais)
"""
fingy.py - Fine-Gray subdistribution hazard model for competing risks via
weighted partial likelihood.

Reference: Fine, J.P. & Gray, R.J. (1999). A proportional hazards model for
the subdistribution of a competing risk. Journal of the American Statistical
Association, 94(446), 496-509.
"""

__all__ = ["fingy"]

import numpy as np


def fingy(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    cause: int = 1,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> dict:
    """
    Fit the Fine-Gray subdistribution hazard model for competing risks.

    The subdistribution hazard for cause k is:
        lambda_k(t) = lim_{dt->0} P(t <= T < t+dt, J=k | T >= t or (T < t and J != k)) / dt

    This is modelled as lambda_k(t | x) = lambda_0k(t) * exp(x^T beta_k).

    IPCW weights are used so that individuals who experience a competing event
    remain in the risk set with appropriate down-weighting.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed times (> 0).
    event : np.ndarray, shape (n,)
        Cause indicators: 0 = censored, positive integers = causes.
    covariates : np.ndarray, shape (n, p) or (n,)
        Covariate matrix (no intercept column needed).
    cause : int, optional
        Cause of interest. Default 1.
    max_iter : int, optional
        Newton-Raphson iterations. Default 50.
    tol : float, optional
        Convergence tolerance (gradient norm). Default 1e-8.

    Returns
    -------
    dict
        beta : np.ndarray, shape (p,)
            Estimated subdistribution log-hazard-ratio coefficients.
        se : np.ndarray, shape (p,)
            Standard errors.
        z : np.ndarray, shape (p,)
            Wald z-statistics.
        p_value : np.ndarray, shape (p,)
            Two-sided p-values.
        log_likelihood : float
            Maximised weighted partial log-likelihood.
        converged : bool

    Raises
    ------
    ValueError
        If inputs have incompatible shapes or cause not found.

    References
    ----------
    Fine, J.P. & Gray, R.J. (1999). Journal of the American Statistical
    Association, 94(446), 496-509.
    """
    from scipy import stats as _stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    X = np.atleast_2d(np.asarray(covariates, dtype=float))
    if X.shape[0] != len(time):
        X = X.T
    if X.shape[0] != len(time):
        raise ValueError("covariates rows must match length of time.")
    if cause not in event:
        raise ValueError(f"Cause {cause} not found in event array.")
    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")

    n, p = X.shape
    # Centre covariates
    X = X - X.mean(axis=0)

    # Compute IPCW weights for the subdistribution risk set.
    # Individuals with competing event (event != cause and event != 0)
    # remain in risk set with weight = KM_C(t_j) / KM_C(t_i), where
    # KM_C is the KM estimator for censoring.
    censoring_indicator = (event == 0).astype(float)
    # KM for censoring: treat censoring as "event", observed events as censored
    km_times_c, km_c = _km_estimator(time, censoring_indicator)

    def _get_km_c(t):
        """Return KM censoring survival at time t."""
        idx = np.searchsorted(km_times_c, t, side="right") - 1
        idx = np.clip(idx, 0, len(km_c) - 1)
        return km_c[idx]

    # Subdistribution risk set at each event time t_j (cause k):
    # i is in risk set if t_i >= t_j  OR  (t_i < t_j and event_i != cause and event_i != 0)
    # Weight for i = 1 if t_i >= t_j (uncensored or not yet event)
    #              = KM_C(t_j) / KM_C(t_i) if competing event occurred at t_i < t_j

    cause_event_times = np.unique(time[(event == cause)])
    beta = np.zeros(p)

    def _weighted_partial_ll(b):
        eta = X @ b
        grad = np.zeros(p)
        info = np.zeros((p, p))
        ll = 0.0

        for t_j in cause_event_times:
            # Subjects with cause-k event at t_j
            event_mask = (time == t_j) & (event == cause)
            d_j = event_mask.sum()
            if d_j == 0:
                continue

            # Build weights for subdistribution risk set at t_j
            in_risk = time >= t_j
            competing_before = (time < t_j) & (event != cause) & (event != 0)
            # Weight
            w = np.zeros(n)
            w[in_risk] = 1.0
            km_c_tj = _get_km_c(t_j)
            km_c_i = np.array([_get_km_c(time[i]) for i in range(n)])
            w[competing_before] = np.where(
                km_c_i[competing_before] > 0,
                km_c_tj / km_c_i[competing_before],
                0.0,
            )
            risk_mask = (in_risk | competing_before) & (w > 0)

            exp_eta = np.exp(eta)
            w_exp_eta = w * exp_eta
            S0 = w_exp_eta[risk_mask].sum()
            if S0 <= 0:
                continue
            X_risk = X[risk_mask]
            we_r = w_exp_eta[risk_mask]
            S1 = (we_r[:, None] * X_risk).sum(axis=0)
            S2 = (we_r[:, None, None] * X_risk[:, :, None] * X_risk[:, None, :]).sum(axis=0)

            ll += eta[event_mask].sum() - d_j * np.log(S0)
            e_bar = S1 / S0
            grad += X[event_mask].sum(axis=0) - d_j * e_bar
            info += d_j * (S2 / S0 - np.outer(e_bar, e_bar))

        return ll, grad, info

    converged = False
    for _iter in range(max_iter):
        ll, grad, info = _weighted_partial_ll(beta)
        try:
            delta = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(info, grad, rcond=None)[0]
        beta = beta + delta
        if np.linalg.norm(grad) < tol:
            converged = True
            break

    ll_final, _, info_final = _weighted_partial_ll(beta)
    try:
        vcov = np.linalg.inv(info_final)
    except np.linalg.LinAlgError:
        vcov = np.linalg.pinv(info_final)

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    z = np.where(se > 0, beta / se, 0.0)
    p_val = 2 * _stats.norm.sf(np.abs(z))

    return {
        "beta": beta,
        "se": se,
        "z": z,
        "p_value": p_val,
        "log_likelihood": float(ll_final),
        "converged": converged,
        "vcov": vcov,
    }


def _km_estimator(time, event):
    """Return (sorted_times, survival) for Kaplan-Meier."""
    order = np.argsort(time)
    t_s = time[order]
    e_s = event[order]
    event_times = np.unique(t_s[e_s == 1])
    S = 1.0
    surv_vals = []
    for t_j in event_times:
        n_risk = np.sum(t_s >= t_j)
        n_event = np.sum((t_s == t_j) & (e_s == 1))
        S *= (1 - n_event / n_risk) if n_risk > 0 else 1.0
        surv_vals.append(S)
    if len(event_times) == 0:
        return np.array([0.0]), np.array([1.0])
    return event_times, np.array(surv_vals)
