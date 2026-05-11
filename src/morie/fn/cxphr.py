# morie.fn — function file (hadesllm/morie)
"""
cxphr.py - Cox proportional hazards model via partial likelihood (Newton-Raphson).

Reference: Cox, D.R. (1972). Regression models and life-tables. Journal of the
Royal Statistical Society, Series B, 34(2), 187-220.
"""

__all__ = ["cxphr"]

import numpy as np


def cxphr(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    ties: str = "breslow",
) -> dict:
    """
    Fit a Cox proportional hazards model via partial likelihood maximisation
    using the Newton-Raphson algorithm.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival times (>0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    covariates : np.ndarray, shape (n, p) or (n,)
        Covariate matrix. Will be column-centred internally for numerical
        stability.
    max_iter : int, optional
        Maximum Newton-Raphson iterations. Default 50.
    tol : float, optional
        Convergence tolerance on the gradient norm. Default 1e-8.
    ties : str, optional
        Tie-handling method: 'breslow' (default) or 'efron'.

    Returns
    -------
    dict
        beta : np.ndarray, shape (p,)
            Estimated log-hazard-ratio coefficients.
        se : np.ndarray, shape (p,)
            Standard errors (sqrt of diagonal of inverse Fisher information).
        z : np.ndarray, shape (p,)
            Wald z-statistics (beta / se).
        p_value : np.ndarray, shape (p,)
            Two-sided p-values from standard normal.
        log_likelihood : float
            Maximised partial log-likelihood.
        iterations : int
            Number of N-R iterations performed.
        converged : bool
            Whether convergence criterion was met.

    Raises
    ------
    ValueError
        If inputs are incompatible or contain invalid values.

    References
    ----------
    Cox, D.R. (1972). Journal of the Royal Statistical Society, Series B,
    34(2), 187-220.
    Breslow, N. (1974). Biometrics, 30, 89-99.
    Efron, B. (1977). Journal of the American Statistical Association, 72, 557-565.
    """
    from scipy import stats as _stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.atleast_2d(np.asarray(covariates, dtype=float))
    if X.shape[0] != len(time):
        X = X.T
    if X.shape[0] != len(time):
        raise ValueError("covariates rows must match length of time.")
    if np.any(time <= 0):
        raise ValueError("All survival times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")
    if ties not in ("breslow", "efron"):
        raise ValueError("ties must be 'breslow' or 'efron'.")

    n, p = X.shape
    # Centre covariates for numerical stability
    X_mean = X.mean(axis=0)
    X = X - X_mean

    beta = np.zeros(p)

    def _partial_loglik_breslow(b):
        """Breslow partial log-likelihood, score, and information."""
        eta = X @ b  # shape (n,)
        # Sort by time (descending for risk-set construction)
        order = np.argsort(time)
        t_s = time[order]
        e_s = event[order]
        X_s = X[order]
        eta_s = eta[order]

        log_lik = 0.0
        score = np.zeros(p)
        info = np.zeros((p, p))

        event_times = np.unique(t_s[e_s == 1])
        for t_j in event_times:
            # Risk set: all i with t_i >= t_j
            risk_mask = t_s >= t_j
            # Event set at t_j
            event_mask = (t_s == t_j) & (e_s == 1)
            d_j = event_mask.sum()  # number of events

            exp_eta_risk = np.exp(eta_s[risk_mask])
            S0 = exp_eta_risk.sum()
            X_risk = X_s[risk_mask]
            S1 = (exp_eta_risk[:, None] * X_risk).sum(axis=0)  # (p,)
            S2 = (exp_eta_risk[:, None, None] * X_risk[:, :, None] * X_risk[:, None, :]).sum(axis=0)  # (p,p)

            # Contribution from event set
            log_lik += eta_s[event_mask].sum() - d_j * np.log(S0)
            e_bar = S1 / S0
            score += X_s[event_mask].sum(axis=0) - d_j * e_bar
            info += d_j * (S2 / S0 - np.outer(e_bar, e_bar))

        return log_lik, score, info

    def _partial_loglik_efron(b):
        """Efron partial log-likelihood, score, and information."""
        eta = X @ b
        order = np.argsort(time)
        t_s = time[order]
        e_s = event[order]
        X_s = X[order]
        eta_s = eta[order]

        log_lik = 0.0
        score = np.zeros(p)
        info = np.zeros((p, p))

        event_times = np.unique(t_s[e_s == 1])
        for t_j in event_times:
            risk_mask = t_s >= t_j
            event_mask = (t_s == t_j) & (e_s == 1)
            d_j = event_mask.sum()

            exp_eta_risk = np.exp(eta_s[risk_mask])
            exp_eta_event = np.exp(eta_s[event_mask])
            S0_R = exp_eta_risk.sum()
            S0_D = exp_eta_event.sum()
            X_risk = X_s[risk_mask]
            X_event = X_s[event_mask]
            S1_R = (exp_eta_risk[:, None] * X_risk).sum(axis=0)
            S1_D = (exp_eta_event[:, None] * X_event).sum(axis=0)

            log_lik += eta_s[event_mask].sum()
            for ell in range(d_j):
                frac = ell / d_j
                denom = S0_R - frac * S0_D
                log_lik -= np.log(denom)
                e_bar_ell = (S1_R - frac * S1_D) / denom
                score += X_s[event_mask].sum(axis=0) / d_j - e_bar_ell / d_j * d_j

            # Information: use Breslow approximation for simplicity
            S2_R = (exp_eta_risk[:, None, None] * X_risk[:, :, None] * X_risk[:, None, :]).sum(axis=0)
            e_bar_0 = S1_R / S0_R
            info += d_j * (S2_R / S0_R - np.outer(e_bar_0, e_bar_0))

        # Recompute score cleanly
        score = np.zeros(p)
        for t_j in event_times:
            risk_mask = t_s >= t_j
            event_mask = (t_s == t_j) & (e_s == 1)
            d_j = event_mask.sum()
            exp_eta_risk = np.exp(eta_s[risk_mask])
            exp_eta_event = np.exp(eta_s[event_mask])
            S0_R = exp_eta_risk.sum()
            S0_D = exp_eta_event.sum()
            X_risk = X_s[risk_mask]
            X_event = X_s[event_mask]
            S1_R = (exp_eta_risk[:, None] * X_risk).sum(axis=0)
            S1_D = (exp_eta_event[:, None] * X_event).sum(axis=0)
            score += X_s[event_mask].sum(axis=0)
            for ell in range(d_j):
                frac = ell / d_j
                denom = S0_R - frac * S0_D
                score -= (S1_R - frac * S1_D) / denom

        return log_lik, score, info

    _loglik_fn = _partial_loglik_breslow if ties == "breslow" else _partial_loglik_efron

    converged = False
    for iteration in range(max_iter):
        ll, grad, hess = _loglik_fn(beta)
        # Newton-Raphson step: beta_new = beta + H^{-1} grad
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta = beta + delta
        if np.linalg.norm(grad) < tol:
            converged = True
            break

    ll_final, _, hess_final = _loglik_fn(beta)
    try:
        vcov = np.linalg.inv(hess_final)
    except np.linalg.LinAlgError:
        vcov = np.linalg.pinv(hess_final)

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    z = np.where(se > 0, beta / se, 0.0)
    p_val = 2 * _stats.norm.sf(np.abs(z))

    return {
        "beta": beta,
        "se": se,
        "z": z,
        "p_value": p_val,
        "log_likelihood": float(ll_final),
        "iterations": iteration + 1,
        "converged": converged,
        "vcov": vcov,
    }
