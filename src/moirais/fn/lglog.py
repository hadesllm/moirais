# moirais.fn — function file (hadesllm/moirais)
"""
lglog.py - Log-logistic AFT survival model via MLE.

Reference: Bennett, S. (1983). Log-logistic regression models for survival data.
Journal of the Royal Statistical Society, Series C, 32(2), 165-171.
"""

__all__ = ["lglog"]

import numpy as np


def lglog(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict:
    """
    Fit a log-logistic AFT model via maximum likelihood estimation.

    The log-logistic survival function is:
        S(t | x) = 1 / (1 + (t / exp(mu + x^T beta))^{1/sigma})

    and the hazard function is non-monotone (first increasing, then decreasing)
    when sigma > 1, making it useful for modelling treatments with delayed effects.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival times (>0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    covariates : np.ndarray, shape (n, p) or (n,), optional
        Covariate matrix. If None, intercept-only model is fitted.
    max_iter : int, optional
        Maximum iterations for optimiser. Default 200.
    tol : float, optional
        Convergence tolerance. Default 1e-8.

    Returns
    -------
    dict
        mu : float
            Intercept (log-scale location).
        beta : np.ndarray, shape (p,)
            Covariate coefficients.
        sigma : float
            Scale parameter (> 0).
        log_sigma : float
            Log scale parameter.
        se_mu : float
            Standard error of mu.
        se_beta : np.ndarray
            Standard errors of beta.
        se_log_sigma : float
            Standard error of log_sigma.
        log_likelihood : float
            Maximised log-likelihood.
        aic : float
        bic : float

    Raises
    ------
    ValueError
        If inputs are incompatible.

    References
    ----------
    Bennett, S. (1983). Journal of the Royal Statistical Society, Series C,
    32(2), 165-171.
    """
    from scipy import optimize as _opt

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if np.any(time <= 0):
        raise ValueError("All survival times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")

    n = len(time)
    log_t = np.log(time)
    has_cov = covariates is not None

    if has_cov:
        X = np.atleast_2d(np.asarray(covariates, dtype=float))
        if X.shape[0] != n:
            X = X.T
        if X.shape[0] != n:
            raise ValueError("covariates rows must match length of time.")
    else:
        X = None

    def _neg_ll(params, X_cov):
        mu = params[0]
        if X_cov is not None:
            beta_c = params[1:-1]
            log_sigma = params[-1]
            eta = mu + X_cov @ beta_c
        else:
            log_sigma = params[1]
            eta = np.full(n, mu)
        sigma = np.exp(log_sigma)
        # Standardised log-time residual under logistic distribution
        w = (log_t - eta) / sigma
        # Log pdf of logistic: -log(sigma) + w - 2*log(1 + exp(w))
        # But numerically stable form: -log(sigma) - log(1+exp(-w)) - log(1+exp(w))
        # Use: log pdf = w - log(sigma) - 2*log1p(exp(w)) for large w issue
        # Numerically safe: logpdf = w - log_sigma - 2*log(1+exp(w))
        # = -log_sigma - w - 2*log(1+exp(-w)) equivalently
        log_pdf = -log_sigma + w - 2.0 * np.log1p(np.exp(w))
        # Survival: S(t) = 1/(1 + exp(w)) = logistic(-w)
        log_surv = -np.log1p(np.exp(w))  # log(1/(1+exp(w)))

        evt = event.astype(bool)
        ll = 0.0
        if evt.any():
            ll += log_pdf[evt].sum()
        if (~evt).any():
            ll += log_surv[~evt].sum()
        return -ll

    if has_cov:
        p_full = 1 + X.shape[1] + 1
        x0 = np.zeros(p_full)
        x0[0] = np.mean(log_t)
        x0[-1] = np.log(np.std(log_t) + 1e-6)
        X_cov = X
    else:
        p_full = 2
        x0 = np.array([np.mean(log_t), np.log(np.std(log_t) + 1e-6)])
        X_cov = None

    res = _opt.minimize(
        _neg_ll, x0, args=(X_cov,), method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    params_hat = res.x
    ll_val = -res.fun
    mu_hat = params_hat[0]
    log_sigma_hat = params_hat[-1]
    beta_hat = params_hat[1:-1] if has_cov else np.array([])

    # SE via numerical Hessian
    try:
        from scipy.optimize import approx_fprime
        eps = 1e-5
        hess = np.zeros((p_full, p_full))
        g0 = approx_fprime(params_hat, lambda p: _neg_ll(p, X_cov), eps)
        for j in range(p_full):
            pp = params_hat.copy(); pp[j] += eps
            g1 = approx_fprime(pp, lambda p: _neg_ll(p, X_cov), eps)
            hess[:, j] = (g1 - g0) / eps
        vcov = np.linalg.pinv((hess + hess.T) / 2)
        se_all = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    except Exception:
        se_all = np.full(p_full, np.nan)

    se_mu = float(se_all[0])
    se_beta = se_all[1:-1] if has_cov else np.array([])
    se_log_sigma = float(se_all[-1])

    aic = -2 * ll_val + 2 * p_full
    bic = -2 * ll_val + p_full * np.log(n)

    return {
        "mu": float(mu_hat),
        "beta": beta_hat,
        "sigma": float(np.exp(log_sigma_hat)),
        "log_sigma": float(log_sigma_hat),
        "se_mu": se_mu,
        "se_beta": se_beta,
        "se_log_sigma": se_log_sigma,
        "log_likelihood": float(ll_val),
        "aic": float(aic),
        "bic": float(bic),
    }
