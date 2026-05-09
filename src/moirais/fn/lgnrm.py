# moirais.fn — function file (hadesllm/moirais)
"""
lgnrm.py - Log-normal accelerated failure time (AFT) survival model via MLE.

Reference: Kalbfleisch, J.D. & Prentice, R.L. (2002). The Statistical Analysis
of Failure Time Data, 2nd ed. Wiley.
"""

__all__ = ["lgnrm"]

import numpy as np


def lgnrm(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict:
    """
    Fit a log-normal AFT model via maximum likelihood estimation.

    The model assumes:
        log(T_i) = mu + x_i^T beta + sigma * epsilon_i,
        epsilon_i ~ N(0, 1).

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival times (>0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    covariates : np.ndarray, shape (n, p) or (n,), optional
        Covariate matrix. If None, intercept-only model fitted.
    max_iter : int, optional
        Maximum iterations for scipy.optimize.minimize. Default 200.
    tol : float, optional
        Gradient convergence tolerance. Default 1e-8.

    Returns
    -------
    dict
        mu : float
            Intercept (location on log-scale).
        beta : np.ndarray, shape (p,)
            Covariate coefficients (empty if no covariates).
        log_sigma : float
            Log of scale parameter (sigma = exp(log_sigma)).
        sigma : float
            Scale parameter.
        se_mu : float
            Standard error of mu.
        se_beta : np.ndarray, shape (p,)
            Standard errors of beta.
        se_log_sigma : float
            Standard error of log_sigma.
        log_likelihood : float
            Maximised log-likelihood.
        aic : float
            Akaike information criterion.
        bic : float
            Bayesian information criterion.

    Raises
    ------
    ValueError
        If inputs are incompatible or contain invalid values.

    References
    ----------
    Kalbfleisch, J.D. & Prentice, R.L. (2002). The Statistical Analysis of
    Failure Time Data (2nd ed.). Wiley.
    """
    from scipy import optimize as _opt
    from scipy import stats as _stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if np.any(time <= 0):
        raise ValueError("All survival times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")

    n = len(time)
    log_t = np.log(time)

    has_covariates = covariates is not None
    if has_covariates:
        X = np.atleast_2d(np.asarray(covariates, dtype=float))
        if X.shape[0] != n:
            X = X.T
        if X.shape[0] != n:
            raise ValueError("covariates rows must match length of time.")
        p = X.shape[1]
    else:
        X = np.ones((n, 1))
        p = 0  # we count mu separately

    # Parameter vector: [mu, beta (p elements), log_sigma]
    def _neg_loglik(params):
        mu = params[0]
        if has_covariates:
            beta = params[1: 1 + X.shape[1] - (0 if has_covariates else 1)]
            log_sigma = params[-1]
        else:
            beta = np.array([])
            log_sigma = params[1]

        sigma = np.exp(log_sigma)
        if has_covariates:
            eta = mu + X[:, 1:] @ beta if X.shape[1] > 1 else np.full(n, mu)
        else:
            eta = np.full(n, mu)

        z = (log_t - eta) / sigma
        # Log-likelihood contributions
        # Event: log pdf of lognormal = -log(sigma) - log(t) + log phi(z)
        # Censored: log survival = log(1 - Phi(z))
        ll = 0.0
        evt = event.astype(bool)
        if evt.any():
            ll += np.sum(-log_sigma - log_t[evt] + _stats.norm.logpdf(z[evt]))
        cen = ~evt
        if cen.any():
            ll += np.sum(_stats.norm.logsf(z[cen]))
        return -ll

    # Build full design if covariates present
    if has_covariates:
        # Intercept absorbed into mu; X does NOT include intercept column
        X_cov = X
        p_full = 1 + X_cov.shape[1] + 1  # mu + p_cov + log_sigma
        x0 = np.zeros(p_full)
        x0[0] = np.mean(log_t)
        x0[-1] = np.log(np.std(log_t) + 1e-6)

        def _neg_loglik_full(params):
            mu = params[0]
            beta_c = params[1:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            eta = mu + X_cov @ beta_c
            z = (log_t - eta) / sigma
            ll = 0.0
            evt = event.astype(bool)
            if evt.any():
                ll += np.sum(-log_sigma - log_t[evt] + _stats.norm.logpdf(z[evt]))
            cen = ~evt
            if cen.any():
                ll += np.sum(_stats.norm.logsf(z[cen]))
            return -ll

        res = _opt.minimize(
            _neg_loglik_full, x0, method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        params_hat = res.x
        mu_hat = params_hat[0]
        beta_hat = params_hat[1:-1]
        log_sigma_hat = params_hat[-1]
        ll_val = -res.fun

        # Hessian via finite differences for SE
        try:
            from scipy.optimize import approx_fprime
            eps = 1e-5
            hess = np.zeros((p_full, p_full))
            g0 = approx_fprime(params_hat, _neg_loglik_full, eps)
            for j in range(p_full):
                params_p = params_hat.copy()
                params_p[j] += eps
                g1 = approx_fprime(params_p, _neg_loglik_full, eps)
                hess[:, j] = (g1 - g0) / eps
            vcov = np.linalg.pinv((hess + hess.T) / 2)
            se_all = np.sqrt(np.maximum(np.diag(vcov), 0.0))
        except Exception:
            se_all = np.full(p_full, np.nan)

        se_mu = float(se_all[0])
        se_beta = se_all[1:-1]
        se_log_sigma = float(se_all[-1])
        k = p_full
    else:
        x0 = np.array([np.mean(log_t), np.log(np.std(log_t) + 1e-6)])
        res = _opt.minimize(
            _neg_loglik, x0, method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        mu_hat = res.x[0]
        beta_hat = np.array([])
        log_sigma_hat = res.x[1]
        ll_val = -res.fun
        se_mu = np.nan
        se_beta = np.array([])
        se_log_sigma = np.nan
        k = 2

    sigma_hat = float(np.exp(log_sigma_hat))
    aic = -2 * ll_val + 2 * k
    bic = -2 * ll_val + k * np.log(n)

    return {
        "mu": float(mu_hat),
        "beta": beta_hat,
        "log_sigma": float(log_sigma_hat),
        "sigma": sigma_hat,
        "se_mu": se_mu,
        "se_beta": se_beta,
        "se_log_sigma": se_log_sigma,
        "log_likelihood": float(ll_val),
        "aic": float(aic),
        "bic": float(bic),
    }
