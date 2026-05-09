# moirais.fn — function file (hadesllm/moirais)
"""
gmprl.py - Gompertz survival model via MLE.

The Gompertz model has hazard h(t) = lambda * exp(gamma * t), lambda > 0.
When gamma > 0 the hazard is monotonically increasing; gamma < 0 gives a
decreasing hazard; gamma = 0 reduces to the exponential model.

Reference: Gompertz, B. (1825). Philosophical Transactions of the Royal Society
of London, 115, 513-583. MLE parameterisation follows Collett (2003).
"""

__all__ = ["gmprl"]

import numpy as np


def gmprl(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict:
    """
    Fit a Gompertz proportional-hazards survival model via MLE.

    Hazard: h(t | x) = lambda * exp(gamma * t) * exp(x^T beta)
    Survival: S(t | x) = exp(-lambda/gamma * exp(x^T beta) * (exp(gamma*t) - 1))
              = exp(-lambda * exp(x^T beta) * t)  when gamma -> 0.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    covariates : np.ndarray, shape (n, p), optional
        Covariate matrix. If None, fits baseline model only.
    max_iter : int, optional
        Maximum optimiser iterations. Default 200.
    tol : float, optional
        Convergence tolerance. Default 1e-8.

    Returns
    -------
    dict
        log_lambda : float
            Log baseline rate parameter.
        gamma : float
            Shape parameter (gamma = 0 => exponential).
        beta : np.ndarray, shape (p,)
            Log-hazard-ratio covariate coefficients.
        se_log_lambda : float
        se_gamma : float
        se_beta : np.ndarray
        log_likelihood : float
        aic : float
        bic : float

    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid values.

    References
    ----------
    Collett, D. (2003). Modelling Survival Data in Medical Research (2nd ed.).
    Chapman & Hall/CRC.
    """
    from scipy import optimize as _opt

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if np.any(time <= 0):
        raise ValueError("All survival times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")

    n = len(time)
    has_cov = covariates is not None
    if has_cov:
        X = np.atleast_2d(np.asarray(covariates, dtype=float))
        if X.shape[0] != n:
            X = X.T
        if X.shape[0] != n:
            raise ValueError("covariates rows must match length of time.")
        p_cov = X.shape[1]
    else:
        X = None
        p_cov = 0

    # params = [log_lambda, gamma, *beta]
    def _neg_ll(params):
        log_lam = params[0]
        gamma = params[1]
        lam = np.exp(log_lam)
        if has_cov:
            beta_c = params[2:]
            phi = np.exp(X @ beta_c)
        else:
            phi = np.ones(n)

        # Cumulative hazard H(t) = lam*phi/gamma*(exp(gamma*t)-1) if gamma!=0
        # else lam*phi*t
        abs_g = np.abs(gamma)
        if abs_g < 1e-10:
            H = lam * phi * time
            log_h = log_lam + np.log(phi)
        else:
            H = lam * phi / gamma * (np.exp(gamma * time) - 1.0)
            log_h = log_lam + gamma * time + np.log(phi)

        # Log-likelihood: sum_i [ d_i * log h(t_i) - H(t_i) ]
        ll = np.sum(event * log_h - H)
        return -ll

    p_full = 2 + p_cov
    x0 = np.zeros(p_full)
    # Initialise log_lambda from mean event rate
    n_events = max(event.sum(), 1)
    x0[0] = np.log(n_events / (n * time.mean()))
    x0[1] = 0.0  # start at exponential

    res = _opt.minimize(
        _neg_ll, x0, method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    params_hat = res.x
    ll_val = -res.fun

    # SE via numerical Hessian
    try:
        from scipy.optimize import approx_fprime
        eps = 1e-5
        hess = np.zeros((p_full, p_full))
        g0 = approx_fprime(params_hat, _neg_ll, eps)
        for j in range(p_full):
            pp = params_hat.copy(); pp[j] += eps
            g1 = approx_fprime(pp, _neg_ll, eps)
            hess[:, j] = (g1 - g0) / eps
        vcov = np.linalg.pinv((hess + hess.T) / 2)
        se_all = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    except Exception:
        se_all = np.full(p_full, np.nan)

    aic = -2 * ll_val + 2 * p_full
    bic = -2 * ll_val + p_full * np.log(n)

    return {
        "log_lambda": float(params_hat[0]),
        "gamma": float(params_hat[1]),
        "beta": params_hat[2:],
        "se_log_lambda": float(se_all[0]),
        "se_gamma": float(se_all[1]),
        "se_beta": se_all[2:],
        "log_likelihood": float(ll_val),
        "aic": float(aic),
        "bic": float(bic),
    }
