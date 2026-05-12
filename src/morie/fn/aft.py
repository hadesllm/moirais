# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Accelerated Failure Time model (Weibull AFT via MLE)."""


import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def aft(time: np.ndarray, event: np.ndarray, X: np.ndarray, distribution: str = "weibull", alpha: float = 0.05, cdf=None) -> dict:
    r"""
    Accelerated Failure Time model via maximum likelihood estimation.

    Fits the parametric survival model:

    .. math::

        \\log T = X\\beta + \\sigma W

    where *W* follows a standard extreme-value distribution (Weibull AFT).

    :param time: 1-D array of observed times (> 0).
    :param event: 1-D binary array (1 = event, 0 = censored).
    :param X: Design matrix (n x p), **without** intercept column.
    :param distribution: Currently only ``"weibull"`` is supported.
    :param alpha: Significance level for Wald p-values. Default 0.05.
    :return: dict with ``coefficients``, ``se``, ``p_values``, ``aic``,
        ``log_sigma``, ``distribution``, ``n``, ``n_events``.
    :raises ValueError: If inputs have incompatible shapes or non-positive times.

    References
    ----------
    Kalbfleisch, J. D. & Prentice, R. L. (2002). The Statistical Analysis
    of Failure Time Data (2nd ed.). Wiley.

    Wei, L. J. (1992). The accelerated failure time model: A useful
    alternative to the Cox regression model in survival analysis.
    *Statistics in Medicine*, 11(14-15), 1871-1879.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    if len(time) != n or len(event) != n:
        raise ValueError(f"Shape mismatch: time({len(time)}), event({len(event)}), X({n},{p}).")
    if np.any(time <= 0):
        raise ValueError("All times must be > 0.")
    if distribution != "weibull":
        raise ValueError(f"Unsupported distribution: {distribution!r}. Use 'weibull'.")

    log_t = np.log(time)

    # Add intercept
    X_aug = np.column_stack([np.ones(n), X])
    n_params = p + 1  # intercept + covariates

    def neg_loglik(params):
        beta = params[:n_params]
        log_sigma = params[n_params]
        sigma = np.exp(log_sigma)
        z = (log_t - X_aug @ beta) / sigma
        # Weibull AFT: log-likelihood
        # event: log(1/sigma) + z - exp(z)
        # censored: -exp(z)
        ll = np.sum(event * (np.log(1.0 / sigma) + z - np.exp(z)))
        ll += np.sum((1 - event) * (-np.exp(z)))
        return -ll

    # Initial values: OLS on log(time) for uncensored
    mask = event == 1
    if mask.sum() < n_params + 1:
        beta0 = np.zeros(n_params)
        log_sigma0 = 0.0
    else:
        try:
            beta0 = np.linalg.lstsq(X_aug[mask], log_t[mask], rcond=None)[0]
        except np.linalg.LinAlgError:
            beta0 = np.zeros(n_params)
        resid = log_t[mask] - X_aug[mask] @ beta0
        log_sigma0 = np.log(max(np.std(resid), 0.1))

    x0 = np.concatenate([beta0, [log_sigma0]])
    result = minimize(neg_loglik, x0, method="BFGS")
    params_hat = result.x
    beta_hat = params_hat[:n_params]
    log_sigma_hat = params_hat[n_params]

    # Hessian-based SE via numerical inverse Hessian
    if result.hess_inv is not None:
        if hasattr(result.hess_inv, "todense"):
            H_inv = np.asarray(result.hess_inv.todense())
        else:
            H_inv = np.asarray(result.hess_inv)
        se_all = np.sqrt(np.maximum(np.diag(H_inv), 0.0))
    else:
        se_all = np.full(len(params_hat), np.nan)

    se_beta = se_all[:n_params]
    z_vals = beta_hat / np.where(se_beta > 0, se_beta, np.inf)
    p_vals = 2.0 * (1.0 - norm.cdf(np.abs(z_vals)))

    names = ["intercept"] + [f"x{i}" for i in range(p)]
    coefficients = dict(zip(names, beta_hat.tolist()))
    se_dict = dict(zip(names, se_beta.tolist()))
    pv_dict = dict(zip(names, p_vals.tolist()))

    loglik = -result.fun
    k_total = len(params_hat)
    aic = 2.0 * k_total - 2.0 * loglik

    return {
        "coefficients": coefficients,
        "se": se_dict,
        "p_values": pv_dict,
        "aic": float(aic),
        "log_sigma": float(log_sigma_hat),
        "distribution": distribution,
        "n": int(n),
        "n_events": int(event.sum()),
    }


def cheatsheet() -> str:
    return "aft({}) -> Accelerated Failure Time model (Weibull AFT via MLE)."
