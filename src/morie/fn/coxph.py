# morie.fn — function file (hadesllm/morie)
"""Cox proportional hazards model (semiparametric)."""

from __future__ import annotations

import numpy as np


def coxph(time: np.ndarray, event: np.ndarray, X: np.ndarray, cdf=None, *, max_iter: int = 100, tol: float = 1e-8) -> dict:
    r"""
    Cox proportional hazards regression via Newton-Raphson.

    The Cox (1972) model specifies the hazard function as

    .. math::

        \lambda(t \mid X_i) = \lambda_0(t) \exp(X_i \beta)

    where :math:`\lambda_0(t)` is an unspecified baseline hazard and
    :math:`\beta` is estimated by maximising the partial log-likelihood:

    .. math::

        \ell(\beta) = \sum_{i:\delta_i=1}
        \Bigl[ X_i \beta - \log \sum_{j \in \mathcal{R}(t_i)}
        \exp(X_j \beta) \Bigr]

    Newton-Raphson updates use the score and observed information:

    .. math::

        U(\beta) = \sum_{i:\delta_i=1}
        \Bigl[ X_i - \frac{\sum_{j \in \mathcal{R}_i} X_j e^{X_j\beta}}
        {\sum_{j \in \mathcal{R}_i} e^{X_j\beta}} \Bigr]

    :param time: Observed times, shape ``(n,)``.
    :param event: Event indicator (1=event, 0=censored), shape ``(n,)``.
    :param X: Covariate matrix, shape ``(n, p)``. Do NOT include an
        intercept (absorbed into :math:`\lambda_0`).
    :param max_iter: Maximum Newton-Raphson iterations. Default 100.
    :param tol: Convergence tolerance on coefficient change. Default 1e-8.
    :return: dict with ``coefficients``, ``se``, ``hazard_ratios``,
        ``z_scores``, ``p_values``, ``log_partial_likelihood``,
        ``n_iter``, ``converged``, ``n_obs``, ``n_events``.
    :raises ValueError: If input dimensions are inconsistent.

    References
    ----------
    Cox, D. R. (1972). Regression models and life-tables. *JRSS-B*,
        34(2), 187--220.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Ch. 6.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    if time.ndim != 1 or event.ndim != 1:
        raise ValueError("time and event must be 1-D.")
    if X.ndim != 2:
        raise ValueError("X must be 2-D.")
    n, p = X.shape
    if n != time.shape[0] or n != event.shape[0]:
        raise ValueError("time, event, and X must have the same number of rows.")

    order = np.argsort(-time)
    time_s = time[order]
    event_s = event[order]
    X_s = X[order]

    beta = np.zeros(p)
    converged = False
    n_iter = 0

    for it in range(max_iter):
        eta = X_s @ beta
        eta -= eta.max()
        exp_eta = np.exp(eta)

        cum_exp = np.cumsum(exp_eta)
        cum_Xexp = np.cumsum(X_s * exp_eta[:, None], axis=0)
        cum_XXexp = np.zeros((n, p, p))
        for i in range(n):
            cum_XXexp[i] = (cum_XXexp[i - 1] if i > 0 else np.zeros((p, p))) + np.outer(X_s[i], X_s[i]) * exp_eta[i]

        score = np.zeros(p)
        hess = np.zeros((p, p))
        for i in range(n):
            if event_s[i] == 0:
                continue
            w = cum_Xexp[i] / cum_exp[i]
            score += X_s[i] - w
            hess -= cum_XXexp[i] / cum_exp[i] - np.outer(w, w)

        try:
            step = np.linalg.solve(hess, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, score, rcond=None)[0]

        beta_new = beta - step
        n_iter = it + 1
        if np.max(np.abs(beta_new - beta)) < tol:
            converged = True
            beta = beta_new
            break
        beta = beta_new

    eta_final = X_s @ beta
    eta_final -= eta_final.max()
    exp_eta_f = np.exp(eta_final)
    cum_exp_f = np.cumsum(exp_eta_f)
    pll = 0.0
    for i in range(n):
        if event_s[i] == 1:
            pll += eta_final[i] - np.log(cum_exp_f[i])

    try:
        info = -hess
        cov = np.linalg.inv(info)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    from scipy.stats import norm

    z = beta / np.where(se > 0, se, np.nan)
    pvals = 2 * (1 - norm.cdf(np.abs(z)))

    return {
        "coefficients": beta,
        "se": se,
        "hazard_ratios": np.exp(beta),
        "z_scores": z,
        "p_values": pvals,
        "log_partial_likelihood": float(pll),
        "n_iter": n_iter,
        "converged": converged,
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


coxph_fn = coxph


def cheatsheet() -> str:
    return "coxph(time, event, X) -> Cox proportional hazards model."
