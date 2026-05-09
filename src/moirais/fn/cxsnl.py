# moirais.fn — function file (hadesllm/moirais)
"""
cxsnl.py - Cox-Snell residuals for survival models.

Cox-Snell residuals are estimates of the cumulative hazard at each observed
time, i.e., r_i = H(t_i | x_i). If the model fits well, these residuals
behave like a censored sample from an Exp(1) distribution.

Reference: Cox, D.R. & Snell, E.J. (1968). A general definition of residuals.
Journal of the Royal Statistical Society, Series B, 30(2), 248-275.
"""

__all__ = ["cxsnl"]

import numpy as np


def cxsnl(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
    model: str = "cox",
    model_params: dict = None,
) -> dict:
    """
    Compute Cox-Snell residuals for a fitted survival model.

    For a Cox model:
        r_i = H_0(t_i) * exp(x_i^T beta)  (= -log S(t_i | x_i))

    For a Weibull model with scale lambda, shape rho, covariates beta:
        r_i = (lambda * exp(x_i^T beta))^rho * t_i^rho

    If the model is correctly specified, r_i should look like censored
    Exp(1) observations and a KM plot of -log KM(r_i) vs r_i should be
    approximately the 45-degree line.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    covariates : np.ndarray, shape (n, p) or (n,)
        Covariate matrix.
    beta : np.ndarray, shape (p,)
        Fitted model coefficients.
    model : str, optional
        Model type: 'cox' (default) or 'weibull'.
    model_params : dict, optional
        For 'weibull': must include 'log_lambda' (float) and 'rho' (float).
        For 'cox': ignored (Breslow baseline hazard used).

    Returns
    -------
    dict
        residuals : np.ndarray, shape (n,)
            Cox-Snell residuals r_i = H(t_i | x_i).
        km_times : np.ndarray
            Time points for KM of residuals (for GOF plot).
        km_neg_log_surv : np.ndarray
            -log(KM survival) of residuals — should equal km_times if fit is good.

    Raises
    ------
    ValueError
        If inputs are incompatible or model is unrecognised.

    References
    ----------
    Cox, D.R. & Snell, E.J. (1968). Journal of the Royal Statistical Society,
    Series B, 30(2), 248-275.
    """
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
    if model not in ("cox", "weibull"):
        raise ValueError("model must be 'cox' or 'weibull'.")

    n = len(time)
    eta = X @ beta
    exp_eta = np.exp(eta)

    if model == "cox":
        # Breslow baseline cumulative hazard
        event_times = np.unique(time[event == 1])
        dH0 = {}
        for t_j in event_times:
            risk_mask = time >= t_j
            d_j = np.sum((time == t_j) & (event == 1))
            denom = exp_eta[risk_mask].sum()
            dH0[t_j] = d_j / denom if denom > 0 else 0.0

        sorted_times = np.array(sorted(dH0.keys()))
        cum_h0 = np.cumsum([dH0[t] for t in sorted_times])
        cum_h0_map = dict(zip(sorted_times, cum_h0))

        H0_i = np.zeros(n)
        for i in range(n):
            eligible = sorted_times[sorted_times <= time[i]]
            if len(eligible) > 0:
                H0_i[i] = cum_h0_map[eligible[-1]]
        residuals = H0_i * exp_eta

    else:  # weibull
        if model_params is None:
            raise ValueError("model_params must be provided for 'weibull' model.")
        log_lam = float(model_params["log_lambda"])
        rho = float(model_params["rho"])
        lam = np.exp(log_lam)
        # H(t | x) = (lambda * exp(x^T beta))^rho * t^rho
        residuals = (lam * exp_eta) ** rho * time ** rho

    # KM of Cox-Snell residuals to assess model fit
    # If fit is good: -log(KM(r)) vs r should be the 45-degree line
    r = residuals
    order = np.argsort(r)
    r_s = r[order]
    e_s = event[order]
    event_r_times = np.unique(r_s[e_s == 1])

    S = 1.0
    km_r_times = []
    km_surv = []
    for t_j in event_r_times:
        n_r = np.sum(r_s >= t_j)
        n_e = np.sum((r_s == t_j) & (e_s == 1))
        km_r_times.append(t_j)
        km_surv.append(S)
        S *= (1 - n_e / n_r) if n_r > 0 else 1.0

    km_r_times = np.array(km_r_times) if km_r_times else np.array([0.0])
    km_surv = np.array(km_surv) if km_surv else np.array([1.0])
    neg_log_surv = -np.log(np.maximum(km_surv, 1e-15))

    return {
        "residuals": residuals,
        "km_times": km_r_times,
        "km_neg_log_surv": neg_log_surv,
    }
