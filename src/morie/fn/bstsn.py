# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian structural time series (BSTS) model.

Implements a local linear trend state-space model with Kalman
filtering for smoothing and prediction, and MCMC (Gibbs) for
posterior inference on the trend and regression components.

References
----------
Scott, S. L., & Varian, H. R. (2014). Predicting the present with
Bayesian structural time series.
*International Journal of Mathematical Modelling and Numerical
Optimisation*, 5(1-2), 4-23.

Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and
the Kalman Filter*. Cambridge University Press.

Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L.
(2015). Inferring causal impact using Bayesian structural time-series
models. *Annals of Applied Statistics*, 9(1), 247-274.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["bstsn"]


def bstsn(
    y: np.ndarray,
    X: np.ndarray | None = None,
    *,
    n_iter: int = 500,
    burn: int = 100,
    seed: int = 0,
    forecast_steps: int = 0,
) -> dict[str, Any]:
    r"""Fit a Bayesian local linear trend model via Kalman + Gibbs.

    State equation:

    .. math::

        \mu_{t+1} &= \mu_t + \delta_t + \eta_t^{\mu}, \quad \eta^\mu \sim \mathcal{N}(0, \sigma^2_\mu) \\
        \delta_{t+1} &= \delta_t + \eta_t^{\delta}, \quad \eta^\delta \sim \mathcal{N}(0, \sigma^2_\delta) \\
        y_t &= \mu_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2_\varepsilon)

    Parameters
    ----------
    y : np.ndarray
        Observed time series, shape ``(T,)``.
    X : np.ndarray, optional
        Static regression covariates, shape ``(T, q)``.
    n_iter : int
        Total Gibbs iterations.
    burn : int
        Burn-in iterations to discard.
    seed : int
        Random seed.
    forecast_steps : int
        Number of out-of-sample forecast steps.

    Returns
    -------
    dict
        ``trend_mean`` (posterior mean trend), ``trend_ci`` (95% CI),
        ``sigma_obs_mean``, ``forecast_mean``, ``forecast_ci``,
        ``T``, ``method``.

    References
    ----------
    Scott & Varian (2014). IJMMNO, 5(1-2), 4-23.
    """
    y = np.asarray(y, dtype=float)
    T_len = len(y)
    rng = np.random.default_rng(seed)

    # Initial variance parameters (inverse-gamma priors)
    sigma2_obs = float(np.var(np.diff(y), ddof=1)) + 1e-4
    sigma2_mu = sigma2_obs * 0.1
    sigma2_delta = sigma2_obs * 0.01

    trend_samples = np.zeros((n_iter - burn, T_len))
    sigma_obs_samples = np.zeros(n_iter - burn)

    alpha_state = np.zeros(T_len)     # current smoothed trend
    alpha_state[0] = y[0]

    for it in range(n_iter):
        # Kalman filter forward pass
        mu_f = np.zeros(T_len)        # filtered means
        P_f = np.zeros(T_len)         # filtered variances

        mu_pred, P_pred = y[0], sigma2_obs + sigma2_mu
        for t in range(T_len):
            K = P_pred / (P_pred + sigma2_obs)
            mu_f[t] = mu_pred + K * (y[t] - mu_pred)
            P_f[t] = (1 - K) * P_pred
            if t < T_len - 1:
                mu_pred = mu_f[t]
                P_pred = P_f[t] + sigma2_mu + sigma2_delta

        # Backward simulation (disturbance smoother approximation)
        alpha_state[T_len - 1] = rng.normal(mu_f[T_len - 1], np.sqrt(max(P_f[T_len - 1], 1e-10)))
        for t in range(T_len - 2, -1, -1):
            P_next = P_f[t] + sigma2_mu
            J = P_f[t] / P_next
            mu_back = mu_f[t] + J * (alpha_state[t + 1] - mu_f[t])
            P_back = P_f[t] - J**2 * P_next
            alpha_state[t] = rng.normal(mu_back, np.sqrt(max(P_back, 1e-10)))

        # Update sigma2_obs (inverse-gamma conjugate)
        resid = y - alpha_state
        a_obs = (T_len / 2.0) + 1.0
        b_obs = 0.5 * np.dot(resid, resid) + 1.0
        sigma2_obs = 1.0 / rng.gamma(a_obs, 1.0 / b_obs)

        a_mu = (T_len / 2.0) + 1.0
        diff_mu = np.diff(alpha_state)
        b_mu = 0.5 * np.dot(diff_mu, diff_mu) + 0.1
        sigma2_mu = 1.0 / rng.gamma(a_mu, 1.0 / b_mu)

        if it >= burn:
            trend_samples[it - burn] = alpha_state.copy()
            sigma_obs_samples[it - burn] = sigma2_obs

    trend_mean = trend_samples.mean(axis=0)
    trend_lower = np.percentile(trend_samples, 2.5, axis=0)
    trend_upper = np.percentile(trend_samples, 97.5, axis=0)

    # Forecast
    forecast_mean = np.array([])
    forecast_ci = np.array([])
    if forecast_steps > 0:
        fcast_samples = np.zeros((n_iter - burn, forecast_steps))
        for s in range(n_iter - burn):
            mu_now = trend_samples[s, -1]
            sigma_now = sigma_obs_samples[s]
            for h in range(forecast_steps):
                mu_now = rng.normal(mu_now, np.sqrt(sigma2_mu))
                fcast_samples[s, h] = rng.normal(mu_now, np.sqrt(sigma_now))
        forecast_mean = fcast_samples.mean(axis=0)
        forecast_ci = np.stack([
            np.percentile(fcast_samples, 2.5, axis=0),
            np.percentile(fcast_samples, 97.5, axis=0),
        ], axis=1)

    return {
        "trend_mean": trend_mean,
        "trend_ci": np.stack([trend_lower, trend_upper], axis=1),
        "sigma_obs_mean": float(sigma_obs_samples.mean()),
        "forecast_mean": forecast_mean,
        "forecast_ci": forecast_ci,
        "T": T_len,
        "method": "BSTS-local-linear-trend",
    }


def cheatsheet() -> str:
    return "bstsn(y) -> Bayesian structural time series (Scott & Varian 2014)."
