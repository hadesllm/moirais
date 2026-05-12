# morie.fn — function file (hadesllm/morie)
"""Doubly-robust ATE estimator (IPW + outcome regression)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def doubly_robust_ate(
    outcome: Union[list, np.ndarray],
    treatment: Union[list, np.ndarray],
    covariates: Union[list, np.ndarray],
) -> dict[str, Any]:
    r"""
    Doubly-robust ATE estimator combining IPW and outcome regression.

    Consistent if *either* the propensity score model or the outcome
    model is correctly specified.

    Uses logistic regression for the propensity score and OLS for the
    outcome model. The influence function is:

    .. math::

        \\hat{\\psi}_i = \\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i)
            + \\frac{T_i(Y_i - \\hat{\\mu}_1(X_i))}{\\hat{e}(X_i)}
            - \\frac{(1-T_i)(Y_i - \\hat{\\mu}_0(X_i))}{1 - \\hat{e}(X_i)}

    :param outcome: Outcome variable (1-D array).
    :param treatment: Binary treatment (0/1, 1-D array).
    :param covariates: Covariate matrix (n x p).
    :return: Dictionary with ate, se, ci_lower, ci_upper.
    :raises ValueError: If dimensions mismatch or treatment not binary.

    References
    ----------
    Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of
    regression coefficients when some regressors are not always observed.
    *JASA*, 89(427), 846--866.

    Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing
    data and causal inference models. *Biometrics*, 61(4), 962--973.
    """
    from scipy.special import expit

    Y = np.asarray(outcome, dtype=float).ravel()
    T = np.asarray(treatment, dtype=float).ravel()
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(Y)
    if len(T) != n or X.shape[0] != n:
        raise ValueError("outcome, treatment, covariates must have the same number of observations.")
    if not set(np.unique(T)).issubset({0.0, 1.0}):
        raise ValueError("treatment must be binary (0/1).")

    # Propensity score via logistic regression (manual IRLS-free: use scipy minimize)
    # Simple approach: add intercept, use lstsq on logit scale (one Newton step from 0)
    Xa = np.column_stack([np.ones(n), X])
    p = Xa.shape[1]

    # Logistic regression via iterative reweighted least squares (3 iterations)
    beta_ps = np.zeros(p)
    for _ in range(10):
        mu = expit(Xa @ beta_ps)
        W = mu * (1 - mu)
        W = np.maximum(W, 1e-6)
        z = Xa @ beta_ps + (T - mu) / W
        WX = Xa * W[:, None]
        beta_ps = np.linalg.lstsq(WX.T @ Xa, WX.T @ z, rcond=None)[0]

    ps = expit(Xa @ beta_ps)
    ps = np.clip(ps, 0.01, 0.99)

    # Outcome models (OLS, separate for treated/control)
    treated = T == 1
    control = T == 0

    beta1 = np.linalg.lstsq(Xa[treated], Y[treated], rcond=None)[0]
    beta0 = np.linalg.lstsq(Xa[control], Y[control], rcond=None)[0]

    mu1 = Xa @ beta1
    mu0 = Xa @ beta0

    # Doubly-robust influence function
    psi = mu1 - mu0 + T * (Y - mu1) / ps - (1 - T) * (Y - mu0) / (1 - ps)

    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    z = 1.959964
    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "n": n,
    }


dr_ = doubly_robust_ate


def cheatsheet() -> str:
    return "doubly_robust_ate({}) -> Doubly-robust ATE estimator (IPW + outcome regression)."
