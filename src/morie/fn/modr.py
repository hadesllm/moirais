# morie.fn -- function file (hadesllm/morie)
"""Moderation (interaction) analysis via OLS."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def moderation_analysis(y, x, moderator, covariates=None):
    """
    Test moderation (interaction effect) of moderator on x -> y relationship.

    Fits y = b0 + b1*x + b2*moderator + b3*x*moderator + covariates.

    :param y: (n,) outcome.
    :param x: (n,) predictor.
    :param moderator: (n,) moderator variable.
    :param covariates: (n, p) optional additional covariates.
    :return: DescriptiveResult with interaction coefficient, p-value.

    References
    ----------
    Aiken LS, West SG (1991). Multiple Regression: Testing and
    Interpreting Interactions. Sage.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(moderator, dtype=np.float64).ravel()
    n = len(y)
    xw = x * w
    X = np.column_stack([np.ones(n), x, w, xw])
    if covariates is not None:
        X = np.column_stack([X, np.asarray(covariates)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    k = X.shape[1]
    sigma2 = np.sum(resid**2) / (n - k)
    cov_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))
    t_stat = beta[3] / se[3] if se[3] > 0 else 0.0
    p_val = 2 * stats.t.sf(abs(t_stat), n - k)

    return DescriptiveResult(
        name="moderation_analysis",
        value=float(beta[3]),
        extra={
            "interaction_coef": float(beta[3]),
            "interaction_se": float(se[3]),
            "interaction_t": float(t_stat),
            "interaction_p": float(p_val),
            "coefficients": beta.tolist(),
            "r_squared": float(1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "moderation_analysis({}) -> Moderation (interaction) analysis via OLS."
