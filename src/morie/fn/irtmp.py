# morie.fn -- function file (hadesllm/morie)
"""MAP theta estimation with normal prior."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from morie.fn._containers import DescriptiveResult


def irt_map_theta(
    responses: np.ndarray,
    item_params: dict,
    *,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
) -> DescriptiveResult:
    """Maximum a posteriori (MAP) theta estimation.

    Combines the likelihood from item responses with a normal prior
    on theta to produce a posterior mode estimate.

    Parameters
    ----------
    responses : ndarray
        Binary response vector (k,) for one person, or matrix (n x k).
    item_params : dict
        {item: {"a": float, "b": float}}.
    prior_mean : float
        Prior mean (default 0).
    prior_sd : float
        Prior SD (default 1).

    Returns
    -------
    DescriptiveResult
        value=dict with theta estimates and SEs.

    References
    ----------
    Mislevy, R. J. (1986). Bayes modal estimation in item response
    models. Psychometrika, 51(2), 177-195.
    """
    X = np.atleast_2d(np.asarray(responses, dtype=np.float64))
    n, k = X.shape
    params = list(item_params.values())

    def neg_log_post(th, x_row):
        ll = -0.5 * ((th - prior_mean) / prior_sd) ** 2
        for j in range(min(k, len(params))):
            a = params[j].get("a", 1.0)
            b = params[j].get("b", 0.0)
            logit = a * (th - b)
            P = 1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700)))
            P = np.clip(P, 1e-10, 1 - 1e-10)
            ll += x_row[j] * np.log(P) + (1 - x_row[j]) * np.log(1 - P)
        return -ll

    thetas = np.zeros(n)
    ses = np.zeros(n)
    for i in range(n):
        res = optimize.minimize_scalar(neg_log_post, bounds=(-6, 6), method="bounded", args=(X[i],))
        thetas[i] = res.x
        h = 1e-5
        f0 = neg_log_post(res.x, X[i])
        f1 = neg_log_post(res.x + h, X[i])
        f2 = neg_log_post(res.x - h, X[i])
        info = (f1 - 2 * f0 + f2) / h**2
        ses[i] = 1.0 / np.sqrt(max(info, 1e-10))

    return DescriptiveResult(
        name="MAP theta",
        value={"theta": thetas.tolist(), "se": ses.tolist()},
        extra={"prior_mean": prior_mean, "prior_sd": prior_sd, "n": n},
    )


map_theta = irt_map_theta


def cheatsheet() -> str:
    return "irt_map_theta({}) -> MAP theta estimation with normal prior."
