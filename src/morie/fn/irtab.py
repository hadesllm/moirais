# morie.fn -- function file (hadesllm/morie)
"""IRT ability (theta) estimation: MLE, MAP, and EAP."""

from __future__ import annotations

import numpy as np
from scipy import optimize


def irtab(
    responses: np.ndarray | list,
    item_params: dict[str, dict],
    *,
    method: str = "MLE",
    model: str = "2PL",
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
    n_quad: int = 61,
) -> dict:
    """Estimate person ability (theta) given item responses and parameters.

    Parameters
    ----------
    responses : array-like
        Binary response vector (0/1) of length n_items, or integer
        responses for polytomous models.  Order must match item_params.
    item_params : dict
        {item_name: {"a": float, "b": float, "c": float (optional)}}.
    method : str
        Estimation method: "MLE", "MAP", or "EAP" (default "MLE").
    model : str
        IRT model: "1PL", "2PL", "3PL" (default "2PL").
    prior_mean, prior_sd : float
        Normal prior parameters for MAP/EAP (default N(0,1)).
    n_quad : int
        Quadrature points for EAP (default 61).

    Returns
    -------
    dict
        {"theta": float, "se": float, "method": str, "loglik": float}.

    References
    ----------
    Bock, R. D. & Mislevy, R. J. (1982). Adaptive EAP estimation of
    ability in a microcomputer environment. Applied Psychological
    Measurement, 6(4), 431-444.
    """
    x = np.asarray(responses, dtype=np.float64).ravel()
    items = list(item_params.keys())
    n_items = len(items)

    if len(x) != n_items:
        raise ValueError(f"Response vector length ({len(x)}) != number of items ({n_items}).")

    # Extract parameters in order
    a_arr = np.array([item_params[it].get("a", 1.0) for it in items])
    b_arr = np.array([item_params[it].get("b", 0.0) for it in items])
    c_arr = np.array([item_params[it].get("c", 0.0) for it in items])

    def _icc(theta_val):
        logit = np.clip(a_arr * (theta_val - b_arr), -700, 700)
        Pstar = 1.0 / (1.0 + np.exp(-logit))
        if model == "3PL":
            return c_arr + (1.0 - c_arr) * Pstar
        return Pstar

    def _loglik(theta_val):
        P = np.clip(_icc(theta_val), 1e-10, 1.0 - 1e-10)
        return np.sum(x * np.log(P) + (1.0 - x) * np.log(1.0 - P))

    def _info_at(theta_val):
        """Fisher information at theta."""
        P = np.clip(_icc(theta_val), 1e-10, 1.0 - 1e-10)
        Q = 1.0 - P
        if model == "3PL":
            Padj = np.clip((P - c_arr) / (1.0 - c_arr), 1e-10, 1.0)
            return np.sum(a_arr**2 * Padj**2 * Q / P)
        return np.sum(a_arr**2 * P * Q)

    # Handle perfect/zero scores for MLE (infinite theta)
    valid = np.isfinite(x)
    score = x[valid].sum()
    if method == "MLE" and (score == 0 or score == valid.sum()):
        # Perfect or zero score: MLE is undefined, fall back to MAP
        method = "MAP"

    if method == "EAP":
        # Expected A Posteriori via quadrature
        quad_pts = np.linspace(prior_mean - 4 * prior_sd, prior_mean + 4 * prior_sd, n_quad)
        log_prior = -0.5 * ((quad_pts - prior_mean) / prior_sd) ** 2

        log_lik = np.array([_loglik(th) for th in quad_pts])
        log_post = log_lik + log_prior
        log_post -= np.max(log_post)  # shift for numerical stability
        post = np.exp(log_post)
        post /= post.sum()

        theta_hat = np.sum(post * quad_pts)
        se = np.sqrt(np.sum(post * (quad_pts - theta_hat) ** 2))
        ll = _loglik(theta_hat)

    elif method == "MAP":
        # Maximum A Posteriori
        def _neg_posterior(theta_val):
            ll = _loglik(theta_val)
            lp = -0.5 * ((theta_val - prior_mean) / prior_sd) ** 2
            return -(ll + lp)

        res = optimize.minimize_scalar(_neg_posterior, bounds=(-6.0, 6.0), method="bounded")
        theta_hat = float(res.x)
        info_val = _info_at(theta_hat) + 1.0 / prior_sd**2
        se = 1.0 / np.sqrt(info_val) if info_val > 1e-10 else np.nan
        ll = _loglik(theta_hat)

    else:
        # MLE via Newton-Raphson
        def _neg_ll(theta_val):
            return -_loglik(theta_val)

        res = optimize.minimize_scalar(_neg_ll, bounds=(-6.0, 6.0), method="bounded")
        theta_hat = float(res.x)
        info_val = _info_at(theta_hat)
        se = 1.0 / np.sqrt(info_val) if info_val > 1e-10 else np.nan
        ll = _loglik(theta_hat)

    return {
        "theta": float(theta_hat),
        "se": float(se),
        "method": method,
        "loglik": float(ll),
    }


ability = irtab


def cheatsheet() -> str:
    return "irtab({}) -> IRT ability (theta) estimation: MLE, MAP, and EAP."
