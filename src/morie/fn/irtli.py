# morie.fn — function file (hadesllm/morie)
"""IRT log-likelihood at given theta."""

from __future__ import annotations

import numpy as np


def irt_likelihood(
    responses: np.ndarray,
    item_params: dict,
    theta: float,
) -> float:
    """Compute log-likelihood of responses at a given theta.

    Parameters
    ----------
    responses : ndarray
        Binary response vector (k,).
    item_params : dict
        {item_name: {'a': ..., 'b': ...}}.
    theta : float
        Ability level.

    Returns
    -------
    float
        Log-likelihood value.

    References
    ----------
    Lord, F. M. (1980). Applications of Item Response Theory to Practical
    Testing Problems. Lawrence Erlbaum.
    """
    r = np.asarray(responses, dtype=np.float64).ravel()
    params_list = list(item_params.values())
    k = len(params_list)
    if len(r) != k:
        raise ValueError(f"responses length {len(r)} != n_items {k}")

    ll = 0.0
    for j in range(k):
        if np.isnan(r[j]):
            continue
        a = params_list[j].get("a", 1.0)
        b = params_list[j].get("b", 0.0)
        exp_val = a * (theta - b)
        if exp_val > 500:
            p = 1.0
        elif exp_val < -500:
            p = 0.0
        else:
            p = 1.0 / (1.0 + np.exp(-exp_val))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        ll += r[j] * np.log(p) + (1 - r[j]) * np.log(1 - p)

    return float(ll)


def cheatsheet() -> str:
    return "irt_likelihood({}) -> IRT log-likelihood at given theta."
