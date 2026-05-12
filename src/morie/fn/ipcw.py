# morie.fn -- function file (hadesllm/morie)
"""Inverse probability of censoring weights (IPCW)."""

from __future__ import annotations

import numpy as np

__all__ = ["ipcw"]


def ipcw(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray | None = None,
) -> dict:
    """Compute IPCW weights using Kaplan-Meier of censoring.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    X : array-like, optional
        Covariates (unused, for API compatibility).

    Returns
    -------
    dict
        weights, censoring_survival, n_obs, n_censored.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)

    cens_indicator = 1 - event
    order = np.argsort(time, kind="stable")
    t_s = time[order]
    c_s = cens_indicator[order]

    unique_t = np.unique(t_s[c_s == 1])
    G = np.ones(n)
    s = 1.0

    for tj in unique_t:
        nj = np.sum(t_s >= tj)
        cj = np.sum((t_s == tj) & (c_s == 1))
        if nj > 0:
            s *= 1 - cj / nj
        mask = t_s >= tj
        G[mask] = s

    inv_order = np.argsort(order)
    G_orig = G[inv_order]

    G_safe = np.where(G_orig > 1e-10, G_orig, 1.0)
    weights = np.where(G_orig > 1e-10, 1.0 / G_safe, 1.0)

    return {
        "weights": weights,
        "censoring_survival": G_orig,
        "n_obs": n,
        "n_censored": int(np.sum(cens_indicator)),
    }


ipcw_fn = ipcw


def cheatsheet() -> str:
    return "ipcw(time, event) -> Inverse probability of censoring weights."
