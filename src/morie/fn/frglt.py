# morie.fn — function file (hadesllm/morie)
"""Frailty model (shared gamma frailty)."""

from __future__ import annotations

import numpy as np

__all__ = ["frglt"]


def frglt(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    cluster: np.ndarray,
    *,
    frailty: str = "gamma",
) -> dict:
    """Shared frailty survival model.

    Fits a Cox-type model with cluster-level frailty using
    penalized partial likelihood approximation.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    X : array-like
        Covariate matrix (n, p).
    cluster : array-like
        Cluster membership indicator (n,).
    frailty : str
        Frailty distribution: "gamma" or "lognormal".

    Returns
    -------
    dict
        coefficients, se, frailty_variance, frailty_values,
        n_clusters, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    cluster = np.asarray(cluster)
    n, p = X.shape
    unique_clusters = np.unique(cluster)
    nc = len(unique_clusters)

    cluster_idx = np.zeros(n, dtype=int)
    for i, c in enumerate(unique_clusters):
        cluster_idx[cluster == c] = i

    from .coxph import coxph
    cox_fit = coxph(time, event, X)
    beta = cox_fit["coefficients"].copy()

    lp = X @ beta
    exp_lp = np.exp(lp)

    order = np.argsort(time, kind="stable")
    t_s = time[order]
    e_s = event[order]
    exp_lp_s = exp_lp[order]
    cl_s = cluster_idx[order]

    event_times = np.unique(t_s[e_s == 1])
    H0 = np.zeros(n)
    cum_h = 0.0
    for tj in event_times:
        at_risk = t_s >= tj
        dj = np.sum((t_s == tj) & (e_s == 1))
        denom = np.sum(exp_lp_s[at_risk])
        if denom > 0:
            cum_h += dj / denom
        H0[t_s <= tj] = cum_h

    inv_order = np.argsort(order)
    H0_orig = H0[inv_order]

    d_c = np.zeros(nc)
    H_c = np.zeros(nc)
    for c in range(nc):
        mask = cluster_idx == c
        d_c[c] = np.sum(event[mask])
        H_c[c] = np.sum(H0_orig[mask] * exp_lp[mask])

    theta = 1.0
    for _ in range(50):
        w = d_c + 1 / theta
        z = H_c + 1 / theta
        frailty_vals = w / z
        theta_new = float(np.var(frailty_vals))
        if abs(theta_new - theta) < 1e-6:
            theta = max(theta_new, 1e-6)
            break
        theta = max(theta_new, 1e-6)

    frailty_vals = (d_c + 1 / theta) / (H_c + 1 / theta)

    return {
        "coefficients": beta,
        "se": cox_fit["se"],
        "frailty_variance": float(theta),
        "frailty_values": frailty_vals,
        "frailty_distribution": frailty,
        "n_clusters": nc,
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


frglt_fn = frglt


def cheatsheet() -> str:
    return "frglt(time, event, X, cluster) -> Shared frailty survival model."
