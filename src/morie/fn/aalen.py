# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Aalen additive hazards model."""

from __future__ import annotations

import numpy as np

__all__ = ["aalen"]


def aalen(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
) -> dict:
    """Aalen additive hazards model.

    h(t|X) = beta_0(t) + beta_1(t)*X_1 + ... + beta_p(t)*X_p

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    X : array-like
        Covariate matrix (n, p).

    Returns
    -------
    dict
        times, cumulative_coefficients, se, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    X_aug = np.column_stack([np.ones(n), X])
    q = p + 1
    event_times = np.sort(np.unique(time[event == 1]))
    nt = len(event_times)

    cum_beta = np.zeros((nt, q))
    cum_var = np.zeros((nt, q))

    for j, tj in enumerate(event_times):
        at_risk = time >= tj
        Y = at_risk.astype(float)
        X_r = X_aug[at_risk]
        nr = X_r.shape[0]
        if nr < q:
            if j > 0:
                cum_beta[j] = cum_beta[j - 1]
                cum_var[j] = cum_var[j - 1]
            continue

        dN = ((time == tj) & (event == 1)).astype(float)

        try:
            XtX_inv = np.linalg.inv(X_r.T @ X_r)
            d_beta = XtX_inv @ X_aug.T @ dN
            cum_beta[j] = (cum_beta[j - 1] if j > 0 else 0) + d_beta
            diag_var = np.diag(XtX_inv @ (X_aug.T @ np.diag(dN)) @ X_aug @ XtX_inv)
            cum_var[j] = (cum_var[j - 1] if j > 0 else 0) + diag_var
        except np.linalg.LinAlgError:
            cum_beta[j] = cum_beta[j - 1] if j > 0 else 0
            cum_var[j] = cum_var[j - 1] if j > 0 else 0

    se = np.sqrt(np.maximum(cum_var, 0))

    return {
        "times": event_times,
        "cumulative_coefficients": cum_beta,
        "se": se,
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


aalen_fn = aalen


def cheatsheet() -> str:
    return "aalen(time, event, X) -> Aalen additive hazards model."
