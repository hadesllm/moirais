# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Brier score for survival prediction."""

from __future__ import annotations

import numpy as np

__all__ = ["brscr"]


def brscr(
    predicted_surv: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    eval_time: float,
) -> dict:
    """Brier score for survival models at a specific time point.

    BS(t) = E[(S(t|X) - I(T > t))^2]

    Parameters
    ----------
    predicted_surv : array-like
        Predicted survival probabilities at eval_time.
    time : array-like
        Observed event/censoring times.
    event : array-like
        Event indicator (1=event).
    eval_time : float
        Time point at which to evaluate.

    Returns
    -------
    dict
        brier_score, ipcw_brier_score, n_obs.
    """
    S_hat = np.asarray(predicted_surv, dtype=float)
    t = np.asarray(time, dtype=float)
    d = np.asarray(event, dtype=float)
    n = len(t)

    Y = (t > eval_time).astype(float)
    bs = np.mean((S_hat - Y) ** 2)

    order = np.argsort(t, kind="stable")
    t_s = t[order]
    cens = 1 - d
    cens_s = cens[order]
    unique_t = np.unique(t_s[cens_s == 1])

    G = np.ones(n)
    s = 1.0
    for tj in sorted(unique_t):
        nj = np.sum(t_s >= tj)
        cj = np.sum((t_s == tj) & (cens_s == 1))
        if nj > 0:
            s *= 1 - cj / nj
        G[t_s >= tj] = s

    inv_order = np.argsort(order)
    G_orig = G[inv_order]
    G_t = np.interp(eval_time, np.sort(t), G_orig[np.argsort(t)])
    G_t = max(G_t, 1e-10)

    w = np.zeros(n)
    for i in range(n):
        if t[i] <= eval_time and d[i] == 1:
            w[i] = 1.0 / max(G_orig[i], 1e-10)
        elif t[i] > eval_time:
            w[i] = 1.0 / G_t

    ipcw_bs = np.mean(w * (S_hat - Y) ** 2)

    return {
        "brier_score": float(bs),
        "ipcw_brier_score": float(ipcw_bs),
        "eval_time": float(eval_time),
        "n_obs": n,
    }


brscr_fn = brscr


def cheatsheet() -> str:
    return "brscr(predicted_surv, time, event, eval_time) -> Brier score for survival."
