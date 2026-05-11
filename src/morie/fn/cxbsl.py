# morie.fn — function file (hadesllm/morie)
"""Cox baseline hazard (Breslow estimator)."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["cxbsl"]


def cxbsl(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> dict[str, Any]:
    r"""
    Estimate the cumulative baseline hazard via the Breslow estimator.

    .. math::

        \hat{\Lambda}_0(t) = \sum_{t_i \le t, \delta_i=1}
        \frac{1}{\sum_{j \in R(t_i)} \exp(\hat\beta^T X_j)}

    :param time: Event/censoring times, shape (n,).
    :param event: Event indicators (1=event, 0=censored), shape (n,).
    :param X: Covariate matrix, shape (n, p).
    :param beta: Fitted Cox regression coefficients, shape (p,).
    :return: Dict with ``times``, ``cumhazard``, ``baseline_survival``, ``n_events``.
    :raises ValueError: If arrays are mismatched or empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 11. Springer.
    Breslow, N.E. (1972). Contribution to discussion of paper by D.R. Cox.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(time)
    if n == 0:
        raise ValueError("Arrays must be non-empty.")

    eta = X @ beta
    exp_eta = np.exp(eta)

    order = np.argsort(time)
    time_sorted = time[order]
    event_sorted = event[order]
    exp_eta_sorted = exp_eta[order]

    unique_times = np.unique(time_sorted[event_sorted == 1])
    cumhazard = np.zeros(len(unique_times))

    running = 0.0
    for k, t in enumerate(unique_times):
        risk_set = exp_eta_sorted[time_sorted >= t]
        denom = np.sum(risk_set)
        n_events_at_t = np.sum((time_sorted == t) & (event_sorted == 1))
        if denom > 0:
            running += n_events_at_t / denom
        cumhazard[k] = running

    baseline_survival = np.exp(-cumhazard)

    return {
        "times": unique_times,
        "cumhazard": cumhazard,
        "baseline_survival": baseline_survival,
        "n_events": int(np.sum(event)),
    }


def cheatsheet() -> str:
    return "cxbsl({time, event, X, beta}) -> Breslow baseline hazard estimator."
