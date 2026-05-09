# moirais.fn — function file (hadesllm/moirais)
"""Cause-specific hazard ratio estimation."""

from __future__ import annotations

import numpy as np

__all__ = ["chzrd"]


def chzrd(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    *,
    cause: int = 1,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict:
    """Cause-specific hazard model for competing risks.

    Fits a Cox model treating events from other causes
    as censored observations.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event type (0=censored, cause=event of interest) (n,).
    X : array-like
        Covariate matrix (n, p).
    cause : int
        Cause of interest.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        coefficients, se, hazard_ratios, p_values, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    X = np.asarray(X, dtype=float)

    cause_event = (event == cause).astype(float)

    from .coxph import coxph
    result = coxph(time, cause_event, X, max_iter=max_iter, tol=tol)

    result["cause"] = cause
    result["n_cause_events"] = int(np.sum(cause_event))
    result["n_competing_events"] = int(np.sum((event > 0) & (event != cause)))

    return result


chzrd_fn = chzrd


def cheatsheet() -> str:
    return "chzrd(time, event, X) -> Cause-specific hazard ratio."
