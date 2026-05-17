"""Compute Shannon entropy and entropy production rate."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def entropy_production(
    probs: np.ndarray,
    *,
    probs_prev: np.ndarray | None = None,
    dt: float = 1.0,
    base: float = 2.0,
) -> DescriptiveResult:
    """Compute Shannon entropy and entropy production rate.

    H = -sum(p * log(p))
    If probs_prev given, computes dH/dt and KL divergence.

    Parameters
    ----------
    probs : array-like
        Probability distribution (sums to 1).
    probs_prev : array-like, optional
        Previous distribution for rate computation.
    dt : float
        Time step between distributions.
    base : float
        Logarithm base (2 for bits, e for nats).

    Returns
    -------
    DescriptiveResult
        With ``value`` = entropy and ``extra`` containing rate and KL.
    """
    p = np.asarray(probs, dtype=float).ravel()
    if abs(p.sum() - 1.0) > 0.01:
        raise ValueError("probs must sum to ~1")
    p = np.clip(p, 1e-30, 1.0)

    if base == 2:
        log_fn = np.log2
    elif base == np.e:
        log_fn = np.log
    else:
        log_fn = lambda x: np.log(x) / np.log(base)

    H = float(-np.sum(p * log_fn(p)))

    extra = {"base": base, "n_states": len(p)}

    if probs_prev is not None:
        q = np.asarray(probs_prev, dtype=float).ravel()
        q = np.clip(q, 1e-30, 1.0)
        H_prev = float(-np.sum(q * log_fn(q)))
        dH_dt = (H - H_prev) / dt
        kl = float(np.sum(p * (log_fn(p) - log_fn(q))))
        extra["H_prev"] = H_prev
        extra["dH_dt"] = dH_dt
        extra["kl_divergence"] = kl
        extra["dt"] = dt

    return DescriptiveResult(
        name="entropy_production",
        value=H,
        extra=extra,
    )


entpro = entropy_production


def cheatsheet() -> str:
    return 'entropy_production({}) -> Entropy production rate.'
