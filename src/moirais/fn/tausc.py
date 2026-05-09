"""Tau scale estimator (Maronna & Zamar, 2002)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def tau_scale(
    x,
    *,
    c1: float = 4.5,
    c2: float = 3.0,
) -> ESRes:
    """Tau scale estimator of dispersion.

    A two-step M-scale that combines high breakdown and high efficiency
    at the normal.  Uses Tukey bisquare rho functions with constants
    *c1* (first step) and *c2* (second step).

    Parameters
    ----------
    x : array-like
        Observations.
    c1 : float
        Tuning constant for the initial M-scale (default 4.5).
    c2 : float
        Tuning constant for the second weight step (default 3.0).

    Returns
    -------
    ESRes
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 finite observations.")

    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad < 1e-12:
        return ESRes(measure="tau_scale", estimate=0.0, n=n, extra={"c1": c1, "c2": c2})

    s0 = 1.4826 * mad
    u = (a - med) / s0

    def _rho_bisq(t, c):
        r = np.where(np.abs(t) <= c, 1.0 - (1.0 - (t / c) ** 2) ** 3, 1.0)
        return r

    w2 = _rho_bisq(u, c2)
    tau2 = s0**2 * (1.0 / n) * np.sum(w2)
    tau = float(np.sqrt(tau2))

    return ESRes(
        measure="tau_scale",
        estimate=tau,
        n=n,
        extra={"c1": c1, "c2": c2, "initial_scale": float(s0)},
    )


tausc = tau_scale


def cheatsheet() -> str:
    return "tau_scale(x) -> Tau scale estimator (Maronna-Zamar)."
