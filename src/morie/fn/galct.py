# morie.fn — function file (hadesllm/morie)
"""Resource depletion model. 'I hunger.' -- Galactus"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def depletion_model(
    t: np.ndarray | list[float],
    stock: np.ndarray | list[float],
    *,
    model: str = "exponential",
) -> DescriptiveResult:
    """Fit a resource depletion model and estimate time to exhaustion.

    Parameters
    ----------
    t : array-like
        Time points.
    stock : array-like
        Resource stock levels at each time point (positive, decreasing).
    model : str
        "exponential" (:math:`S(t) = S_0 e^{-\\lambda t}`) or
        "linear" (:math:`S(t) = S_0 - rt`).

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``rate``, ``S0``, ``time_to_exhaustion``,
        ``fitted``, ``half_life`` (exponential only), ``r_squared``.
    """
    t_arr = np.asarray(t, dtype=float)
    s_arr = np.asarray(stock, dtype=float)
    if len(t_arr) != len(s_arr) or len(t_arr) < 3:
        raise ValueError("t and stock must have same length >= 3")
    if np.any(s_arr <= 0):
        raise ValueError("All stock values must be positive")

    if model == "exponential":
        log_s = np.log(s_arr)
        slope, intercept = np.polyfit(t_arr, log_s, 1)
        lam = -slope
        S0 = np.exp(intercept)
        fitted = S0 * np.exp(-lam * t_arr)
        half_life = np.log(2) / lam if lam > 0 else np.inf
        tte = np.inf if lam <= 0 else -np.log(0.01) / lam + t_arr[0]
    elif model == "linear":
        slope, intercept = np.polyfit(t_arr, s_arr, 1)
        rate = -slope
        S0 = intercept
        fitted = S0 - rate * t_arr
        half_life = None
        lam = rate
        tte = S0 / rate + t_arr[0] if rate > 0 else np.inf
    else:
        raise ValueError(f"Unknown model: {model}")

    ss_res = float(np.sum((s_arr - fitted) ** 2))
    ss_tot = float(np.sum((s_arr - s_arr.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result = {
        "rate": float(lam),
        "S0": float(S0),
        "time_to_exhaustion": float(tte),
        "fitted": fitted,
        "r_squared": r2,
    }
    if half_life is not None:
        result["half_life"] = float(half_life)

    return DescriptiveResult(
        name="depletion_model",
        value=result,
        extra={"model": model, "n": len(t_arr)},
    )


galct = depletion_model


def cheatsheet() -> str:
    return "depletion_model({}) -> Resource depletion model. 'I hunger.' -- Galactus"
