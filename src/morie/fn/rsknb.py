# morie.fn — function file (hadesllm/morie)
"""Nagelkerke R-squared for risk model."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def risk_nagelkerke(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
) -> ESRes:
    """Nagelkerke R-squared for a risk prediction model.

    Parameters
    ----------
    predicted_probs : ndarray
        Predicted probabilities.
    outcomes : ndarray
        Binary outcomes.

    Returns
    -------
    ESRes
    """
    p = np.asarray(predicted_probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = len(o)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    ll_full = np.sum(o * np.log(p) + (1 - o) * np.log(1 - p))
    p0 = np.mean(o)
    p0 = np.clip(p0, 1e-15, 1 - 1e-15)
    ll_null = np.sum(o * np.log(p0) + (1 - o) * np.log(1 - p0))
    cox_snell = 1 - np.exp(-2 * (ll_full - ll_null) / n)
    max_cs = 1 - np.exp(2 * ll_null / n)
    nagelkerke = cox_snell / max_cs if max_cs > 0 else 0.0
    return ESRes(
        measure="risk_nagelkerke_r2",
        estimate=float(nagelkerke),
        n=n,
        extra={"cox_snell_r2": float(cox_snell), "ll_full": float(ll_full), "ll_null": float(ll_null)},
    )


rsknb = risk_nagelkerke


def cheatsheet() -> str:
    return "risk_nagelkerke({}) -> Nagelkerke R-squared for risk model."
