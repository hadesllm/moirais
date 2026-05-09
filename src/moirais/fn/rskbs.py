# moirais.fn — function file (hadesllm/moirais)
"""Brier score for risk predictions."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def risk_brier(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
) -> ESRes:
    """Brier score for risk probability calibration.

    Parameters
    ----------
    predicted_probs : ndarray
        Predicted probabilities in [0, 1].
    outcomes : ndarray
        Binary outcomes.

    Returns
    -------
    ESRes
        estimate is the Brier score (lower is better).
    """
    p = np.asarray(predicted_probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    brier = float(np.mean((p - o) ** 2))
    return ESRes(measure="risk_brier", estimate=brier, n=len(o))


rskbs = risk_brier


def cheatsheet() -> str:
    return "risk_brier({}) -> Brier score for risk predictions."
