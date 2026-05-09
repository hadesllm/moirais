# moirais.fn — function file (hadesllm/moirais)
"""Concept drift detection (ADWIN). 'Things are changing...' -- Drift"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def adwin_drift(
    stream: np.ndarray,
    *,
    delta: float = 0.002,
    min_window: int = 10,
) -> DescriptiveResult:
    """Detect concept drift using ADWIN (ADaptive WINdowing) algorithm.

    ADWIN (Bifet & Gavalda, 2007) maintains a variable-length window and
    detects distributional change by comparing sub-window means.

    Parameters
    ----------
    stream : array-like
        Sequential data stream.
    delta : float
        Confidence parameter. Smaller = fewer false alarms.
    min_window : int
        Minimum window size before testing for drift.

    Returns
    -------
    DescriptiveResult
        With ``value`` = list of drift point indices and
        ``extra`` containing number of drifts detected.
    """
    x = np.asarray(stream, dtype=float).ravel()
    n = len(x)
    if n < 2 * min_window:
        return DescriptiveResult(
            name="adwin_drift",
            value=[],
            extra={"n_drifts": 0, "n": n},
        )

    drift_points = []
    window_start = 0

    for i in range(min_window, n):
        window = x[window_start : i + 1]
        w_len = len(window)
        if w_len < 2 * min_window:
            continue

        best_cut = None
        best_eps = 0.0

        for cut in range(min_window, w_len - min_window + 1):
            left = window[:cut]
            right = window[cut:]
            n0, n1 = len(left), len(right)
            mu0, mu1 = left.mean(), right.mean()
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps_cut = np.sqrt(np.log(4.0 / delta) / (2.0 * m))
            if abs(mu0 - mu1) >= eps_cut and abs(mu0 - mu1) > best_eps:
                best_eps = abs(mu0 - mu1)
                best_cut = cut

        if best_cut is not None:
            drift_points.append(window_start + best_cut)
            window_start = window_start + best_cut

    return DescriptiveResult(
        name="adwin_drift",
        value=drift_points,
        extra={"n_drifts": len(drift_points), "n": n, "delta": delta},
    )


drift = adwin_drift


def cheatsheet() -> str:
    return "adwin_drift({}) -> Concept drift detection (ADWIN). 'Things are changing...' --"
