# moirais.fn — function file (hadesllm/moirais)
"""Dose-response: number of placements to outcome."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import DescriptiveResult


def otis_dose_response(
    n_placements: np.ndarray,
    outcomes: np.ndarray,
) -> DescriptiveResult:
    """Dose-response curve: placements to outcome.

    Parameters
    ----------
    n_placements : ndarray
        Number of placements (dose).
    outcomes : ndarray
        Outcome values.

    Returns
    -------
    DescriptiveResult
    """
    doses = np.asarray(n_placements, dtype=int)
    y = np.asarray(outcomes, dtype=float)
    unique_d = np.sort(np.unique(doses))
    means = []
    counts = []
    for d in unique_d:
        mask = doses == d
        means.append(float(np.mean(y[mask])))
        counts.append(int(np.sum(mask)))
    n = len(doses)
    X = np.column_stack([np.ones(n), doses.astype(float)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return DescriptiveResult(
        name="otis_dose_response",
        value=float(beta[1]),
        extra={
            "slope": float(beta[1]),
            "intercept": float(beta[0]),
            "dose_levels": unique_d.tolist(),
            "dose_means": means,
            "dose_counts": counts,
        },
    )


odose = otis_dose_response


def cheatsheet() -> str:
    return "otis_dose_response({}) -> Dose-response: number of placements to outcome."
