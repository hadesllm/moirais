# morie.fn — function file (hadesllm/morie)
"""Burden index: weighted recidivism accounting for severity."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def recidivism_burden(
    counts: np.ndarray,
    severity_weights: np.ndarray,
) -> ESRes:
    """Burden index: weighted recidivism accounting for offense severity.

    Parameters
    ----------
    counts : ndarray
        Number of recidivism events per category.
    severity_weights : ndarray
        Severity weight for each category.

    Returns
    -------
    ESRes
        Weighted burden index.
    """
    counts = np.asarray(counts, dtype=float)
    severity_weights = np.asarray(severity_weights, dtype=float)
    burden = float(np.sum(counts * severity_weights))
    total = float(np.sum(counts))
    avg_weight = burden / total if total > 0 else 0.0
    return ESRes(
        measure="recidivism_burden",
        estimate=burden,
        n=int(total),
        extra={"avg_severity_weight": avg_weight, "total_events": total},
    )


rcdbi = recidivism_burden


def cheatsheet() -> str:
    return "recidivism_burden({}) -> Burden index: weighted recidivism accounting for severity."
