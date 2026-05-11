"""Victimization severity scale."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def victim_severity(
    incident_types: list[str] | np.ndarray,
    severity_weights: dict[str, float],
) -> ESRes:
    """Compute weighted victimization severity index.

    Parameters
    ----------
    incident_types : array-like
        Incident type labels.
    severity_weights : dict
        Type -> severity weight mapping.

    Returns
    -------
    ESRes
    """
    types = list(incident_types)
    if len(types) == 0:
        raise ValueError("incident_types must be non-empty")
    total = sum(severity_weights.get(t, 1.0) for t in types)
    index = total / len(types)
    return ESRes(
        measure="victimization_severity_index",
        estimate=index,
        n=len(types),
        extra={"total_severity": total},
    )


vctsv = victim_severity


def cheatsheet() -> str:
    return "victim_severity({}) -> Victimization severity scale."
