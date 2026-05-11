"""Crime severity index."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def tps_crime_severity(
    offenses: list[str] | np.ndarray,
    severity_weights: dict[str, float],
) -> ESRes:
    """Compute crime severity index (CSI-style).

    Weighted sum of offenses by severity, normalised by total offenses.

    Parameters
    ----------
    offenses : array-like
        List of offense type labels.
    severity_weights : dict
        Mapping of offense type to severity weight.

    Returns
    -------
    ESRes
    """
    off = list(offenses)
    if len(off) == 0:
        raise ValueError("offenses must be non-empty")
    total_weight = sum(severity_weights.get(o, 1.0) for o in off)
    csi = total_weight / len(off)
    return ESRes(
        measure="crime_severity_index",
        estimate=csi,
        n=len(off),
        extra={"total_weight": total_weight, "n_offenses": len(off)},
    )


tpscrs = tps_crime_severity


def cheatsheet() -> str:
    return "tps_crime_severity({}) -> Crime severity index."
