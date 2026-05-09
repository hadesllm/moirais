"""Water quality index (WQI)."""

import numpy as np

from ._containers import ESRes


def water_quality_index(
    parameters: dict[str, float],
    weights: dict[str, float] | None = None,
) -> ESRes:
    """Compute weighted water quality index.

    Uses weighted arithmetic mean of sub-index values (0-100 scale).

    Parameters
    ----------
    parameters : dict
        {parameter_name: sub-index value (0-100)}.
    weights : dict or None
        {parameter_name: weight}. Default: equal weights.

    Returns
    -------
    ESRes
    """
    if not parameters:
        raise ValueError("No parameters provided")

    names = list(parameters.keys())
    vals = np.array([parameters[n] for n in names], dtype=float)

    if weights is not None:
        w = np.array([weights.get(n, 1.0) for n in names], dtype=float)
    else:
        w = np.ones(len(vals))

    wqi = float(np.average(vals, weights=w))

    if wqi >= 90:
        quality = "excellent"
    elif wqi >= 70:
        quality = "good"
    elif wqi >= 50:
        quality = "medium"
    elif wqi >= 25:
        quality = "bad"
    else:
        quality = "very_bad"

    return ESRes(
        measure="WQI",
        estimate=wqi,
        extra={"quality": quality, "n_parameters": len(parameters)},
    )


wqidx = water_quality_index


def cheatsheet() -> str:
    return "water_quality_index({}) -> Water quality index (WQI)."
