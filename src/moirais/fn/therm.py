"""Feeling thermometer standardization. 'Dragon of the Darkness Flame!' -- Hiei, Yu Yu Hakusho"""

from __future__ import annotations

from ._containers import DescriptiveResult


def feeling_thermometer_scale(ratings, lo=0, hi=100):
    """Standardize feeling thermometer ratings to [0, 1] range.

    Parameters
    ----------
    ratings : array-like
        Raw thermometer ratings.
    lo : float
        Minimum of original scale.
    hi : float
        Maximum of original scale.

    Returns
    -------
    DescriptiveResult
        value = standardized ratings (ndarray in [0, 1]).
    """
    import numpy as np

    R = np.asarray(ratings, dtype=float)
    R_std = (R - lo) / (hi - lo)
    R_std = np.clip(R_std, 0.0, 1.0)
    return DescriptiveResult(
        name="feeling_thermometer_scale",
        value=R_std,
        extra={"lo": lo, "hi": hi, "mean_std": float(np.mean(R_std))},
    )


therm = feeling_thermometer_scale


def cheatsheet() -> str:
    return "feeling_thermometer_scale({}) -> Feeling thermometer standardization. 'Dragon of the Darkness"
