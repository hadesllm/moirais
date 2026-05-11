# morie.fn — function file (hadesllm/morie)
"""Coefficient of variation (CV)."""

from typing import Union

import numpy as np
import pandas as pd

from ._containers import ESRes
from ._helpers import _arr


def coefficient_of_variation(
    x: Union[np.ndarray, pd.Series, list],
) -> ESRes:
    """Coefficient of variation (CV).

    CV = s / |x_bar|

    Parameters
    ----------
    x : array-like

    Returns
    -------
    ESRes
    """
    x = _arr(x)
    mean = x.mean()
    sd = x.std(ddof=1)
    cv_val = sd / abs(mean) if abs(mean) > 0 else np.inf
    return ESRes(
        measure="Coefficient of variation",
        estimate=float(cv_val),
        n=len(x),
    )


cv = coefficient_of_variation


def cheatsheet() -> str:
    return "coefficient_of_variation({}) -> Coefficient of variation (CV)."
