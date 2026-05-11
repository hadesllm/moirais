# morie.fn — function file (hadesllm/morie)
"""R-squared (coefficient of determination) as effect size."""

from typing import Union

import numpy as np
import pandas as pd

from ._containers import ESRes
from .r_es import r_effect_size


def r_squared(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
) -> ESRes:
    """Coefficient of determination R-squared.

    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    ESRes
    """
    r_res = r_effect_size(x, y)
    r2 = r_res.estimate**2
    return ESRes(
        measure="R-squared",
        estimate=float(r2),
        ci_lower=float(r_res.ci_lower**2) if r_res.ci_lower else None,
        ci_upper=float(r_res.ci_upper**2) if r_res.ci_upper else None,
        n=r_res.n,
    )


rsq = r_squared


def cheatsheet() -> str:
    return "r_squared({}) -> R-squared (coefficient of determination) as effect size."
