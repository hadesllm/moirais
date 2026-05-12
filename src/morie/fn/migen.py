# morie.fn -- function file (hadesllm/morie)
"""Full measurement invariance ladder by gender."""

from __future__ import annotations

import pandas as pd

from morie.fn.mi_cf import mi_configural
from morie.fn.mi_mt import mi_metric
from morie.fn.mi_sc import mi_scalar
from morie.fn.mi_st import mi_strict


def mi_by_gender(
    data: pd.DataFrame,
    *,
    gender_col: str = "gender",
    items: list[str] | None = None,
) -> list[dict]:
    """Run the full measurement invariance ladder by gender.

    Tests configural -> metric -> scalar -> strict invariance sequentially.

    Parameters
    ----------
    data : DataFrame
        Item response data with a gender column.
    gender_col : str
        Column name for gender (default 'gender').
    items : list of str, optional
        Override item names.

    Returns
    -------
    list of dict
        One dict per level: configural, metric, scalar, strict.
        Each has keys: level, fit, delta_fit, passed.

    References
    ----------
    Vandenberg, R.J. & Lance, C.E. (2000). A review and synthesis of the
        measurement invariance literature. Org. Research Methods, 3(1), 4-70.
    """
    results = []

    cfg = mi_configural(data, gender_col, items=items)
    results.append(cfg)

    met = mi_metric(data, gender_col, items=items, configural_fit=cfg)
    results.append(met)

    scl = mi_scalar(data, gender_col, items=items, metric_fit=met)
    results.append(scl)

    strt = mi_strict(data, gender_col, items=items, scalar_fit=scl)
    results.append(strt)

    return results


def cheatsheet() -> str:
    return "mi_by_gender({}) -> Full measurement invariance ladder by gender."
