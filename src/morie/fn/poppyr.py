# morie.fn — function file (hadesllm/morie)
"""Population pyramid data preparation."""

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def population_pyramid(
    ages: list | np.ndarray,
    sexes: list | np.ndarray,
    bins: list | None = None,
) -> DescriptiveResult:
    """Prepare population pyramid data from individual records.

    Parameters
    ----------
    ages : array-like
        Individual ages.
    sexes : array-like
        Sex labels (e.g. 'M', 'F').
    bins : list or None
        Age bin edges. Default: 0-90 by 5.

    Returns
    -------
    DescriptiveResult
    """
    a = np.asarray(ages, dtype=float)
    s = np.asarray(sexes, dtype=str)
    if bins is None:
        bins = list(range(0, 95, 5))

    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
    age_groups = pd.cut(a, bins=bins, labels=labels, right=False, include_lowest=True)

    df = pd.DataFrame({"age_group": age_groups, "sex": s})
    tbl = df.groupby(["age_group", "sex"], observed=True).size().unstack(fill_value=0)

    return DescriptiveResult(
        name="population_pyramid",
        value=tbl,
        extra={"n": len(a), "n_bins": len(labels)},
    )


poppyr = population_pyramid


def cheatsheet() -> str:
    return "population_pyramid({}) -> Population pyramid data preparation."
