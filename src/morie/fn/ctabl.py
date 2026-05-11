# morie.fn — function file (hadesllm/morie)
"""Hope in reality is the worst of all evils because it prolongs the torments of man. — Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def contingency_table(x: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """
    Cross-tabulate two categorical variables and compute Pearson's
    chi-squared test of independence.

    :param x: 1-D array of categorical labels (first variable).
    :type x: numpy.ndarray
    :param y: 1-D array of categorical labels (second variable).
    :type y: numpy.ndarray
    :return: DescriptiveResult with observed table, chi2, p-value.
    :rtype: DescriptiveResult
    :raises ValueError: If x and y have different lengths.

    References
    ----------
    Pearson K. (1900). On the criterion that a given system of deviations
    from the probable in the case of a correlated system of variables is
    such that it can be reasonably supposed to have arisen from random
    sampling. *Philosophical Magazine*, 50(302), 157-175.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if len(x) != len(y):
        raise ValueError(f"x ({len(x)}) and y ({len(y)}) must have equal length.")
    labels_x = np.unique(x)
    labels_y = np.unique(y)
    table = np.zeros((len(labels_x), len(labels_y)), dtype=int)
    x_idx = {v: i for i, v in enumerate(labels_x)}
    y_idx = {v: i for i, v in enumerate(labels_y)}
    for xi, yi in zip(x, y):
        table[x_idx[xi], y_idx[yi]] += 1
    chi2, p, dof, expected = _st.chi2_contingency(table)
    return DescriptiveResult(
        name="contingency_table",
        value=float(chi2),
        extra={
            "table": table,
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "expected": expected,
            "labels_x": labels_x,
            "labels_y": labels_y,
        },
    )


ctabl = contingency_table


def cheatsheet() -> str:
    return "contingency_table({}) -> Contingency table with chi-squared test. 'You were the chose"
