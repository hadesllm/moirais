# moirais.fn — function file (hadesllm/moirais)
"""Decision tree split criterion. 'There is no spoon.' -- Spoon Boy"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def decision_split(
    data: pd.DataFrame,
    *,
    target: str = "outcome",
    feature: str = "x",
    criterion: str = "gini",
) -> DescriptiveResult:
    """Find the optimal binary split for *feature* predicting *target*.

    Evaluates every unique midpoint of *feature* and returns the threshold
    that minimises impurity (Gini or entropy) in the resulting child nodes.

    Parameters
    ----------
    data : DataFrame
        Must contain *target* (binary 0/1) and *feature* (numeric).
    target : str
        Column name for the binary class label.
    feature : str
        Column name for the numeric splitting variable.
    criterion : str
        ``"gini"`` or ``"entropy"``.

    Returns
    -------
    DescriptiveResult
        ``value`` is the best threshold; ``extra`` contains impurity
        reduction and child-node sizes.
    """
    if criterion not in ("gini", "entropy"):
        raise ValueError(f"criterion must be 'gini' or 'entropy', got '{criterion}'")
    df = data[[target, feature]].dropna()
    y = df[target].to_numpy(dtype=float)
    x = df[feature].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("target must have at least 2 unique values")

    def _impurity(arr):
        if len(arr) == 0:
            return 0.0
        p = arr.mean()
        if criterion == "gini":
            return 2 * p * (1 - p)
        eps = 1e-15
        return -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))

    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    thresholds = (x_s[:-1] + x_s[1:]) / 2
    mask = x_s[:-1] != x_s[1:]
    thresholds = thresholds[mask]

    if len(thresholds) == 0:
        raise ValueError("feature has no variability for splitting")

    parent_imp = _impurity(y)
    best_t, best_gain = float(thresholds[0]), -np.inf
    best_nl = best_nr = 0
    for t in thresholds:
        left = y[x <= t]
        right = y[x > t]
        wl, wr = len(left) / len(y), len(right) / len(y)
        gain = parent_imp - wl * _impurity(left) - wr * _impurity(right)
        if gain > best_gain:
            best_gain, best_t = gain, float(t)
            best_nl, best_nr = len(left), len(right)

    return DescriptiveResult(
        name=f"DecisionSplit ({criterion})",
        value=best_t,
        extra={
            "criterion": criterion,
            "impurity_reduction": float(best_gain),
            "parent_impurity": float(parent_imp),
            "n_left": best_nl,
            "n_right": best_nr,
            "feature": feature,
        },
    )


neom = decision_split


def cheatsheet() -> str:
    return "decision_split({}) -> Decision tree split criterion. 'There is no spoon.' -- Spoon"
