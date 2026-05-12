# morie.fn -- function file (hadesllm/morie)
"""Information gain / mutual information. 'One door, one key.' -- The Keymaker"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def information_gain(
    data: pd.DataFrame,
    *,
    target: str = "outcome",
    feature: str = "x",
    n_bins: int = 10,
) -> DescriptiveResult:
    """Compute mutual information / information gain between a feature and target.

    Discretises the feature into *n_bins* equal-width bins (if continuous)
    and computes I(X; Y) = H(Y) - H(Y|X) using empirical frequencies.

    Parameters
    ----------
    data : DataFrame
        Input data.
    target : str
        Target column (categorical or binary).
    feature : str
        Feature column (numeric or categorical).
    n_bins : int
        Number of bins for discretising continuous features.

    Returns
    -------
    DescriptiveResult
        ``value`` is the mutual information in nats; ``extra`` has
        information gain ratio and entropies.
    """
    if target not in data.columns or feature not in data.columns:
        raise ValueError(f"columns '{target}' and '{feature}' must exist in data")
    df = data[[target, feature]].dropna()
    if len(df) < 2:
        raise ValueError("Need at least 2 non-null rows")

    y = df[target]
    x = df[feature]

    if pd.api.types.is_numeric_dtype(x) and x.nunique() > n_bins:
        x = pd.cut(x, bins=n_bins, labels=False)

    def _entropy(series):
        p = series.value_counts(normalize=True).to_numpy()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    h_y = _entropy(y)
    h_x = _entropy(x)

    h_y_given_x = 0.0
    n = len(df)
    for val, grp in df.groupby(x.values):
        w = len(grp) / n
        h_y_given_x += w * _entropy(grp[target])

    mi = h_y - h_y_given_x
    gain_ratio = mi / h_x if h_x > 0 else 0.0

    return DescriptiveResult(
        name="Information Gain",
        value=float(mi),
        extra={
            "h_target": h_y,
            "h_feature": h_x,
            "h_target_given_feature": h_y_given_x,
            "gain_ratio": float(gain_ratio),
            "n": n,
            "n_bins": n_bins,
        },
    )


kymkr = information_gain


def cheatsheet() -> str:
    return "information_gain({}) -> Information gain / mutual information. 'One door, one key.' "
