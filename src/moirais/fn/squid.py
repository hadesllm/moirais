"""That which does not kill us makes us stronger. — Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def threat_score(
    data: pd.DataFrame,
    *,
    features: list[str] | None = None,
    weights: dict[str, float] | None = None,
    normalize: bool = True,
) -> DescriptiveResult:
    """Compute composite threat scores from multiple risk indicators.

    Produces a weighted z-score composite. Each feature is standardised
    to zero mean and unit variance, then combined with weights.

    Parameters
    ----------
    data : DataFrame
        Input data with numeric risk indicator columns.
    features : list of str or None
        Columns to include (defaults to all numeric).
    weights : dict or None
        Per-feature weights (defaults to equal).
    normalize : bool
        If True, rescale final scores to [0, 1].

    Returns
    -------
    DescriptiveResult
        ``value`` is the mean threat score; ``extra`` has per-row scores
        and feature contributions.
    """
    numeric = data.select_dtypes(include="number")
    if features is not None:
        missing = [f for f in features if f not in numeric.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        numeric = numeric[features]
    if numeric.shape[1] == 0:
        raise ValueError("No numeric features to score")

    if weights is None:
        w = np.ones(numeric.shape[1])
    else:
        w = np.array([weights.get(c, 1.0) for c in numeric.columns])

    means = numeric.mean()
    stds = numeric.std()
    stds = stds.replace(0, 1)
    z = (numeric - means) / stds

    scores = z.to_numpy() @ w
    if normalize:
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.zeros_like(scores)

    contributions = {}
    for i, col in enumerate(numeric.columns):
        contributions[col] = float(np.abs(z[col].mean() * w[i]))

    return DescriptiveResult(
        name="Threat Score",
        value=float(scores.mean()),
        extra={
            "scores": scores.tolist(),
            "contributions": contributions,
            "n": len(scores),
            "n_features": numeric.shape[1],
            "normalized": normalize,
        },
    )


squid = threat_score


def cheatsheet() -> str:
    return "threat_score({}) -> Threat scoring model. 'They are everywhere and they are no o"
