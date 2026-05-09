# moirais.fn — function file (hadesllm/moirais)
"""Feature normalization (z-score, min-max, robust)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Great, kid. Don't get cocky."


def feature_normalize(X, method="zscore", **kwargs) -> DescriptiveResult:
    """Normalize features column-wise.

    Parameters
    ----------
    X : array-like of shape (n, p)
    method : str
        "zscore", "minmax", or "robust" (default "zscore").

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    if method == "minmax":
        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        denom = xmax - xmin
        denom[denom == 0] = 1.0
        X_norm = (X - xmin) / denom
        params = {"min": xmin, "max": xmax}
    elif method == "robust":
        med = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        X_norm = (X - med) / iqr
        params = {"median": med, "iqr": iqr}
    else:
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        X_norm = (X - mu) / sd
        params = {"mean": mu, "std": sd}

    return DescriptiveResult(
        name="feature_normalize",
        value=float(np.mean(X_norm.std(axis=0))),
        extra={"X_normalized": X_norm, "method": method, **params},
    )


fnorm = feature_normalize


def cheatsheet() -> str:
    return "feature_normalize({}) -> Feature normalization (z-score, min-max, robust)."
