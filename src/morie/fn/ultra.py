"""Ensemble model aggregation. 'I am Ultra Magnus, and I will lead.' -- Ultra Magnus"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ensemble_aggregate(
    predictions: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    method: str = "mean",
) -> DescriptiveResult:
    """Aggregate predictions from multiple models.

    Parameters
    ----------
    predictions : ndarray of shape (n_models, n_samples)
        Predictions from each model.
    weights : ndarray of shape (n_models,), optional
        Model weights. Defaults to uniform.
    method : str
        Aggregation method: 'mean', 'median', 'trimmed_mean', or 'vote'.

    Returns
    -------
    DescriptiveResult
        With ``value`` = aggregated prediction vector.
    """
    P = np.asarray(predictions, dtype=float)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    n_models, n_samples = P.shape

    if weights is None:
        w = np.ones(n_models) / n_models
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if len(w) != n_models:
            raise ValueError("weights length must match n_models")
        if w.sum() <= 0:
            raise ValueError("weights must have positive sum")
        w = w / w.sum()

    if method == "mean":
        agg = np.average(P, axis=0, weights=w)
    elif method == "median":
        agg = np.median(P, axis=0)
    elif method == "trimmed_mean":
        if n_models < 3:
            agg = np.average(P, axis=0, weights=w)
        else:
            sorted_p = np.sort(P, axis=0)
            trim = max(1, n_models // 5)
            agg = sorted_p[trim:-trim].mean(axis=0) if n_models - 2 * trim > 0 else sorted_p.mean(axis=0)
    elif method == "vote":
        agg = np.apply_along_axis(
            lambda col: np.bincount(col.astype(int), minlength=int(P.max()) + 1).argmax(), axis=0, arr=P
        ).astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")

    disagreement = float(np.mean(np.std(P, axis=0)))

    return DescriptiveResult(
        name="ensemble_aggregate",
        value=agg,
        extra={"n_models": n_models, "n_samples": n_samples, "method": method, "disagreement": disagreement},
    )


ultra = ensemble_aggregate


def cheatsheet() -> str:
    return "ensemble_aggregate({}) -> Ensemble model aggregation. 'I am Ultra Magnus, and I will l"
