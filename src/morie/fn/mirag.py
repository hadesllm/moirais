# morie.fn -- function file (hadesllm/morie)
"""Differential privacy (Laplace mechanism). 'Now you see me...' -- Mirage"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dp_laplace(
    values: np.ndarray,
    *,
    epsilon: float = 1.0,
    sensitivity: float | None = None,
    query: str = "mean",
    seed: int = 42,
) -> DescriptiveResult:
    """Apply the Laplace mechanism for differential privacy.

    Adds Laplace noise calibrated to the query sensitivity and privacy
    budget epsilon (Dwork & Roth, 2014).

    Parameters
    ----------
    values : array-like
        Raw data values.
    epsilon : float
        Privacy budget. Smaller = more private.
    sensitivity : float, optional
        Global sensitivity of the query. Auto-computed if None.
    query : str
        Type of query: 'mean', 'sum', 'count', or 'median'.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        With ``value`` = privatized query result and
        ``extra`` containing noise magnitude and true value.
    """
    x = np.asarray(values, dtype=float).ravel()
    if len(x) == 0:
        raise ValueError("Input must be non-empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if query == "mean":
        true_val = float(x.mean())
        if sensitivity is None:
            sensitivity = float((x.max() - x.min()) / len(x))
    elif query == "sum":
        true_val = float(x.sum())
        if sensitivity is None:
            sensitivity = float(x.max() - x.min())
    elif query == "count":
        true_val = float(len(x))
        if sensitivity is None:
            sensitivity = 1.0
    elif query == "median":
        true_val = float(np.median(x))
        if sensitivity is None:
            sensitivity = float(x.max() - x.min())
    else:
        raise ValueError(f"Unknown query: {query}")

    rng = np.random.default_rng(seed)
    scale = sensitivity / epsilon
    noise = rng.laplace(0, scale)
    private_val = true_val + noise

    return DescriptiveResult(
        name="dp_laplace",
        value=float(private_val),
        extra={
            "true_value": true_val,
            "noise": float(noise),
            "scale": scale,
            "epsilon": epsilon,
            "sensitivity": sensitivity,
            "query": query,
        },
    )


mirag = dp_laplace


def cheatsheet() -> str:
    return "dp_laplace({}) -> Differential privacy (Laplace mechanism). 'Now you see me..."
