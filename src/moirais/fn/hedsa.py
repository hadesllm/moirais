# moirais.fn — function file (hadesllm/moirais)
"""Deterministic one-way sensitivity analysis."""

import numpy as np

from ._containers import DescriptiveResult


def deterministic_sensitivity(
    base_result: float,
    param_name: str,
    param_range: list | np.ndarray,
    results_at_range: list | np.ndarray,
) -> DescriptiveResult:
    """One-way deterministic sensitivity analysis (tornado diagram data).

    Parameters
    ----------
    base_result : float
        Base-case result.
    param_name : str
    param_range : array-like
        Values of the parameter tested.
    results_at_range : array-like
        Result at each parameter value.

    Returns
    -------
    DescriptiveResult
    """
    r = np.asarray(results_at_range, dtype=float)
    p = np.asarray(param_range, dtype=float)
    if len(r) != len(p):
        raise ValueError("param_range and results must match")

    return DescriptiveResult(
        name="DSA",
        value=float(np.max(r) - np.min(r)),
        extra={
            "param_name": param_name,
            "base_result": float(base_result),
            "min_result": float(np.min(r)),
            "max_result": float(np.max(r)),
            "param_range": p.tolist(),
            "results": r.tolist(),
        },
    )


hedsa = deterministic_sensitivity


def cheatsheet() -> str:
    return "deterministic_sensitivity({}) -> Deterministic one-way sensitivity analysis."
