# moirais.fn — function file (hadesllm/moirais)
"""DW-NOMINATE dynamic weighted estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def dw_nominate_estimate(
    votes,
    n_dims: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """Out of chaos, comes order. — Friedrich Nietzsche"""
    from moirais._spatial_voting import dw_nominate as _fn

    result = _fn(votes, n_dims=n_dims, max_iter=max_iter, tol=tol)
    return DescriptiveResult(
        name="dw_nominate_estimate",
        value=result["gmp"],
        extra=result,
    )


dwnom = dw_nominate_estimate


def cheatsheet() -> str:
    return "dw_nominate_estimate({}) -> DW-NOMINATE dynamic weighted estimation."
