# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Blackbox scaling for roll-call voting data."""

from __future__ import annotations

from ._containers import DescriptiveResult


def blackbox_scaling(
    X,
    n_dims: int = 2,
) -> DescriptiveResult:
    """Blackbox transpose scaling (Poole 1998).

    :param X: Vote matrix (legislators x roll calls).
    :param n_dims: Number of dimensions to recover.
    :return: DescriptiveResult with ideal points in ``extra``.

    .. epigraph:: "He who controls the spice controls the universe." -- Baron Harkonnen, Dune
    """
    from morie._spatial_voting import blackbox_scaling as _fn

    result = _fn(X, n_dims=n_dims)
    return DescriptiveResult(
        name="blackbox_scaling",
        value=result["explained_variance"],
        extra=result,
    )


bblck = blackbox_scaling


def cheatsheet() -> str:
    return "blackbox_scaling({}) -> Blackbox scaling for roll-call voting data."
