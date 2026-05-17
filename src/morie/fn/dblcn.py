# morie.fn -- function file (hadesllm/morie)
"""Double centering of a distance matrix."""

from __future__ import annotations

from ._containers import DescriptiveResult


def double_centering(D) -> DescriptiveResult:
    """Apply double-centering transform to a distance matrix.

    :param D: Square distance matrix.
    :return: DescriptiveResult with centered matrix in ``extra``.

    .. epigraph:: Errors using inadequate data are much less than those using none. -- Charles Babbage
    """
    from morie._spatial_voting import double_centering as _fn

    result = _fn(D)
    return DescriptiveResult(
        name="double_centering",
        value=float(result.shape[0]),
        extra={"matrix": result},
    )


dblcn = double_centering


def cheatsheet() -> str:
    return "double_centering({}) -> Double centering of a distance matrix."
