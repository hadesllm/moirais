# moirais.fn — function file (hadesllm/moirais)
"""Number needed to harm (NNH) from a 2x2 table."""

from ._containers import ESRes
from .nnt import number_needed_to_treat


def number_needed_to_harm(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> ESRes:
    """Number needed to harm (NNH) -- same as NNT with reversed sign convention.

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    result = number_needed_to_treat(a, b, c, d, confidence)
    return ESRes(
        measure="NNH",
        estimate=result.estimate,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        n=result.n,
    )


nnh = number_needed_to_harm


def cheatsheet() -> str:
    return "number_needed_to_harm({}) -> Number needed to harm (NNH) from a 2x2 table."
