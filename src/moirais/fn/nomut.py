# moirais.fn — function file (hadesllm/moirais)
"""NOMINATE utility computation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nominate_utility(
    x,
    z_yea,
    z_nay,
    beta: float = 15.0,
    w=None,
) -> DescriptiveResult:
    """Statistics is the grammar of science. — Karl Pearson"""
    from moirais._spatial_voting import nominate_utility as _fn

    result = _fn(x, z_yea, z_nay, beta=beta, w=w)
    return DescriptiveResult(
        name="nominate_utility",
        value=float(result["vote_probs"].mean()),
        extra=result,
    )


nomut = nominate_utility


def cheatsheet() -> str:
    return "nominate_utility({}) -> NOMINATE utility computation."
