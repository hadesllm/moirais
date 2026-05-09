# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Nonparametric bootstrap for scaling standard errors."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bootstrap_scaling_se(
    Z,
    scale_fn: str = "am",
    n_boot: int = 200,
    seed: int = 42,
) -> DescriptiveResult:
    """Bootstrap SEs for AM/blackbox scaling (Efron & Tibshirani 1993).

    :param Z: Respondent x stimulus perception matrix.
    :param scale_fn: Scaling function ("am", "blackbox", "blackbox_t").
    :param n_boot: Number of bootstrap replications.
    :param seed: Random seed.
    :return: DescriptiveResult with standard errors and CIs.

    .. epigraph:: "The will of D. cannot be stopped." -- Dr. Kureha, One Piece
    """
    from moirais._spatial_voting import nonparametric_bootstrap_scaling as _fn

    result = _fn(Z, scale_fn=scale_fn, n_boot=n_boot, seed=seed)
    return DescriptiveResult(
        name="bootstrap_scaling_se",
        value=result["n_boot"],
        extra=result,
    )


btsca = bootstrap_scaling_se


def cheatsheet() -> str:
    return "bootstrap_scaling_se({}) -> Nonparametric bootstrap for scaling standard errors."
