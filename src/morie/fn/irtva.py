# morie.fn — function file (hadesllm/morie)
"""Per-legislator IRT variance from posterior."""

from __future__ import annotations

from ._containers import DescriptiveResult


def irt_variance_legislator(chain_theta) -> DescriptiveResult:
    """Posterior variance of theta per legislator.

    .. epigraph:: "Stick them with the pointy end." -- Jon, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain_theta, dtype=float)
    if chain.ndim == 1:
        chain = chain.reshape(-1, 1)
    variances = np.var(chain, axis=0, ddof=1)
    return DescriptiveResult(
        name="irt_variance_legislator",
        value=float(np.mean(variances)),
        extra={
            "variances": variances.tolist(),
            "mean_variance": float(np.mean(variances)),
            "n_legislators": chain.shape[1],
            "n_samples": chain.shape[0],
        },
    )


irtva = irt_variance_legislator


def cheatsheet() -> str:
    return "irt_variance_legislator({}) -> Per-legislator IRT variance from posterior."
