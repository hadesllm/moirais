# morie.fn — function file (hadesllm/morie)
"""Non-parametric PDF estimation."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Character is destiny. — Heraclitus"


def pdf_estimate(x, method: str = "histogram", bins: int = 50, **kwargs) -> DescriptiveResult:
    """
    Estimate the probability density function non-parametrically.

    :param x: array-like of observations.
    :param method: ``'histogram'`` for histogram-based density estimation.
    :param bins: Number of bins for histogram method.
    :return: DescriptiveResult with bin centres and density values in extra.

    References
    ----------
    Silverman BW (1986). Density Estimation for Statistics and
    Data Analysis. Chapman & Hall, London.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 2:
        raise ValueError("Need at least 2 observations.")
    counts, edges = np.histogram(x, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return DescriptiveResult(
        name="pdf_estimate",
        value=float(np.max(counts)),
        extra={
            "centres": centres.tolist(),
            "density": counts.tolist(),
            "method": method,
            "n": len(x),
        },
    )


pdfen = pdf_estimate


def cheatsheet() -> str:
    return "pdf_estimate({}) -> Non-parametric PDF estimation."
