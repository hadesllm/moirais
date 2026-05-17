# morie.fn -- function file (hadesllm/morie)
"""ICC curve data for plotting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def icc_curve_data(alpha, beta, theta_range=(-3, 3)) -> DescriptiveResult:
    """Generate ICC (item characteristic curve) plot data.

    .. epigraph:: The Analytical Engine weaves algebraic patterns. -- Ada Lovelace
    """
    import numpy as np

    theta = np.linspace(theta_range[0], theta_range[1], 200)
    logit = alpha * (theta - beta)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return DescriptiveResult(
        name="icc_curve_data",
        value=float(alpha),
        extra={
            "theta": theta,
            "probability": prob,
            "alpha": float(alpha),
            "beta": float(beta),
            "theta_range": list(theta_range),
        },
    )


iccfn = icc_curve_data


def cheatsheet() -> str:
    return "icc_curve_data({}) -> ICC curve data for plotting."
