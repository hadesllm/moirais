"""Sample size bound."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "In a dark place we find ourselves, and a little more knowledge lights our way."


def sample_bound(confidence=0.95, margin=0.05, p=0.5, **kwargs) -> DescriptiveResult:
    """Compute sample size bound for a given confidence and margin.

    Uses the normal approximation: n = (z^2 * p * (1-p)) / margin^2.

    Parameters
    ----------
    confidence : float
        Confidence level (default 0.95).
    margin : float
        Margin of error (default 0.05).
    p : float
        Estimated proportion (default 0.5 for maximum).

    Returns
    -------
    DescriptiveResult
    """
    from scipy.stats import norm

    if not (0 < confidence < 1):
        raise ValueError("Confidence must be in (0, 1).")
    if margin <= 0:
        raise ValueError("Margin must be positive.")
    z = norm.ppf(1 - (1 - confidence) / 2.0)
    n = (z**2 * p * (1 - p)) / margin**2
    import math

    n_ceil = math.ceil(n)
    return DescriptiveResult(
        name="sample_bound",
        value=float(n_ceil),
        extra={
            "confidence": confidence,
            "margin": margin,
            "p": p,
            "z": float(z),
            "n_exact": float(n),
            "n_ceil": n_ceil,
        },
    )


smpbd = sample_bound


def cheatsheet() -> str:
    return "sample_bound({}) -> Sample size bound."
