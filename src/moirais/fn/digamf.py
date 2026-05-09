# moirais.fn — function file (hadesllm/moirais)
"""Digamma function."""

from scipy.special import digamma
def digamf(x):
    """Digamma function ψ(x) = d/dx ln Γ(x)."""
    return float(digamma(x)) if isinstance(x, (int, float)) else digamma(x)
