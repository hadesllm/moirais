# morie.fn -- function file (hadesllm/morie)
"""Gamma function."""

from scipy.special import gamma as _g
def gammfn(x):
    """Gamma function Γ(x) = (x-1)!."""
    return float(_g(x)) if isinstance(x, (int, float)) else _g(x)
