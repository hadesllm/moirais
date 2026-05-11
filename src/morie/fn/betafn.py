# morie.fn — function file (hadesllm/morie)
"""Beta function."""

from scipy.special import beta as _b
def betafn(a, b):
    """Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)."""
    return float(_b(a, b))
