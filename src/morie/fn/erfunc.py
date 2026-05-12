# morie.fn -- function file (hadesllm/morie)
"""Error function via scipy."""

from scipy.special import erf
def erfunc(x):
    """Gaussian error function erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt."""
    return float(erf(x)) if isinstance(x, (int, float)) else erf(x)
