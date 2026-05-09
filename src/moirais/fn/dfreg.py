# moirais.fn — function file (hadesllm/moirais)
"""Regression degrees of freedom."""

def dfreg(k: int) -> int:
    """Regression degrees of freedom: number of slope params."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    return k
