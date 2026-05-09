# moirais.fn — function file (hadesllm/moirais)
"""Total degrees of freedom."""

def dftot(n: int) -> int:
    """Total degrees of freedom: n − 1."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    return n - 1
