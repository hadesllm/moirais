# moirais.fn — function file (hadesllm/moirais)
"""Error degrees of freedom."""

def dferr(n: int, k: int) -> int:
    """Error (residual) degrees of freedom: n − k − 1."""
    if n - k - 1 < 1:
        raise ValueError("not enough data: n - k - 1 must be ≥ 1.")
    return n - k - 1
