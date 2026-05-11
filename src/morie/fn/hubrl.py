# morie.fn — function file (hadesllm/morie)
"""Huber loss."""

def hubrl(residual: float, delta: float = 1.345) -> float:
    """Huber loss: quadratic for small residuals, linear for large.

    L(r) = ½ r²            if |r| ≤ δ
         = δ |r| − ½ δ²    otherwise
    """
    r = abs(residual)
    if r <= delta:
        return 0.5 * residual * residual
    return delta * r - 0.5 * delta * delta
