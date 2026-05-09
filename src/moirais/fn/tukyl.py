# moirais.fn — function file (hadesllm/moirais)
"""Tukey biweight loss."""

def tukyl(residual: float, c: float = 4.685) -> float:
    """Tukey biweight (bisquare) loss.

    L(r) = (c²/6)(1 − (1 − (r/c)²)³)   if |r| ≤ c
         =  c²/6                          otherwise

    Bounded — totally rejects outliers beyond c.
    """
    r = residual
    if abs(r) <= c:
        u = (r / c) ** 2
        return (c * c / 6.0) * (1.0 - (1.0 - u) ** 3)
    return c * c / 6.0
