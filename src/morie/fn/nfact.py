# morie.fn — function file (hadesllm/morie)
"""n factorial as integer."""

import math
def nfact(n: int) -> int:
    """n! — factorial."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    return math.factorial(n)
