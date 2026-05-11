# morie.fn — function file (hadesllm/morie)
"""Binomial coefficient C(n, k)."""

import math
def ncombo(n: int, k: int) -> int:
    """C(n, k) — unordered combinations."""
    if k < 0 or n < 0 or k > n:
        raise ValueError("require 0 ≤ k ≤ n.")
    return math.comb(n, k)
