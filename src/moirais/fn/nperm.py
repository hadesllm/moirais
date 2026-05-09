# moirais.fn — function file (hadesllm/moirais)
"""Permutations P(n, k)."""

import math
def nperm(n: int, k: int) -> int:
    """P(n, k) — ordered permutations."""
    if k < 0 or n < 0 or k > n:
        raise ValueError("require 0 ≤ k ≤ n.")
    return math.perm(n, k)
