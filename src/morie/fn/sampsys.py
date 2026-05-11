# morie.fn — function file (hadesllm/morie)
"""Systematic sample (every k-th unit)."""

def sampsys(population, k: int, start: int = 0):
    """Systematic sample: take every k-th unit starting at index `start`."""
    pop = list(population)
    if k < 1:
        raise ValueError("k must be ≥ 1.")
    if start < 0 or start >= len(pop):
        raise ValueError("start out of range.")
    return pop[start::k]
