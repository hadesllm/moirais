# morie.fn — function file (hadesllm/morie)
"""Simple random sample (without replacement)."""

from typing import Sequence, Union
import numpy as np
def sampsrs(population, n: int, seed: int = 42):
    """Simple random sample of size n, without replacement."""
    rng = np.random.default_rng(seed)
    pop = list(population)
    if n > len(pop):
        raise ValueError("n exceeds population size.")
    idx = rng.choice(len(pop), size=n, replace=False)
    return [pop[i] for i in idx]
