# moirais.fn — function file (hadesllm/moirais)
"""Block randomization sequence generation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def randomization(
    n: int,
    *,
    block_sizes: list[int] | None = None,
    n_arms: int = 2,
    seed: int | None = None,
) -> DescriptiveResult:
    """
    Generate a block randomization sequence for a clinical trial.

    Parameters
    ----------
    n : int
        Total number of subjects.
    block_sizes : list of int, optional
        Permissible block sizes (default [4, 6]).
    n_arms : int
        Number of treatment arms.
    seed : int, optional
        Random seed.

    Returns
    -------
    DescriptiveResult
        extra has 'sequence', 'block_assignments'.

    References
    ----------
    Efird, J. (2011). Blocked randomization with randomly selected
    block sizes. *Int J Environ Res Public Health*, 8(1), 15-20.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n_arms < 2:
        raise ValueError("Need at least 2 arms.")
    if block_sizes is None:
        block_sizes = [4, 6]
    for bs in block_sizes:
        if bs % n_arms != 0:
            raise ValueError(f"Block size {bs} not divisible by n_arms={n_arms}.")

    rng = np.random.default_rng(seed)
    sequence = []
    while len(sequence) < n:
        bs = rng.choice(block_sizes)
        per_arm = bs // n_arms
        block = []
        for arm in range(n_arms):
            block.extend([arm] * per_arm)
        rng.shuffle(block)
        sequence.extend(block)

    sequence = sequence[:n]

    return DescriptiveResult(
        name="block_randomization",
        value=float(n),
        extra={
            "sequence": sequence,
            "n_arms": n_arms,
            "block_sizes": block_sizes,
        },
    )


rndm = randomization


def cheatsheet() -> str:
    return "randomization({}) -> Block randomization sequence generation."
