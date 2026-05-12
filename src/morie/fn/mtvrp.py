# morie.fn -- function file (hadesllm/morie)
"""
Vehicle routing movement

Category: MovTyp
"""

import numpy as np


def mtvrp(trajectory=None, n=50, dt=1.0):
    """Vehicle routing movement

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if trajectory is None:
        steps = np.random.default_rng(0).standard_normal((n, 2))
        trajectory = np.cumsum(steps, axis=0)
    diffs = np.diff(trajectory, axis=0)
    step_lens = np.sqrt(np.sum(diffs**2, axis=1))
    stat = float(np.mean(step_lens))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n_steps": len(trajectory) - 1,
            "total_distance": float(np.sum(step_lens)),
            "mean_step": float(np.mean(step_lens)),
            "max_step": float(np.max(step_lens)),
        },
    )


short = "mtvrp"
alias = "mtvrp"
quote = "The needs of the many outweigh the few. -- Spock"
mtvrp = mtvrp


def cheatsheet() -> str:
    return "mtvrp({}) -> Vehicle routing movement"
