# morie.fn -- function file (hadesllm/morie)
"""
Transitivity spatial network

Category: MovTyp
"""

import numpy as np


def mttrs(trajectory=None, n=50, dt=1.0):
    """Transitivity spatial network

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


short = "mttrs"
alias = "mttrs"
quote = "An investment in knowledge pays the best interest. -- Benjamin Franklin"
mttrs = mttrs


def cheatsheet() -> str:
    return "mttrs({}) -> Transitivity spatial network"
