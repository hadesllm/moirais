# morie.fn -- function file (hadesllm/morie)
"""
Radius spatial network

Category: MovTyp
"""

import numpy as np


def mtrad(trajectory=None, n=50, dt=1.0):
    """Radius spatial network

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


short = "mtrad"
alias = "mtrad"
quote = "One does not simply walk. -- Boromir"
mtrad = mtrad


def cheatsheet() -> str:
    return "mtrad({}) -> Radius spatial network"
