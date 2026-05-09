# moirais.fn — function file (hadesllm/moirais)
r"""Learning rate scheduler with cosine annealing.

Decays learning rate following cosine annealing schedule.

Notes
-----
Cosine annealing schedule::

    lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

References
----------
Loshchilov, I., & Hutter, F. (2016).
SGDR: Stochastic gradient descent with warm restarts.
In ICLR.
"""

__all__ = ["lradp"]

import numpy as np


def lradp(
    t,
    t_max,
    learning_rate_min=1e-6,
    learning_rate_max=0.1,
):
    """
    Cosine annealing learning rate scheduler.

    Parameters
    ----------
    t : int
        Current timestep (iteration number).
    t_max : int
        Total number of iterations.
    learning_rate_min : float, optional
        Minimum learning rate. Default 1e-6.
    learning_rate_max : float, optional
        Maximum learning rate. Default 0.1.

    Returns
    -------
    float
        Scheduled learning rate at step t.
    """
    if not (0 <= t <= t_max):
        raise ValueError(f"t must be in [0, {t_max}]")

    if t_max <= 0:
        raise ValueError("t_max must be positive")

    if learning_rate_min < 0 or learning_rate_max <= 0:
        raise ValueError("learning rates must be non-negative")

    if learning_rate_min >= learning_rate_max:
        raise ValueError("learning_rate_min must be < learning_rate_max")

    lr = learning_rate_min + 0.5 * (learning_rate_max - learning_rate_min) * (
        1 + np.cos(np.pi * t / t_max)
    )

    return float(lr)
