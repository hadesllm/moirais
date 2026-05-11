# morie.fn — function file (hadesllm/morie)
r"""Dropout regularization (forward pass).

Randomly zeros activations with probability p during training.

References
----------
Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
Dropout: a simple way to prevent neural networks from overfitting.
JMLR, 15(56), 1929-1958.
"""

__all__ = ["drupt"]

import numpy as np


def drupt(
    x,
    rate=0.5,
    training=True,
    seed=None,
):
    """
    Dropout layer (forward pass).

    Parameters
    ----------
    x : ndarray
        Input activations, shape (...,).
    rate : float, optional
        Dropout probability. Default 0.5.
    training : bool, optional
        If False, no dropout applied. Default True.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray
        Output with dropout applied (scaled by 1/(1-rate)).
    """
    x = np.asarray(x, dtype=float)

    if not (0 <= rate < 1):
        raise ValueError("rate must be in [0, 1)")

    if not training or rate == 0:
        return x

    rng = np.random.RandomState(seed)
    mask = rng.binomial(1, 1 - rate, size=x.shape)

    return x * mask / (1 - rate)
