# moirais.fn — function file (hadesllm/moirais)
r"""Group normalization.

Normalizes activations within groups.

References
----------
Yuxin Wu & Kaiming He (2018).
Group normalization.
In ECCV (pp. 3-19).
"""

__all__ = ["gnorm"]

import numpy as np


def gnorm(
    x,
    num_groups=32,
    epsilon=1e-6,
    gamma=None,
    beta=None,
):
    """
    Group normalization.

    Parameters
    ----------
    x : ndarray
        Input, shape (batch, channels, *spatial).
    num_groups : int, optional
        Number of groups. Default 32.
    epsilon : float, optional
        Numerical stability. Default 1e-6.
    gamma : ndarray, optional
        Scale, shape (channels,).
    beta : ndarray, optional
        Shift, shape (channels,).

    Returns
    -------
    dict
        Keys: 'output', 'mean', 'var'.
    """
    x = np.asarray(x, dtype=float)

    batch, channels = x.shape[0], x.shape[1]

    if channels % num_groups != 0:
        raise ValueError(f"channels {channels} not divisible by num_groups {num_groups}")

    x_reshaped = x.reshape(batch, num_groups, channels // num_groups, *x.shape[2:])

    axes = tuple(range(2, x_reshaped.ndim))
    mean = np.mean(x_reshaped, axis=axes, keepdims=True)
    var = np.var(x_reshaped, axis=axes, keepdims=True)

    x_norm = (x_reshaped - mean) / np.sqrt(var + epsilon)
    x_norm = x_norm.reshape(x.shape)

    if gamma is not None:
        x_norm = x_norm * gamma.reshape(1, channels, *[1] * (x.ndim - 2))

    if beta is not None:
        x_norm = x_norm + beta.reshape(1, channels, *[1] * (x.ndim - 2))

    return {"output": x_norm, "mean": mean.reshape(batch, num_groups, -1),
            "var": var.reshape(batch, num_groups, -1)}
