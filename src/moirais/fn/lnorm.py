# moirais.fn — function file (hadesllm/moirais)
r"""Layer normalization.

Normalizes activations independently for each sample.

References
----------
Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).
Layer normalization.
arXiv preprint arXiv:1607.06450.
"""

__all__ = ["lnorm"]

import numpy as np
from ._richresult import RichResult


def lnorm(
    x,
    epsilon=1e-6,
    gamma=None,
    beta=None,
):
    """
    Layer normalization.

    Parameters
    ----------
    x : ndarray
        Input, shape (batch, ..., features).
    epsilon : float, optional
        Numerical stability. Default 1e-6.
    gamma : ndarray, optional
        Scale parameter, shape (features,).
    beta : ndarray, optional
        Shift parameter, shape (features,).

    Returns
    -------
    dict
        Keys: 'output', 'mean', 'var'.
    """
    x = np.asarray(x, dtype=float)

    axis = tuple(range(1, x.ndim))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + epsilon)

    if gamma is not None:
        x_norm = x_norm * gamma

    if beta is not None:
        x_norm = x_norm + beta

    return RichResult(payload={"output": x_norm, "mean": mean, "var": var})
