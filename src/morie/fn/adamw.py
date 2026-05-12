# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
r"""AdamW optimizer with decoupled weight decay.

Adaptive moment estimation with weight decay (L2 regularization).

References
----------
Loshchilov, I., & Hutter, F. (2019).
Decoupled weight decay regularization.
In ICLR.
"""

__all__ = ["adamw"]

import numpy as np
from ._richresult import RichResult


def adamw(
    weights,
    gradients,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    m=None,
    v=None,
    t=0,
):
    """
    AdamW optimizer step.

    Parameters
    ----------
    weights : ndarray
        Model parameters.
    gradients : ndarray
        Gradient of loss w.r.t. weights.
    learning_rate : float, optional
        Step size. Default 0.001.
    beta1 : float, optional
        Exponential decay for 1st moment. Default 0.9.
    beta2 : float, optional
        Exponential decay for 2nd moment. Default 0.999.
    epsilon : float, optional
        Numerical stability constant. Default 1e-8.
    weight_decay : float, optional
        L2 regularization strength. Default 0.01.
    m : ndarray, optional
        First moment (mean) buffer.
    v : ndarray, optional
        Second moment (variance) buffer.
    t : int, optional
        Timestep (for bias correction). Default 0.

    Returns
    -------
    dict
        Keys: 'weights', 'm', 'v', 't'.
    """
    weights = np.asarray(weights, dtype=float)
    gradients = np.asarray(gradients, dtype=float)

    if weights.shape != gradients.shape:
        raise ValueError("weights and gradients must have same shape")

    if m is None:
        m = np.zeros_like(weights)
    if v is None:
        v = np.zeros_like(weights)

    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    t = int(t) + 1

    m = beta1 * m + (1 - beta1) * gradients
    v = beta2 * v + (1 - beta2) * (gradients**2)

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    weights = weights - learning_rate * weight_decay * weights

    return RichResult(payload={"weights": weights, "m": m, "v": v, "t": t})
