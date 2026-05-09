r"""Stochastic gradient descent optimizer.

Implements SGD with momentum and Nesterov acceleration.

Parameters
----------
weights : ndarray
    Model parameters, shape (n_params,).
gradients : ndarray
    Gradient of loss w.r.t. weights, shape (n_params,).
learning_rate : float
    Step size. Default 0.01.
momentum : float
    Momentum coefficient, in [0, 1). Default 0.9.
nesterov : bool
    Apply Nesterov acceleration. Default False.
velocity : ndarray, optional
    Accumulated velocity (momentum buffer). Shape (n_params,).
    If None, initialized as zeros.

Returns
-------
updated_weights : ndarray
    Updated parameters after one SGD step.
velocity_out : ndarray
    Updated momentum buffer for next iteration.

Notes
-----
Standard SGD update with momentum::

    v_{t+1} = momentum * v_t - lr * grad
    w_{t+1} = w_t + v_{t+1}

With Nesterov::

    w_{t+1} = w_t + momentum * v_t - lr * grad

References
----------
Ruder, S. (2016). An overview of gradient descent optimization algorithms.
arXiv preprint arXiv:1609.04747.
Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013).
On the importance of initialization and momentum in deep learning.
In ICML (pp. 1139-1147).
"""

__all__ = ["sgdop"]

import numpy as np
from ._richresult import RichResult


def sgdop(
    weights,
    gradients,
    learning_rate=0.01,
    momentum=0.9,
    nesterov=False,
    velocity=None,
):
    """
    Stochastic gradient descent with momentum.

    Parameters
    ----------
    weights : ndarray
        Model parameters.
    gradients : ndarray
        Gradient of loss w.r.t. weights.
    learning_rate : float, optional
        Step size. Default 0.01.
    momentum : float, optional
        Momentum coefficient. Default 0.9.
    nesterov : bool, optional
        Use Nesterov acceleration. Default False.
    velocity : ndarray, optional
        Momentum buffer from previous step.

    Returns
    -------
    dict
        Keys: 'weights' (updated), 'velocity' (momentum buffer).
    """
    weights = np.asarray(weights, dtype=float)
    gradients = np.asarray(gradients, dtype=float)

    if weights.shape != gradients.shape:
        raise ValueError("weights and gradients must have same shape")

    if velocity is None:
        velocity = np.zeros_like(weights)
    else:
        velocity = np.asarray(velocity, dtype=float)

    if not (0.0 <= momentum < 1.0):
        raise ValueError("momentum must be in [0, 1)")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    velocity = momentum * velocity - learning_rate * gradients

    if nesterov:
        weights_new = weights + momentum * velocity - learning_rate * gradients
    else:
        weights_new = weights + velocity

    return RichResult(payload={"weights": weights_new, "velocity": velocity})
