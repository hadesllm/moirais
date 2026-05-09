# moirais.fn — function file (hadesllm/moirais)
"""
RMSprop optimizer for stochastic optimization.

Adaptive learning rate method with exponential moving average of squared gradients.
"""

import numpy as np

__all__ = ['rmspf']


def rmspf(f, grad_f, x0, learning_rate=0.01, decay=0.9, epsilon=1e-8, max_iter=1000,
          full_output=False, seed=None):
    """
    RMSprop optimizer for unconstrained minimization.

    Maintains exponential moving average of squared gradients;
    adapts per-parameter learning rates.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    x0 : ndarray
        Initial point.
    learning_rate : float, optional
        Initial learning rate (default 0.01).
    decay : float, optional
        Decay rate for squared gradient moving average (default 0.9).
    epsilon : float, optional
        Numerical stability term (default 1e-8).
    max_iter : int, optional
        Maximum iterations (default 1000).
    full_output : bool, optional
        If True, return (x_min, info_dict).
    seed : int, optional
        Random seed.

    Returns
    -------
    x_min : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'final_value'.

    References
    ----------
    Tieleman, T., & Hinton, G. (2012). Lecture 6.5 - RMSprop: Divide the
    gradient by a running average of its recent magnitude. COURSERA: Neural
    Networks for Machine Learning.

    Examples
    --------
    >>> import numpy as np
    >>> from moirais.fn import rmspf
    >>> f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    >>> gf = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
    >>> x0 = np.array([0.0, 0.0])
    >>> x_min = rmspf(f, gf, x0)
    >>> np.allclose(x_min, [1, 2], atol=1e-3)
    True
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.atleast_1d(x0).astype(float)
    v = np.zeros_like(x)  # Moving average of squared gradients

    for iteration in range(max_iter):
        g = grad_f(x)

        # Update moving average of squared gradients
        v = decay * v + (1 - decay) * g**2

        # Adaptive update
        x_new = x - learning_rate * g / (np.sqrt(v) + epsilon)

        residual = np.linalg.norm(x_new - x)
        if residual < 1e-6:
            if full_output:
                return x_new, {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_value': f(x_new)
                }
            return x_new

        x = x_new

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f(x)
        }
    return x
