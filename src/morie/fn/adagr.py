# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Adagrad optimizer with per-parameter adaptive learning rates.

Accumulates squared gradients for adaptive learning rate scheduling.
"""

import numpy as np

__all__ = ['adagr']


def adagr(f, grad_f, x0, learning_rate=0.01, epsilon=1e-8, max_iter=1000,
          full_output=False, seed=None):
    """
    Adagrad optimizer for unconstrained minimization.

    Adapts learning rate per parameter based on accumulated squared gradients.
    Parameters with large gradients get smaller steps.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    x0 : ndarray
        Initial point.
    learning_rate : float, optional
        Base learning rate (default 0.01).
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
    Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods
    for online learning and stochastic optimization. Journal of Machine
    Learning Research, 12, 2121-2159.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import adagr
    >>> f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    >>> gf = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
    >>> x0 = np.array([0.0, 0.0])
    >>> x_min = adagr(f, gf, x0)
    >>> np.allclose(x_min, [1, 2], atol=1e-3)
    True
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.atleast_1d(x0).astype(float)
    G = np.zeros_like(x)  # Accumulated squared gradients

    for iteration in range(max_iter):
        g = grad_f(x)

        # Accumulate squared gradients
        G += g**2

        # Adaptive update
        x_new = x - learning_rate * g / (np.sqrt(G) + epsilon)

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
