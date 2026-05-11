# morie.fn — function file (hadesllm/morie)
"""
Interior point (barrier) method for constrained optimization.

Solves constrained optimization via logarithmic barrier penalties.
"""

import numpy as np
from scipy.optimize import minimize

__all__ = ['ipopt']


def ipopt(f, grad_f, constraints, x0, tol=1e-6, max_iter=100, mu_init=1.0, full_output=False):
    """
    Interior point barrier method for constrained optimization.

    Solves: minimize f(x) s.t. c_i(x) >= 0 via logarithmic barrier:
    minimize f(x) - mu * sum(log(c_i(x))).

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient of f(x).
    constraints : list of dict
        Each dict has 'type': 'ineq', 'fun': c_i(x) callable, 'jac': grad c_i(x).
    x0 : ndarray
        Initial feasible point.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Barrier iterations (default 100).
    mu_init : float, optional
        Initial barrier parameter (default 1.0).
    full_output : bool, optional
        If True, return (x_min, info_dict).

    Returns
    -------
    x_min : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'final_value'.

    References
    ----------
    Fiacco, A. V., & McCormick, G. P. (1968). Nonlinear Programming:
    Sequential Unconstrained Minimization Techniques. Wiley.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import ipopt
    >>> f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    >>> gf = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
    >>> c1 = {'type': 'ineq', 'fun': lambda x: x[0]}
    >>> c2 = {'type': 'ineq', 'fun': lambda x: x[1]}
    >>> x0 = np.array([0.5, 0.5])
    >>> x_min = ipopt(f, gf, [c1, c2], x0)
    >>> x_min[0] >= -1e-4 and x_min[1] >= -1e-4
    True
    """
    x = np.atleast_1d(x0).astype(float)
    mu = mu_init
    rho = 0.1

    for iteration in range(max_iter):
        def barrier_f(x_):
            # f(x) - mu * sum(log(c_i(x)))
            barrier_val = f(x_)
            for c_dict in constraints:
                c_val = c_dict['fun'](x_)
                if c_val <= 0:
                    return 1e10
                barrier_val -= mu * np.log(c_val)
            return barrier_val

        def barrier_grad(x_):
            # grad f - mu * sum(grad(log(c_i)))
            g = grad_f(x_)
            for c_dict in constraints:
                c_val = c_dict['fun'](x_)
                jac = c_dict.get('jac', lambda x: np.gradient(c_dict['fun'](x)))
                g -= mu * jac(x_) / (c_val + 1e-14)
            return g

        # Minimize barrier subproblem
        result = minimize(barrier_f, x, method='BFGS', jac=barrier_grad, options={'maxiter': 50})
        x = result.x

        if mu < tol:
            if full_output:
                return x, {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_value': f(x)
                }
            return x

        mu *= rho

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f(x)
        }
    return x
