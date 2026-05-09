"""
Trust region method for unconstrained optimization.

Uses a trust radius to guarantee convergence to local minima.
"""

import numpy as np
from scipy.linalg import solve

__all__ = ['trnrg']


def trnrg(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100, radius=1.0, full_output=False):
    """
    Trust region method for unconstrained minimization.

    Uses quadratic model m_k(p) = f(x_k) + g_k^T p + 0.5 p^T H_k p
    within trust region ||p|| <= Delta_k.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    hess_f : callable
        Hessian function hess_f(x) -> matrix.
    x0 : ndarray
        Initial point.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 100).
    radius : float, optional
        Initial trust region radius (default 1.0).
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
    Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
    Springer.

    Examples
    --------
    >>> import numpy as np
    >>> from moirais.fn import trnrg
    >>> f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    >>> gf = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
    >>> hf = lambda x: np.array([[2, 0], [0, 2]])
    >>> x0 = np.array([0.0, 0.0])
    >>> x_min = trnrg(f, gf, hf, x0)
    >>> np.allclose(x_min, [1, 2], atol=1e-4)
    True
    """
    x = np.atleast_1d(x0).astype(float)
    n = len(x)
    Delta = radius
    eta = 0.15

    for iteration in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            if full_output:
                return x, {
                    'iterations': iteration,
                    'converged': True,
                    'final_value': f(x)
                }
            return x

        H = hess_f(x)

        # Solve trust region subproblem via Cauchy point + dogleg
        g_norm = np.linalg.norm(g)
        p_cauchy = -(Delta / g_norm) * g

        # Dogleg: Newton step + line search
        try:
            p_newton = solve(H, -g)
        except np.linalg.LinAlgError:
            p_newton = p_cauchy

        p_norm = np.linalg.norm(p_newton)
        if p_norm <= Delta:
            p = p_newton
        else:
            # Dogleg interpolation
            tau = 1.0
            for _ in range(10):
                p_dog = tau * p_newton
                if np.linalg.norm(p_dog) <= Delta:
                    tau *= 2.0
                else:
                    break
            p = p_dog

        # Ratio of actual to predicted reduction
        x_new = x + p
        f_x = f(x)
        f_new = f(x_new)

        quad_pred = -np.dot(g, p) - 0.5 * np.dot(p, H @ p)
        rho = (f_x - f_new) / (quad_pred + 1e-14)

        if rho > eta:
            x = x_new
            if rho > 0.9:
                Delta = 2.0 * Delta
            elif rho < 0.1:
                Delta *= 0.25

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f(x)
        }
    return x
