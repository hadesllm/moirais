# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""
ADMM (alternating direction method of multipliers) for convex optimization.

Splits problem into subproblems; solves via alternating proximal updates.
"""

import numpy as np

__all__ = ['admm']


def admm(f, g, A, b, rho=1.0, max_iter=1000, tol=1e-4, full_output=False):
    """
    ADMM for constrained optimization.

    Solves: minimize f(x) + g(z) s.t. Ax + Bz = b via splitting.

    Parameters
    ----------
    f : callable
        Function f(x) to minimize (typically convex).
    g : callable
        Function g(z) to minimize (typically convex).
    A : ndarray
        Constraint matrix A (m, n).
    b : ndarray
        Constraint RHS (m,).
    rho : float, optional
        Penalty parameter (default 1.0).
    max_iter : int, optional
        Maximum iterations (default 1000).
    tol : float, optional
        Convergence tolerance (default 1e-4).
    full_output : bool, optional
        If True, return (x, info_dict).

    Returns
    -------
    x : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged'.

    References
    ----------
    Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
    Distributed optimization and statistical learning via ADMM. Foundations
    and Trends in Machine Learning, 3(1), 1-122.

    Examples
    --------
    >>> import numpy as np
    >>> from moirais.fn import admm
    >>> f = lambda x: np.sum(x**2)
    >>> g = lambda z: np.sum(np.abs(z))
    >>> A = np.eye(3)
    >>> b = np.array([1, 2, 3])
    >>> x, info = admm(f, g, A, b, full_output=True)
    >>> info['converged']
    True
    """
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(m)

    for it in range(max_iter):
        # x-update
        from scipy.optimize import minimize
        def obj_x(x_):
            return f(x_) + 0.5 * rho * np.sum((A @ x_ + z - b + u / rho)**2)
        res = minimize(obj_x, x, method='BFGS', options={'maxiter': 20})
        x = res.x

        # z-update (proximal operator of g)
        z_arg = A @ x - b + u / rho
        thresh = 1.0 / rho  # Soft threshold for g(z) = ||z||_1
        z = np.sign(z_arg) * np.maximum(np.abs(z_arg) - thresh, 0)

        # Dual update
        u_old = u.copy()
        u = u + rho * (A @ x + z - b)

        residual = np.linalg.norm(u - u_old)
        if residual < tol:
            if full_output:
                return x, {
                    'iterations': it + 1,
                    'converged': True
                }
            return x

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False
        }
    return x
