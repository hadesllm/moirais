# morie.fn — function file (hadesllm/morie)
"""
Newton-Raphson root finding.

Solves f(x) = 0 using Newton's method with derivative information.
"""

import numpy as np

__all__ = ['nwtrp']


def nwtrp(
    f,
    fprime,
    x0,
    tol=1e-6,
    max_iter=100,
    full_output=False
):
    """
    Newton-Raphson root finding method.

    Iteratively refines root estimate via x_{n+1} = x_n - f(x_n) / f'(x_n).

    Parameters
    ----------
    f : callable
        Function f(x).
    fprime : callable
        Derivative f'(x).
    x0 : float or ndarray
        Initial guess.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 100).
    full_output : bool, optional
        If True, return (root, info_dict). If False, return root only.

    Returns
    -------
    root : float or ndarray
        Estimated root.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'final_residual'.

    References
    ----------
    Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.).
    Cengage Learning.

    Examples
    --------
    >>> # Solve x^2 - 2 = 0 (root = sqrt(2))
    >>> import numpy as np
    >>> from morie.fn import nwtrp
    >>> f = lambda x: x**2 - 2
    >>> fp = lambda x: 2*x
    >>> root = nwtrp(f, fp, 1.5)
    >>> np.isclose(root, np.sqrt(2))
    True
    """
    x = np.atleast_1d(x0).astype(float)
    for iteration in range(max_iter):
        fx = np.atleast_1d(f(x))
        fpx = np.asarray(fprime(x))

        # Dispatch: scalar/elementwise vs full Jacobian (multivariate Newton).
        if fpx.ndim == 2 and fpx.shape[0] == fpx.shape[1] == fx.size:
            try:
                step = np.linalg.solve(fpx, fx)
            except np.linalg.LinAlgError:
                if full_output:
                    return x, {
                        'iterations': iteration,
                        'converged': False,
                        'final_residual': float(np.linalg.norm(fx)),
                    }
                return x
        else:
            if np.any(np.abs(fpx) < 1e-14):
                if full_output:
                    return x, {
                        'iterations': iteration,
                        'converged': False,
                        'final_residual': float(np.linalg.norm(fx)),
                    }
                return x
            step = fx / fpx

        x_new = x - step
        residual = np.linalg.norm(x_new - x)

        if residual < tol:
            if full_output:
                return x_new, {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_residual': float(np.linalg.norm(np.atleast_1d(f(x_new)))),
                }
            return x_new

        x = x_new

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False,
            'final_residual': float(np.linalg.norm(np.atleast_1d(f(x)))),
        }
    return x
