# morie.fn — function file (hadesllm/morie)
"""
Nelder-Mead simplex optimization.

Derivative-free optimization using a dynamic simplex in n dimensions.
"""

import numpy as np

__all__ = ['nmead']


def nmead(
    f,
    x0,
    alpha=1.0,
    beta=0.5,
    gamma=2.0,
    tol=1e-8,
    max_iter=5000,
    full_output=False
):
    """
    Nelder-Mead simplex method for unconstrained minimization.

    Maintains a simplex of n+1 points in n dimensions. Iteratively reflects,
    expands, contracts, or shrinks the simplex based on function values.

    Parameters
    ----------
    f : callable
        Objective function f(x) to minimize.
    x0 : ndarray
        Initial point (shape (n,)).
    alpha : float, optional
        Reflection coefficient (default 1.0).
    beta : float, optional
        Contraction coefficient (default 0.5).
    gamma : float, optional
        Expansion coefficient (default 2.0).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 1000).
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
    Nelder, J. A., & Mead, R. (1965). A simplex method for function
    minimization. The Computer Journal, 7(4), 308-313.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import nmead
    >>> f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
    >>> x0 = np.array([0.0, 0.0])
    >>> x_min = nmead(f, x0)
    >>> np.allclose(x_min, [2, 3], atol=1e-4)
    True
    """
    x0 = np.atleast_1d(x0).astype(float)
    n = len(x0)

    scale = max(float(np.max(np.abs(x0))), 1.0)
    simplex = np.vstack([x0, x0[np.newaxis, :] + scale * np.eye(n)])
    f_vals = np.array([f(point) for point in simplex])

    for iteration in range(max_iter):
        # Sort by function value
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        variance = np.var(f_vals)
        diameter = float(np.max(np.linalg.norm(simplex - simplex[0], axis=1)))
        if variance < tol and diameter < tol:
            if full_output:
                return simplex[0], {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_value': f_vals[0]
                }
            return simplex[0]

        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_refl = centroid + alpha * (centroid - simplex[-1])
        f_refl = f(x_refl)

        if f_refl < f_vals[0]:
            # Expansion
            x_exp = centroid + gamma * (x_refl - centroid)
            f_exp = f(x_exp)
            if f_exp < f_refl:
                simplex[-1] = x_exp
                f_vals[-1] = f_exp
            else:
                simplex[-1] = x_refl
                f_vals[-1] = f_refl
        elif f_refl < f_vals[-2]:
            simplex[-1] = x_refl
            f_vals[-1] = f_refl
        else:
            # Contraction
            x_contr = centroid + beta * (simplex[-1] - centroid)
            f_contr = f(x_contr)
            if f_contr < f_vals[-1]:
                simplex[-1] = x_contr
                f_vals[-1] = f_contr
            else:
                # Shrink
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                f_vals[1:] = np.array([f(p) for p in simplex[1:]])

    if full_output:
        best_idx = np.argmin(f_vals)
        return simplex[best_idx], {
            'iterations': max_iter,
            'converged': False,
            'final_value': f_vals[best_idx]
        }
    return simplex[np.argmin(f_vals)]
