# moirais.fn — function file (hadesllm/moirais)
"""Maximum entropy distribution subject to moment constraints."""

__all__ = ["mxent"]

import numpy as np
from scipy.optimize import minimize


def mxent(
    support: np.ndarray,
    constraints: list,
    *,
    tol: float = 1e-10,
) -> dict:
    """
    Find the maximum entropy distribution on a finite support.

    Solves:

    .. math::

        \\max_p -\\sum_i p_i \\log p_i \\quad
        \\text{s.t.} \\sum_i p_i f_k(x_i) = c_k, \\;
        \\sum_i p_i = 1

    Parameters
    ----------
    support : np.ndarray
        Support points, shape (n,).
    constraints : list of (callable, float)
        Each tuple (f_k, c_k) where f_k maps support -> values and
        c_k is the constraint value: E[f_k(X)] = c_k.

    Returns
    -------
    dict
        'pmf' (np.ndarray, maximum entropy distribution),
        'entropy' (float, bits),
        'lagrange_multipliers' (np.ndarray),
        'constraint_errors' (list of float).

    Raises
    ------
    ValueError
        If support is empty or constraints infeasible.

    References
    ----------
    Jaynes, E. T. (1957). Information theory and statistical mechanics.
    Physical Review, 106(4), 620-630.
    Cover & Thomas (2006). Elements of Information Theory, Ch. 12.
    """
    support = np.asarray(support, dtype=np.float64).ravel()
    n = len(support)
    if n == 0:
        raise ValueError("support must be non-empty.")

    n_constraints = len(constraints)
    f_values = np.zeros((n_constraints, n))
    c_values = np.zeros(n_constraints)
    for k, (fk, ck) in enumerate(constraints):
        f_values[k] = np.array([fk(x) for x in support])
        c_values[k] = ck

    def neg_entropy(lambdas):
        log_unnorm = -f_values.T @ lambdas
        log_z = np.log(np.sum(np.exp(log_unnorm - log_unnorm.max())) + 1e-300) + log_unnorm.max()
        return log_z + lambdas @ c_values

    def neg_entropy_grad(lambdas):
        log_unnorm = -f_values.T @ lambdas
        log_unnorm -= log_unnorm.max()
        p = np.exp(log_unnorm)
        p /= p.sum()
        return c_values - f_values @ p

    lam0 = np.zeros(n_constraints)
    result = minimize(neg_entropy, lam0, jac=neg_entropy_grad,
                      method="L-BFGS-B", options={"maxiter": 1000, "ftol": tol})

    lambdas = result.x
    log_unnorm = -f_values.T @ lambdas
    log_unnorm -= log_unnorm.max()
    pmf = np.exp(log_unnorm)
    pmf /= pmf.sum()

    eps = 1e-300
    entropy = -np.sum(pmf[pmf > eps] * np.log2(pmf[pmf > eps]))

    errors = []
    for k in range(n_constraints):
        errors.append(float(np.sum(pmf * f_values[k]) - c_values[k]))

    return {
        "pmf": pmf,
        "entropy": entropy,
        "lagrange_multipliers": lambdas,
        "constraint_errors": errors,
    }
