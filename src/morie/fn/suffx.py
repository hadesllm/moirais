"""Sufficient statistic test via Fisher-Neyman factorization."""

__all__ = ["suffx"]

import numpy as np


def suffx(
    data: np.ndarray,
    statistic: callable,
    log_likelihood: callable,
    theta_grid: np.ndarray,
    *,
    n_bins: int = 20,
) -> dict:
    """
    Test whether T(X) is a sufficient statistic for theta via factorization.

    The Fisher-Neyman factorization theorem states T(X) is sufficient iff:

    .. math::

        f(x|\\theta) = g(T(x), \\theta) \\cdot h(x)

    This function checks numerically whether the likelihood ratio
    f(x|theta)/f(x'|theta) depends only on T(x) vs T(x').

    Parameters
    ----------
    data : np.ndarray
        Observed data, shape (n,) or (n, d).
    statistic : callable
        Function(x) -> scalar or array. The candidate sufficient statistic.
    log_likelihood : callable
        Function(x, theta) -> float. Log-likelihood of observation x at theta.
    theta_grid : np.ndarray
        Grid of parameter values to test over.

    Returns
    -------
    dict
        'is_sufficient' (bool, True if factorization holds approximately),
        'max_violation' (float, maximum deviation from factorization),
        'mean_violation' (float),
        'n_comparisons' (int).

    References
    ----------
    Fisher, R. A. (1922). On the mathematical foundations of theoretical
    statistics. Phil. Trans. R. Soc. A, 222, 309-368.
    Lehmann & Casella (1998). Theory of Point Estimation, Theorem 1.6.5.
    """
    data = np.asarray(data, dtype=np.float64)
    theta_grid = np.asarray(theta_grid, dtype=np.float64).ravel()

    if data.ndim == 1:
        n = len(data)
        get_x = lambda i: data[i]
    else:
        n = data.shape[0]
        get_x = lambda i: data[i]

    t_values = np.array([statistic(get_x(i)) for i in range(n)])

    sorted_idx = np.argsort(t_values)
    t_sorted = t_values[sorted_idx]

    max_violation = 0.0
    total_violation = 0.0
    n_comp = 0

    bin_edges = np.linspace(t_sorted.min() - 1e-10, t_sorted.max() + 1e-10, n_bins + 1)

    for b in range(n_bins):
        mask = (t_sorted >= bin_edges[b]) & (t_sorted < bin_edges[b + 1])
        bin_indices = sorted_idx[mask]
        if len(bin_indices) < 2:
            continue

        for ii in range(min(len(bin_indices), 5)):
            for jj in range(ii + 1, min(len(bin_indices), 5)):
                i_idx = bin_indices[ii]
                j_idx = bin_indices[jj]
                xi = get_x(i_idx)
                xj = get_x(j_idx)

                for theta in theta_grid[:10]:
                    ll_i = log_likelihood(xi, theta)
                    ll_j = log_likelihood(xj, theta)
                    ratio = ll_i - ll_j

                for t1_idx, theta1 in enumerate(theta_grid[:10]):
                    ll_i_1 = log_likelihood(xi, theta1)
                    ll_j_1 = log_likelihood(xj, theta1)
                    r1 = ll_i_1 - ll_j_1
                    for t2_idx in range(t1_idx + 1, min(len(theta_grid), 10)):
                        theta2 = theta_grid[t2_idx]
                        ll_i_2 = log_likelihood(xi, theta2)
                        ll_j_2 = log_likelihood(xj, theta2)
                        r2 = ll_i_2 - ll_j_2
                        violation = abs(r1 - r2)
                        max_violation = max(max_violation, violation)
                        total_violation += violation
                        n_comp += 1

    mean_violation = total_violation / n_comp if n_comp > 0 else 0.0
    is_sufficient = max_violation < 0.01

    return {
        "is_sufficient": is_sufficient,
        "max_violation": max_violation,
        "mean_violation": mean_violation,
        "n_comparisons": n_comp,
    }
