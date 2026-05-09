# moirais.fn — function file (hadesllm/moirais)
"""
Expectation-Maximization (EM) algorithm for mixture models.

General-purpose iterative algorithm for maximum likelihood with latent variables.
"""

import numpy as np
from scipy.stats import multivariate_normal

__all__ = ['emfit']


def emfit(X, n_components=2, max_iter=100, tol=1e-6, seed=None, full_output=False):
    """
    EM algorithm for Gaussian mixture models.

    Fits mixture of n_components Gaussian distributions to data X.

    Parameters
    ----------
    X : ndarray
        Data matrix (n_samples, n_features).
    n_components : int, optional
        Number of mixture components (default 2).
    max_iter : int, optional
        Maximum EM iterations (default 100).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    seed : int, optional
        Random seed.
    full_output : bool, optional
        If True, return (params_dict, info_dict).

    Returns
    -------
    params_dict : dict
        Dictionary with keys: 'means', 'covars', 'weights', 'llh' (log-likelihood).
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged'.

    References
    ----------
    Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood
    from incomplete data via the EM algorithm. Journal of the Royal Statistical
    Society, 39(1), 1-38.

    Examples
    --------
    >>> import numpy as np
    >>> from moirais.fn import emfit
    >>> X = np.random.randn(100, 2)
    >>> result = emfit(X, n_components=2, seed=42)
    >>> isinstance(result[0]['means'], np.ndarray)
    True
    """
    if seed is not None:
        np.random.seed(seed)

    n, d = X.shape
    X = X.astype(float)

    # Initialize
    idx = np.random.choice(n, n_components, replace=False)
    means = X[idx].copy()
    covars = np.array([np.eye(d) for _ in range(n_components)])
    weights = np.ones(n_components) / n_components
    resp = np.ones((n, n_components)) / n_components

    llh_prev = -np.inf

    for it in range(max_iter):
        # E-step: compute responsibilities
        for k in range(n_components):
            mvn = multivariate_normal(mean=means[k], cov=covars[k] + 1e-6 * np.eye(d))
            resp[:, k] = weights[k] * mvn.pdf(X)
        resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-10)

        # M-step: update parameters
        Nk = resp.sum(axis=0)
        weights = Nk / n

        for k in range(n_components):
            means[k] = np.dot(resp[:, k], X) / (Nk[k] + 1e-10)
            diff = X - means[k]
            cov_k = np.zeros((d, d))
            for i in range(n):
                cov_k += resp[i, k] * np.outer(diff[i], diff[i])
            covars[k] = cov_k / (Nk[k] + 1e-10)

        # Log-likelihood
        llh = 0.0
        for k in range(n_components):
            mvn = multivariate_normal(mean=means[k], cov=covars[k] + 1e-6 * np.eye(d))
            llh += np.sum(resp[:, k] * np.log(weights[k] * mvn.pdf(X) + 1e-10))

        if np.abs(llh - llh_prev) < tol:
            params = {'means': means, 'covars': covars, 'weights': weights, 'llh': llh}
            if full_output:
                return params, {
                    'iterations': it + 1,
                    'converged': True
                }
            return params, {'iterations': it + 1, 'converged': True}

        llh_prev = llh

    params = {'means': means, 'covars': covars, 'weights': weights, 'llh': llh}
    if full_output:
        return params, {
            'iterations': max_iter,
            'converged': False
        }
    return params, {'iterations': max_iter, 'converged': False}
