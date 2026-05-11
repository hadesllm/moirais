# morie.fn — function file (hadesllm/morie)
"""
Iteratively reweighted least squares (IRLS) for GLM fitting.

Solves GLM via weighted least squares with iteratively updated weights.
"""

import numpy as np

__all__ = ['irlsf']


def irlsf(X, y, family='gaussian', max_iter=100, tol=1e-6, full_output=False):
    """
    IRLS (iteratively reweighted least squares) for GLM fitting.

    Solves generalized linear model via iterated weighted LS.

    Parameters
    ----------
    X : ndarray
        Design matrix (n_samples, n_features). First column should be 1 for intercept.
    y : ndarray
        Response vector (n_samples,).
    family : str, optional
        GLM family: 'gaussian', 'binomial', 'poisson' (default 'gaussian').
    max_iter : int, optional
        Maximum IRLS iterations (default 100).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    full_output : bool, optional
        If True, return (beta, info_dict).

    Returns
    -------
    beta : ndarray
        Fitted coefficients.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'deviance'.

    References
    ----------
    Green, P. J. (1984). Iteratively reweighted least squares for maximum
    likelihood estimation. Journal of the Royal Statistical Society, 46, 149-192.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import irlsf
    >>> X = np.column_stack([np.ones(20), np.linspace(0, 1, 20)])
    >>> y = 1 + 2*X[:, 1] + np.random.randn(20)*0.1
    >>> beta = irlsf(X, y)
    >>> np.allclose(beta, [1, 2], atol=0.5)
    True
    """
    X = np.atleast_2d(X).astype(float)
    y = np.atleast_1d(y).astype(float)

    n, p = X.shape
    beta = np.zeros(p)

    for it in range(max_iter):
        eta = X @ beta
        mu = eta.copy()

        if family == 'gaussian':
            g_mu = mu
            g_prime = np.ones(n)
            w = np.ones(n)
        elif family == 'binomial':
            mu = 1.0 / (1.0 + np.exp(-eta))
            g_mu = np.log(mu / (1 - mu + 1e-10))
            g_prime = 1.0 / (mu * (1 - mu) + 1e-10)
            w = mu * (1 - mu)
        elif family == 'poisson':
            mu = np.exp(eta)
            g_mu = np.log(mu + 1e-10)
            g_prime = 1.0 / (mu + 1e-10)
            w = mu
        else:
            raise ValueError(f"Unknown family: {family}")

        # Adjusted response
        z = eta + (y - mu) / (g_prime + 1e-10)

        # Weighted LS: solve (X^T W X) beta = X^T W z
        W = np.diag(np.sqrt(np.maximum(w, 1e-10)))
        Xw = W @ X
        zw = W @ z

        try:
            beta_new = np.linalg.lstsq(Xw, zw, rcond=None)[0]
        except:
            beta_new = beta

        residual = np.linalg.norm(beta_new - beta)
        beta = beta_new

        if residual < tol:
            if full_output:
                eta = X @ beta
                if family == 'gaussian':
                    dev = np.sum((y - eta)**2)
                elif family == 'binomial':
                    mu = 1.0 / (1.0 + np.exp(-eta))
                    dev = 2 * np.sum(y * np.log(mu + 1e-10) + (1 - y) * np.log(1 - mu + 1e-10))
                elif family == 'poisson':
                    mu = np.exp(eta)
                    dev = 2 * np.sum(y * np.log(y / (mu + 1e-10) + 1e-10) - (y - mu))
                return beta, {
                    'iterations': it + 1,
                    'converged': True,
                    'deviance': dev
                }
            return beta

    if full_output:
        eta = X @ beta
        if family == 'gaussian':
            dev = np.sum((y - eta)**2)
        return beta, {
            'iterations': max_iter,
            'converged': False,
            'deviance': dev
        }
    return beta
