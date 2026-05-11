"""
Stochastic gradient Langevin dynamics for Bayesian sampling.

Adds noise to SGD for posterior sampling without convergence to point estimate.
"""

import numpy as np

__all__ = ['sgld']


def sgld(log_likelihood, log_prior, X, y, param_shape, learning_rate=0.01,
         friction=0.01, n_iter=1000, batch_size=32, seed=None, full_output=False):
    """
    Stochastic gradient Langevin dynamics (SGLD) for Bayesian inference.

    Samples from posterior via SGD with added Langevin noise.

    Parameters
    ----------
    log_likelihood : callable
        Function log_likelihood(params, X_batch, y_batch) -> log p(y|X, params).
    log_prior : callable
        Function log_prior(params) -> log p(params).
    X : ndarray
        Feature matrix (n_samples, n_features).
    y : ndarray
        Target vector (n_samples,).
    param_shape : tuple
        Shape of parameter array.
    learning_rate : float, optional
        Learning rate / step size (default 0.01).
    friction : float, optional
        Friction coefficient (default 0.01).
    n_iter : int, optional
        Number of SGLD iterations (default 1000).
    batch_size : int, optional
        Mini-batch size (default 32).
    seed : int, optional
        Random seed.
    full_output : bool, optional
        If True, return (samples, info_dict).

    Returns
    -------
    samples : ndarray
        Posterior samples (n_iter, *param_shape).
    info_dict : dict, optional
        Dictionary with keys: 'n_iterations'.

    References
    ----------
    Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic
    gradient Langevin dynamics. In ICML (pp. 681-688).

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import sgld
    >>> ll = lambda p, X, y: -0.5*np.sum((y - X @ p)**2)
    >>> lp = lambda p: -0.5*np.sum(p**2)
    >>> X = np.random.randn(100, 2)
    >>> y = np.random.randn(100)
    >>> samples = sgld(ll, lp, X, y, (2,), n_iter=100, seed=42)
    >>> samples.shape == (100, 2)
    True
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = X.shape[0]
    params = np.zeros(param_shape)
    samples = np.zeros((n_iter,) + param_shape)

    for it in range(n_iter):
        # Mini-batch
        idx = np.random.choice(n_samples, batch_size, replace=True)
        X_batch = X[idx]
        y_batch = y[idx]

        # Gradient of log likelihood + log prior (scaled by minibatch)
        def grad_log_post(p):
            grad_ll = np.zeros(param_shape)
            # Numerical gradient (simple approach; could use autodiff)
            eps = 1e-5
            for i in np.ndindex(param_shape):
                p_eps = p.copy()
                p_eps[i] += eps
                grad_ll[i] = (log_likelihood(p_eps, X_batch, y_batch) -
                             log_likelihood(p, X_batch, y_batch)) / eps

            grad_lp = np.zeros(param_shape)
            for i in np.ndindex(param_shape):
                p_eps = p.copy()
                p_eps[i] += eps
                grad_lp[i] = (log_prior(p_eps) - log_prior(p)) / eps

            return (n_samples / batch_size) * grad_ll + grad_lp

        g = grad_log_post(params)

        # Langevin noise
        noise = np.random.normal(0, np.sqrt(2 * learning_rate * friction), param_shape)

        # Update
        params = params + learning_rate * g + noise

        samples[it] = params.copy()

    if full_output:
        return samples, {
            'n_iterations': n_iter
        }
    return samples
