# moirais.fn — function file (hadesllm/moirais)
"""Fisher information matrix."""

__all__ = ["finfo"]

import numpy as np


def finfo(
    log_likelihood_grad: callable,
    theta: np.ndarray,
    *,
    n_samples: int = 10000,
    data_generator: callable = None,
    seed: int = 42,
    h: float = 1e-5,
) -> dict:
    """
    Estimate the Fisher information matrix via Monte Carlo or numerical diff.

    .. math::

        \\mathcal{I}(\\theta)_{ij} = E\\left[
        \\frac{\\partial \\log f}{\\partial \\theta_i}
        \\frac{\\partial \\log f}{\\partial \\theta_j}
        \\right]

    Parameters
    ----------
    log_likelihood_grad : callable
        Function(x, theta) -> np.ndarray of shape (p,) returning the
        score vector (gradient of log-likelihood w.r.t. theta).
    theta : np.ndarray
        Parameter vector, shape (p,).
    n_samples : int
        Number of Monte Carlo samples.
    data_generator : callable
        Function(n, theta, rng) -> np.ndarray generating n samples from
        the model at parameter theta. Required for Monte Carlo estimation.
    seed : int
        Random seed.
    h : float
        Step size for numerical differentiation (unused if data_generator given).

    Returns
    -------
    dict
        'fisher_info' (np.ndarray, p x p Fisher information matrix),
        'crlb' (np.ndarray, diagonal of inverse = Cramer-Rao lower bounds),
        'n_samples'.

    Raises
    ------
    ValueError
        If theta is not 1-D or data_generator is None.

    References
    ----------
    Fisher, R. A. (1925). Theory of statistical estimation. Proc. Cambridge
    Phil. Soc., 22, 700-725.
    Lehmann, E. L. & Casella, G. (1998). Theory of Point Estimation, Ch. 2.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    p = len(theta)

    if data_generator is None:
        raise ValueError("data_generator is required for Fisher information estimation.")

    rng = np.random.default_rng(seed)
    samples = data_generator(n_samples, theta, rng)

    fim = np.zeros((p, p))
    for i in range(n_samples):
        x_i = samples[i] if samples.ndim > 1 else samples[i:i+1]
        score = log_likelihood_grad(x_i, theta)
        score = np.asarray(score, dtype=np.float64).ravel()
        fim += np.outer(score, score)
    fim /= n_samples

    try:
        fim_inv = np.linalg.inv(fim)
        crlb = np.diag(fim_inv)
    except np.linalg.LinAlgError:
        crlb = np.full(p, np.inf)

    return {
        "fisher_info": fim,
        "crlb": crlb,
        "n_samples": n_samples,
    }
