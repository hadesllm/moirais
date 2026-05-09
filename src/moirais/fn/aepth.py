# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""AEP (asymptotic equipartition property) verification."""

__all__ = ["aepth"]

import numpy as np


def aepth(
    pmf: np.ndarray,
    n: int,
    *,
    n_samples: int = 10000,
    epsilon: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Verify the AEP via Monte Carlo simulation.

    The AEP states that for i.i.d. X_1, ..., X_n ~ p(x):

    .. math::

        -\\frac{1}{n} \\log_2 p(X_1, \\ldots, X_n) \\to H(X)
        \\quad \\text{in probability}

    Parameters
    ----------
    pmf : np.ndarray
        Source PMF, shape (k,). Must sum to 1.
    n : int
        Sequence length.
    n_samples : int
        Number of Monte Carlo samples.
    epsilon : float
        Typicality tolerance.
    seed : int
        Random seed.

    Returns
    -------
    dict
        'entropy' (H(X), bits),
        'mean_neg_log_rate' (mean of -1/n log2 p(x^n)),
        'std_neg_log_rate' (std of the above),
        'typical_fraction' (fraction of samples in typical set),
        'n_samples', 'n', 'epsilon'.

    References
    ----------
    Cover & Thomas (2006). Elements of Information Theory, Theorem 3.1.2.
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    if pmf.ndim != 1 or not np.isclose(pmf.sum(), 1.0):
        raise ValueError("pmf must be a 1-D array summing to 1.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    eps_c = 1e-300
    h = -np.sum(pmf[pmf > eps_c] * np.log2(pmf[pmf > eps_c]))

    rng = np.random.default_rng(seed)
    k = len(pmf)

    samples = rng.choice(k, size=(n_samples, n), p=pmf)

    log2_pmf = np.log2(pmf + eps_c)
    log_probs = log2_pmf[samples].sum(axis=1)
    neg_log_rate = -log_probs / n

    mean_rate = float(np.mean(neg_log_rate))
    std_rate = float(np.std(neg_log_rate, ddof=1))

    typical_mask = np.abs(neg_log_rate - h) <= epsilon
    typical_frac = float(np.mean(typical_mask))

    return {
        "entropy": h,
        "mean_neg_log_rate": mean_rate,
        "std_neg_log_rate": std_rate,
        "typical_fraction": typical_frac,
        "n_samples": n_samples,
        "n": n,
        "epsilon": epsilon,
    }
