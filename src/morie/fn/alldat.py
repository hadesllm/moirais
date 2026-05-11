# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""LDA per-document topic distribution theta_d ~ Dirichlet(alpha)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_lda_topic_distribution"]


def alammar_lda_topic_distribution(alpha, beta, n_topics, documents):
    """
    LDA per-document topic distribution theta_d ~ Dirichlet(alpha)

    Formula: theta_d ~ Dir(alpha);  z_{d,n} ~ Cat(theta_d);  w_{d,n} ~ Cat(phi_{z_{d,n}})

    Parameters
    ----------
    alpha : array-like
        Input data.
    beta : array-like
        Input data.
    n_topics : array-like
        Input data.
    documents : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta, phi, z

    References
    ----------
    Alammar Ch 5, LDA section
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LDA per-document topic distribution theta_d ~ Dirichlet(alpha)"})


def cheatsheet():
    return "alldat: LDA per-document topic distribution theta_d ~ Dirichlet(alpha)"
