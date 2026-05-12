# morie.fn — function file (hadesllm/morie)
"""Tangent-space dimension for parametric submodels (Kosorok 2008, Ch 6).

The tangent space T at P is the closure of {score functions s along
all smooth one-parameter paths through P}.  Its empirical analogue is
the rank of the centred-score covariance matrix.  We use the basis
of moment scores {x - mean(x), x^2 - mean(x^2)} and return its rank.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_tangent_space"]


def kosorok_tangent_space(x):
    """Empirical tangent-space dimension via moment-score rank.

    Parameters
    ----------
    x : array-like.

    Returns
    -------
    RichResult with keys estimate (rank), n, method.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    s1 = x - x.mean()
    s2 = x ** 2 - (x ** 2).mean()
    S = np.column_stack([s1, s2])
    G = (S.T @ S) / n
    rank = int(np.linalg.matrix_rank(G, tol=1e-10))
    return RichResult(payload={
        "estimate": rank,
        "n":        n,
        "method":   "Tangent-space dim via empirical-score rank",
    })


def cheatsheet():
    return "ksr13: tangent-space dimension via empirical-score rank"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(kosorok_tangent_space(rng.normal(size=200)))
