"""First and second moments of partitioning-set probabilities under a Polya tree process with parameters alpha_epsilon.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_polya_tree_first_two_moments"]


def ghosal_ch3_polya_tree_first_two_moments(alpha_epsilon, epsilon, m):
    """
    First and second moments of partitioning-set probabilities under a Polya tree process with parameters alpha_epsilon.

    Formula: E( P(A_{epsilon_1 ... epsilon_m}) ) = prod_{j=1}^{m} alpha_{epsilon_1 ... epsilon_j} / ( alpha_{epsilon_1 ... epsilon_{j-1} 0} + alpha_{epsilon_1 ... epsilon_{j-1} 1} );   E( P(A_{epsilon_1 ... epsilon_m})^2 ) = prod_{j=1}^{m} alpha_{epsilon_1 ... epsilon_j} ( alpha_{epsilon_1 ... epsilon_j} + 1 ) / ( ( alpha_{epsilon_1 ... epsilon_{j-1} 0} + alpha_{epsilon_1 ... epsilon_{j-1} 1} ) ( alpha_{epsilon_1 ... epsilon_{j-1} 0} + alpha_{epsilon_1 ... epsilon_{j-1} 1} + 1 ) )

    Parameters
    ----------
    alpha_epsilon : array-like
        Input data.
    epsilon : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.21, p. 49
    """
    alpha_epsilon = np.atleast_1d(np.asarray(alpha_epsilon, dtype=float))
    n = len(alpha_epsilon)
    result = float(np.mean(alpha_epsilon))
    se = float(np.std(alpha_epsilon, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "First and second moments of partitioning-set probabilities under a Polya tree process with parameters alpha_epsilon."})


def cheatsheet():
    return "ghs028: First and second moments of partitioning-set probabilities under a Polya tree process with parameters alpha_epsilon."
